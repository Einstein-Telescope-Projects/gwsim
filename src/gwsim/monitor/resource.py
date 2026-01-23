"""Module for monitoring resource usage during code execution."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager, suppress
from datetime import timedelta
from pathlib import Path
from threading import Event, Thread

import psutil

from gwsim.utils.io import atomic_writer


class ResourceMonitor:
    """Class to monitor resource usage during code execution."""

    def __init__(self, sample_interval: float = 1.0):
        """Initialize the resource monitor.

        Args:
            sample_interval: How often (in seconds) to sample memory usage.
                Defaults to 1.0 second.
        """
        if sample_interval <= 0:
            raise ValueError("sample_interval must be > 0")

        self.metrics: dict[str, float | str | dict] = {}
        self.sample_interval = sample_interval
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._cgroup_samples: list[float] = []
        self._stop_event = Event()

    def _get_all_processes(self, parent: psutil.Process) -> list[psutil.Process]:
        """Get the parent process and all its descendant processes that are still running.

        Args:
            parent: The parent psutil.Process to start from.

        Returns:
            List of running psutil.Process objects (parent + children recursively).
        """
        processes = [parent]
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            processes.extend(parent.children(recursive=True))
        return [p for p in processes if p.is_running()]

    def _aggregate_metric(self, processes: list[psutil.Process], attr: str, sub_attr: str | None = None) -> float:
        """Aggregate a metric across multiple processes.

        Args:
            processes: List of psutil.Process objects.
            attr: Name of the psutil method to call (e.g., 'memory_info', 'cpu_times').
            sub_attr: Optional sub-attribute to extract (e.g., 'rss' for memory_info).

        Returns:
            Sum of the requested metric across all processes (0.0 on errors).
        """
        total = 0.0
        for p in processes:
            try:
                val = getattr(p, attr)()
                total += getattr(val, sub_attr) if sub_attr else val
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                pass
        return total

    def _monitor_loop(self, parent: psutil.Process):
        """Background thread that periodically samples memory usage.

        Args:
            parent: The parent process to monitor.
        """
        # Take immediate sample before waiting
        processes = self._get_all_processes(parent)
        mem_rss = self._aggregate_metric(processes, "memory_info", "rss")
        self._samples["memory_gb"].append(mem_rss / (1024**3))

        while not self._stop_event.wait(self.sample_interval):
            processes = self._get_all_processes(parent)
            mem_rss = self._aggregate_metric(processes, "memory_info", "rss")
            self._samples["memory_gb"].append(mem_rss / (1024**3))

    def _calculate_total_cpu_seconds(self, processes: list[psutil.Process]) -> float:
        """Calculate total CPU time (user + system) used by the given processes.

        Args:
            processes: List of psutil.Process objects.

        Returns:
            Total CPU seconds consumed (user + system time).
        """
        total = 0.0
        for p in processes:
            try:
                cpu = p.cpu_times()
                total += cpu.user + cpu.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def _calculate_io_operations(
        self,
        parent: psutil.Process,
        parent_start_io: psutil._common.pio | None,
    ) -> dict[str, int]:
        """Calculate physical disk IO delta for the parent process only.

        Returns an empty dict if IO counters are not supported or an error occurs.

        Args:
            parent: The parent psutil.Process.
            parent_start_io: IO counters snapshot taken at the start (or None).

        Returns:
            Dictionary with IO deltas (read_count, write_count, read_bytes, write_bytes),
            or {} if not available.
        """
        if parent_start_io is None or not hasattr(parent, "io_counters"):
            return {}

        io_totals = {
            "read_count": 0,
            "write_count": 0,
            "read_bytes": 0,
            "write_bytes": 0,
        }

        try:
            parent_end = parent.io_counters()
            io_totals["read_count"] = parent_end.read_count - parent_start_io.read_count
            io_totals["write_count"] = parent_end.write_count - parent_start_io.write_count
            io_totals["read_bytes"] = parent_end.read_bytes - parent_start_io.read_bytes
            io_totals["write_bytes"] = parent_end.write_bytes - parent_start_io.write_bytes
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return {}

        return io_totals

    def _get_cgroup_memory_info(self) -> dict[str, float] | None:
        """Get current cgroup memory usage (SLURM-like) using psutil.

        Returns None if not available (old psutil or not in cgroup).
        """
        try:
            p = psutil.Process()
            cgroup_mem = p.cgroup_memory_info()  # Requires psutil >= 5.9.5
            return {
                "current_bytes": cgroup_mem.current or cgroup_mem.usage_in_bytes,
                "peak_bytes": cgroup_mem.peak or cgroup_mem.max_usage_in_bytes,
            }
        except (AttributeError, NotImplementedError):
            # cgroup support not available
            return None

    @contextmanager
    def measure(self):
        """Context manager to measure resource usage during the wrapped code block.

        Collects:
        - CPU time (total and core-hours)
        - Peak and average **per-process** memory usage (RSS)
        - Peak and average **cgroup** memory usage (SLURM-like MaxRSS)
        - Disk IO operations (parent process only, delta-based)
        - Wall-clock time

        The metrics are stored in `self.metrics` after the context exits.
        """
        parent = psutil.Process()
        self._stop_event.clear()
        self._samples.clear()
        self._cgroup_samples.clear()

        start_time = time.time()

        start_cpu_total = self._calculate_total_cpu_seconds(self._get_all_processes(parent))

        start_io = None
        if hasattr(parent, "io_counters"):
            try:
                start_io = parent.io_counters()
            except (AttributeError, psutil.AccessDenied):
                start_io = None

        monitor_thread = Thread(target=self._monitor_loop, args=(parent,), daemon=True)
        monitor_thread.start()

        try:
            yield
        finally:
            self._stop_event.set()
            monitor_thread.join()

            end_time = time.time()
            end_cpu_total = self._calculate_total_cpu_seconds(self._get_all_processes(parent))
            total_cpu_seconds = end_cpu_total - start_cpu_total

            io_operations = self._calculate_io_operations(parent, start_io)

            # Per-process RSS
            mem_samples = self._samples["memory_gb"]
            peak_memory_gb = max(mem_samples) if mem_samples else 0.0
            average_memory_gb = sum(mem_samples) / len(mem_samples) if mem_samples else 0.0

            # Cgroup memory (SLURM-like) - measured only at the end
            cgroup_info = self._get_cgroup_memory_info()
            cgroup_peak_gb = 0.0
            cgroup_average_gb = 0.0

            if cgroup_info:
                # Take one final sample for current cgroup memory
                current_cgroup_gb = cgroup_info["current_bytes"] / (1024**3)
                cgroup_peak_gb = cgroup_info["peak_bytes"] / (1024**3)
                # For average, we use the last known value (since we don't sample continuously)
                cgroup_average_gb = current_cgroup_gb

            wall_seconds = end_time - start_time

            self.metrics = {
                "cpu_core_hours": round(total_cpu_seconds / 3600.0, 6),
                "peak_memory_gb": round(peak_memory_gb, 3),  # per-process RSS peak
                "average_memory_gb": round(average_memory_gb, 3),  # per-process RSS avg
                "cgroup_peak_memory_gb": round(cgroup_peak_gb, 3),  # SLURM-like MaxRSS
                "cgroup_average_memory_gb": round(cgroup_average_gb, 3),  # approx
                "io_operations": io_operations,
                "wall_time_seconds": round(wall_seconds, 3),
                "wall_time": str(timedelta(seconds=wall_seconds)),
                "total_cpu_seconds": round(total_cpu_seconds, 3),
            }

    def log_summary(self, logger: logging.Logger) -> None:
        """Log a human-readable summary of the collected resource usage metrics.

        Args:
            logger: The logger instance to write the summary to.
        """
        formatted_names = {
            "cpu_core_hours": "CPU Core Hours",
            "peak_memory_gb": "Peak Memory (GB) [per-process RSS]",
            "average_memory_gb": "Average Memory (GB) [per-process RSS]",
            "cgroup_peak_memory_gb": "Peak Memory (GB) [cgroup / SLURM MaxRSS]",
            "cgroup_average_memory_gb": "Average Memory (GB) [cgroup approx]",
            "io_operations": "IO Operations",
            "wall_time_seconds": "Wall Time (seconds)",
            "wall_time": "Wall Time",
            "total_cpu_seconds": "Total CPU Seconds",
        }

        formatted_io_names = {
            "read_count": "Read Count",
            "write_count": "Write Count",
            "read_bytes": "Read Bytes",
            "write_bytes": "Write Bytes",
        }

        logger.info("Resource Usage Summary:")
        for key, value in self.metrics.items():
            if key == "io_operations" and isinstance(value, dict) and value:
                logger.info("  IO Operations:")
                for io_key, io_value in value.items():
                    logger.info("    %s: %d", formatted_io_names.get(io_key, io_key), io_value)
            else:
                logger.info("  %s: %s", formatted_names.get(key, key), value)

    def save_metrics(self, file_name: Path | str, encoding: str = "utf-8", overwrite: bool = False) -> None:
        """Save the collected resource usage metrics to a JSON file.

        Args:
            file_name: Path or string name of the file to write.
            encoding: File encoding (default: "utf-8").
            overwrite: If False, raises FileExistsError if the file already exists.

        Raises:
            FileExistsError: If the file exists and overwrite is False.
        """
        file_name = Path(file_name)
        if not overwrite and file_name.exists():
            raise FileExistsError(f"File '{file_name}' already exists and overwrite is set to False.")

        with atomic_writer(file_name, mode="w", encoding=encoding) as f:
            json.dump(self.metrics, f, indent=4)
