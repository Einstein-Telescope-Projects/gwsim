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
        self.metrics: dict[str, float | str | dict] = {}
        self.sample_interval = sample_interval
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._stop_event = Event()

    def _get_all_processes(self, parent: psutil.Process) -> list[psutil.Process]:
        processes = [parent]
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            processes.extend(parent.children(recursive=True))
        return [p for p in processes if p.is_running()]

    def _aggregate_metric(self, processes: list[psutil.Process], attr: str, sub_attr: str | None = None) -> float:
        total = 0.0
        for p in processes:
            try:
                val = getattr(p, attr)()
                total += getattr(val, sub_attr) if sub_attr else val
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                pass
        return total

    def _monitor_loop(self, parent: psutil.Process):
        while not self._stop_event.wait(self.sample_interval):
            processes = self._get_all_processes(parent)
            mem_rss = self._aggregate_metric(processes, "memory_info", "rss")
            self._samples["memory_gb"].append(mem_rss / (1024**3))

    def _calculate_total_cpu_seconds(self, processes: list[psutil.Process]) -> float:
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
        """Calculate physical disk IO for the parent process only (delta-based)."""
        io_totals = {
            "read_count": 0,
            "write_count": 0,
            "read_bytes": 0,
            "write_bytes": 0,
        }

        if parent_start_io is None or not hasattr(parent, "io_counters"):
            return io_totals

        try:
            parent_end = parent.io_counters()
            io_totals["read_count"] = parent_end.read_count - parent_start_io.read_count
            io_totals["write_count"] = parent_end.write_count - parent_start_io.write_count
            io_totals["read_bytes"] = parent_end.read_bytes - parent_start_io.read_bytes
            io_totals["write_bytes"] = parent_end.write_bytes - parent_start_io.write_bytes
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            pass

        return io_totals

    @contextmanager
    def measure(self):
        parent = psutil.Process()
        start_time = time.time()

        start_cpu_total = self._calculate_total_cpu_seconds([parent])
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

            mem_samples = self._samples["memory_gb"]
            peak_memory_gb = max(mem_samples) if mem_samples else 0.0
            average_memory_gb = sum(mem_samples) / len(mem_samples) if mem_samples else 0.0

            wall_seconds = end_time - start_time

            self.metrics = {
                "cpu_core_hours": round(total_cpu_seconds / 3600.0, 6),
                "peak_memory_gb": round(peak_memory_gb, 3),
                "average_memory_gb": round(average_memory_gb, 3),
                "io_operations": io_operations,
                "wall_time_seconds": round(wall_seconds, 3),
                "wall_time": str(timedelta(seconds=wall_seconds)),
                "total_cpu_seconds": round(total_cpu_seconds, 3),
            }

    def log_summary(self, logger: logging.Logger) -> None:
        """Log the resource usage summary."""
        formatted_names = {
            "cpu_core_hours": "CPU Core Hours",
            "peak_memory_gb": "Peak Memory (GB)",
            "average_memory_gb": "Average Memory (GB)",
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
        """Save the resource usage metrics to a JSON file."""
        file_name = Path(file_name)
        if not overwrite and file_name.exists():
            raise FileExistsError(f"File '{file_name}' already exists and overwrite is set to False.")

        with atomic_writer(file_name, mode="w", encoding=encoding) as f:
            json.dump(self.metrics, f, indent=4)
