"""Module for monitoring resource usage during code execution."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import psutil

from gwsim.utils.io import atomic_writer


class ResourceMonitor:  # pylint: disable=too-few-public-methods
    """Class to monitor resource usage during code execution."""

    def __init__(self):
        """Initialize ResourceMonitor."""
        self.metrics: dict[str, float | str | dict] = {}

    def _calculate_total_cpu_seconds(
        self,
        start_cpu: psutil._common.pcputimes,
        end_cpu: psutil._common.pcputimes,
    ) -> float:
        """Calculate total CPU seconds used between two cpu_times snapshots.

        Args:
            start_cpu: CPU times at the start.
            end_cpu: CPU times at the end.

        Returns:
            float: Total CPU seconds used.
        """
        total_user = end_cpu.user - start_cpu.user
        total_system = end_cpu.system - start_cpu.system
        return total_user + total_system

    def _calculate_memory_usage(
        self,
        start_mem: int,
        end_mem: int,
    ) -> tuple[float, float]:
        """Calculate peak and average memory usage in GB.

        Args:
            start_mem: Memory usage at the start in bytes.
            end_mem: Memory usage at the end in bytes.

        Returns:
            tuple[float, float]: Peak and average memory usage in GB.
        """
        peak_memory_gb = max(start_mem, end_mem) / (1024**3)  # Bytes to GB
        average_memory_gb = (start_mem + end_mem) / 2 / (1024**3)  # Simple average
        return peak_memory_gb, average_memory_gb

    def _calculate_io_operations(
        self,
        process: psutil.Process,
        start_io: psutil._common.pio,
        end_io: psutil._common.pio,
    ) -> dict[str, int]:
        """Calculate IO operations between two io_counters snapshots.

        Args:
            process: The psutil Process object.
            start_io: IO counters at the start.
            end_io: IO counters at the end.

        Returns:
            dict[str, int]: Dictionary with counts of read/write operations and bytes.
        """
        io_operations = {}
        if start_io is not None and hasattr(process, "io_counters"):
            try:
                end_io = process.io_counters()
                io_operations = {
                    "read_count": end_io.read_count - start_io.read_count,
                    "write_count": end_io.write_count - start_io.write_count,
                    "read_bytes": end_io.read_bytes - start_io.read_bytes,
                    "write_bytes": end_io.write_bytes - start_io.write_bytes,
                }
            except (AttributeError, psutil.AccessDenied):
                pass  # Leave io_operations empty
        return io_operations

    @contextmanager
    def measure(self):  # pylint: disable=too-many-locals
        """Context manager to measure resource usage."""
        process = psutil.Process()
        start_time = time.time()
        start_cpu = process.cpu_times()
        start_mem = process.memory_info().rss  # Resident set size in bytes

        # IO counters may not be available on all platforms (e.g., macOS)
        start_io = None
        if hasattr(process, "io_counters"):
            try:
                start_io = process.io_counters()
            except (AttributeError, psutil.AccessDenied):
                start_io = None

        try:
            yield
        finally:
            end_time = time.time()
            end_cpu = process.cpu_times()
            end_mem = process.memory_info().rss

            total_cpu_seconds = self._calculate_total_cpu_seconds(start_cpu=start_cpu, end_cpu=end_cpu)
            core_hours = total_cpu_seconds / 3600.0

            peak_memory_gb, average_memory_gb = self._calculate_memory_usage(start_mem=start_mem, end_mem=end_mem)

            cpu_percent = process.cpu_percent(interval=None)  # CPU usage percentage (0-100)

            io_operations = self._calculate_io_operations(process=process, start_io=start_io, end_io=None)

            wall_seconds = end_time - start_time

            self.metrics = {
                "cpu_core_hours": round(core_hours, 6),
                "peak_memory_gb": round(peak_memory_gb, 3),
                "average_memory_gb": round(average_memory_gb, 3),
                "cpu_percent": round(cpu_percent, 2),
                "io_operations": io_operations,
                "wall_time_seconds": round(wall_seconds, 3),
                "wall_time": str(timedelta(seconds=int(wall_seconds))),
                "total_cpu_seconds": round(total_cpu_seconds, 3),
            }

    def log_summary(self, logger: logging.Logger) -> None:
        """Log the resource usage summary.

        Args:
            logger: Logger to use for logging.
        """
        formatted_names = {
            "cpu_core_hours": "CPU Core Hours",
            "peak_memory_gb": "Peak Memory (GB)",
            "average_memory_gb": "Average Memory (GB)",
            "cpu_percent": "CPU Usage (%)",
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
        """Save the resource usage metrics to a JSON file.

        Args:
            file_name: Path to the output JSON file.
            encoding: File encoding (default is 'utf-8').
            overwrite: Whether to overwrite existing file (default is False).
        """
        file_name = Path(file_name)
        if not overwrite and file_name.exists():
            raise FileExistsError(f"File '{file_name}' already exists and overwrite is set to False.")

        with atomic_writer(file_name, mode="w", encoding=encoding) as f:
            json.dump(self.metrics, f, indent=4)
