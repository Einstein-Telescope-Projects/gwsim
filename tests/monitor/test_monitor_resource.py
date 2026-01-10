"""Unit tests for resource monitoring utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from gwsim.monitor.resource import ResourceMonitor


class TestResourceMonitor:
    """Test suite for ResourceMonitor class."""

    def test_measure_successful(self):
        """Test successful measurement with all metrics populated."""
        monitor = ResourceMonitor()

        # Mock psutil.Process and its methods
        with patch("psutil.Process") as mock_process_class:
            mock_process = MagicMock()
            mock_process_class.return_value = mock_process

            # Mock CPU times - start and end
            mock_start_cpu_times = MagicMock()
            mock_start_cpu_times.user = 0.5
            mock_start_cpu_times.system = 0.25
            mock_end_cpu_times = MagicMock()
            mock_end_cpu_times.user = 1.0
            mock_end_cpu_times.system = 0.5
            mock_process.cpu_times.side_effect = [mock_start_cpu_times, mock_end_cpu_times]

            # Mock memory info - same for start and end since no operation
            mock_mem_info = MagicMock()
            mock_mem_info.rss = 1024 * 1024 * 1024  # 1 GB in bytes
            mock_process.memory_info.side_effect = [mock_mem_info, mock_mem_info]

            # Mock IO counters - start and end
            mock_start_io_counters = MagicMock()
            mock_start_io_counters.read_count = 0
            mock_start_io_counters.write_count = 0
            mock_start_io_counters.read_bytes = 0
            mock_start_io_counters.write_bytes = 0
            mock_end_io_counters = MagicMock()
            mock_end_io_counters.read_count = 100
            mock_end_io_counters.write_count = 50
            mock_end_io_counters.read_bytes = 1024 * 100
            mock_end_io_counters.write_bytes = 1024 * 50
            mock_process.io_counters.side_effect = [mock_start_io_counters, mock_end_io_counters]

            # Mock CPU percent
            mock_process.cpu_percent.return_value = 25.0

            # Simulate measurement
            with monitor.measure():
                pass  # No operation

            # Verify metrics
            metrics = monitor.metrics
            assert "cpu_core_hours" in metrics
            assert "peak_memory_gb" in metrics
            assert "average_memory_gb" in metrics
            assert "cpu_percent" in metrics
            assert "io_operations" in metrics
            assert "wall_time_seconds" in metrics
            assert "wall_time" in metrics
            assert "total_cpu_seconds" in metrics

            # Check specific values (approximate due to timing)
            assert metrics["peak_memory_gb"] == 1.0  # 1 GB
            assert metrics["average_memory_gb"] == 1.0  # Average of same value
            assert metrics["cpu_percent"] == mock_process.cpu_percent.return_value
            assert isinstance(metrics["io_operations"], dict)
            assert metrics["io_operations"]["read_count"] == mock_end_io_counters.read_count
            assert metrics["io_operations"]["write_count"] == mock_end_io_counters.write_count

    def test_measure_with_exception(self):
        """Test measurement when an exception occurs inside the context."""
        monitor = ResourceMonitor()

        with patch("psutil.Process") as mock_process_class:
            mock_process = MagicMock()
            mock_process_class.return_value = mock_process

            # Mock basic attributes
            mock_cpu_times = MagicMock()
            mock_cpu_times.user = 0.5
            mock_cpu_times.system = 0.25
            mock_process.cpu_times.return_value = mock_cpu_times

            mock_mem_info = MagicMock()
            mock_mem_info.rss = 512 * 1024 * 1024  # 512 MB
            mock_process.memory_info.return_value = mock_mem_info

            mock_process.cpu_percent.return_value = 10.0

            # Simulate exception
            def _raise_exception():
                with monitor.measure():
                    raise ValueError("Test exception")

            with pytest.raises(ValueError, match="Test exception"):
                _raise_exception()

            # Metrics should still be populated
            metrics = monitor.metrics
            expected_peak_memory = 0.5
            assert "peak_memory_gb" in metrics
            assert metrics["peak_memory_gb"] == expected_peak_memory  # 512 MB in GB

    def test_measure_without_io_counters(self):
        """Test measurement on platforms without IO counters support."""
        monitor = ResourceMonitor()

        with patch("psutil.Process") as mock_process_class:
            mock_process = MagicMock()
            mock_process_class.return_value = mock_process

            # Mock basic attributes
            mock_cpu_times = MagicMock()
            mock_cpu_times.user = 0.1
            mock_cpu_times.system = 0.05
            mock_process.cpu_times.return_value = mock_cpu_times

            mock_mem_info = MagicMock()
            mock_mem_info.rss = 256 * 1024 * 1024  # 256 MB
            mock_process.memory_info.return_value = mock_mem_info

            mock_process.cpu_percent.return_value = 5.0

            # Simulate no io_counters attribute
            del mock_process.io_counters  # Remove the attribute

            with monitor.measure():
                pass

            metrics = monitor.metrics
            assert metrics["io_operations"] == {}  # Should be empty dict

    def test_measure_io_counters_exception(self):
        """Test measurement when IO counters raise an exception."""
        monitor = ResourceMonitor()

        with patch("psutil.Process") as mock_process_class:
            mock_process = MagicMock()
            mock_process_class.return_value = mock_process

            # Mock basic attributes
            mock_cpu_times = MagicMock()
            mock_cpu_times.user = 0.2
            mock_cpu_times.system = 0.1
            mock_process.cpu_times.return_value = mock_cpu_times

            mock_mem_info = MagicMock()
            mock_mem_info.rss = 128 * 1024 * 1024  # 128 MB
            mock_process.memory_info.return_value = mock_mem_info

            mock_process.cpu_percent.return_value = 2.0

            # Mock io_counters to raise exception
            mock_process.io_counters.side_effect = AttributeError("Not supported")

            with monitor.measure():
                pass

            metrics = monitor.metrics
            assert metrics["io_operations"] == {}  # Should be empty dict

    def test_log_summary(self):
        """Test logging of resource usage summary."""
        monitor = ResourceMonitor()
        monitor.metrics = {
            "cpu_core_hours": 1.5,
            "peak_memory_gb": 2.0,
            "io_operations": {"read_count": 100, "write_count": 50},
            "wall_time": "00:00:10",
        }

        mock_logger = MagicMock()
        monitor.log_summary(mock_logger)

        expected_calls = [
            call("Resource Usage Summary:"),
            call("  %s: %s", "CPU Core Hours", 1.5),
            call("  %s: %s", "Peak Memory (GB)", 2.0),
            call("  IO Operations:"),
            call("    %s: %d", "Read Count", 100),
            call("    %s: %d", "Write Count", 50),
            call("  %s: %s", "Wall Time", "00:00:10"),
        ]
        mock_logger.info.assert_has_calls(expected_calls)

    def test_log_summary_empty_io_operations(self):
        """Test logging when io_operations is empty."""
        monitor = ResourceMonitor()
        monitor.metrics = {
            "cpu_core_hours": 0.5,
            "io_operations": {},
        }

        mock_logger = MagicMock()
        monitor.log_summary(mock_logger)

        expected_calls = [
            call("Resource Usage Summary:"),
            call("  %s: %s", "CPU Core Hours", 0.5),
            call("  %s: %s", "IO Operations", {}),
        ]
        mock_logger.info.assert_has_calls(expected_calls)

    def test_save_metrics_success(self):
        """Test successful saving of metrics to a new file."""
        monitor = ResourceMonitor()
        monitor.metrics = {"cpu_core_hours": 1.0, "peak_memory_gb": 2.0}

        with (
            patch.object(Path, "exists", return_value=False),
            patch("gwsim.monitor.resource.atomic_writer") as mock_atomic_writer,
            patch("json.dump") as mock_json_dump,
        ):
            mock_file = MagicMock()
            mock_atomic_writer.return_value.__enter__.return_value = mock_file

            monitor.save_metrics("test.json")

            file_name = Path("test.json")
            mock_atomic_writer.assert_called_once_with(file_name, mode="w", encoding="utf-8")
            mock_json_dump.assert_called_once_with(monitor.metrics, mock_file, indent=4)

    def test_save_metrics_overwrite(self):
        """Test saving metrics with overwrite=True when file exists."""
        monitor = ResourceMonitor()
        monitor.metrics = {"cpu_core_hours": 1.0}

        with (
            patch.object(Path, "exists", return_value=True),
            patch("gwsim.monitor.resource.atomic_writer") as mock_atomic_writer,
            patch("json.dump") as mock_json_dump,
        ):
            mock_file = MagicMock()
            mock_atomic_writer.return_value.__enter__.return_value = mock_file

            monitor.save_metrics("test.json", overwrite=True)

            file_name = Path("test.json")
            mock_atomic_writer.assert_called_once_with(file_name, mode="w", encoding="utf-8")
            mock_json_dump.assert_called_once_with(monitor.metrics, mock_file, indent=4)

    def test_save_metrics_file_exists_no_overwrite(self):
        """Test that FileExistsError is raised when file exists and overwrite=False."""
        monitor = ResourceMonitor()
        monitor.metrics = {"cpu_core_hours": 1.0}

        with (
            patch.object(Path, "exists", return_value=True),
            pytest.raises(FileExistsError, match=r"File 'test.json' already exists and overwrite is set to False."),
        ):
            monitor.save_metrics("test.json", overwrite=False)

    def test_save_metrics_custom_encoding(self):
        """Test saving metrics with custom encoding."""
        monitor = ResourceMonitor()
        monitor.metrics = {"cpu_core_hours": 1.0}

        with (
            patch.object(Path, "exists", return_value=False),
            patch("gwsim.monitor.resource.atomic_writer") as mock_atomic_writer,
            patch("json.dump") as mock_json_dump,
        ):
            mock_file = MagicMock()
            mock_atomic_writer.return_value.__enter__.return_value = mock_file

            monitor.save_metrics("test.json", encoding="latin-1")

            file_name = Path("test.json")
            mock_atomic_writer.assert_called_once_with(file_name, mode="w", encoding="latin-1")
            mock_json_dump.assert_called_once_with(monitor.metrics, mock_file, indent=4)
