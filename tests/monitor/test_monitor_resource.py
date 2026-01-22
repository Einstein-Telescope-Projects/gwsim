"""Unit tests for resource monitoring utilities."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from gwsim.monitor.resource import ResourceMonitor

# Constants for test assertions to avoid magic values
COUNTER_50 = 50
COUNTER_100 = 100
N_BYTES_51200 = 51200
N_BYTES_102400 = 102400
WALL_TIME_SEC_025 = 0.25
WALL_TIME_SEC_02 = 0.2


class TestResourceMonitor:
    """Test suite for ResourceMonitor class."""

    def test_measure_successful(self):
        """Test successful measurement with all metrics populated."""
        monitor = ResourceMonitor(sample_interval=0.1)  # faster sampling for test

        # Mock psutil.Process and its methods
        with patch("psutil.Process") as mock_process_class:
            mock_parent = MagicMock()
            mock_process_class.return_value = mock_parent

            # Mock children
            mock_child = MagicMock()
            mock_parent.children.return_value = [mock_child]

            # Mock processes list returned by _get_all_processes
            def get_all_processes(parent):
                """
                Return a list containing the parent process and its mocked child.

                Parameters:
                    parent: The parent process mock.

                Returns:
                    list: A two-element list [parent, mock_child] with the parent mock first and its child mock second.
                """
                return [parent, mock_child]

            with patch.object(monitor, "_get_all_processes", side_effect=get_all_processes):
                # CPU times - start and end (parent + child)
                mock_parent_start_cpu = MagicMock(user=0.5, system=0.3)
                mock_parent_end_cpu = MagicMock(user=1.2, system=0.7)
                mock_child_start_cpu = MagicMock(user=0.4, system=0.2)
                mock_child_end_cpu = MagicMock(user=0.9, system=0.5)

                mock_parent.cpu_times.side_effect = [mock_parent_start_cpu, mock_parent_end_cpu]
                mock_child.cpu_times.side_effect = [mock_child_start_cpu, mock_child_end_cpu]

                # Memory sampling (rss in bytes)
                mock_mem_parent = MagicMock(rss=1024 * 1024 * 1024)  # 1 GB
                mock_mem_child = MagicMock(rss=512 * 1024 * 1024)  # 0.5 GB

                mock_parent.memory_info.side_effect = [mock_mem_parent] * 4  # called multiple times
                mock_child.memory_info.side_effect = [mock_mem_child] * 4

                # IO counters - only for parent process (delta)
                mock_start_io = MagicMock(read_count=10, write_count=5, read_bytes=2048, write_bytes=1024)
                mock_end_io = MagicMock(
                    read_count=110, write_count=55, read_bytes=2048 + 102400, write_bytes=1024 + 51200
                )

                mock_parent.io_counters.side_effect = [mock_start_io, mock_end_io]

                # Run measurement
                with monitor.measure():
                    time.sleep(0.3)  # give monitor thread time to take a few samples

                metrics = monitor.metrics

                # Basic presence checks
                assert "cpu_core_hours" in metrics
                assert "peak_memory_gb" in metrics
                assert "average_memory_gb" in metrics
                assert "io_operations" in metrics
                assert "wall_time_seconds" in metrics
                assert "wall_time" in metrics
                assert "total_cpu_seconds" in metrics

                # CPU: (parent 0.8 + child 0.8) = 1.6 seconds → ~0.000444 core-hours
                assert round(metrics["total_cpu_seconds"], 3) == pytest.approx(1.6, abs=0.2)
                assert metrics["cpu_core_hours"] > 0

                # Memory: samples should see ~1.5 GB total
                assert metrics["peak_memory_gb"] == pytest.approx(1.5, abs=0.1)
                assert metrics["average_memory_gb"] == pytest.approx(1.5, abs=0.1)

                # IO: only parent delta
                io = metrics["io_operations"]
                assert io["read_count"] == COUNTER_100
                assert io["write_count"] == COUNTER_50
                assert io["read_bytes"] == N_BYTES_102400
                assert io["write_bytes"] == N_BYTES_51200

                # Wall time should be > 0.3s
                assert metrics["wall_time_seconds"] > WALL_TIME_SEC_025

    def test_measure_with_exception(self):
        """Test measurement when an exception occurs inside the context."""
        monitor = ResourceMonitor(sample_interval=0.1)

        with patch("psutil.Process") as mock_process_class:
            mock_parent = MagicMock()
            mock_process_class.return_value = mock_parent

            mock_child = MagicMock()
            mock_parent.children.return_value = [mock_child]

            # Mock processes list
            def get_all_processes(parent):
                """
                Return a list containing the parent process and its mocked child.

                Parameters:
                    parent: The parent process mock.

                Returns:
                    list: A two-element list [parent, mock_child] with the parent mock first and its child mock second.
                """
                return [parent, mock_child]

            with patch.object(monitor, "_get_all_processes", side_effect=get_all_processes):
                # CPU times
                mock_parent.cpu_times.return_value = MagicMock(user=0.5, system=0.3)
                mock_child.cpu_times.return_value = MagicMock(user=0.4, system=0.2)

                # Memory
                mock_parent.memory_info.return_value = MagicMock(rss=1024 * 1024 * 1024)  # 1 GB
                mock_child.memory_info.return_value = MagicMock(rss=512 * 1024 * 1024)  # 0.5 GB

                # IO
                mock_parent.io_counters.side_effect = [
                    MagicMock(read_count=10, write_count=5, read_bytes=2048, write_bytes=1024),
                    MagicMock(read_count=60, write_count=25, read_bytes=2048 + 51200, write_bytes=1024 + 25600),
                ]

                def code_that_raises():
                    """
                    Enter the monitor.measure() context, pause briefly, and then raise a ValueError.

                    This helper function is used in tests to exercise the measurement context while forcing an exception to propagate. It sleeps for approximately 0.25 seconds inside the context before raising the error.

                    Raises:
                        ValueError: Always raised with the message "Test exception".
                    """
                    with monitor.measure():
                        time.sleep(0.25)
                        raise ValueError("Test exception")

                with pytest.raises(ValueError, match="Test exception"):
                    code_that_raises()

                metrics = monitor.metrics

                # Metrics should still be collected
                assert "peak_memory_gb" in metrics
                assert "average_memory_gb" in metrics
                assert "cpu_core_hours" in metrics
                assert "io_operations" in metrics
                assert metrics["wall_time_seconds"] > WALL_TIME_SEC_02
                assert metrics["peak_memory_gb"] == pytest.approx(1.5, abs=0.1)

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
