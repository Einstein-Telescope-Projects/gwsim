"""Unit tests for the Detector class."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gwsim.detector.base import Detector


class TestDetector:
    """Test suite for the Detector class."""

    def test_init_with_name(self):
        """Test initialization with a built-in detector name."""
        with patch("gwsim.detector.base.PyCBCDetector") as mock_det_class:
            mock_det = mock_det_class.return_value
            det = Detector(name="H1")
            mock_det_class.assert_called_once_with("H1")
            assert det.name == "H1"
            assert det._detector is mock_det

    def test_init_with_config_file_absolute_path(self):
        """Test initialization with an absolute path to a config file."""
        with (
            patch("gwsim.detector.base.load_interferometer_config") as mock_load,
            patch("gwsim.detector.base.PyCBCDetector") as mock_det_class,
        ):
            mock_load.return_value = "V1"
            mock_det = mock_det_class.return_value

            with tempfile.NamedTemporaryFile(suffix=".interferometer", delete=False) as f:
                file_path = Path(f.name)

            try:
                det = Detector(configuration_file=file_path)
                mock_load.assert_called_once_with(config_file=file_path)
                mock_det_class.assert_called_once_with("V1")
                assert det.name == "V1"
                assert det._detector is mock_det
            finally:
                file_path.unlink()

    def test_init_with_config_file_relative_path(self):
        """Test initialization with a relative path to a config file in DEFAULT_DETECTOR_BASE_PATH."""
        with (
            patch("gwsim.detector.base.load_interferometer_config") as mock_load,
            patch("gwsim.detector.base.PyCBCDetector") as mock_det_class,
            patch("gwsim.detector.base.DEFAULT_DETECTOR_BASE_PATH", Path("/fake/base")),
        ):
            mock_load.return_value = "L1"

            # Mock the file existence check: relative path doesn't exist, but absolute does
            def mock_is_file(self):
                return str(self) == "/fake/base/L1.interferometer"

            with patch.object(Path, "is_file", mock_is_file):
                det = Detector(configuration_file="L1.interferometer")
                expected_path = Path("/fake/base/L1.interferometer")
                mock_load.assert_called_once_with(config_file=expected_path)
                mock_det_class.assert_called_once_with("L1")
                assert det.name == "L1"

    def test_init_with_config_file_relative_path_not_found(self):
        """Test initialization with relative path when file doesn't exist in DEFAULT_DETECTOR_BASE_PATH."""
        with (
            patch("gwsim.detector.base.DEFAULT_DETECTOR_BASE_PATH", Path("/fake/base")),
            patch.object(Path, "is_file", return_value=False),
            pytest.raises(FileNotFoundError, match=re.escape("Configuration file 'L1.interferometer' not found")),
        ):
            Detector(configuration_file="L1.interferometer")

    def test_init_both_name_and_config_file(self):
        """Test that specifying both name and config_file raises ValueError."""
        with pytest.raises(ValueError, match="Specify either 'name' or 'configuration_file', not both"):
            Detector(name="H1", configuration_file="file.interferometer")

    def test_init_neither_name_nor_config_file(self):
        """Test that specifying neither name nor config_file raises ValueError."""
        with pytest.raises(ValueError, match="Either 'name' or 'configuration_file' must be provided"):
            Detector()

    def test_init_config_file_absolute_not_found(self):
        """Test FileNotFoundError for non-existent absolute config file path."""
        nonexistent = Path("/nonexistent/file.interferometer")
        with pytest.raises(FileNotFoundError, match=f"Configuration file '{nonexistent}' not found"):
            Detector(configuration_file=nonexistent)

    def test_antenna_pattern(self):
        """Test antenna_pattern method delegates to underlying detector."""
        with patch("gwsim.detector.base.PyCBCDetector") as mock_det_class:
            mock_det = mock_det_class.return_value
            mock_det.antenna_pattern.return_value = (0.7, 0.4)

            det = Detector(name="H1")
            result = det.antenna_pattern(1.0, 2.0, 3.0, 1000000000)

            mock_det.antenna_pattern.assert_called_once_with(1.0, 2.0, 3.0, 1000000000, 0, "tensor")
            assert result == (0.7, 0.4)

    def test_time_delay_from_earth_center(self):
        """Test time_delay_from_earth_center method delegates to underlying detector."""
        with patch("gwsim.detector.base.PyCBCDetector") as mock_det_class:
            mock_time_delay = 0.02
            mock_det = mock_det_class.return_value
            mock_det.time_delay_from_earth_center.return_value = mock_time_delay

            det = Detector(name="H1")
            result = det.time_delay_from_earth_center(1.0, 2.0, 1000000000)

            mock_det.time_delay_from_earth_center.assert_called_once_with(1.0, 2.0, 1000000000)
            assert result == mock_time_delay

    def test_getattr_delegation(self):
        """Test __getattr__ delegates attribute access to underlying detector."""
        with patch("gwsim.detector.base.PyCBCDetector") as mock_det_class:
            mock_det = mock_det_class.return_value
            mock_det.latitude = 46.4551

            det = Detector(name="H1")
            assert det.latitude == mock_det.latitude

    def test_str_method(self):
        """Test __str__ returns the detector name."""
        with patch("gwsim.detector.base.PyCBCDetector"):
            det = Detector(name="H1")
            assert str(det) == "H1"
