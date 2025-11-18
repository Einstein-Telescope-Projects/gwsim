"""Unit tests for the DetectorMixin class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from gwsim.mixin.detector import DetectorMixin
from gwsim.simulator.base import Simulator


class MockSimulator(DetectorMixin, Simulator):
    """Mock simulator class for testing DetectorMixin."""

    def simulate(self, *args, **kwargs):
        """Mock simulate method."""
        return "mock_sample"

    def _save_data(self, data, file_name, **kwargs):
        """Mock _save_data method."""
        pass


class TestDetectorMixin:
    """Test suite for the DetectorMixin class."""

    def test_init_with_detectors_none(self):
        """Test initialization with detectors=None."""
        sim = MockSimulator(detectors=None)
        assert sim.detectors is None

    def test_init_with_detectors_list_of_names(self):
        """Test initialization with a list of detector names."""
        with patch("gwsim.simulator.mixin.detector.Detector") as mock_detector_class:
            mock_detector = mock_detector_class.return_value
            detectors = ["H1", "L1"]
            sim = MockSimulator(detectors=detectors)
            assert sim._detectors == [mock_detector, mock_detector]
            assert mock_detector_class.call_count == 2
            mock_detector_class.assert_any_call(name="H1")
            mock_detector_class.assert_any_call(name="L1")

    def test_init_with_detectors_list_of_config_files(self):
        """Test initialization with a list of config file paths."""
        with (
            patch("gwsim.simulator.mixin.detector.Detector") as mock_detector_class,
            patch("pathlib.Path.is_file", return_value=True),
        ):
            mock_detector = mock_detector_class.return_value
            detectors = ["H1.interferometer", "L1.interferometer"]
            sim = MockSimulator(detectors=detectors)
            assert sim._detectors == [mock_detector, mock_detector]
            assert mock_detector_class.call_count == 2
            mock_detector_class.assert_any_call(config_file="H1.interferometer")
            mock_detector_class.assert_any_call(config_file="L1.interferometer")

    def test_init_with_detectors_list_of_relative_config_files(self):
        """Test initialization with a list of relative config file paths in DEFAULT_DETECTOR_BASE_PATH."""
        with (
            patch("gwsim.simulator.mixin.detector.Detector") as mock_detector_class,
            patch("gwsim.simulator.mixin.detector.DEFAULT_DETECTOR_BASE_PATH", Path("/fake/base")),
            patch.object(Path, "is_file", lambda self: str(self).startswith("/fake/base/")),
        ):
            mock_detector = mock_detector_class.return_value

            detectors = ["H1.interferometer", "L1.interferometer"]
            sim = MockSimulator(detectors=detectors)
            assert sim._detectors == [mock_detector, mock_detector]
            assert mock_detector_class.call_count == 2
            mock_detector_class.assert_any_call(config_file="H1.interferometer")
            mock_detector_class.assert_any_call(config_file="L1.interferometer")

    def test_detectors_property_getter(self):
        """Test the detectors property getter."""
        sim = MockSimulator(detectors=None)
        assert sim.detectors is None

        with patch("gwsim.simulator.mixin.detector.Detector"):
            sim = MockSimulator(detectors=["H1"])
            assert sim.detectors is not None
            assert len(sim.detectors) == 1

    def test_detectors_property_setter(self):
        """Test the detectors property setter."""
        sim = MockSimulator()
        sim.detectors = None
        assert sim._detectors is None

        with patch("gwsim.simulator.mixin.detector.Detector") as mock_detector_class:
            mock_detector = mock_detector_class.return_value
            sim.detectors = ["H1", "L1"]
            assert sim._detectors == [mock_detector, mock_detector]

    def test_metadata_property(self):
        """Test the metadata property."""
        sim = MockSimulator(detectors=None)
        metadata = sim.metadata
        assert metadata == {"detectors": None}

        with patch("gwsim.simulator.mixin.detector.Detector"):
            sim = MockSimulator(detectors=["H1"])
            metadata = sim.metadata
            assert "detectors" in metadata
            assert len(metadata["detectors"]) == 1
