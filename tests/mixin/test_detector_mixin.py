"""Unit tests for the DetectorMixin class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwsim.detector import Detector
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

    @property
    def metadata(self):
        """Mock metadata property."""
        meta = super().metadata
        return meta


class TestDetectorMixin:
    """Test suite for the DetectorMixin class."""

    def test_init_with_detectors_none(self):
        """Test initialization with detectors=None."""
        sim = MockSimulator(detectors=None)
        assert sim.detectors == []

    def test_init_with_detectors_list_of_names(self):
        """Test initialization with a list of detector names."""
        with patch("gwsim.mixin.detector.Detector") as mock_detector_class:
            mock_detector = mock_detector_class.return_value
            detectors = ["H1", "L1"]
            sim = MockSimulator(detectors=detectors)

            assert sim._detectors == [mock_detector, mock_detector]

            assert mock_detector_class.call_count == len(detectors)
            mock_detector_class.assert_any_call(name="H1")
            mock_detector_class.assert_any_call(name="L1")

    def test_init_with_detectors_list_of_config_files(self):
        """Test initialization with a list of config file paths."""
        with (
            patch("gwsim.mixin.detector.Detector") as mock_detector_class,
            patch("pathlib.Path.is_file", return_value=True),
        ):
            mock_detector = mock_detector_class.return_value
            detectors = ["H1.interferometer", "L1.interferometer"]
            sim = MockSimulator(detectors=detectors)
            assert sim._detectors == [mock_detector, mock_detector]
            assert mock_detector_class.call_count == len(detectors)
            mock_detector_class.assert_any_call(configuration_file="H1.interferometer")
            mock_detector_class.assert_any_call(configuration_file="L1.interferometer")

    def test_init_with_detectors_list_of_relative_config_files(self):
        """Test initialization with a list of relative config file paths in DEFAULT_DETECTOR_BASE_PATH."""
        with (
            patch("gwsim.mixin.detector.Detector") as mock_detector_class,
            patch("gwsim.mixin.detector.DEFAULT_DETECTOR_BASE_PATH", Path("/fake/base")),
            patch.object(Path, "is_file", lambda self: str(self).startswith("/fake/base/")),
        ):
            mock_detector = mock_detector_class.return_value

            detectors = ["H1.interferometer", "L1.interferometer"]
            sim = MockSimulator(detectors=detectors)
            assert sim._detectors == [mock_detector, mock_detector]
            assert mock_detector_class.call_count == len(detectors)
            mock_detector_class.assert_any_call(configuration_file="H1.interferometer")
            mock_detector_class.assert_any_call(configuration_file="L1.interferometer")

    def test_detectors_property_getter(self):
        """Test the detectors property getter."""
        sim = MockSimulator(detectors=None)
        assert sim.detectors == []

        with patch("gwsim.mixin.detector.Detector"):
            sim = MockSimulator(detectors=["H1"])
            assert sim.detectors is not None
            assert len(sim.detectors) == 1

    def test_detectors_property_setter(self):
        """Test the detectors property setter."""
        sim = MockSimulator()
        sim.detectors = None
        assert sim._detectors == []

        with patch("gwsim.mixin.detector.Detector") as mock_detector_class:
            mock_detector = mock_detector_class.return_value
            sim.detectors = ["H1", "L1"]
            assert sim._detectors == [mock_detector, mock_detector]

    def test_metadata_property(self):
        """Test the metadata property."""
        sim = MockSimulator(detectors=None)
        metadata = sim.metadata
        assert metadata == {"detector": {"arguments": {"detectors": None}}}

        with patch("gwsim.mixin.detector.Detector"):
            sim = MockSimulator(detectors=["H1"])
            metadata = sim.metadata
            assert "detector" in metadata
            assert len(metadata["detector"]["arguments"]["detectors"]) == 1


class TestProjectPolarizationsEarthRotation:
    """Test suite for earth rotation effects in polarization projection."""

    @pytest.fixture
    def h1_detector(self):
        """Create a real H1 detector once for all tests in this class."""
        return Detector(name="H1")

    @pytest.fixture
    def sine_wave_polarizations(self):
        """Create synthetic sine wave polarizations for testing.

        Returns a dictionary with 'plus' and 'cross' polarizations as simple
        sine waves to avoid expensive waveform calculations.
        """
        # Create a long signal spanning many hours to observe earth rotation effects
        duration = 12 * 3600  # 12 hours
        sampling_rate = 256  # Hz (lower for computational efficiency)
        num_samples = int(duration * sampling_rate)
        t0 = 1000000000  # GPS time reference

        # Create time array
        times = np.arange(num_samples) / sampling_rate + t0

        # Create simple sine wave polarizations
        frequency = 100  # Hz
        hp_data = np.sin(2 * np.pi * frequency * (times - t0))
        hc_data = 0.5 * np.cos(2 * np.pi * frequency * (times - t0))

        # Convert to GWpy TimeSeries
        hp = GWpyTimeSeries(hp_data, times=times, sample_rate=sampling_rate)
        hc = GWpyTimeSeries(hc_data, times=times, sample_rate=sampling_rate)

        return {"plus": hp, "cross": hc}

    @pytest.mark.slow
    def test_earth_rotation_produces_different_results(self, sine_wave_polarizations, h1_detector):
        """Test that earth_rotation=True produces different results than earth_rotation=False."""

        # Use the h1_detector fixture instead of creating a new one
        detector = h1_detector

        with patch("gwsim.mixin.detector.Detector") as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.is_configured.return_value = True

            # Use real antenna pattern calculations
            def mock_antenna_pattern(*args, **kwargs):
                # Call the real detector's antenna pattern method
                return detector.antenna_pattern(*args, **kwargs)

            # Use real time delay calculations from detector
            def mock_time_delay(*args, **kwargs):
                # Call the real detector's time_delay_from_earth_center method
                return detector.time_delay_from_earth_center(*args, **kwargs)

            mock_detector.antenna_pattern.side_effect = mock_antenna_pattern
            mock_detector.time_delay_from_earth_center.side_effect = mock_time_delay
            mock_detector_class.return_value = mock_detector

            sim = MockSimulator(detectors=["H1"])

            # Project with earth rotation enabled
            result_with_rotation = sim.project_polarizations(
                polarizations=sine_wave_polarizations,
                right_ascension=1.5,
                declination=0.5,
                polarization_angle=0.0,
                earth_rotation=True,
            )

            # Project with earth rotation disabled
            result_without_rotation = sim.project_polarizations(
                polarizations=sine_wave_polarizations,
                right_ascension=1.5,
                declination=0.5,
                polarization_angle=0.0,
                earth_rotation=False,
            )

            # Results should be different due to time-dependent antenna patterns
            assert not np.allclose(result_with_rotation._data, result_without_rotation._data, atol=1e-5)

    @pytest.mark.slow
    def test_earth_rotation_center_point_consistency(self, sine_wave_polarizations, h1_detector):
        """Test that segments near the center time are consistent between earth_rotation modes.

        When earth_rotation=False, the antenna pattern and time delay are computed
        at the middle time. Segments near the middle should match closely between
        earth_rotation=True and earth_rotation=False, while segments far from the
        center should differ significantly.
        """

        # Use the h1_detector fixture instead of creating a new one
        detector = h1_detector

        with patch("gwsim.mixin.detector.Detector") as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.is_configured.return_value = True

            # Use real antenna pattern calculations
            def mock_antenna_pattern(*args, **kwargs):
                # Call the real detector's antenna pattern method
                return detector.antenna_pattern(*args, **kwargs)

            # Use real time delay calculations from detector
            def mock_time_delay(*args, **kwargs):
                # Call the real detector's time_delay_from_earth_center method
                return detector.time_delay_from_earth_center(*args, **kwargs)

            mock_detector.antenna_pattern.side_effect = mock_antenna_pattern
            mock_detector.time_delay_from_earth_center.side_effect = mock_time_delay
            mock_detector_class.return_value = mock_detector

            sim = MockSimulator(detectors=["H1"])

            # Project with earth rotation enabled
            result_with_rotation = sim.project_polarizations(
                polarizations=sine_wave_polarizations,
                right_ascension=1.5,
                declination=0.5,
                polarization_angle=0.0,
                earth_rotation=True,
            )

            # Project with earth rotation disabled
            result_without_rotation = sim.project_polarizations(
                polarizations=sine_wave_polarizations,
                right_ascension=1.5,
                declination=0.5,
                polarization_angle=0.0,
                earth_rotation=False,
            )

            # Get middle time segment (should be consistent)
            n_samples = len(result_with_rotation._data[0])
            center_idx = n_samples // 2
            window = 256  # samples around center

            center_with = result_with_rotation._data[0][center_idx - window : center_idx + window]
            center_without = result_without_rotation._data[0][center_idx - window : center_idx + window]

            # Center should be similar (within reasonable tolerance due to interpolation and time-dependent effects)
            # With real antenna patterns and time delays, differences are very small at the center
            np.testing.assert_allclose(center_with.value, center_without.value, rtol=0.05, atol=3e-4)

            # Get edge segments - they may differ due to real antenna pattern and time delay time dependence
            edge_with = result_with_rotation._data[0][:256]
            edge_without = result_without_rotation._data[0][:256]

            # Edges should show some difference (at least 1% relative difference for real patterns)
            relative_error = np.mean(
                np.abs(edge_with.value - edge_without.value) / (np.abs(edge_without.value) + 1e-10)
            )
            expected_difference_threshold = 0.01  # 1% relative difference
            assert relative_error > expected_difference_threshold  # At least 1% relative difference at edges

    @pytest.mark.slow
    def test_earth_rotation_parameter_affects_antenna_pattern_calls(self, sine_wave_polarizations, h1_detector):
        """Test that earth_rotation parameter controls whether antenna patterns are computed at multiple times."""

        # Use the h1_detector fixture instead of creating a new one
        detector = h1_detector

        with patch("gwsim.mixin.detector.Detector") as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.is_configured.return_value = True

            # Use real antenna pattern from detector
            def mock_antenna_pattern(*args, **kwargs):
                return detector.antenna_pattern(*args, **kwargs)

            mock_detector.antenna_pattern.side_effect = mock_antenna_pattern
            mock_detector.time_delay_from_earth_center.return_value = 0.01
            mock_detector_class.return_value = mock_detector

            sim = MockSimulator(detectors=["H1"])

            # With earth_rotation=True, antenna_pattern should be called with array
            mock_detector.antenna_pattern.reset_mock()
            sim.project_polarizations(
                polarizations=sine_wave_polarizations,
                right_ascension=1.5,
                declination=0.5,
                polarization_angle=0.0,
                earth_rotation=True,
            )

            # Check that antenna_pattern was called with t_gps as array
            calls_with_rotation = mock_detector.antenna_pattern.call_args_list
            assert len(calls_with_rotation) > 0
            assert "t_gps" in calls_with_rotation[0][1]
            t_gps_value = calls_with_rotation[0][1]["t_gps"]
            assert isinstance(t_gps_value, np.ndarray)

            # With earth_rotation=False, antenna_pattern should be called with scalar
            mock_detector.antenna_pattern.reset_mock()
            sim.project_polarizations(
                polarizations=sine_wave_polarizations,
                right_ascension=1.5,
                declination=0.5,
                polarization_angle=0.0,
                earth_rotation=False,
            )

            calls_without_rotation = mock_detector.antenna_pattern.call_args_list
            assert len(calls_without_rotation) > 0
            assert "t_gps" in calls_without_rotation[0][1]
            t_gps_value = calls_without_rotation[0][1]["t_gps"]
            assert isinstance(t_gps_value, (int, float, np.number))
