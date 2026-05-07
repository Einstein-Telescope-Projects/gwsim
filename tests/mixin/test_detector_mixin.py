"""Unit tests for the DetectorMixin class."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock.mixin.detector import DetectorMixin
from gwmock.simulator.base import Simulator


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
        with patch("gwmock.mixin.detector.SignalAdapter.resolve_detector_network") as mock_resolve:
            detectors = ["H1", "L1"]
            mock_resolve.return_value = SimpleNamespace(detector_names=tuple(detectors))
            sim = MockSimulator(detectors=detectors)

            assert sim._detectors == detectors
            mock_resolve.assert_called_once_with(detectors)

    def test_init_with_detectors_list_of_config_files(self, tmp_path: Path):
        """Test initialization with a list of config file paths."""
        config_path = tmp_path / "custom.interferometer"
        sentinel_detector = SimpleNamespace(name="custom")
        with (
            patch.object(Path, "is_file", return_value=True),
            patch(
                "gwmock.mixin.detector.SignalAdapter.resolve_detector_network",
                return_value=SimpleNamespace(detector_names=(sentinel_detector,)),
            ) as mock_resolve,
        ):
            sim = MockSimulator(detectors=[str(config_path)])

        assert sim._detectors == [sentinel_detector]
        mock_resolve.assert_called_once_with([str(config_path)])

    def test_init_with_detectors_list_of_preset_names(self):
        """Test initialization with a public preset name."""
        sentinel_detectors = (
            SimpleNamespace(name="ET1_SARD"),
            SimpleNamespace(name="ET2_SARD"),
            SimpleNamespace(name="ET3_SARD"),
        )
        with patch(
            "gwmock.mixin.detector.SignalAdapter.resolve_detector_network",
            return_value=SimpleNamespace(detector_names=sentinel_detectors),
        ) as mock_resolve:
            sim = MockSimulator(detectors=["ET-Triangle-Sardinia"])

        assert sim._detectors == list(sentinel_detectors)
        mock_resolve.assert_called_once_with(["ET-Triangle-Sardinia"])

    def test_detectors_property_getter(self):
        """Test the detectors property getter."""
        sim = MockSimulator(detectors=None)
        assert sim.detectors == []

        with patch(
            "gwmock.mixin.detector.SignalAdapter.resolve_detector_network",
            return_value=SimpleNamespace(detector_names=("H1",)),
        ):
            sim = MockSimulator(detectors=["H1"])
            assert sim.detectors is not None
            assert len(sim.detectors) == 1

    def test_detectors_property_setter(self):
        """Test the detectors property setter."""
        sim = MockSimulator()
        sim.detectors = None
        assert sim._detectors == []

        with patch(
            "gwmock.mixin.detector.SignalAdapter.resolve_detector_network",
            return_value=SimpleNamespace(detector_names=("H1", "L1")),
        ):
            sim.detectors = ["H1", "L1"]
            assert sim._detectors == ["H1", "L1"]

    def test_metadata_property(self):
        """Test the metadata property."""
        sim = MockSimulator(detectors=None)
        metadata = sim.metadata
        assert metadata == {"detector": {"arguments": {"detectors": None}}}

        with patch(
            "gwmock.mixin.detector.SignalAdapter.resolve_detector_network",
            return_value=SimpleNamespace(detector_names=("H1",)),
        ):
            sim = MockSimulator(detectors=["H1"])
            metadata = sim.metadata
            assert "detector" in metadata
            assert len(metadata["detector"]["arguments"]["detectors"]) == 1

    def test_detectors_are_configured_for_known_builtin_detector(self):
        """Known built-in detectors should be treated as configured."""
        sim = MockSimulator(detectors=["H1"])
        assert sim.detectors_are_configured() is True

    def test_detectors_are_configured_is_false_for_unknown_detector(self):
        """Unknown string detectors should not be treated as configured."""
        with patch(
            "gwmock.mixin.detector.SignalAdapter.resolve_detector_network",
            return_value=SimpleNamespace(detector_names=("NOT_A_DETECTOR",)),
        ):
            sim = MockSimulator(detectors=["NOT_A_DETECTOR"])
        assert sim.detectors_are_configured() is False


class TestProjectPolarizationsEarthRotation:
    """Test suite for earth rotation effects in polarization projection."""

    @pytest.fixture(scope="class")
    def sine_wave_polarizations(self):
        """Create synthetic sine wave polarizations for testing.

        Returns a dictionary with 'plus' and 'cross' polarizations as simple
        sine waves to avoid expensive waveform calculations.
        """
        # Keep the segment long enough to capture earth-rotation drift without making the
        # projection tests prohibitively expensive.
        duration = 30 * 60  # 30 minutes
        sampling_rate = 4  # Hz
        num_samples = int(duration * sampling_rate)
        t0 = 1000000000  # GPS time reference

        # Create time array
        times = np.arange(num_samples) / sampling_rate + t0

        # Create simple sine wave polarizations
        frequency = 0.5  # Hz
        hp_data = np.sin(2 * np.pi * frequency * (times - t0))
        hc_data = 0.5 * np.cos(2 * np.pi * frequency * (times - t0))

        # Convert to GWpy TimeSeries
        hp = GWpyTimeSeries(hp_data, times=times, sample_rate=sampling_rate)
        hc = GWpyTimeSeries(hc_data, times=times, sample_rate=sampling_rate)

        return {"plus": hp, "cross": hc}

    @pytest.fixture(scope="class")
    def projection_results(self, sine_wave_polarizations):
        """Compute the with/without earth-rotation projections once for the slow tests."""
        sim = MockSimulator(detectors=["H1"])
        result_with_rotation = sim.project_polarizations(
            polarizations=sine_wave_polarizations,
            right_ascension=1.5,
            declination=0.5,
            polarization_angle=0.0,
            earth_rotation=True,
        )
        result_without_rotation = sim.project_polarizations(
            polarizations=sine_wave_polarizations,
            right_ascension=1.5,
            declination=0.5,
            polarization_angle=0.0,
            earth_rotation=False,
        )
        return result_with_rotation, result_without_rotation

    @pytest.mark.slow
    def test_earth_rotation_produces_different_results(self, projection_results):
        """Test that earth_rotation=True produces different results than earth_rotation=False."""
        result_with_rotation, result_without_rotation = projection_results
        assert not np.allclose(result_with_rotation[0].value, result_without_rotation[0].value, atol=1e-5)

    @pytest.mark.slow
    def test_earth_rotation_center_point_consistency(self, projection_results):
        """Test that segments near the center time are consistent between earth_rotation modes.

        When earth_rotation=False, the antenna pattern and time delay are computed
        at the middle time. Segments near the middle should match closely between
        earth_rotation=True and earth_rotation=False, while segments far from the
        center should differ significantly.
        """

        result_with_rotation, result_without_rotation = projection_results
        n_samples = len(result_with_rotation._data[0])
        center_idx = n_samples // 2
        window = 16

        center_with = result_with_rotation._data[0][center_idx - window : center_idx + window]
        center_without = result_without_rotation._data[0][center_idx - window : center_idx + window]
        np.testing.assert_allclose(center_with.value, center_without.value, rtol=0.05, atol=3e-4)

        edge_with = result_with_rotation._data[0][:64]
        edge_without = result_without_rotation._data[0][:64]
        relative_error = np.mean(np.abs(edge_with.value - edge_without.value) / (np.abs(edge_without.value) + 1e-10))
        assert relative_error > 0.001

    @pytest.mark.slow
    def test_earth_rotation_parameter_is_forwarded_to_public_projection(self, sine_wave_polarizations):
        """The mixin should pass the earth_rotation flag through to the public projection API."""

        projected_strain = GWpyTimeSeries([1.0, 2.0, 3.0, 4.0], t0=1000000000, sample_rate=4)
        sim = MockSimulator(detectors=["H1"])

        with patch(
            "gwmock.mixin.detector.project_polarizations_to_network",
            return_value={"H1": projected_strain},
        ) as mock_project:
            sim.project_polarizations(
                polarizations=sine_wave_polarizations,
                right_ascension=1.5,
                declination=0.5,
                polarization_angle=0.0,
                earth_rotation=True,
            )
            sim.project_polarizations(
                polarizations=sine_wave_polarizations,
                right_ascension=1.5,
                declination=0.5,
                polarization_angle=0.0,
                earth_rotation=False,
            )

        assert mock_project.call_args_list[0].kwargs["earth_rotation"] is True
        assert mock_project.call_args_list[1].kwargs["earth_rotation"] is False
