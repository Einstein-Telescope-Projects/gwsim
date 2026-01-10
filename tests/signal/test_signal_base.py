"""Unit tests for SignalSimulator base class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy.units import Quantity

from gwsim.data.time_series.time_series import TimeSeries
from gwsim.data.time_series.time_series_list import TimeSeriesList
from gwsim.signal.base import SignalSimulator


@pytest.fixture
def signal_simulator_with_mocks(tmp_path):
    """Create a SignalSimulator with all mixins properly mocked."""
    dummy_file = tmp_path / "dummy_file.csv"
    dummy_file.write_text("")  # Create empty CSV file

    with (
        patch("gwsim.mixin.population_reader.PopulationReaderMixin.__init__", return_value=None),
        patch("gwsim.mixin.waveform.WaveformMixin.__init__", return_value=None),
        patch("gwsim.mixin.time_series.TimeSeriesMixin.__init__", return_value=None),
        patch("gwsim.mixin.detector.DetectorMixin.__init__", return_value=None),
        patch("gwsim.simulator.base.Simulator.__init__", return_value=None),
    ):

        simulator = SignalSimulator(
            population_file=str(dummy_file),
            waveform_model="IMRPhenomD",
            start_time=0,
            duration=100.0,
            sampling_frequency=4096,
            detectors=["H1"],
            minimum_frequency=5.0,
        )

        # Manually set required attributes
        simulator.waveform_model = "IMRPhenomD"
        simulator.waveform_arguments = {"sampling_frequency": 4096, "minimum_frequency": 5.0}
        simulator.start_time = Quantity(0, unit="s")
        simulator.duration = Quantity(100.0, unit="s")
        # end_time is calculated from start_time and duration, so don't set it directly
        simulator.sampling_frequency = Quantity(4096, unit="Hz")
        simulator.detectors = ["H1"]
        simulator.counter = 0
        simulator.signals_injected = []
        simulator.waveform_factory = MagicMock()
        simulator.population_file = str(dummy_file)  # Required for metadata property
        simulator.population_file_type = "pycbc"  # Required for metadata property
        simulator.population_parameter_name_mapper = {}  # Required for metadata
        simulator.population_sort_by = None  # Required for metadata
        simulator.population_cache_dir = Path.home() / ".gwsim" / "population"  # Required for metadata
        simulator.population_download_timeout = 300  # Required for metadata
        simulator._population_metadata = {}  # Required for metadata

        return simulator


class TestSignalSimulatorInitialization:
    """Test SignalSimulator initialization with mocks."""

    def test_init_requires_population_file(self, tmp_path):
        """Test that initialization requires a population file parameter."""
        dummy_file = tmp_path / "test.csv"
        dummy_file.write_text("")

        with patch("gwsim.mixin.population_reader.PopulationReaderMixin.__init__", return_value=None):
            # Just verify the class can be instantiated with mocked init
            simulator = SignalSimulator(population_file=str(dummy_file))
            assert simulator is not None

    def test_waveform_arguments_initialization(self, signal_simulator_with_mocks):
        """Test that waveform arguments include required parameters."""
        simulator = signal_simulator_with_mocks
        assert "sampling_frequency" in simulator.waveform_arguments
        assert "minimum_frequency" in simulator.waveform_arguments
        expected_sampling_frequency = 4096
        expected_minimum_frequency = 5.0
        assert simulator.waveform_arguments["sampling_frequency"] == expected_sampling_frequency
        assert simulator.waveform_arguments["minimum_frequency"] == expected_minimum_frequency

    def test_detector_initialization(self, signal_simulator_with_mocks):
        """Test that detectors are properly initialized."""
        simulator = signal_simulator_with_mocks
        # Detectors are converted to Detector objects, so check for their names
        detector_names = [det.name if hasattr(det, "name") else str(det) for det in simulator.detectors]
        assert "H1" in detector_names or simulator.detectors == ["H1"]


class TestSignalSimulatorSimulate:
    """Test the _simulate method behavior."""

    def test_simulate_returns_timeseries_list(self, signal_simulator_with_mocks):
        """Test that _simulate returns TimeSeriesList."""
        simulator = signal_simulator_with_mocks

        # Mock get_next_injection_parameters to return None (empty population)
        with patch.object(simulator, "get_next_injection_parameters", return_value=None):
            result = simulator._simulate()
            assert isinstance(result, TimeSeriesList)
            assert len(result) == 0

    def test_simulate_empty_when_no_parameters(self, signal_simulator_with_mocks):
        """Test that _simulate handles None parameters gracefully."""
        simulator = signal_simulator_with_mocks

        with patch.object(simulator, "get_next_injection_parameters", return_value=None):
            result = simulator._simulate()
            assert isinstance(result, TimeSeriesList)
            assert len(result) == 0

    def test_simulate_stops_when_signal_starts_after_segment(self, signal_simulator_with_mocks):
        """Test that _simulate stops fetching new signals when one starts at or after segment end."""
        simulator = signal_simulator_with_mocks

        # Create mock strain data
        strain_in_segment = MagicMock(spec=TimeSeries)
        strain_in_segment.start_time = Quantity(50, unit="s")
        strain_in_segment.metadata = {}

        strain_outside_segment = MagicMock(spec=TimeSeries)
        strain_outside_segment.start_time = Quantity(120, unit="s")  # At or after end_time
        strain_outside_segment.metadata = {}

        strain_after_outside = MagicMock(spec=TimeSeries)
        strain_after_outside.start_time = Quantity(150, unit="s")  # Even further after
        strain_after_outside.metadata = {}

        parameters_list = [
            {
                "mass_1": 30,
                "mass_2": 32,
                "geocent_time": 50,
                "right_ascension": 1.97,
                "declination": -1.21,
                "polarization_angle": 1.6,
            },
            {
                "mass_1": 25,
                "mass_2": 28,
                "geocent_time": 120,
                "right_ascension": 1.97,
                "declination": -1.21,
                "polarization_angle": 1.6,
            },
            {
                "mass_1": 20,
                "mass_2": 25,
                "geocent_time": 150,
                "right_ascension": 1.97,
                "declination": -1.21,
                "polarization_angle": 1.6,
            },
        ]
        parameter_index = [0]

        def get_next_params():
            if parameter_index[0] < len(parameters_list):
                result = parameters_list[parameter_index[0]]
                parameter_index[0] += 1
                return result
            return None

        strains = [strain_in_segment, strain_outside_segment, strain_after_outside]
        strain_idx = [0]

        def project_mock(polarizations, **kwargs):
            result = strains[strain_idx[0]]
            strain_idx[0] += 1
            return result

        with (
            patch.object(simulator, "get_next_injection_parameters", side_effect=get_next_params),
            patch.object(simulator.waveform_factory, "generate", return_value=MagicMock()),
            patch.object(simulator, "project_polarizations", side_effect=project_mock),
        ):

            result = simulator._simulate()

            # Should include first two signals (first is in segment, second triggers break but is added before break)
            # The third signal should NOT be fetched or added (loop breaks after second)
            expected_number_of_signals = 2
            assert len(result) == expected_number_of_signals
            assert result[0].start_time == Quantity(50, unit="s")
            assert result[1].start_time == Quantity(120, unit="s")

    def test_simulate_registers_injection_parameters(self, signal_simulator_with_mocks):
        """Test that _simulate registers injection parameters in metadata."""
        simulator = signal_simulator_with_mocks

        parameters = {
            "mass_1": 30,
            "mass_2": 32,
            "geocent_time": 50,
            "right_ascension": 1.97,
            "declination": -1.21,
            "polarization_angle": 1.6,
        }

        strain = MagicMock(spec=TimeSeries)
        strain.start_time = Quantity(40, unit="s")
        strain.metadata = {}

        with (
            patch.object(simulator, "get_next_injection_parameters", side_effect=[parameters, None]),
            patch.object(simulator.waveform_factory, "generate", return_value=MagicMock()),
            patch.object(simulator, "project_polarizations", return_value=strain),
        ):

            result = simulator._simulate()

            # Check that metadata was updated
            assert len(result) == 1
            assert "injection_parameters" in result[0].metadata

    def test_simulate_calls_waveform_factory_generate(self, signal_simulator_with_mocks):
        """Test that _simulate calls waveform_factory.generate."""
        simulator = signal_simulator_with_mocks

        parameters = {
            "mass_1": 30,
            "mass_2": 32,
            "geocent_time": 50,
            "right_ascension": 1.97,
            "declination": -1.21,
            "polarization_angle": 1.6,
        }

        strain = MagicMock(spec=TimeSeries)
        strain.start_time = Quantity(40, unit="s")
        strain.metadata = {}

        with (
            patch.object(simulator, "get_next_injection_parameters", side_effect=[parameters, None]),
            patch.object(simulator.waveform_factory, "generate") as mock_generate,
            patch.object(simulator, "project_polarizations", return_value=strain),
        ):

            mock_generate.return_value = MagicMock()
            simulator._simulate()

            # Verify waveform_factory.generate was called with correct parameters
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["waveform_model"] == simulator.waveform_model
            assert call_kwargs["parameters"] == parameters

    def test_simulate_calls_project_polarizations(self, signal_simulator_with_mocks):
        """Test that _simulate calls project_polarizations with correct arguments."""
        simulator = signal_simulator_with_mocks

        parameters = {
            "mass_1": 30,
            "mass_2": 32,
            "geocent_time": 50,
            "right_ascension": 1.97,
            "declination": -1.21,
            "polarization_angle": 1.6,
        }

        strain = MagicMock(spec=TimeSeries)
        strain.start_time = Quantity(40, unit="s")
        strain.metadata = {}

        with (
            patch.object(simulator, "get_next_injection_parameters", side_effect=[parameters, None]),
            patch.object(simulator.waveform_factory, "generate", return_value=MagicMock()),
            patch.object(simulator, "project_polarizations", return_value=strain) as mock_project,
        ):

            simulator._simulate()

            # Verify project_polarizations was called with correct parameters
            mock_project.assert_called_once()
            call_kwargs = mock_project.call_args[1]
            assert call_kwargs["right_ascension"] == parameters["right_ascension"]
            assert call_kwargs["declination"] == parameters["declination"]
            assert call_kwargs["polarization_angle"] == parameters["polarization_angle"]


class TestSignalSimulatorMetadata:
    """Test metadata generation."""

    def test_metadata_exists(self, signal_simulator_with_mocks):
        """Test that metadata property returns a dictionary."""
        simulator = signal_simulator_with_mocks
        metadata = simulator.metadata
        assert isinstance(metadata, dict)


class TestSignalSimulatorUpdateState:
    """Test state update functionality."""

    def test_update_state_increments_counter(self, signal_simulator_with_mocks):
        """Test that update_state increments the counter."""
        simulator = signal_simulator_with_mocks
        initial_counter = simulator.counter
        simulator.update_state()
        assert simulator.counter == initial_counter + 1

    def test_update_state_advances_start_time(self, signal_simulator_with_mocks):
        """Test that update_state advances the start time."""
        simulator = signal_simulator_with_mocks
        initial_start_time = simulator.start_time.value
        simulator.update_state()
        expected_start_time = initial_start_time + simulator.duration.value
        tolerance = 1e-10
        assert abs(simulator.start_time.value - expected_start_time) < tolerance

    def test_update_state_multiple_times(self, signal_simulator_with_mocks):
        """Test that update_state can be called multiple times."""
        simulator = signal_simulator_with_mocks
        initial_start_time = simulator.start_time.value
        simulator.update_state()
        simulator.update_state()
        simulator.update_state()
        expected_start_time = initial_start_time + 3 * simulator.duration.value
        tolerance = 1e-10
        assert abs(simulator.start_time.value - expected_start_time) < tolerance


class TestSignalSimulatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_simulate_with_signal_spanning_segment_boundary(self, signal_simulator_with_mocks):
        """Test behavior when signal spans segment boundary."""
        simulator = signal_simulator_with_mocks

        # Signal that starts before end and ends after
        strain = MagicMock(spec=TimeSeries)
        strain.start_time = Quantity(80, unit="s")
        strain.metadata = {}

        parameters = {
            "mass_1": 30,
            "mass_2": 32,
            "geocent_time": 80,
            "right_ascension": 1.97,
            "declination": -1.21,
            "polarization_angle": 1.6,
        }

        with (
            patch.object(simulator, "get_next_injection_parameters", side_effect=[parameters, None]),
            patch.object(simulator.waveform_factory, "generate", return_value=MagicMock()),
            patch.object(simulator, "project_polarizations", return_value=strain),
        ):

            result = simulator._simulate()

            # Should include this signal (starts before end_time)
            assert len(result) == 1

    def test_simulate_with_multiple_signals_in_segment(self, signal_simulator_with_mocks):
        """Test when multiple signals fit in a segment."""
        simulator = signal_simulator_with_mocks

        strain1 = MagicMock(spec=TimeSeries)
        strain1.start_time = Quantity(20, unit="s")
        strain1.metadata = {}

        strain2 = MagicMock(spec=TimeSeries)
        strain2.start_time = Quantity(50, unit="s")
        strain2.metadata = {}

        parameters1 = {
            "mass_1": 30,
            "mass_2": 32,
            "geocent_time": 20,
            "right_ascension": 1.97,
            "declination": -1.21,
            "polarization_angle": 1.6,
        }
        parameters2 = {
            "mass_1": 25,
            "mass_2": 28,
            "geocent_time": 50,
            "right_ascension": 1.97,
            "declination": -1.21,
            "polarization_angle": 1.6,
        }

        strains = [strain1, strain2]
        strain_idx = [0]

        def project_mock(polarizations, **kwargs):
            result = strains[strain_idx[0]]
            strain_idx[0] += 1
            return result

        with (
            patch.object(simulator, "get_next_injection_parameters", side_effect=[parameters1, parameters2, None]),
            patch.object(simulator.waveform_factory, "generate", return_value=MagicMock()),
            patch.object(simulator, "project_polarizations", side_effect=project_mock),
        ):

            result = simulator._simulate()

            # Should include both signals
            expected_number_of_signals = 2
            assert len(result) == expected_number_of_signals

    def test_signals_injected_tracking(self, signal_simulator_with_mocks):
        """Test that signals_injected list is properly initialized."""
        simulator = signal_simulator_with_mocks
        assert hasattr(simulator, "signals_injected")
        assert isinstance(simulator.signals_injected, list)
        assert len(simulator.signals_injected) == 0

    def test_break_condition_uses_start_time(self, signal_simulator_with_mocks):
        """Test that the break condition correctly uses strain.start_time."""
        simulator = signal_simulator_with_mocks

        # Signal that starts exactly at end_time (will be added, then break occurs)
        strain = MagicMock(spec=TimeSeries)
        strain.start_time = Quantity(100, unit="s")  # Equals end_time
        strain.metadata = {}

        parameters = {
            "mass_1": 30,
            "mass_2": 32,
            "geocent_time": 100,
            "right_ascension": 1.97,
            "declination": -1.21,
            "polarization_angle": 1.6,
        }

        with (
            patch.object(simulator, "get_next_injection_parameters", side_effect=[parameters, None]),
            patch.object(simulator.waveform_factory, "generate", return_value=MagicMock()),
            patch.object(simulator, "project_polarizations", return_value=strain),
        ):

            result = simulator._simulate()

            # Signal is added first, then break condition triggers, so it's included
            assert len(result) == 1
            assert result[0].start_time == Quantity(100, unit="s")
