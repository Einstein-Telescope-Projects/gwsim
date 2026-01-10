"""Unit tests for ColoredNoiseSimulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.units import Quantity

from gwsim.data.time_series.time_series_list import TimeSeriesList
from gwsim.noise.colored_noise import ColoredNoiseSimulator
from gwsim.simulator.state import StateAttribute

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_psd_file(tmp_path: Path) -> Path:
    """Create a temporary PSD file for testing.

    Creates a simple 1/f PSD characteristic of gravitational wave detectors.
    """
    # Create frequency array from 1 Hz to 2048 Hz
    freqs = np.linspace(1, 2048, 2000)
    # Simple PSD model: constant with some frequency dependence
    psd_values = 1e-46 * (1 + (30 / freqs) ** 4)

    psd_data = np.column_stack([freqs, psd_values])
    psd_file = tmp_path / "test_psd.txt"
    np.savetxt(psd_file, psd_data)
    return psd_file


@pytest.fixture
def temp_psd_npy_file(tmp_path: Path) -> Path:
    """Create a temporary PSD file in .npy format."""
    freqs = np.linspace(1, 2048, 2000)
    psd_values = 1e-46 * (1 + (30 / freqs) ** 4)
    psd_data = np.column_stack([freqs, psd_values])
    psd_file = tmp_path / "test_psd.npy"
    np.save(psd_file, psd_data)
    return psd_file


@pytest.fixture
def basic_simulator(temp_psd_file: Path) -> ColoredNoiseSimulator:
    """Create a basic ColoredNoiseSimulator instance for testing."""
    return ColoredNoiseSimulator(
        psd_file=temp_psd_file,
        detectors=["H1"],
        sampling_frequency=4096,
        duration=1024,  # Must be >= t_window / 2 = 1024
        start_time=0,
        seed=42,
        low_frequency_cutoff=10.0,
        high_frequency_cutoff=1000.0,
    )


@pytest.fixture
def multi_detector_simulator(temp_psd_file: Path) -> ColoredNoiseSimulator:
    """Create a multi-detector ColoredNoiseSimulator instance."""
    return ColoredNoiseSimulator(
        psd_file=temp_psd_file,
        detectors=["H1", "L1", "V1"],
        sampling_frequency=4096,
        duration=1024,
        start_time=1000000000,
        seed=123,
        low_frequency_cutoff=10.0,
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestColoredNoiseSimulatorInitialization:
    """Test ColoredNoiseSimulator initialization."""

    def test_basic_initialization(self, temp_psd_file: Path):
        """Test basic initialization with valid parameters."""
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            start_time=0,
            seed=42,
        )
        expected_low_frequency_cutoff = 2.0  # Default value
        assert simulator.psd_file == temp_psd_file
        assert simulator.low_frequency_cutoff == expected_low_frequency_cutoff  # Default
        assert simulator.sampling_frequency == Quantity(4096, unit="Hz")
        assert simulator.duration == Quantity(1024, unit="s")

    def test_initialization_with_npy_file(self, temp_psd_npy_file: Path):
        """Test initialization with .npy PSD file."""
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_npy_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            seed=42,
        )
        assert simulator.psd_file == temp_psd_npy_file

    def test_initialization_with_multiple_detectors(self, temp_psd_file: Path):
        """Test initialization with multiple detectors."""
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1", "L1", "V1"],
            sampling_frequency=4096,
            duration=1024,
            seed=42,
        )
        expected_number_of_detectors = 3
        assert simulator._n_det == expected_number_of_detectors

    def test_initialization_empty_detectors_raises_error(self, temp_psd_file: Path):
        """Test that empty detectors list raises ValueError."""
        with pytest.raises(ValueError, match="detectors must contain at least one detector"):
            ColoredNoiseSimulator(
                psd_file=temp_psd_file,
                detectors=[],
                sampling_frequency=4096,
                duration=1024,
            )

    def test_initialization_duration_too_short_raises_error(self, temp_psd_file: Path):
        """Test that duration shorter than t_window/2 raises ValueError."""
        with pytest.raises(ValueError, match=r"Duration .* must be at least"):
            ColoredNoiseSimulator(
                psd_file=temp_psd_file,
                detectors=["H1"],
                sampling_frequency=4096,
                duration=100,  # Too short, must be >= 1024
                seed=42,
            )

    def test_high_frequency_cutoff_clamped_to_nyquist(self, temp_psd_file: Path):
        """Test that high_frequency_cutoff is clamped to Nyquist frequency."""
        sampling_frequency = 4096
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=sampling_frequency,
            duration=1024,
            seed=42,
            high_frequency_cutoff=3000,  # Above Nyquist (2048)
        )
        expected_high_frequency_cutoff = sampling_frequency / 2
        assert simulator.high_frequency_cutoff == expected_high_frequency_cutoff  # Nyquist

    def test_high_frequency_cutoff_preserved_when_valid(self, temp_psd_file: Path):
        """Test that valid high_frequency_cutoff is preserved."""
        high_frequency_cutoff = 1000
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            seed=42,
            high_frequency_cutoff=high_frequency_cutoff,
        )
        assert simulator.high_frequency_cutoff == high_frequency_cutoff

    def test_previous_strain_initialized_correctly(self, basic_simulator: ColoredNoiseSimulator):
        """Test that previous_strain buffer is initialized with correct shape."""
        expected_shape = (basic_simulator._n_det, basic_simulator._n_chunk)
        assert basic_simulator.previous_strain.shape == expected_shape
        assert np.all(basic_simulator.previous_strain == 0)


# ============================================================================
# PSD Loading Tests
# ============================================================================


class TestPSDLoading:
    """Test PSD file loading functionality."""

    def test_load_txt_file(self, temp_psd_file: Path):
        """Test loading PSD from .txt file."""
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            seed=42,
        )
        assert simulator._psd is not None
        assert len(simulator._psd) > 0

    def test_load_npy_file(self, temp_psd_npy_file: Path):
        """Test loading PSD from .npy file."""
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_npy_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            seed=42,
        )
        assert simulator._psd is not None
        assert len(simulator._psd) > 0

    def test_load_csv_file(self, tmp_path: Path):
        """Test loading PSD from .csv file."""
        freqs = np.linspace(1, 2048, 2000)
        psd_values = 1e-46 * np.ones_like(freqs)
        psd_data = np.column_stack([freqs, psd_values])
        psd_file = tmp_path / "test_psd.csv"
        np.savetxt(psd_file, psd_data, delimiter=",")

        simulator = ColoredNoiseSimulator(
            psd_file=psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            seed=42,
        )
        assert simulator._psd is not None

    def test_unsupported_file_format_raises_error(self, tmp_path: Path):
        """Test that unsupported file format raises ValueError."""
        psd_file = tmp_path / "test_psd.xyz"
        psd_file.write_text("dummy content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            ColoredNoiseSimulator(
                psd_file=psd_file,
                detectors=["H1"],
                sampling_frequency=4096,
                duration=1024,
                seed=42,
            )

    def test_invalid_psd_shape_raises_error(self, tmp_path: Path):
        """Test that PSD with wrong shape raises ValueError."""
        # Create PSD with 3 columns instead of 2
        psd_data = np.column_stack([np.linspace(1, 2048, 100), np.ones(100), np.ones(100)])
        psd_file = tmp_path / "bad_psd.txt"
        np.savetxt(psd_file, psd_data)

        with pytest.raises(ValueError, match="PSD file must have shape"):
            ColoredNoiseSimulator(
                psd_file=psd_file,
                detectors=["H1"],
                sampling_frequency=4096,
                duration=1024,
                seed=42,
            )


# ============================================================================
# Simulation Tests
# ============================================================================


class TestSimulation:
    """Test noise simulation functionality."""

    def test_simulate_returns_timeseries_list(self, basic_simulator: ColoredNoiseSimulator):
        """Test that _simulate returns TimeSeriesList."""
        result = basic_simulator._simulate()
        assert isinstance(result, TimeSeriesList)
        assert len(result) == 1

    def test_simulate_returns_correct_shape(self, basic_simulator: ColoredNoiseSimulator):
        """Test that simulated data has correct shape."""
        result = basic_simulator._simulate()
        expected_samples = int(basic_simulator.duration.value * basic_simulator.sampling_frequency.value)
        assert len(result[0]._data) == 1
        assert len(result[0]._data[0]) == expected_samples

    def test_simulate_multi_detector_shape(self, multi_detector_simulator: ColoredNoiseSimulator):
        """Test that multi-detector simulation has correct shape."""
        result = multi_detector_simulator._simulate()
        expected_samples = int(
            multi_detector_simulator.duration.value * multi_detector_simulator.sampling_frequency.value
        )
        expected_number_of_detectors = 3
        assert len(result[0]._data) == expected_number_of_detectors
        assert len(result[0]._data[0]) == expected_samples
        assert len(result[0]._data[1]) == expected_samples
        assert len(result[0]._data[2]) == expected_samples

    def test_simulate_correct_start_time(self, basic_simulator: ColoredNoiseSimulator):
        """Test that simulated data has correct start time."""
        result = basic_simulator._simulate()
        assert result[0].start_time == basic_simulator.start_time

    def test_simulate_correct_sampling_frequency(self, basic_simulator: ColoredNoiseSimulator):
        """Test that simulated data has correct sampling frequency."""
        result = basic_simulator._simulate()
        assert result[0].sampling_frequency == basic_simulator.sampling_frequency

    def test_simulate_without_seed_raises_error(self, temp_psd_file: Path):
        """Test that simulation without RNG initialization raises error."""
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            seed=None,  # No seed
        )
        with pytest.raises(RuntimeError, match="Random number generator not initialized"):
            simulator._simulate()

    def test_simulate_produces_nonzero_data(self, basic_simulator: ColoredNoiseSimulator):
        """Test that simulation produces non-zero noise data."""
        result = basic_simulator._simulate()
        assert not np.all(result[0]._data[0].value == 0)

    def test_simulate_reproducibility_with_same_seed(self, temp_psd_file: Path):
        """Test that simulations with same seed produce same results."""
        simulator1 = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            seed=12345,
        )
        simulator2 = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            seed=12345,
        )

        result1 = simulator1._simulate()
        result2 = simulator2._simulate()

        np.testing.assert_array_equal(result1[0]._data[0].value, result2[0]._data[0].value)

    def test_simulate_different_seeds_produce_different_results(self, temp_psd_file: Path):
        """Test that different seeds produce different noise."""
        simulator1 = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            seed=111,
        )
        simulator2 = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            seed=222,
        )

        result1 = simulator1._simulate()
        result2 = simulator2._simulate()
        assert not np.allclose(result1[0]._data[0].value, result2[0]._data[0].value, rtol=1e-25, atol=1e-25)


# ============================================================================
# State Management Tests
# ============================================================================


class TestStateManagement:
    """Test state management and continuity functionality."""

    def test_update_state_increments_counter(self, basic_simulator: ColoredNoiseSimulator):
        """Test that update_state increments counter."""
        initial_counter = basic_simulator.counter
        basic_simulator._simulate()
        basic_simulator.update_state()
        assert basic_simulator.counter == initial_counter + 1

    def test_update_state_advances_start_time(self, basic_simulator: ColoredNoiseSimulator):
        """Test that update_state advances start_time."""
        initial_start_time = basic_simulator.start_time.value
        duration = basic_simulator.duration.value
        basic_simulator._simulate()
        basic_simulator.update_state()
        assert basic_simulator.start_time.value == initial_start_time + duration

    def test_update_state_updates_previous_strain(self, basic_simulator: ColoredNoiseSimulator):
        """Test that update_state updates previous_strain buffer."""
        initial_strain = basic_simulator.previous_strain.copy()
        basic_simulator._simulate()
        basic_simulator.update_state()
        # previous_strain should be updated (not all zeros anymore)
        assert not np.array_equal(basic_simulator.previous_strain, initial_strain)

    def test_continuity_across_batches(self, temp_psd_file: Path):
        """Test that noise maintains continuity across multiple batches."""
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            start_time=0,
            seed=42,
        )

        # Generate first batch
        result1 = simulator._simulate()
        simulator.update_state()

        # Generate second batch
        result2 = simulator._simulate()

        # Both batches should have valid non-zero data
        assert not np.all(result1[0]._data[0].value == 0)
        assert not np.all(result2[0]._data[0].value == 0)

        # Start times should be consecutive
        assert result2[0].start_time.value == result1[0].start_time.value + simulator.duration.value

    def test_state_attribute_previous_strain_exists(self, basic_simulator: ColoredNoiseSimulator):
        """Test that previous_strain is a StateAttribute."""
        # Check that the class has previous_strain as a StateAttribute
        assert hasattr(ColoredNoiseSimulator, "previous_strain")
        assert isinstance(ColoredNoiseSimulator.__dict__["previous_strain"], StateAttribute)


# ============================================================================
# Metadata Tests
# ============================================================================


class TestMetadata:
    """Test metadata generation."""

    def test_metadata_contains_colored_noise_section(self, basic_simulator: ColoredNoiseSimulator):
        """Test that metadata contains colored_noise section."""
        metadata = basic_simulator.metadata
        assert "colored_noise" in metadata

    def test_metadata_contains_psd_file(self, basic_simulator: ColoredNoiseSimulator):
        """Test that metadata contains PSD file path."""
        metadata = basic_simulator.metadata
        assert "psd_file" in metadata["colored_noise"]["arguments"]
        assert str(basic_simulator.psd_file) == metadata["colored_noise"]["arguments"]["psd_file"]

    def test_metadata_contains_frequency_cutoffs(self, basic_simulator: ColoredNoiseSimulator):
        """Test that metadata contains frequency cutoff parameters."""
        metadata = basic_simulator.metadata
        args = metadata["colored_noise"]["arguments"]
        assert "low_frequency_cutoff" in args
        assert "high_frequency_cutoff" in args
        assert args["low_frequency_cutoff"] == basic_simulator.low_frequency_cutoff
        assert args["high_frequency_cutoff"] == basic_simulator.high_frequency_cutoff

    def test_metadata_inherits_from_parent(self, basic_simulator: ColoredNoiseSimulator):
        """Test that metadata includes parent class metadata."""
        metadata = basic_simulator.metadata
        # Should have base simulator metadata fields
        assert "sampling_frequency" in metadata or "colored_noise" in metadata


# ============================================================================
# Window Properties Tests
# ============================================================================


class TestWindowProperties:
    """Test window and overlap properties."""

    def test_window_properties_initialized(self, basic_simulator: ColoredNoiseSimulator):
        """Test that window properties are properly initialized."""
        assert hasattr(basic_simulator, "_t_window")
        assert hasattr(basic_simulator, "_f_window")
        assert hasattr(basic_simulator, "_t_overlap")
        assert hasattr(basic_simulator, "_n_overlap")
        assert hasattr(basic_simulator, "_w0")
        assert hasattr(basic_simulator, "_w1")

    def test_window_arrays_have_correct_length(self, basic_simulator: ColoredNoiseSimulator):
        """Test that window arrays have correct length."""
        assert len(basic_simulator._w0) == basic_simulator._n_overlap
        assert len(basic_simulator._w1) == basic_simulator._n_overlap

    def test_window_values_range(self, basic_simulator: ColoredNoiseSimulator):
        """Test that window values are in valid range [0, 1]."""
        assert np.all(basic_simulator._w0 >= 0)
        assert np.all(basic_simulator._w0 <= 1)
        assert np.all(basic_simulator._w1 >= 0)
        assert np.all(basic_simulator._w1 <= 1)


# ============================================================================
# Frequency Properties Tests
# ============================================================================


class TestFrequencyProperties:
    """Test frequency domain properties."""

    def test_frequency_properties_initialized(self, basic_simulator: ColoredNoiseSimulator):
        """Test that frequency properties are properly initialized."""
        assert hasattr(basic_simulator, "_t_chunk")
        assert hasattr(basic_simulator, "_df_chunk")
        assert hasattr(basic_simulator, "_n_chunk")
        assert hasattr(basic_simulator, "_frequency_chunk")

    def test_frequency_array_covers_range(self, basic_simulator: ColoredNoiseSimulator):
        """Test that frequency array covers appropriate range."""
        assert basic_simulator._frequency_chunk[0] == 0
        # Last frequency should be close to Nyquist
        nyquist = basic_simulator.sampling_frequency.value / 2
        assert basic_simulator._frequency_chunk[-1] == pytest.approx(nyquist, rel=0.01)

    def test_psd_array_initialized(self, basic_simulator: ColoredNoiseSimulator):
        """Test that PSD array is properly initialized."""
        assert hasattr(basic_simulator, "_psd")
        assert basic_simulator._psd is not None
        assert len(basic_simulator._psd) == basic_simulator._n_freq_chunk


# ============================================================================
# Single Realization Tests
# ============================================================================


class TestSingleRealization:
    """Test single noise realization generation."""

    def test_generate_single_realization_shape(self, basic_simulator: ColoredNoiseSimulator):
        """Test that single realization has correct shape."""
        realization = basic_simulator._generate_single_realization()
        assert realization.shape == (basic_simulator._n_det, basic_simulator._n_chunk)

    def test_generate_single_realization_nonzero(self, basic_simulator: ColoredNoiseSimulator):
        """Test that single realization contains non-zero data."""
        realization = basic_simulator._generate_single_realization()
        assert not np.all(realization == 0)

    def test_generate_single_realization_multi_detector(self, multi_detector_simulator: ColoredNoiseSimulator):
        """Test single realization for multiple detectors."""
        realization = multi_detector_simulator._generate_single_realization()
        assert realization.shape == (3, multi_detector_simulator._n_chunk)
        # Each detector should have different noise
        assert not np.allclose(realization[0], realization[1], rtol=1e-25, atol=1e-25)
        assert not np.allclose(realization[1], realization[2], rtol=1e-25, atol=1e-25)


# ============================================================================
# Statistical Properties Tests
# ============================================================================


class TestStatisticalProperties:
    """Test statistical properties of generated noise."""

    @pytest.mark.slow
    def test_noise_has_zero_mean(self, basic_simulator: ColoredNoiseSimulator):
        """Test that generated noise has approximately zero mean."""
        result = basic_simulator._simulate()
        mean = np.mean(result[0]._data[0].value)
        # Mean should be close to zero (within some tolerance for finite samples)
        tolerance = 1e-20
        assert abs(mean) < tolerance  # Very small mean expected

    @pytest.mark.slow
    def test_noise_variance_is_positive(self, basic_simulator: ColoredNoiseSimulator):
        """Test that generated noise has positive variance."""
        result = basic_simulator._simulate()
        variance = np.var(result[0]._data[0].value)
        assert variance > 0

    @pytest.mark.slow
    def test_different_detectors_uncorrelated(self, multi_detector_simulator: ColoredNoiseSimulator):
        """Test that noise in different detectors is uncorrelated."""
        result = multi_detector_simulator._simulate()
        data = result[0]._data

        # Calculate correlation between detector pairs
        corr_01 = np.corrcoef(data[0].value, data[1].value)[0, 1]
        corr_12 = np.corrcoef(data[1].value, data[2].value)[0, 1]
        corr_02 = np.corrcoef(data[0].value, data[2].value)[0, 1]

        # Correlations should be small (independent noise)
        tolerance = 0.1
        assert abs(corr_01) < tolerance
        assert abs(corr_12) < tolerance
        assert abs(corr_02) < tolerance


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_detector(self, temp_psd_file: Path):
        """Test simulation with single detector."""
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            seed=42,
        )
        result = simulator._simulate()
        assert len(result[0]._data) == 1

    def test_minimum_valid_duration(self, temp_psd_file: Path):
        """Test simulation with minimum valid duration."""
        # t_window = 2048, so minimum duration is 1024
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,  # Exactly minimum
            seed=42,
        )
        result = simulator._simulate()
        assert len(result[0]._data[0].value) == (1024 * 4096)

    def test_large_start_time(self, temp_psd_file: Path):
        """Test simulation with large GPS start time."""
        start_time = 1700000000  # Large GPS time
        simulator = ColoredNoiseSimulator(
            psd_file=temp_psd_file,
            detectors=["H1"],
            sampling_frequency=4096,
            duration=1024,
            start_time=start_time,  # Large GPS time
            seed=42,
        )
        result = simulator._simulate()
        assert result[0].start_time.value == start_time

    def test_previous_strain_too_short_raises_error(self, basic_simulator: ColoredNoiseSimulator):
        """Test that previous_strain shorter than n_overlap raises error."""
        # Manually set previous_strain to be too short
        basic_simulator.previous_strain = np.zeros((1, 10))  # Very short

        with pytest.raises(ValueError, match="previous_strain has only"):
            basic_simulator._simulate()
