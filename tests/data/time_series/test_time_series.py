"""Unit tests for the TimeSeries class."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.units import Quantity

from gwsim.data.time_series.time_series import TimeSeries
from gwsim.data.time_series.time_series_list import TimeSeriesList


@pytest.fixture
def sample_timeseries() -> TimeSeries:
    """Fixture for a sample TimeSeries instance."""
    np.random.seed(42)
    data = np.random.rand(2, 1024)  # 2 channels, 1024 samples
    start_time = Quantity(1234567890, unit="s")
    sampling_frequency = Quantity(4096, unit="Hz")
    return TimeSeries(data=data, start_time=start_time, sampling_frequency=sampling_frequency)


@pytest.fixture
def small_timeseries() -> TimeSeries:
    """Fixture for a smaller TimeSeries for injection tests."""
    data = np.ones((2, 512))  # 2 channels, 512 samples
    start_time = Quantity(1234567890.1, unit="s")  # Overlaps with sample
    sampling_frequency = Quantity(4096, unit="Hz")
    return TimeSeries(data=data, start_time=start_time, sampling_frequency=sampling_frequency)


class TestTimeSeriesInitialization:
    """Test TimeSeries initialization."""

    def test_init_with_valid_data(self, sample_timeseries: TimeSeries):
        """Test initialization with valid 2D array."""
        expected_num_of_channels = 2
        assert sample_timeseries.num_of_channels == expected_num_of_channels
        assert sample_timeseries.dtype == np.float64
        expected_shape = (2, 1024)
        assert len(sample_timeseries) == expected_shape[0]

    def test_init_with_int_start_time(self):
        """Test initialization with int start_time."""
        data = np.random.rand(1, 100)
        ts = TimeSeries(data, start_time=1000, sampling_frequency=100)
        assert ts.start_time == Quantity(1000, unit="s")

    def test_init_with_float_sampling_freq(self):
        """Test initialization with float sampling_frequency."""
        data = np.random.rand(1, 100)
        ts = TimeSeries(data, start_time=1000, sampling_frequency=100.0)
        assert ts.sampling_frequency == Quantity(100.0, unit="Hz")

    def test_init_raises_for_1d_data(self):
        """Test that 1D data raises ValueError."""
        data = np.random.rand(100)
        with pytest.raises(ValueError, match="Data must be a 2D"):
            TimeSeries(data, start_time=1000, sampling_frequency=100)


class TestTimeSeriesProperties:
    """Test TimeSeries properties."""

    def test_start_time_property(self, sample_timeseries: TimeSeries):
        """Test start_time property."""
        assert sample_timeseries.start_time == Quantity(1234567890, unit="s")

    def test_duration_property(self, sample_timeseries: TimeSeries):
        """Test duration property."""
        expected_duration = Quantity(1024 / 4096, unit="s")  # samples / freq
        assert sample_timeseries.duration == expected_duration

    def test_end_time_property(self, sample_timeseries: TimeSeries):
        """Test end_time property."""
        assert sample_timeseries.end_time == sample_timeseries.start_time + sample_timeseries.duration

    def test_sampling_frequency_property(self, sample_timeseries: TimeSeries):
        """Test sampling_frequency property."""
        assert sample_timeseries.sampling_frequency == Quantity(4096, unit="Hz")

    def test_time_array_property(self, sample_timeseries: TimeSeries):
        """Test time_array property."""
        times = sample_timeseries.time_array
        expected_len = 1024
        assert len(times) == expected_len
        assert times[0] == sample_timeseries.start_time


class TestTimeSeriesIndexing:
    """Test TimeSeries indexing and iteration."""

    def test_getitem(self, sample_timeseries: TimeSeries):
        """Test __getitem__."""
        channel = sample_timeseries[0]
        assert hasattr(channel, "value")  # GWpy TimeSeries

    def test_len(self, sample_timeseries: TimeSeries):
        """Test __len__."""
        expected_len = 2
        assert len(sample_timeseries) == expected_len

    def test_iter(self, sample_timeseries: TimeSeries):
        """Test __iter__."""
        channels = list(sample_timeseries)
        expected_num_of_channels = 2
        assert len(channels) == expected_num_of_channels


class TestTimeSeriesCrop:
    """Test TimeSeries crop method."""

    def test_crop_with_start_end(self, sample_timeseries: TimeSeries):
        """Test cropping with start and end times."""
        original_start = sample_timeseries.start_time
        original_duration = sample_timeseries.duration
        cropped = sample_timeseries.crop(
            start_time=original_start + Quantity(0.1, unit="s"), end_time=original_start + Quantity(0.2, unit="s")
        )
        assert cropped.start_time > original_start
        assert cropped.duration < original_duration

    def test_crop_returns_self(self, sample_timeseries: TimeSeries):
        """Test that crop returns self."""
        result = sample_timeseries.crop()
        assert result is sample_timeseries


class TestTimeSeriesInject:
    """Test TimeSeries inject method."""

    def test_inject_overlapping(self, sample_timeseries: TimeSeries, small_timeseries: TimeSeries):
        """Test injecting an overlapping TimeSeries."""
        original_value = sample_timeseries[0].value[512]  # Middle sample
        sample_timeseries.inject(small_timeseries)

        # Check that injection modified the data (assuming small_timeseries has ones)
        assert sample_timeseries[0].value[512] != original_value

    def test_inject_mismatched_channels_raises(self, sample_timeseries: TimeSeries):
        """Test that mismatched channel count raises ValueError."""
        wrong_channels = TimeSeries(
            np.ones((3, 100)),  # 3 channels vs 2
            start_time=sample_timeseries.start_time,
            sampling_frequency=sample_timeseries.sampling_frequency,
        )
        with pytest.raises(ValueError, match="Number of channels"):
            sample_timeseries.inject(wrong_channels)

    def test_inject_extending_beyond(self, sample_timeseries: TimeSeries):
        """Test injecting a TimeSeries that extends beyond the end."""
        extending_ts = TimeSeries(
            np.ones((2, 100)),
            start_time=sample_timeseries.end_time - Quantity(50 / 4096, unit="s"),  # Overlaps end
            sampling_frequency=sample_timeseries.sampling_frequency,
        )
        remaining = sample_timeseries.inject(extending_ts)
        assert remaining is not None
        assert isinstance(remaining, TimeSeries)


class TestTimeSeriesInjectFromList:
    """Test TimeSeries inject_from_list method."""

    def test_inject_from_list(self, sample_timeseries: TimeSeries, small_timeseries: TimeSeries):
        """Test injecting from a TimeSeriesList."""
        ts_list = TimeSeriesList([small_timeseries])
        remaining_list = sample_timeseries.inject_from_list(ts_list)
        assert isinstance(remaining_list, TimeSeriesList)


class TestTimeSeriesSerialization:
    """Test TimeSeries serialization."""

    def test_to_json_dict(self, sample_timeseries: TimeSeries):
        """Test to_json_dict produces correct structure."""
        data = sample_timeseries.to_json_dict()
        assert data["__type__"] == "TimeSeries"
        assert "data" in data
        assert "start_time" in data
        assert "start_time_unit" in data
        assert "sampling_frequency" in data
        assert "sampling_frequency_unit" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) == sample_timeseries.num_of_channels

    def test_from_json_dict_round_trip(self, sample_timeseries: TimeSeries):
        """Test round-trip serialization."""
        json_data = sample_timeseries.to_json_dict()
        reconstructed = TimeSeries.from_json_dict(json_data)
        assert reconstructed.num_of_channels == sample_timeseries.num_of_channels
        assert reconstructed.start_time == sample_timeseries.start_time
        assert reconstructed.sampling_frequency == sample_timeseries.sampling_frequency
        np.testing.assert_array_equal(reconstructed[0].value, sample_timeseries[0].value)
