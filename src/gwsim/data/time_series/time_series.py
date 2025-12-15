"""Module for handling time series data for multiple channels."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from numbers import Number
from typing import TYPE_CHECKING, overload

import numpy as np
from astropy.units.quantity import Quantity
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from gwpy.types.index import Index
from scipy.interpolate import interp1d

from gwsim.data.serialize.serializable import JSONSerializable
from gwsim.data.time_series.inject import inject

logger = logging.getLogger("gwsim")


if TYPE_CHECKING:
    from gwsim.data.time_series.time_series_list import TimeSeriesList


class TimeSeries(JSONSerializable):
    """Class representing a time series data for multiple channels."""

    def __init__(self, data: np.ndarray, start_time: int | float | Quantity, sampling_frequency: float | Quantity):
        """Initialize the TimeSeries with a list of GWPy TimeSeries objects.

        Args:
            data: 2D numpy array of shape (num_of_channels, num_samples) containing the time series data.
            start_time: Start time of the time series in GPS seconds.
            sampling_frequency: Sampling frequency of the time series in Hz.
        """
        if data.ndim != 2:
            raise ValueError("Data must be a 2D numpy array with shape (num_of_channels, num_samples).")

        if isinstance(start_time, Number):
            start_time = Quantity(start_time, unit="s")
        if isinstance(sampling_frequency, (int, float)):
            sampling_frequency = Quantity(sampling_frequency, unit="Hz")

        self._data: list[GWpyTimeSeries] = [
            GWpyTimeSeries(
                data=data[i],
                t0=start_time,
                sample_rate=sampling_frequency,
            )
            for i in range(data.shape[0])
        ]
        self.num_of_channels = int(data.shape[0])
        self.dtype = data.dtype
        self.metadata = {}

    def __len__(self) -> int:
        """Get the number of channels in the time series.

        Returns:
            Number of channels in the time series.
        """
        return self.num_of_channels

    @overload
    def __getitem__(self, index: int) -> GWpyTimeSeries: ...

    @overload
    def __getitem__(self, index: slice) -> TimeSeries: ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> np.floating: ...

    @overload
    def __getitem__(self, index: tuple[int, slice]) -> np.ndarray: ...

    @overload
    def __getitem__(self, index: tuple[slice, int]) -> np.ndarray: ...

    @overload
    def __getitem__(self, index: tuple[slice, slice]) -> np.ndarray: ...

    def _validate_channel_index(self, channel_idx: int) -> int:
        """Validate and normalize a channel index.

        Args:
            channel_idx: Channel index (may be negative).

        Returns:
            Normalized non-negative channel index.

        Raises:
            IndexError: If index is out of bounds.
        """
        if channel_idx < 0:
            channel_idx = self.num_of_channels + channel_idx
        if channel_idx < 0 or channel_idx >= self.num_of_channels:
            raise IndexError(f"Channel index {channel_idx} out of range for {self.num_of_channels} channels.")
        return channel_idx

    def _validate_sample_index(self, sample_idx: int, size: int) -> int:
        """Validate and normalize a sample index.

        Args:
            sample_idx: Sample index (may be negative).
            size: Total number of samples.

        Returns:
            Normalized non-negative sample index.

        Raises:
            IndexError: If index is out of bounds.
        """
        if sample_idx < 0:
            sample_idx = size + sample_idx
        if sample_idx < 0 or sample_idx >= size:
            raise IndexError(f"Sample index {sample_idx} out of range.")
        return sample_idx

    def _getitem_1d_int(self, index: int) -> GWpyTimeSeries:
        """Get a single channel by integer index."""
        index = self._validate_channel_index(index)
        return self._data[index]

    def _getitem_1d_slice(self, index: slice) -> TimeSeries:
        """Get multiple channels by slice."""
        sliced_data = self._data[index]
        if not sliced_data:
            raise IndexError("Slice resulted in no channels.")
        data_array = np.array([ts.value for ts in sliced_data])
        return TimeSeries(
            data=data_array,
            start_time=self.start_time,
            sampling_frequency=self.sampling_frequency,
        )

    def _getitem_2d_int_int(self, channel_idx: int, sample_idx: int) -> np.floating:
        """Get a scalar value at (channel, sample)."""
        channel_idx = self._validate_channel_index(channel_idx)
        sample_idx = self._validate_sample_index(sample_idx, self._data[channel_idx].size)
        return self._data[channel_idx].value[sample_idx]

    def _getitem_2d_int_slice(self, channel_idx: int, sample_idx: slice) -> np.ndarray:
        """Get a 1D array for a single channel with sample slice."""
        channel_idx = self._validate_channel_index(channel_idx)
        return self._data[channel_idx].value[sample_idx]

    def _getitem_2d_slice_int(self, channel_idx: slice, sample_idx: int) -> np.ndarray:
        """Get a 1D array for a single sample across multiple channels."""
        selected_channels = self._data[channel_idx]
        if not selected_channels:
            raise IndexError("Channel slice resulted in no channels.")
        sample_idx = self._validate_sample_index(sample_idx, selected_channels[0].size)
        return np.array([ts.value[sample_idx] for ts in selected_channels])

    def _getitem_2d_slice_slice(self, channel_idx: slice, sample_idx: slice) -> np.ndarray:
        """Get a 2D array for multiple channels and samples."""
        selected_channels = self._data[channel_idx]
        if not selected_channels:
            raise IndexError("Channel slice resulted in no channels.")
        return np.array([ts.value[sample_idx] for ts in selected_channels])

    def __getitem__(
        self, index: int | slice | tuple[int | slice, int | slice]
    ) -> GWpyTimeSeries | TimeSeries | np.ndarray | np.floating:
        """Get time series data by channel index, slice, or 2D indexing.

        Supports:
        - Single channel: `ts[0]` → `GWpyTimeSeries`
        - Channel slice: `ts[0:2]` → `TimeSeries` with 2 channels
        - Negative indices: `ts[-1]` → last channel
        - 2D indexing (channel, sample):
            - `ts[0, 5]` → scalar value at channel 0, sample 5
            - `ts[0, :]` → `np.ndarray` (all samples of channel 0)
            - `ts[:, 5]` → `np.ndarray` (sample 5 across all channels)
            - `ts[0:2, 3:10]` → `np.ndarray` (2D slice)

        Args:
            index: Channel index (int), slice, or tuple of (channel_index, sample_index).

        Returns:
            GWpyTimeSeries for single channel, TimeSeries for channel slice,
            np.ndarray for 2D indexing.

        Raises:
            IndexError: If index is out of bounds.
            TypeError: If index type is unsupported.
        """
        # 1D indexing: channel selection
        if isinstance(index, int):
            return self._getitem_1d_int(index)

        if isinstance(index, slice):
            return self._getitem_1d_slice(index)

        # 2D indexing: (channel, sample) selection
        if isinstance(index, tuple) and len(index) == 2:
            channel_idx, sample_idx = index

            if isinstance(channel_idx, int) and isinstance(sample_idx, int):
                return self._getitem_2d_int_int(channel_idx, sample_idx)
            if isinstance(channel_idx, int) and isinstance(sample_idx, slice):
                return self._getitem_2d_int_slice(channel_idx, sample_idx)
            if isinstance(channel_idx, slice) and isinstance(sample_idx, int):
                return self._getitem_2d_slice_int(channel_idx, sample_idx)
            if isinstance(channel_idx, slice) and isinstance(sample_idx, slice):
                return self._getitem_2d_slice_slice(channel_idx, sample_idx)

            raise TypeError(f"Channel index must be int or slice, got {type(channel_idx).__name__}.")

        raise TypeError(f"Index must be int, slice, or tuple of (int|slice, int|slice), got {type(index).__name__}.")

    @overload
    def __setitem__(self, index: int, value: GWpyTimeSeries) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: TimeSeries) -> None: ...

    @overload
    def __setitem__(self, index: tuple[int, int], value: float | np.floating) -> None: ...

    @overload
    def __setitem__(self, index: tuple[int, slice], value: np.ndarray) -> None: ...

    @overload
    def __setitem__(self, index: tuple[slice, int], value: np.ndarray) -> None: ...

    @overload
    def __setitem__(self, index: tuple[slice, slice], value: np.ndarray) -> None: ...

    def _setitem_1d_int(self, index: int, value: GWpyTimeSeries) -> None:
        """Set a single channel by integer index."""
        index = self._validate_channel_index(index)

        if value.t0 != self.start_time:
            raise ValueError(f"Start time mismatch on channel {index}. " f"Expected {self.start_time}, got {value.t0}.")

        if value.sample_rate != self.sampling_frequency:
            raise ValueError(
                f"Sampling frequency mismatch on channel {index}. "
                f"Expected {self.sampling_frequency}, got {value.sample_rate}."
            )

        if value.duration != self.duration:
            raise ValueError(
                f"Duration mismatch on channel {index}. " f"Expected {self.duration}, got {value.duration}."
            )

        self._data[index] = value

    def _setitem_1d_slice(self, index: slice, value: TimeSeries) -> None:
        """Set multiple channels by slice."""
        sliced_indices = range(*index.indices(self.num_of_channels))
        sliced_indices_list = list(sliced_indices)

        if len(sliced_indices_list) != value.num_of_channels:
            raise ValueError(
                f"Slice selects {len(sliced_indices_list)} channels, "
                f"but value has {value.num_of_channels} channels."
            )

        if value.start_time != self.start_time:
            raise ValueError(
                f"Start time mismatch for slice assignment. " f"Expected {self.start_time}, got {value.start_time}."
            )

        if value.sampling_frequency != self.sampling_frequency:
            raise ValueError(
                f"Sampling frequency mismatch for slice assignment. "
                f"Expected {self.sampling_frequency}, got {value.sampling_frequency}."
            )

        if value.duration != self.duration:
            raise ValueError(
                f"Duration mismatch for slice assignment. " f"Expected {self.duration}, got {value.duration}."
            )

        for i, channel_idx in enumerate(sliced_indices_list):
            self._data[channel_idx] = value[i]

        logger.debug("Assigned %d channels via slice", len(sliced_indices_list))

    def _setitem_2d_int_int(self, channel_idx: int, sample_idx: int, value: float | np.floating) -> None:
        """Set a scalar value at (channel, sample)."""
        channel_idx = self._validate_channel_index(channel_idx)
        sample_idx = self._validate_sample_index(sample_idx, self._data[channel_idx].size)
        self._data[channel_idx].value[sample_idx] = value

    def _setitem_2d_int_slice(self, channel_idx: int, sample_idx: slice, value: np.ndarray) -> None:
        """Set a 1D array for a single channel with sample slice."""
        channel_idx = self._validate_channel_index(channel_idx)
        self._data[channel_idx].value[sample_idx] = value

    def _setitem_2d_slice_int(self, channel_idx: slice, sample_idx: int, value: np.ndarray) -> None:
        """Set a 1D array for a single sample across multiple channels."""
        selected_channels = self._data[channel_idx]
        if not selected_channels:
            raise IndexError("Channel slice resulted in no channels.")
        if len(value) != len(selected_channels):
            raise ValueError(f"Value has {len(value)} elements, but {len(selected_channels)} channels selected.")
        sample_idx = self._validate_sample_index(sample_idx, selected_channels[0].size)
        for i, ts in enumerate(selected_channels):
            ts.value[sample_idx] = value[i]

    def _setitem_2d_slice_slice(self, channel_idx: slice, sample_idx: slice, value: np.ndarray) -> None:
        """Set a 2D array for multiple channels and samples."""
        selected_channels = self._data[channel_idx]
        if not selected_channels:
            raise IndexError("Channel slice resulted in no channels.")
        if value.ndim != 2:
            raise ValueError(f"Value must be 2D, got {value.ndim}D.")
        if value.shape[0] != len(selected_channels):
            raise ValueError(f"Value has {value.shape[0]} channels, but {len(selected_channels)} channels selected.")
        for i, ts in enumerate(selected_channels):
            ts.value[sample_idx] = value[i]

    def _dispatch_setitem_2d(self, channel_idx: int | slice, sample_idx: int | slice, value) -> None:
        """Dispatch 2D indexing assignment to the appropriate handler."""
        if isinstance(channel_idx, int) and isinstance(sample_idx, int):
            if not isinstance(value, (int, float, np.floating)):
                raise TypeError(f"For single sample assignment, value must be numeric, got {type(value).__name__}.")
            self._setitem_2d_int_int(channel_idx, sample_idx, value)

        elif isinstance(channel_idx, int) and isinstance(sample_idx, slice):
            if not isinstance(value, np.ndarray):
                raise TypeError(f"For sample slice assignment, value must be np.ndarray, got {type(value).__name__}.")
            self._setitem_2d_int_slice(channel_idx, sample_idx, value)

        elif isinstance(channel_idx, slice) and isinstance(sample_idx, int):
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    "For multi-channel sample assignment, value must be np.ndarray," f"got {type(value).__name__}."
                )
            self._setitem_2d_slice_int(channel_idx, sample_idx, value)

        elif isinstance(channel_idx, slice) and isinstance(sample_idx, slice):
            if not isinstance(value, np.ndarray):
                raise TypeError(f"For 2D slice assignment, value must be np.ndarray, got {type(value).__name__}.")
            self._setitem_2d_slice_slice(channel_idx, sample_idx, value)

        else:
            raise TypeError(f"Channel index must be int or slice, got {type(channel_idx).__name__}.")

    def __setitem__(
        self,
        index: int | slice | tuple[int | slice, int | slice],
        value: GWpyTimeSeries | TimeSeries | np.ndarray | float | np.floating,
    ) -> None:
        """Set time series data by channel index, slice, or 2D indexing.

        Supports:
        - Single channel: `ts[0] = gwpy_ts`
        - Channel slice: `ts[0:2] = other_ts`
        - 2D indexing (channel, sample):
          - `ts[0, 5] = 1.5` → set scalar value
          - `ts[0, :] = array` → set all samples of channel 0
          - `ts[:, 5] = array` → set sample 5 across all channels
          - `ts[0:2, 3:10] = array` → set 2D slice

        Args:
            index: Channel index (int), slice, or tuple of (channel_index, sample_index).
            value: Value to set (GWpyTimeSeries, TimeSeries, np.ndarray, or scalar).

        Raises:
            ValueError: If time/frequency parameters or shapes don't match.
            TypeError: If value type doesn't match index type.
            IndexError: If index is out of bounds.
        """
        # 1D indexing: single channel assignment
        if isinstance(index, int):
            if not isinstance(value, GWpyTimeSeries):
                raise TypeError(
                    f"For single channel assignment, value must be GWpyTimeSeries, got {type(value).__name__}."
                )
            self._setitem_1d_int(index, value)

        # 1D indexing: multi-channel slice assignment
        elif isinstance(index, slice):
            if not isinstance(value, TimeSeries):
                raise TypeError(f"For channel slice assignment, value must be TimeSeries, got {type(value).__name__}.")
            self._setitem_1d_slice(index, value)

        # 2D indexing: (channel, sample) assignment
        elif isinstance(index, tuple) and len(index) == 2:
            self._dispatch_setitem_2d(index[0], index[1], value)

        else:
            raise TypeError(f"Index must be int, slice, or tuple of (channel, sample), got {type(index).__name__}.")

    def __iter__(self):
        """Iterate over the channels in the time series.

        Returns:
            Iterator over the GWPy TimeSeries objects in the time series.
        """
        return iter(self._data)

    def __eq__(self, other: object) -> bool:
        """Check equality with another TimeSeries object.

        Args:
            other: Another TimeSeries object to compare with.

        Returns:
            True if the two TimeSeries objects are equal, False otherwise.
        """
        if not isinstance(other, TimeSeries):
            return False
        if self.num_of_channels != other.num_of_channels:
            return False
        for i in range(self.num_of_channels):
            if not np.array_equal(self[i].value, other[i].value):
                return False
            if self[i].t0 != other[i].t0:
                return False
            if self[i].sample_rate != other[i].sample_rate:
                return False
        return True

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the time series data.

        Returns:
            Tuple representing the shape of the time series data (num_of_channels, num_samples).
        """
        return (self.num_of_channels, self[0].size)

    @property
    def start_time(self) -> Quantity:
        """Get the start time of the time series.

        Returns:
            Start time of the time series.
        """
        return Quantity(self._data[0].t0)

    @property
    def duration(self) -> Quantity:
        """Get the duration of the time series.

        Returns:
            Duration of the time series.
        """
        return Quantity(self._data[0].duration)

    @property
    def end_time(self) -> Quantity:
        """Get the end time of the time series.

        Returns:
            End time of the time series.
        """
        end_time: Quantity = self.start_time + self.duration
        return end_time

    @property
    def sampling_frequency(self) -> Quantity:
        """Get the sampling frequency of the time series.

        Returns:
            Sampling frequency of the time series.
        """
        return Quantity(self._data[0].sample_rate)

    @property
    def time_array(self) -> Index:
        """Get the time array of the time series.

        Returns:
            Time array of the time series.
        """
        return self[0].times

    def crop(
        self,
        start_time: Quantity | None = None,
        end_time: Quantity | None = None,
    ) -> TimeSeries:
        """Crop the time series to the specified start and end times.

        Args:
            start_time: Start time of the cropped segment in GPS seconds. If None, use the
                original start time.
            end_time: End time of the cropped segment in GPS seconds. If None, use the
                original end time.

        Returns:
            Cropped TimeSeries instance.
        """
        for i in range(self.num_of_channels):
            self._data[i] = GWpyTimeSeries(self._data[i].crop(start=start_time, end=end_time, copy=True))
        return self

    def inject(self, other: TimeSeries) -> TimeSeries | None:
        """Inject another TimeSeries into the current TimeSeries.

        Args:
            other: TimeSeries instance to inject.

        Returns:
            Remaining TimeSeries instance if the injected TimeSeries extends beyond the current
            TimeSeries end time, otherwise None.
        """
        if len(other) != len(self):
            raise ValueError(
                f"Number of channels of other ({other.num_of_channels}) must "
                f"match number of channels of self ({self.num_of_channels})."
            )

        # Enforce that other has the same sampling frequency as self
        if not other.sampling_frequency == self.sampling_frequency:
            raise ValueError(
                f"Sampling frequency of chunk ({other.sampling_frequency}) must match "
                f"sampling frequency of segment ({self.sampling_frequency}). "
                "This ensures time grid alignment and avoids rounding errors."
            )

        if other.end_time < self.start_time:
            logger.warning(
                "The time series to inject ends before the current time series starts. No injection performed."
                "The start time of this segment is %s, while the end time of the other segment is %s",
                self.start_time,
                other.end_time,
            )
            return other

        if other.start_time > self.end_time:
            logger.warning(
                "The time series to inject starts after the current time series ends. No injection performed."
                "The end time of this segment is %s, while the start time of the other segment is %s",
                self.end_time,
                other.start_time,
            )
            return other

        # Check whether there is any offset in times
        other_start_time = other.start_time.to(self.start_time.unit)
        idx = ((other_start_time - self.start_time) * self.sampling_frequency).value
        if not np.isclose(idx, np.round(idx)):
            logger.warning("Chunk time grid does not align with segment time grid.")
            logger.warning("Interpolation will be used to align the chunk to the segment grid.")

            other_end_time = other.end_time.to(self.start_time.unit)
            other_new_times = self.time_array.value[
                (self.time_array.value >= other_start_time.value) & (self.time_array.value <= other_end_time.value)
            ]

            other = TimeSeries(
                data=np.array(
                    [
                        interp1d(
                            other.time_array.value, other[i].value, kind="linear", bounds_error=False, fill_value=0.0
                        )(other_new_times)
                        for i in range(len(other))
                    ]
                ),
                start_time=Quantity(other_new_times[0], unit=self.start_time.unit),
                sampling_frequency=self.sampling_frequency,
            )

        for i in range(self.num_of_channels):
            self[i] = inject(self[i], other[i])

        if other.end_time > self.end_time:
            return other.crop(start_time=self.end_time)
        return None

    def inject_from_list(self, ts_iterable: Iterable[TimeSeries]) -> TimeSeriesList:
        """Inject multiple TimeSeries from an iterable into the current TimeSeries.

        Args:
            ts_iterable: Iterable of TimeSeries instances to inject.

        Returns:
            TimeSeriesList of remaining TimeSeries instances that extend beyond the current TimeSeries end time.
        """
        from gwsim.data.time_series.time_series_list import TimeSeriesList  # pylint: disable=import-outside-toplevel

        remaining_ts: list[TimeSeries] = []
        for ts in ts_iterable:
            remaining_chunk = self.inject(ts)
            if remaining_chunk is not None:
                remaining_ts.append(remaining_chunk)
        return TimeSeriesList(remaining_ts)

    def to_json_dict(self) -> dict:
        """Convert the TimeSeries to a JSON-serializable dictionary.

        Assume the unit

        Returns:
            JSON-serializable dictionary representation of the TimeSeries.
        """
        return {
            "__type__": "TimeSeries",
            "data": [self[i].value.tolist() for i in range(self.num_of_channels)],
            "start_time": self.start_time.value,
            "start_time_unit": str(self.start_time.unit),
            "sampling_frequency": self.sampling_frequency.value,
            "sampling_frequency_unit": str(self.sampling_frequency.unit),
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> TimeSeries:
        """Create a TimeSeries object from a JSON-serializable dictionary.

        Args:
            json_dict: JSON-serializable dictionary representation of the TimeSeries.

        Returns:
            TimeSeries: An instance of the TimeSeries class created from the dictionary.
        """
        data = np.array(json_dict["data"])
        start_time = Quantity(json_dict["start_time"], unit=json_dict["start_time_unit"])
        sampling_frequency = Quantity(json_dict["sampling_frequency"], unit=json_dict["sampling_frequency_unit"])
        return cls(data=data, start_time=start_time, sampling_frequency=sampling_frequency)

    @property
    def num_of_samples(self) -> int:
        """Get the number of samples in each channel.

        Returns:
            Number of samples in each channel.
        """
        return self[0].size
