"""Module for handling time series data for multiple channels."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from numbers import Number
from typing import TYPE_CHECKING

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

    __hash__ = None

    def __init__(self, data: np.ndarray, start_time: int | float | Quantity, sampling_frequency: float | Quantity):
        """Initialize the TimeSeries with a list of GWPy TimeSeries objects.

        Args:
            data: 2D numpy array of shape (num_of_channels, num_samples) containing the time series data.
            start_time: Start time of the time series in GPS seconds.
            sampling_frequency: Sampling frequency of the time series in Hz.
        """
        expected_ndim = 2
        if data.ndim != expected_ndim:
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
        self.num_of_channels = data.shape[0]
        self.dtype = data.dtype
        self.metadata = {}

    def __len__(self) -> int:
        """Get the number of channels in the time series.

        Returns:
            Number of channels in the time series.
        """
        return self.num_of_channels

    def __getitem__(self, index: int) -> GWpyTimeSeries:
        """Get the GWPy TimeSeries object for a specific channel.

        Args:
            index: Index of the channel to retrieve.

        Returns:
            GWPy TimeSeries object for the specified channel.
        """
        return self._data[index]

    def __setitem__(self, index: int, value: GWpyTimeSeries) -> None:
        """Set the GWPy TimeSeries object for a specific channel.

        Args:
            index: Index of the channel to set.
            value: GWPy TimeSeries object to set for the specified channel.
        """
        # First check whether the start time and sampling frequency match
        if value.t0 != self.start_time:
            raise ValueError(
                "Start time of the provided TimeSeries does not match."
                f"The start time of this instance is {self.start_time}, "
                f"while that of the provided TimeSeries is {value.t0}."
            )

        # Debug: log the sampling frequencies
        logger.debug(
            "Assigning to channel %d: value.sample_rate=%.15f, self.sampling_frequency=%.15f",
            index,
            float(value.sample_rate.value),
            float(self.sampling_frequency.value),
        )

        if value.sample_rate != self.sampling_frequency:
            # Additional debug info
            logger.warning(
                "Sampling frequency mismatch on channel %d. "
                "Difference: %.15e Hz. "
                "Value times: %s to %s (%d samples, dt=%.15f). "
                "Self times span should match.",
                index,
                float(value.sample_rate.value) - float(self.sampling_frequency.value),
                value.times[0],
                value.times[-1],
                len(value),
                float(value.dt.value),
            )
            raise ValueError(
                "Sampling frequency of the provided TimeSeries does not match."
                f"The sampling frequency of this instance is {self.sampling_frequency}, "
                f"while that of the provided TimeSeries is {value.sample_rate}."
            )
        # Check the duration
        if value.duration != self.duration:
            raise ValueError(
                "Duration of the provided TimeSeries does not match."
                f"The duration of this instance is {self.duration}, "
                f"while that of the provided TimeSeries is {value.duration}."
            )

        if not isinstance(value, GWpyTimeSeries):
            raise TypeError(f"Value must be a GWpy TimeSeries instance, got {type(value)}")

        self._data[index] = value

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
        from gwsim.data.time_series.time_series_list import TimeSeriesList  # noqa: PLC0415

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
