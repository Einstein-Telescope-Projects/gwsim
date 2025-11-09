"""Module for handling time series data for multiple channels."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from astropy.units.quantity import Quantity
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from gwpy.types.index import Index
from scipy.interpolate import interp1d

logger = logging.getLogger("gwsim")


if TYPE_CHECKING:
    from gwsim.data.time_series_list import TimeSeriesList


class TimeSeries:
    """Class representing a time series data for multiple channels."""

    def __init__(self, data: np.ndarray, start_time: int | Quantity, sampling_frequency: float | Quantity):
        """Initialize the TimeSeries with a list of GWPy TimeSeries objects.

        Args:
            channels: List of GWPy TimeSeries objects representing different channels.
        """
        if data.ndim != 2:
            raise ValueError("Data must be a 2D numpy array with shape (num_channels, num_samples).")
        self._data: list[GWpyTimeSeries] = [
            GWpyTimeSeries(
                data=data[i],
                t0=start_time,
                sample_rate=sampling_frequency,
            )
            for i in range(data.shape[0])
        ]
        self.num_channels = data.shape[0]
        self.dtype = data.dtype

    def __len__(self) -> int:
        """Get the number of channels in the time series.

        Returns:
            Number of channels in the time series.
        """
        return self.num_channels

    def __getitem__(self, index: int) -> GWpyTimeSeries:
        """Get the GWPy TimeSeries object for a specific channel.

        Args:
            index: Index of the channel to retrieve.

        Returns:
            GWPy TimeSeries object for the specified channel.
        """
        return self._data[index]

    def __iter__(self):
        """Iterate over the channels in the time series.

        Returns:
            Iterator over the GWPy TimeSeries objects in the time series.
        """
        return iter(self._data)

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
        return self.start_time + self.duration

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
        for i in range(self.num_channels):
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
            raise ValueError("Number of channels in chunk must match number of channels in segment.")

        if other.end_time < self.start_time or other.start_time > self.end_time:
            raise ValueError("The time series to inject does not overlap with the current time series.")

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
                start_time=other.start_time,
                sampling_frequency=other.sampling_frequency,
            )

        for i in range(self.num_channels):
            self[i].inject(other[i])

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
        from gwsim.data.time_series_list import TimeSeriesList  # pylint: disable=import-outside-toplevel

        remaining_ts: list[TimeSeries] = []
        for ts in ts_iterable:
            remaining_chunk = self.inject(ts)
            if remaining_chunk is not None:
                remaining_ts.append(remaining_chunk)
        return TimeSeriesList(remaining_ts)
