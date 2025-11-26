"""Mixins for simulator classes providing optional functionality."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from gwpy.timeseries import TimeSeries as GWPyTimeSeries

from gwsim.data.time_series import TimeSeries
from gwsim.data.time_series.time_series_list import TimeSeriesList
from gwsim.utils.datetime_parser import parse_duration_to_seconds

logger = logging.getLogger("gwsim")


class TimeSeriesMixin:  # pylint: disable=too-few-public-methods
    """Mixin providing timing and duration management.

    This mixin adds time-based parameters commonly used
    in gravitational wave simulations.
    """

    start_time = 0
    cached_data_chunks = TimeSeriesList()

    def __init__(
        self,
        start_time: int = 0,
        duration: float = 4,
        total_duration: float | str | None = None,
        sampling_frequency: float = 4096,
        num_of_channels: int | None = None,
        dtype: type = np.float64,
        **kwargs,
    ):
        """Initialize timing parameters.

        Args:
            start_time: Start time in GPS seconds. Default is 0.
            duration: Duration of simulation in seconds. Default is 4.
            total_duration
            sampling_frequency: Sampling frequency in Hz. Default is 4096.
            dtype: Data type for the time series data. Default is np.float64.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(**kwargs)
        # TimeSeriesMixin is the last mixin in the hierarchy, so no super().__init__() call needed
        self.start_time = start_time
        self.duration = duration
        self.total_duration = total_duration
        self.sampling_frequency = sampling_frequency
        self.dtype = dtype

        # Get the number of channels.
        if num_of_channels is not None:
            self.num_of_channels = num_of_channels
            if "detectors" in kwargs and kwargs["detectors"] is not None:
                if len(kwargs["detectors"]) != num_of_channels:
                    raise ValueError("Number of detectors does not match num_of_channels.")
        elif "detectors" in kwargs and kwargs["detectors"] is not None:
            self.num_of_channels = len(kwargs["detectors"])
        else:
            self.num_of_channels = 1

    @property
    def total_duration(self) -> float:
        """Get the total duration of the simulation.

        Returns:
            Total duration in seconds.
        """
        return self._total_duration

    @total_duration.setter
    def total_duration(self, value: int | float | str | None) -> None:
        """Set the total duration of the simulation.

        Args:
            value: Total duration in seconds.
        """
        if value is not None:
            if isinstance(value, (float, int)):
                self._total_duration = float(value)
            elif isinstance(value, str):
                self._total_duration = parse_duration_to_seconds(value)
            else:
                raise ValueError("total_duration must be a float, int, or str representing duration.")

            if self.total_duration < 0:
                raise ValueError("total_duration must be non-negative.")

            if self.total_duration < self.duration:
                raise ValueError("total_duration must be greater than or equal to duration.")

            # Round the total_duration to the nearest multiple of duration
            num_segments = round(self.total_duration / self.duration)
            self._total_duration = num_segments * self.duration

            logger.info("Total duration set to %s seconds.", self.total_duration)

            # Set the max_samples based on total_duration and duration
            self.max_samples = int(self.total_duration / self.duration)
            logger.info("Setting max_samples to %s based on total_duration and duration.", self.max_samples)
        else:
            self._total_duration = self.duration * self.max_samples
            logger.info("total_duration not set, using duration * max_samples = %f seconds.", self.total_duration)

    @property
    def end_time(self) -> float:
        """Calculate the end time of the current segment.

        Returns:
            End time in GPS seconds.
        """
        return self.start_time + self.duration

    def _simulate(self, *args, **kwargs) -> TimeSeriesList:
        """Generate time series data chunks.

        This method should be implemented by subclasses to generate
        the actual time series data.
        """
        raise NotImplementedError("Subclasses must implement the _simulate method.")

    def simulate(self, *args, **kwargs) -> TimeSeries:
        """
        Simulate a segment of time series data.

        Args:
            *args: Positional arguments for the _simulate method.
            **kwargs: Keyword arguments for the _simulate method.

        Returns:
            TimeSeries: Simulated time series segment.
        """
        # First create a new segment
        segment = TimeSeries(
            data=np.zeros((self.num_of_channels, int(self.duration * self.sampling_frequency)), dtype=self.dtype),
            start_time=self.start_time,
            sampling_frequency=self.sampling_frequency,
        )

        # Inject cached data chunks into the segment
        self.cached_data_chunks = segment.inject_from_list(self.cached_data_chunks)

        # Generate new chunks of data
        new_chunks = self._simulate(*args, **kwargs)

        # Add the new chunks to the segment
        remaining_chunks = segment.inject_from_list(new_chunks)

        # Add the remaining chunks to the cache
        self.cached_data_chunks.extend(remaining_chunks)

        return segment

    @property
    def metadata(self) -> dict:
        """Get metadata including timing information.

        Returns:
            Dictionary containing timing parameters and other metadata.
        """
        metadata = {
            "time_series": {
                "arguments": {
                    "start_time": self.start_time,
                    "duration": self.duration,
                    "sampling_frequency": self.sampling_frequency,
                    "num_of_channels": self.num_of_channels,
                    "dtype": str(self.dtype),
                }
            }
        }
        return metadata

    def _save_data(  # pylint: disable=unused-argument
        self,
        data: TimeSeries,
        file_name: str | Path | np.ndarray[Any, np.dtype[np.object_]],
        channel_names: str | None = None,
        **kwargs,
    ) -> None:
        """Save time series data to a file.

        Args:
            data: Time series data to save.
            file_name: Path to the output file.
            **kwargs: Additional arguments for the saving function.
        """
        if data.num_of_channels == 1 and isinstance(file_name, (str, Path)):
            self._save_gwf_data(data=data[0], file_name=file_name, **kwargs)
        elif (
            data.num_of_channels > 1
            and isinstance(file_name, np.ndarray)
            and len(file_name.shape) == 1
            and file_name.shape[0] == data.num_of_channels
        ):
            for i in range(data.num_of_channels):
                single_file_name = cast(Path, file_name[i])
                self._save_gwf_data(data=data[i], file_name=single_file_name, **kwargs)
        else:
            raise ValueError(
                "file_name must be a single path for single-channel data or an array of paths for multi-channel data."
            )

    def _save_gwf_data(  # pylint: disable=unused-argument
        self, data: GWPyTimeSeries, file_name: str | Path, channel: str | None = None, **kwargs
    ) -> None:
        """Save GWPy TimeSeries data to a GWF file.

        Args:
            data: GWPy TimeSeries data to save.
            file_name: Path to the output GWF file.
            channel: Optional channel name to set in the data.
            **kwargs: Additional arguments for the write function.
        """
        if channel is not None:
            data.channel = channel
        data.write(str(file_name))
