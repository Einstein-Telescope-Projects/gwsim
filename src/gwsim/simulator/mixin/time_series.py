"""Mixins for simulator classes providing optional functionality."""

from __future__ import annotations

import numpy as np

from gwsim.data.time_series import TimeSeries
from gwsim.data.time_series_list import TimeSeriesList


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
        sampling_frequency: float = 4096,
        num_of_channels: int | None = None,
        dtype: type = np.float64,
        **kwargs,
    ):
        """Initialize timing parameters.

        Args:
            start_time: Start time in GPS seconds. Default is 0.
            duration: Duration of simulation in seconds. Default is 4.
            sampling_frequency: Sampling frequency in Hz. Default is 4096.
            dtype: Data type for the time series data. Default is np.float64.
            **kwargs: Additional arguments passed to parent classes.
        """
        # TimeSeriesMixin is the last mixin in the hierarchy, so no super().__init__() call needed
        self.start_time = start_time
        self.duration = duration
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
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "dtype": str(self.dtype),
        }
        return metadata
