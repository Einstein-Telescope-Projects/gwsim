"""Mixins for simulator classes providing optional functionality."""

from __future__ import annotations

from gwsim.simulator.state import StateAttribute


class TimeSeriesMixin:  # pylint: disable=too-few-public-methods
    """Mixin providing timing and duration management.

    This mixin adds time-based parameters commonly used
    in gravitational wave simulations.
    """

    start_time = StateAttribute(0)

    def __init__(self, start_time: float = 0, duration: float | None = None, sampling_frequency: float | None = None):
        """Initialize timing parameters.

        Args:
            start_time: Start time in GPS seconds. Default is 0.
            duration: Duration of simulation in seconds.
            sampling_frequency: Sampling frequency in Hz.
            **kwargs: Additional arguments passed to parent classes.
        """
        self.start_time = start_time
        self.duration = duration
        self.sampling_frequency = sampling_frequency

    @property
    def metadata(self) -> dict:
        """Get metadata including timing information.

        Returns:
            Dictionary containing timing parameters and other metadata.
        """
        metadata = {
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
        }
        return metadata
