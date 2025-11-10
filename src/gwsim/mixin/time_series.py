"""Mixins for simulator classes providing optional functionality."""

from __future__ import annotations


class TimeSeriesMixin:
    """Mixin providing timing and duration management.

    This mixin adds time-based parameters commonly used
    in gravitational wave simulations.
    """

    def __init__(self, duration: float | None = None, sample_rate: float | None = None, **kwargs):
        """Initialize timing parameters.

        Args:
            duration: Duration of simulation in seconds.
            sample_rate: Sample rate in Hz.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(**kwargs)
        self.duration = duration
        self.sample_rate = sample_rate

    @property
    def num_samples(self) -> int | None:
        """Calculate number of samples from duration and sample rate.

        Returns:
            Number of samples or None if parameters not set.
        """
        if self.duration is not None and self.sample_rate is not None:
            return int(self.duration * self.sample_rate)
        return None

    @property
    def metadata(self) -> dict:
        """Get metadata including timing information.

        Returns:
            Dictionary containing timing parameters and other metadata.
        """
        metadata = super().metadata if hasattr(super(), "metadata") else {}
        metadata.update(
            {
                "duration": self.duration,
                "sample_rate": self.sample_rate,
                "num_samples": self.num_samples,
            }
        )
        return metadata
