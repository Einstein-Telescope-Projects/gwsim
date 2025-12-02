"""White noise simulator implementation."""

from __future__ import annotations

import numpy as np

from gwsim.noise.base import NoiseSimulator


class WhiteNoiseSimulator(NoiseSimulator):
    """White noise simulator."""

    def __init__(
        self,
        loc: float,
        scale: float,
        sampling_frequency: float,
        duration: float,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """Initialize the white noise simulator.

        Args:
            loc: Mean of the normal distribution.
            scale: Standard deviation of the normal distribution.
            sampling_frequency: Sampling frequency of the noise in Hz.
            duration: Duration of each noise segment in seconds.
            start_time: Start time of the first noise segment in GPS seconds. Default is 0
            max_samples: Maximum number of samples to generate. None means infinite.
            seed: Seed for the random number generator. If None, the RNG is not initialized.
            **kwargs: Additional arguments absorbed by subclasses and mixins.
        """
        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
            **kwargs,
        )
        self.loc = loc
        self.scale = scale

    def next(self) -> np.ndarray:
        """Generate the next batch of white noise data."""
        if self.rng is None:
            raise RuntimeError("Random number generator not initialized. Set seed in constructor.")
        return self.rng.normal(loc=self.loc, scale=self.scale, size=int(self.duration * self.sampling_frequency))
