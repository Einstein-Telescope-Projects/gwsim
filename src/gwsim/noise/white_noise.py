""" "White noise simulator implementation."""

from __future__ import annotations

import numpy as np

from .base import NoiseSimulator


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
        if self.rng is None:
            raise RuntimeError("Random number generator not initialized. Set seed in constructor.")
        return self.rng.normal(loc=self.loc, scale=self.scale, size=int(self.duration * self.sampling_frequency))
