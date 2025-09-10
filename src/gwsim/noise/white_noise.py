from __future__ import annotations

from multiprocessing import Value
from pathlib import Path

import h5py
import numpy as np

from .base import BaseNoise


class WhiteNoise(BaseNoise):
    def __init__(
        self,
        loc: float,
        scale: float,
        sampling_frequency: float,
        duration: float,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
        )
        self.loc = loc
        self.scale = scale

    def next(self) -> np.ndarray:
        return self.rng.normal(loc=self.loc, scale=self.scale, size=int(self.duration * self.sampling_frequency))
