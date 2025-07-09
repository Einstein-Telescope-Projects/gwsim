from __future__ import annotations

from ..generator.base import Generator


class BaseNoise(Generator):
    def __init__(
        self,
        sampling_frequency: float,
        duration: float,
        batch_size: int = 1,
        max_samples: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(batch_size=batch_size, max_samples=max_samples, seed=seed)
        self.sampling_frequency = sampling_frequency
        self.duration = duration

    def next(self):
        raise NotImplementedError("Not implemented.")

    def update_state(self):
        raise NotImplementedError("Not Implemented.")
