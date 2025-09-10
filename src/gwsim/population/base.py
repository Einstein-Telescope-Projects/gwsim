from __future__ import annotations

from typing import Any

from ..generator.base import Generator


class BasePopulation(Generator):
    def __init__(
        self,
        max_samples: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(max_samples=max_samples, seed=seed)

    def next(self):
        raise NotImplementedError("Not implemented.")

    def update_state(self):
        raise NotImplementedError("Not implemented.")

    def save_batch(self, batch: Any, file_name: str, overwrite: bool = False) -> None:
        raise NotImplementedError("Not implemented.")
