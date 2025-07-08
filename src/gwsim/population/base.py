from __future__ import annotations

from abc import ABC, abstractmethod


class BasePopulation(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, n_sample: int) -> dict:
        pass
