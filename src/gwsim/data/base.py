from __future__ import annotations

from abc import ABC, abstractmethod


class Data(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, filename: str):
        pass
