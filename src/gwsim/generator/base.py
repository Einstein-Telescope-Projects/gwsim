from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path


class Generator(ABC):
    def __init__(self, batch_size: int = 1, max_samples: int | None = None, seed: int | None = None):
        self._state = {"current_sample": 0}
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.seed = seed

    def __iter__(self):
        self._state["current_sample"] = 0
        return self

    def __next__(self):
        if self.max_samples is not None and self._state["current_sample"] >= self.max_samples:
            raise StopIteration
        result = self.next()
        self._state["current_sample"] += self.batch_size
        return result

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def update_state(self):
        pass

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        if value < 1:
            raise ValueError("Batch size must be at least 1.")
        self._batch_size = value

    @property
    def max_samples(self):
        return self._max_samples

    @max_samples.setter
    def max_samples(self, value: int | None):
        if value is not None and value < 0:
            raise ValueError("Max samples cannot be negative.")
        self._max_samples = value

    @property
    def state(self):
        return self._state

    @property
    def metadata(self):
        return {"batch_size": self.batch_size, "max_samples": self.max_samples, "seed": self.seed}

    def save_state(self, file_name: str, overwrite=False):
        file_path = Path(file_name)
        if not overwrite and file_path.exists():
            raise FileExistsError(f"File {file_name} already exists.")

        file_extension = file_path.suffix.lower()
        state = self.state

        if file_extension in [".pkl", ".pickle"]:
            with file_path.open("wb") as f:
                pickle.dump(state, f)
        elif file_extension == ".json":
            with file_path.open("w") as f:
                json.dump(state, f)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}. Use .pkl, .pickle, or .json.")

    def load_state(self, file_name: str):
        file_path = Path(file_name)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_name} does not exist.")

        file_extension = file_path.suffix.lower()

        if file_extension in [".pkl", ".pickle"]:
            with file_path.open("rb") as f:
                state = pickle.load(f)
        elif file_extension == ".json":
            with file_path.open("r") as f:
                state = json.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}. Use .pkl or .json.")

        # Restore state
        self._state = state
