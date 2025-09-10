"""
A base class for generators.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random._generator import Generator as RNG

from ..utils.io import check_file_exist, check_file_overwrite
from ..utils.random import get_rng, get_state
from ..utils.random import seed as set_seed
from ..utils.random import set_state
from ..version import __version__
from .state import StateAttribute

logger = logging.getLogger("gwsim")


class Generator(ABC):
    # A list of state attributes.
    _state_attributes = []

    # It counts the number of samples of data has been generated.
    sample_counter = StateAttribute(0)

    # It records the state of the random number generator.
    # The post set hook is used to reinitialize the random number generator when the state is set.
    rng_state = StateAttribute(get_state(), post_set_hook=lambda self, state: self._init_rng(state))

    def __init__(self, max_samples: int | None = None, seed: int | None = None):
        """Generator.

        Args:
            max_samples (int | None, optional): Maximum number of samples.
                None implies that this is an infinite iterator. Defaults to None.
            seed (int | None, optional): Seed to initialize the random number generator. Defaults to None.
        """
        # Save the attributes
        self.max_samples = max_samples

        # Calculate the number of samples
        if self.max_samples is None:
            self.max_samples = np.inf

        # Set the seed

        self.seed = seed

        # Initialize the random number generator if seed is not None.
        if seed is not None:
            set_seed(seed)
            self.rng = get_rng()
            self.rng_state = get_state()
        else:
            self.rng = None

            # Remove rng_state from _state_attributes
            self._state_attributes.remove("rng_state")

    def __iter__(self) -> Generator:
        """Get an instance of the iterator.

        Returns:
            Generator: An instance of the iterator.
        """
        return self

    def __next__(self) -> Any:
        """Generate one batch of data.

        Raises:
            StopIteration: If sample_counter is greater or equal to max_samples, raise StopIteration.

        Returns:
            Any: One batch of data.
        """
        if self.sample_counter >= self.max_samples:
            raise StopIteration
        result = self.next()
        return result

    @abstractmethod
    def next(self) -> Any:
        """An abstract class to be defined by the subclass.
        It should define the generation process for one batch of data.

        Returns:
            Any: One batch of data.
        """

    def update_state(self) -> None:
        """Update the current state."""
        self.sample_counter += 1
        if self.rng is not None:
            self.rng_state = get_state()

    def __len__(self) -> int | float:
        """Get the number of samples.

        Returns:
            int | None: Number of samples. None implies that this is an infinite iterator.
        """
        return self.max_samples

    @property
    def state(self) -> dict:
        """Get the state of the generator.

        Returns:
            dict: A dictionary of values that defines the current state.
        """
        return {key: getattr(self, key) for key in self._state_attributes}

    @state.setter
    def state(self, state: dict) -> None:
        """Set the state.

        Args:
            state (dict): A dictionary of the state.

        Raises:
            ValueError: If the key is not registered as a state attribute, raise ValueError.
        """
        for key, value in state.items():
            if key not in self._state_attributes:
                raise ValueError(f"The attribute {key} is not registered as a state attribute.")
            setattr(self, key, value)

    @property
    def rng(self) -> RNG | None:
        """Get the random number generator.

        Returns:
            RNG | None: If seed is not None, a random number generator is returned,
                or otherwise None is returned.
        """
        return self._rng

    @rng.setter
    def rng(self, value: RNG | None) -> None:
        """Set the random number generator.

        Args:
            value (RNG | None): Random number generator.
        """
        self._rng = value

    def _init_rng(self, state: dict | None) -> None:
        if state is not None and self.rng is not None:
            set_state(state)
            self.rng = get_rng()
        else:
            logger.debug(
                "_init_rng(self, state) is called but state is %s and self.rng is %s.", state, self.rng)

    @property
    def max_samples(self) -> int | None:
        """Get the maximum number of samples.

        Returns:
            int | None: Maximum number of samples.
        """
        return self._max_samples

    @max_samples.setter
    def max_samples(self, value: int | None) -> None:
        """Set the maximum number of samples.

        Args:
            value (int | None): Maximum number of samples.

        Raises:
            ValueError: If value is not None and is smaller than 0, raise a ValueError.
        """
        if value is not None and value < 0:
            raise ValueError("Max samples cannot be negative.")
        self._max_samples = value

    @property
    def metadata(self) -> dict:
        """Get a dictionary of metadata.
        This can be overridden by the subclass.

        Returns:
            dict: A dictionary of metadata.
        """
        return {
            "max_samples": self.max_samples,
            "seed": self.seed,
            "version": __version__,
        }

    @check_file_overwrite()
    def save_state(self, file_name: Path, overwrite=False) -> None:
        """Save the state of the iterator to file.

        Supported file format: '.json'.

        Args:
            file_name (Path): File name.
            overwrite (bool, optional): If True, overwrite the existing file. Defaults to False.

        Raises:
            FileExistsError: If 'overwrite' is False, and 'file_name' exists, raise FileExistsError.
            ValueError: If the suffix of 'file_name' is not '.json', raise ValueError.
        """
        file_extension = file_name.suffix.lower()
        state = self.state

        if file_extension == ".json":
            with file_name.open("w") as f:
                json.dump(state, f)
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. Supported file format: .json.")

    @check_file_exist()
    def load_state(self, file_name: Path) -> None:
        """Load the state from file.

        Args:
            file_name (Path): File name.

        Raises:
            FileNotFoundError: If 'file_name' is not found, raise FileNotFoundError.
            ValueError: If the suffix is not '.json', raise ValueError.
        """

        file_extension = file_name.suffix.lower()

        if file_extension == ".json":
            with file_name.open("r") as f:
                state = json.load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. Supported file format: .json.")

        # Restore state
        self.state = state

    @abstractmethod
    def save_batch(self, batch: Any, file_name: str | Path, overwrite: bool = False, **kwargs) -> None:
        """Save a batch of samples to file.

        Args:
            batch (Any): One batch of data.
            file_name (str | Path): File name.
            overwrite (bool, optional): If True, overwrite existing file. Defaults to False.
        """

    @check_file_overwrite()
    def save_metadata(self, file_name: Path, overwrite: bool = False) -> None:
        """Save the metadata file file.

        Supported file format: .json

        Args:
            file_name (str | Path): File name.
            overwrite (bool, optional): If True, overwrite the existing file. Defaults to False.
        """
        if file_name.suffix != ".json":
            raise ValueError(f"Suffix of {file_name} is not supported. Supported: .json.")

        with open(file_name, "w") as f:
            json.dump(self.metadata, f)
