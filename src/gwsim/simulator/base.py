"""Refactored base simulator with clean separation of concerns."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from gwsim import __version__

from ..utils.io import check_file_exist
from .state import StateAttribute

logger = logging.getLogger(__name__)


class Simulator(ABC):
    """Base simulator class providing core interface and iteration capabilities.

    This class provides the minimal common interface that all simulators share:
    - State management and persistence
    - Iterator protocol for data generation
    - Metadata handling
    - File I/O operations

    Specialized functionality (randomness, timing, etc.) should be added
    via mixins to avoid bloating the base interface.

    Args:
        max_samples: Maximum number of samples to generate. None means infinite.
        **kwargs: Additional arguments absorbed by subclasses and mixins.
    """

    # State attributes using StateAttribute descriptor
    counter = StateAttribute(default=0)

    def __init__(self, max_samples: int | float | None = None, **kwargs):
        """Initialize the base simulator.

        Args:
            max_samples: Maximum number of samples to generate.
            **kwargs: Additional arguments for subclasses and mixins.
        """
        # Absorb unused kwargs to enable flexible parameter passing
        if kwargs:
            logger.debug("Unused kwargs in Simulator.__init__: %s", kwargs)

        # Initialize StateAttribute system
        super().__init__()

        # Non-state attributes
        self.max_samples = max_samples

    @property
    def max_samples(self) -> int | float:
        """Get the maximum number of samples.

        Returns:
            Maximum number of samples (np.inf for unlimited).
        """
        return self._max_samples

    @max_samples.setter
    def max_samples(self, value: int | float | None) -> None:
        """Set the maximum number of samples.

        Args:
            value: Maximum number of samples. None interpreted as infinite.

        Raises:
            ValueError: If value is negative.
        """
        if value is None:
            self._max_samples = np.inf
            logger.debug("max_samples set to None, interpreted as infinite.")
            return
        if value < 0:
            raise ValueError("Max samples cannot be negative.")
        self._max_samples = value

    @property
    def state(self) -> dict:
        """Get the current simulator state.

        Returns:
            Dictionary containing all state attributes.
        """
        # Get state attributes from the class (set by StateAttribute descriptors)
        state_attrs = getattr(self.__class__, "_state_attributes", [])
        return {key: getattr(self, key) for key in state_attrs}

    @state.setter
    def state(self, state: dict) -> None:
        """Set the simulator state.

        Args:
            state: Dictionary of state values.

        Raises:
            ValueError: If state contains unregistered attributes.
        """
        # Get state attributes from the class (set by StateAttribute descriptors)
        state_attrs = getattr(self.__class__, "_state_attributes", [])
        for key, value in state.items():
            if key not in state_attrs:
                raise ValueError(f"Attribute {key} is not registered as a state attribute.")
            setattr(self, key, value)

    @property
    def metadata(self) -> dict:
        """Get simulator metadata.

        This can be overridden by subclasses to include additional metadata.
        Mixins should call super().metadata and update the returned dictionary.

        Returns:
            Dictionary containing metadata.
        """
        return {
            "max_samples": self.max_samples,
            "counter": self.counter,
            "version": __version__,
        }

    # Iterator protocol
    def __iter__(self):
        """Return iterator interface."""
        return self

    def __next__(self):
        """Generate next sample.

        Returns:
            Next generated sample.

        Raises:
            StopIteration: When max_samples is reached.
        """
        if self.counter >= self.max_samples:
            raise StopIteration("Maximum number of samples reached.")

        sample = self.simulate()
        self.counter += 1
        return sample

    # # State persistence
    # @check_file_overwrite()
    # def save_state(self, file_name: Path, overwrite: bool = False) -> None:
    #     """Save simulator state to file.

    #     Args:
    #         file_name: Output file path (must have .json extension).
    #         overwrite: Whether to overwrite existing files.

    #     Raises:
    #         ValueError: If file extension is not .json.
    #         FileExistsError: If file exists and overwrite=False.
    #     """
    #     if file_name.suffix.lower() != ".json":
    #         raise ValueError(f"Unsupported file format: {file_name.suffix}. Supported: .json")

    #     with file_name.open("w") as f:
    #         json.dump(self.state, f)

    @check_file_exist()
    def load_state(self, file_name: Path) -> None:
        """Load simulator state from file.

        Args:
            file_name: Input file path (must have .json extension).

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file extension is not .json.
        """
        if file_name.suffix.lower() != ".json":
            raise ValueError(f"Unsupported file format: {file_name.suffix}. Supported: .json")

        with file_name.open("r") as f:
            state = json.load(f)

        self.state = state

    # @check_file_overwrite()
    # def save_metadata(self, file_name: Path, overwrite: bool = False) -> None:
    #     """Save simulator metadata to file.

    #     Args:
    #         file_name: Output file path (must have .json extension).
    #         overwrite: Whether to overwrite existing files.

    #     Raises:
    #         ValueError: If file extension is not .json.
    #         FileExistsError: If file exists and overwrite=False.
    #     """
    #     if file_name.suffix.lower() != ".json":
    #         raise ValueError(f"Unsupported file format: {file_name.suffix}. Supported: .json")

    #     with file_name.open("w") as f:
    #         json.dump(self.metadata, f)

    # Abstract methods that subclasses must implement
    @abstractmethod
    def simulate(self, *args, **kwargs) -> Any:
        """Generate a single sample.

        This method must be implemented by all simulator subclasses.

        Returns:
            A single generated sample.
        """

    @abstractmethod
    def save_batch(self, batch: Any, file_name: str | Path, overwrite: bool = False, **kwargs) -> None:
        """Save a batch of samples to file.

        This method must be implemented by all simulator subclasses.

        Args:
            batch: Batch of generated samples.
            file_name: Output file path.
            overwrite: Whether to overwrite existing files.
            **kwargs: Additional arguments for specific file formats.
        """
