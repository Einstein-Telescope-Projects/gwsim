"""Refactored base simulator with clean separation of concerns."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from gwsim import __version__
from gwsim.simulator.state import StateAttribute
from gwsim.utils.io import get_file_name_from_template

logger = logging.getLogger("gwsim")


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

        # Non-state attributes
        if max_samples is None:
            self.max_samples = np.inf
            logger.debug("max_samples set to None, interpreted as infinite.")
        else:
            self.max_samples = max_samples

    @property
    def max_samples(self) -> int | float:
        """Get the maximum number of samples.

        Returns:
            Maximum number of samples (np.inf for unlimited).
        """
        return self._max_samples

    @max_samples.setter
    def max_samples(self, value: int | float) -> None:
        """Set the maximum number of samples.

        Args:
            value: Maximum number of samples. None interpreted as infinite.

        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError("Max samples cannot be negative.")
        self._max_samples = value

    @property
    def state(self) -> dict:
        """Get the current simulator state.

        Returns:
            Dictionary containing all state attributes.
        """
        # Get state attributes from all classes in MRO (set by StateAttribute descriptors)
        state_attrs: list[Any] = []
        for cls in self.__class__.__mro__:
            state_attrs.extend(getattr(cls, "_state_attributes", []))
        # Remove duplicates while preserving order
        seen = set()
        state_attrs = [x for x in state_attrs if not (x in seen or seen.add(x))]
        return {key: getattr(self, key) for key in state_attrs}

    @state.setter
    def state(self, state: dict) -> None:
        """Set the simulator state.

        Args:
            state: Dictionary of state values.

        Raises:
            ValueError: If state contains unregistered attributes.
        """
        # Get state attributes from all classes in MRO (set by StateAttribute descriptors)
        state_attrs: list[Any] = []
        for cls in self.__class__.__mro__:
            state_attrs.extend(getattr(cls, "_state_attributes", []))
        # Remove duplicates while preserving order
        seen = set()
        state_attrs = [x for x in state_attrs if not (x in seen or seen.add(x))]

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
        metadata = {
            "max_samples": self.max_samples,
            "counter": self.counter,
            "version": __version__,
            "state": {},
        }
        metadata["state"].update(self.state)
        return metadata

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
        self.update_state()
        return sample

    def save_state(self, file_name: str | Path, overwrite: bool = False, encoding: str = "utf-8") -> None:
        """Save simulator state to file.

        Args:
            file_name: Output file path (must have .yml or .yaml extension).
            overwrite: Whether to overwrite existing files.
            encoding: File encoding to use when writing the file.

        Raises:
            ValueError: If file extension is not .yml or .yaml.
            FileExistsError: If file exists and overwrite=False.
        """
        file_name = Path(file_name)

        if file_name.suffix.lower() not in [".yml", ".yaml"]:
            raise ValueError(f"Unsupported file format: {file_name.suffix}. Supported: [.yml, .yaml]")

        if not overwrite and file_name.exists():
            raise FileExistsError(f"File '{file_name}' already exists. Use overwrite=True to overwrite it.")

        with file_name.open("w", encoding=encoding) as f:
            yaml.safe_dump(self.state, f)

    def load_state(self, file_name: str | Path, encoding: str = "utf-8") -> None:
        """Load simulator state from file.

        Args:
            file_name: Input file path (must have .yml or .yaml extension).
            encoding: File encoding to use when reading the file.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file extension is not .yml or .yaml.
        """
        file_name = Path(file_name)

        if file_name.suffix.lower() not in [".yml", ".yaml"]:
            raise ValueError(f"Unsupported file format: {file_name.suffix}. Supported: [.yml, .yaml]")

        if not file_name.exists():
            raise FileNotFoundError(f"File '{file_name}' does not exist.")

        with file_name.open("r", encoding=encoding) as f:
            state = yaml.safe_load(f)

        self.state = state

    def save_metadata(
        self,
        file_name: str | Path,
        output_directory: str | Path | None = None,
        overwrite: bool = False,
        encoding: str = "utf-8",
    ) -> None:
        """Save simulator metadata to file.

        Args:
            file_name: Output file path (must have .yml or .yaml extension).
            output_directory: Optional output directory to prepend to the file name.
            overwrite: Whether to overwrite existing files.
            encoding: File encoding to use when writing the file.

        Raises:
            ValueError: If file extension is not .yml or .yaml.
            FileExistsError: If file exists and overwrite=False.
        """
        file_name_resolved = get_file_name_from_template(
            template=str(file_name),
            instance=self,
            output_directory=output_directory,
        )
        if not isinstance(file_name_resolved, Path):
            raise ValueError("Resolved file name for metadata must be a single Path.")

        if file_name_resolved.suffix.lower() not in [".yml", ".yaml"]:
            raise ValueError(f"Unsupported file format: {file_name_resolved.suffix}. Supported: [.yml, .yaml]")

        if not overwrite and file_name_resolved.exists():
            raise FileExistsError(f"File '{file_name_resolved}' already exists. Use overwrite=True to overwrite it.")

        with file_name_resolved.open("w", encoding=encoding) as f:
            yaml.safe_dump(self.metadata, f)

    def update_state(self) -> None:
        """Update internal state after each sample generation.

        This method must be implemented by all simulator subclasses.
        """
        self.counter = cast(int, self.counter) + 1

    # Abstract methods that subclasses must implement
    @abstractmethod
    def simulate(self, *args, **kwargs) -> Any:
        """Generate a single sample.

        This method must be implemented by all simulator subclasses.

        Returns:
            A single generated sample.
        """

    @abstractmethod
    def _save_data(self, data: Any, file_name: str | Path | np.ndarray[Any, np.dtype[np.object_]], **kwargs) -> None:
        """Internal method to save data to a file.

        This method must be implemented by all simulator subclasses.

        Args:
            batch: Batch of generated samples.
            file_name: Output file path.
            overwrite: Whether to overwrite existing files.
            **kwargs: Additional arguments for specific file formats.
        """

    def save_data(
        self,
        data: Any,
        file_name: str | Path,
        output_directory: str | Path | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """Save data to a file.

        This method must be implemented by all simulator subclasses.

        Args:
            batch: Batch of generated samples.
            file_name: Output file path.
                If the file_name contains placeholders (e.g., {{detector}}, {{duration}}),
                they are filled by the attributes of the simulator.
            output_directory: Optional output directory to prepend to the file name.
            overwrite: Whether to overwrite existing files.
            save_metadata: Whether to save metadata alongside the data.
            **kwargs: Additional arguments for specific file formats.
        """
        file_name_resolved = get_file_name_from_template(
            template=str(file_name),
            instance=self,
            output_directory=output_directory,
        )

        if not overwrite:
            if isinstance(file_name_resolved, Path):
                if file_name_resolved.exists():
                    raise FileExistsError(
                        f"File '{file_name_resolved}' already exists. " f"Use overwrite=True to overwrite it."
                    )
            else:
                for single_file in file_name_resolved.flatten():
                    if single_file.exists():
                        raise FileExistsError(
                            f"File '{single_file}' already exists. " f"Use overwrite=True to overwrite it."
                        )

        self._save_data(data=data, file_name=file_name_resolved, **kwargs)
