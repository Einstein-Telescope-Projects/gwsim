"""State helpers for simulator and orchestration checkpoint persistence."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, overload

import yaml

logger = logging.getLogger("gwmock")


class StateAttribute[T]:  # pylint: disable=duplicate-code
    """A state attribute."""

    def __init__(
        self,
        default: T | None = None,
        default_factory: Callable[[], T] | None = None,
        post_set_hook: Callable[[Any, T], None] | None = None,
    ) -> None:
        """A state attribute.

        Args:
            default (T | None, optional): The default value of the attribute. Defaults to None.
            default_factory (Callable[[], T] | None, optional): A factory to create the default value.
                This is useful when you want to have a list as a state attribute, and
                do not want to share the list across instances. Defaults to None.
            post_set_hook (Callable[[Any, T], None] | None): Called after the value of the attribute is set.
        """
        self.default = default
        self.default_factory = default_factory
        self.post_set_hook = post_set_hook
        self.name = None

    def __set_name__(self, owner: type, name: str) -> None:
        """Set the name of the attribute.

        Args:
            owner (type): Owner of the attribute.
            name (str): Name.
        """
        self.name = name
        # Ensure each class has its OWN _state_attributes list (not inherited from parent).
        # We check __dict__ directly to avoid inheriting a parent's list.
        if "_state_attributes" not in owner.__dict__:
            owner._state_attributes = []
        if name not in owner._state_attributes:
            owner._state_attributes.append(name)

    @overload
    def __get__(self, instance: None, owner: type) -> StateAttribute[T]: ...

    @overload
    def __get__(self, instance: Any, owner: type) -> T: ...

    def __get__(self, instance: Any, owner: type) -> T | StateAttribute[T]:
        """Get the value of the attribute.

        Args:
            instance (Any): Instance of the owner.
            owner (type): Placeholder.

        Returns:
            T | StateAttribute: Value of the attribute.
        """
        if instance is None:
            return self
        if self.name not in instance.__dict__:
            if self.default_factory is not None:
                instance.__dict__[self.name] = self.default_factory()
            else:
                instance.__dict__[self.name] = self.default
        return instance.__dict__[self.name]

    def __set__(self, instance: Any, value: T) -> None:
        """Set the value of the attribute.

        Args:
            instance (Any): An instance of the owner.
            value (T): Value to set.
        """
        instance.__dict__[self.name] = value
        if self.post_set_hook is not None:
            self.post_set_hook(instance, value)


class PopulationIterationState:  # pylint: disable=too-few-public-methods
    """Manage resumable population iteration state loaded from legacy checkpoints."""

    def __init__(self, checkpoint_file: str | Path | None = None, encoding: str = "utf-8") -> None:
        self.checkpoint_file = checkpoint_file
        self.encoding = encoding
        self.current_index = 0
        self.injected_indices: list[int] = []
        self.segment_map: dict[int, list[int]] = {}
        self._load_checkpoint()

    @property
    def checkpoint_file(self) -> Path | None:
        """Return the checkpoint file path, if configured."""
        return self._checkpoint_file

    @checkpoint_file.setter
    def checkpoint_file(self, value: str | Path | None) -> None:
        """Normalize the checkpoint file path to a ``Path`` instance."""
        self._checkpoint_file = None if value is None else Path(value)

    def to_checkpoint_data(self) -> dict[str, Any]:
        """Serialize the iteration state using the legacy checkpoint shape."""
        return {
            "population": {
                "current_index": int(self.current_index),
                "injected_indices": [int(index) for index in self.injected_indices],
                "segment_map": {
                    int(segment_index): [int(event_index) for event_index in event_indices]
                    for segment_index, event_indices in self.segment_map.items()
                },
            }
        }

    def load_checkpoint_data(self, data: Mapping[str, Any]) -> None:
        """Load the legacy population checkpoint payload into the in-memory state."""
        self.current_index = int(data.get("current_index", 0))
        self.injected_indices = [int(index) for index in data.get("injected_indices", [])]
        raw_segment_map = data.get("segment_map", {})
        if not isinstance(raw_segment_map, Mapping):
            raise TypeError("segment_map must be a mapping.")
        self.segment_map = {
            int(segment_index): [int(event_index) for event_index in event_indices]
            for segment_index, event_indices in raw_segment_map.items()
        }

    def _load_checkpoint(self) -> None:
        if self.checkpoint_file is None or not self.checkpoint_file.is_file():
            return

        try:
            with self.checkpoint_file.open(encoding=self.encoding) as file:
                payload = yaml.safe_load(file) or {}
            if not isinstance(payload, Mapping):
                raise TypeError("Checkpoint payload must be a mapping.")
            population_state = payload["population"]
            if not isinstance(population_state, Mapping):
                raise TypeError("population checkpoint payload must be a mapping.")
            self.load_checkpoint_data(population_state)
            logger.info(
                "Loaded checkpoint: current_index=%s, injected=%s",
                self.current_index,
                self.injected_indices,
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError, KeyError) as error:
            logger.warning("Failed to load checkpoint %s: %s. Starting fresh.", self.checkpoint_file, error)
