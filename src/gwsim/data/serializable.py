"""Protocols for serializable objects in the simulator."""

from __future__ import annotations

from typing import Any, Protocol


class JSONSerializable(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for JSON serializable objects.

    Classes implementing this protocol should provide a method to convert
    the object to a JSON-serializable dictionary.
    """

    def to_json_dict(self) -> dict[str, Any]:
        """Convert the object to a JSON-serializable dictionary.

        Returns:
            dict[str, Any]: JSON-serializable dictionary representation of the object.
        """
        raise NotImplementedError("to_json_dict method must be implemented by subclasses.")
