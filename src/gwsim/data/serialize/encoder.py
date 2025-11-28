"""Custom JSON encoder for JSONSerializable objects."""

from __future__ import annotations

import json
from typing import Any

from astropy.units import Quantity


class Encoder(json.JSONEncoder):
    """Custom JSON encoder for JSONSerializable objects."""

    def default(self, o: Any) -> Any:
        """Serialize JSONSerializable objects to JSON.

        Args:
            o: Object to serialize.
        """
        if hasattr(o, "to_json_dict"):
            encoded = o.to_json_dict()
            if "__type__" not in encoded:
                encoded["__type__"] = o.__class__.__name__
            return encoded
        if isinstance(o, Quantity):
            return {
                "__type__": "Quantity",
                "value": o.value,
                "unit": str(o.unit),
            }
        return super().default(o)
