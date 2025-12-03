"""Custom JSON decoder for JSONSerializable objects."""

from __future__ import annotations

import base64
import importlib
import json
from typing import Any

import numpy as np
from astropy.units import Quantity


class Decoder(json.JSONDecoder):
    """Custom JSON decoder for JSONSerializable objects.

    Automatically reconstructs objects that have been serialized with
    the Encoder class by checking for the "__type__" key and calling
    the appropriate from_json_dict class method.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the decoder with custom object_hook.

        Args:
            *args: Positional arguments passed to json.JSONDecoder.
            **kwargs: Keyword arguments passed to json.JSONDecoder.
        """
        super().__init__(*args, object_hook=self._object_hook, **kwargs)

    def _object_hook(self, obj: dict[str, Any]) -> Any:
        """Object hook to reconstruct JSONSerializable objects.

        Args:
            obj: Dictionary from JSON.

        Returns:
            Reconstructed object or original dict if not a known type.
        """
        if "__type__" in obj:
            type_name = obj["__type__"]

            if type_name == "Quantity":
                return Quantity(value=obj["value"], unit=obj["unit"])

            if type_name == "ndarray":
                encoded_data = obj["data"]
                bytes_data = base64.b64decode(encoded_data)
                array = np.frombuffer(bytes_data, dtype=obj["dtype"])
                array = array.reshape(obj["shape"])
                return array

            # Assume all serializable classes are in gwsim.data module
            module = importlib.import_module("gwsim.data")
            cls = getattr(module, type_name, None)
            if cls and hasattr(cls, "from_json_dict"):
                return cls.from_json_dict(obj)

        return obj
