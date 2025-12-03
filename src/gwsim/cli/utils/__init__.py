"""Utility functions for the gwsim CLI."""

from __future__ import annotations

import base64

import numpy as np
import yaml
from astropy.units import Quantity


def represent_quantity(dumper: yaml.SafeDumper, obj: Quantity) -> yaml.nodes.MappingNode:
    """Represent Quantity for YAML serialization.

    Args:
        dumper: YAML dumper.
        obj: Quantity object to represent.

    Returns:
        YAML node representing the Quantity.
    """
    return dumper.represent_mapping("!Quantity", {"value": float(obj.value), "unit": str(obj.unit)})


def construct_quantity(loader: yaml.Loader, node: yaml.MappingNode) -> Quantity:
    """Construct Quantity from YAML representation.

    Args:
        loader: YAML loader.
        node: YAML node to construct from.

    Returns:
        Quantity object.
    """
    data = loader.construct_mapping(node)
    return Quantity(data["value"], data["unit"])


yaml.SafeDumper.add_multi_representer(Quantity, represent_quantity)
yaml.SafeLoader.add_constructor("!Quantity", construct_quantity)


def represent_numpy_array(dumper: yaml.SafeDumper, obj: np.ndarray) -> yaml.nodes.MappingNode:
    """Represent numpy array for YAML serialization.

    Args:
        dumper: YAML dumper.
        obj: Numpy array to represent.

    Returns:
        YAML node representing the numpy array.
    """
    bytes_data = obj.tobytes()
    encoded_data = base64.b64encode(bytes_data).decode("ascii")
    data = {
        "data": encoded_data,
        "dtype": str(obj.dtype),
        "shape": list(obj.shape),
        "encoding": "base64",
    }
    return dumper.represent_mapping("!ndarray", data)


def construct_numpy_array(loader: yaml.Loader, node: yaml.MappingNode) -> np.ndarray:
    """Construct numpy array from YAML representation.

    Args:
        loader: YAML loader.
        node: YAML node to construct from.

    Returns:
        Numpy array.
    """
    data = loader.construct_mapping(node)
    if data.get("encoding") != "base64":
        raise ValueError("Expected base64 encoding in YAML data")
    dtype = np.dtype(data["dtype"])
    shape = tuple(data["shape"])
    decoded_bytes = base64.b64decode(data["data"])
    array = np.frombuffer(decoded_bytes, dtype=dtype).reshape(shape)
    return array


yaml.SafeDumper.add_multi_representer(np.ndarray, represent_numpy_array)
yaml.SafeLoader.add_constructor("!ndarray", construct_numpy_array)
