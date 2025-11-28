"""Utility functions for the gwsim CLI."""

from __future__ import annotations

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
