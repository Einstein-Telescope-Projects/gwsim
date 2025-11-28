"""Utility functions for file input/output operations with safety checks."""

from __future__ import annotations

import itertools
import logging
import re
from functools import wraps
from pathlib import Path

import numpy as np
from astropy.units import Quantity
from numpy.typing import NDArray

logger = logging.getLogger("gwsim")


def check_file_overwrite():
    """A decorator to check the existence of the file,
    and avoid overwriting it unintentionally.

    Provides safe file handling with clear error messages and logging.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, file_name: str | Path, overwrite: bool = False, **kwargs):
            file_name = Path(file_name)

            # Create parent directories if they don't exist
            file_name.parent.mkdir(parents=True, exist_ok=True)

            if file_name.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"File '{file_name}' already exists. "
                        f"Use overwrite=True or --overwrite flag to overwrite it."
                    )
                file_size = file_name.stat().st_size
                logger.warning("File '%s' already exists (size: %d bytes). Overwriting...", file_name, file_size)

            return func(*args, file_name=file_name, overwrite=overwrite, **kwargs)

        return wrapper

    return decorator


def check_file_exist():
    """A decorator to check the existence of a file."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, file_name: str | Path, **kwargs):
            file_name = Path(file_name)
            if not file_name.is_file():
                raise FileNotFoundError(f"File {file_name} does not exist.")
            return func(*args, file_name=file_name, **kwargs)

        return wrapper

    return decorator


def get_file_name_from_template(  # pylint: disable=too-many-locals
    template: str,
    instance: object | None = None,
    output_directory: str | Path | None = None,
    exclude: set[str] | None = None,
) -> Path | NDArray[Path]:
    """Get the file name(s) from a template string.

    The template string uses double curly brackets for placeholders (e.g., '{{ x }}-{{ y }}.txt').
    If any placeholder refers to an array-like attribute (list, tuple, or iterable),
    the function generates all combinations of file names by iterating over the Cartesian product.
    For example, '{{ non-list }}-{{ list-A }}-{{ list-B }}' with list-A having X elements
    and list-B having Y elements returns a list of X * Y file names.

    Excluded placeholders are left unsubstituted (e.g., '{{ excluded }}' remains as-is).

    Args:
        template (str): A template string with placeholders.
        instance (object): An instance from which to retrieve attribute values.
        output_directory (str | Path | None): Optional output directory to prepend to the file names.
        exclude (set[str] | None): Set of attribute names to exclude from expansion. Defaults to None.

    Returns:
        str | np.ndarray: A single file name string if no array-like attributes are used,
        otherwise a nested list structure matching the dimensionality of the array placeholders.

    Raises:
        ValueError: If a placeholder refers to a non-existent attribute.
    """
    if exclude is None:
        exclude = set()

    # Find all unique placeholders in the template
    placeholders = list(dict.fromkeys(re.findall(r"\{\{\s*(\w+)\s*\}\}", template)))

    # Remove excluded placeholders from the list
    placeholders = [p for p in placeholders if p not in exclude]

    # Collect values for each placeholder
    values_dict = {}
    for label in placeholders:
        if label in exclude:
            continue
        try:
            value = getattr(instance, label)
        except AttributeError as e:
            raise ValueError(f"Attribute '{label}' not found in instance of type {type(instance).__name__}") from e

        if isinstance(value, Quantity):
            x = value.value
            if x.is_integer():
                x = int(x)
            values_dict[label] = [str(x)]
        # Check if value is array-like (list, tuple, or iterable but not str)
        elif isinstance(value, (list, tuple)) or (hasattr(value, "__iter__") and not isinstance(value, str)):
            values_dict[label] = [str(ele) for ele in list(value)]
        else:
            values_dict[label] = [str(value)]

    # Prepare lists for Cartesian product
    product_lists = [values_dict[label] for label in placeholders]

    # Generate all combinations
    combinations = list(itertools.product(*product_lists))

    # Helper function for substitution
    def substitute_template(combo_dict: dict) -> str:
        def replace(matched):
            label = matched.group(1).strip()
            if label in exclude:
                return matched.group(0)  # Return the original placeholder unchanged
            return str(combo_dict[label])

        return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace, template)

    # Substitute for each combination
    results = [Path(substitute_template(dict(zip(placeholders, combo)))) for combo in combinations]

    if output_directory is not None:
        output_directory = Path(output_directory)
        results = [output_directory / result for result in results]

    # Reshape into nested list if there are array placeholders
    array_placeholders = [p for p in placeholders if len(values_dict[p]) > 1]
    if array_placeholders:
        lengths = [len(values_dict[p]) for p in array_placeholders]

        def reshape_to_nested(flat_list: list, lengths: list[int]):
            if not lengths:
                return flat_list[0] if flat_list else ""
            size = lengths[0]
            sub_size = len(flat_list) // size
            nested = []
            for i in range(size):
                start = i * sub_size
                end = (i + 1) * sub_size
                sub_list = flat_list[start:end]
                nested.append(reshape_to_nested(sub_list, lengths[1:]))
            return nested

        return np.array(reshape_to_nested(results, lengths))
    # No arrays: return single string
    return results[0] if results else Path("")
