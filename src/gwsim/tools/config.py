"""
Utility functions to load and save configuration files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from ..utils.io import check_file_exist, check_file_overwrite

logger = logging.getLogger("gwsim")


@check_file_exist()
def load_config(file_name: Path) -> dict:
    """Load configuration file.

    Args:
        file_name (Path): File name.

    Returns:
        dict: A dictionary of the configuration.
    """
    with open(file_name) as f:
        config = yaml.safe_load(f)
    return config


@check_file_overwrite()
def save_config(file_name: Path, config: dict, overwrite: bool = False) -> None:
    """Save configuration file.

    Args:
        file_name (Path): File name.
        config (dict): A dictionary of configuration.
        overwrite (bool, optional): If True, overwrite the existing file, or otherwise raise an error.
            Defaults to False.

    Raises:
        FileExistsError: If file_name exists and overwrite is False, raise an error.
    """
    with open(file_name, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def get_config_value(config: dict, key: str, default_value: Any | None = None) -> Any:
    """Get the argument

    Args:
        config (dict): A dictionary of configuration.
        key (str): Key of the entry.
        default_value (Any | None, optional): Default value if key is not present. Defaults to None.

    Returns:
        Any: Value of the corresponding key in config.
    """
    if key in config:
        return config[key]
    return default_value
