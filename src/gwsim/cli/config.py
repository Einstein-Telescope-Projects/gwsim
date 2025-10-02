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
def load_config(file_name: Path, encoding: str = "utf-8") -> dict:
    """Load configuration file.

    Args:
        file_name (Path): File name.
        encoding (str, optional): File encoding. Defaults to "utf-8".

    Returns:
        dict: A dictionary of the configuration.
    """
    with open(file_name, encoding=encoding) as f:
        config = yaml.safe_load(f)
    return config


@check_file_overwrite()
def save_config(
    file_name: Path, config: dict, overwrite: bool = False, encoding: str = "utf-8", backup: bool = True
) -> None:
    """Save configuration file safely with optional backup.

    Args:
        file_name (Path): File name.
        config (dict): A dictionary of configuration.
        overwrite (bool, optional): If True, overwrite the existing file, or otherwise raise an error.
            Defaults to False.
        encoding (str, optional): File encoding. Defaults to "utf-8".
        backup (bool, optional): If True and overwriting, create a backup of the existing file.
            Defaults to True.

    Raises:
        FileExistsError: If file_name exists and overwrite is False, raise an error.
    """
    # Create backup if file exists and we're overwriting
    if file_name.exists() and overwrite and backup:
        backup_path = file_name.with_suffix(f"{file_name.suffix}.backup")
        logger.info("Creating backup: %s", backup_path)
        backup_path.write_text(file_name.read_text(encoding=encoding), encoding=encoding)

    # Atomic write using temporary file
    temp_file = file_name.with_suffix(f"{file_name.suffix}.tmp")
    try:
        with open(temp_file, "w", encoding=encoding) as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        # Atomic move (rename) - this is atomic on most filesystems
        temp_file.replace(file_name)
        logger.info("Configuration saved to: %s", file_name)

    except Exception:
        # Clean up temp file if something went wrong
        if temp_file.exists():
            temp_file.unlink()
        raise


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
