"""Utility functions for file input/output operations with safety checks."""

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path

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
