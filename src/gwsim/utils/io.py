from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path

logger = logging.getLogger("gwsim")


def check_file_overwrite():
    """A decorator to check the existence of the file,
    and avoid overwriting it unintentionally.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, file_name: str | Path, overwrite: bool = False, **kwargs):
            file_name = Path(file_name)
            if file_name.is_file():
                if not overwrite:
                    raise FileExistsError(f"{file_name} already exists.")
                else:
                    logger.warning("%s already exists. Overwriting it.", file_name)
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
