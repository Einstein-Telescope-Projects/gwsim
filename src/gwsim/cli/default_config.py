"""
A tool to generate default configuration file.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .config import save_config

_DEFAULT_CONFIG = {
    "working-directory": ".",
    "generator": {"class": None, "arguments": None},
    "output": {"file_name": None},
}


def default_config_command(
    output: str = typer.Option("config.yaml", "--output", help="File name of the output"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite the existing file"),
) -> None:
    """Write the default configuration file to disk.

    Args:
        output (str): Name of the output file.
        overwrite (bool): If True, overwrite the existing file, otherwise raise an error if output already exists.

    Raises:
        FileExistsError: If file_name exists and overwrite is False, raise an error.
    """
    save_config(file_name=Path(output), config=_DEFAULT_CONFIG, overwrite=overwrite)
