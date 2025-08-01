"""
A tool to generate default configuration file.
"""

from __future__ import annotations

from pathlib import Path

import click

from .config import save_config

_DEFAULT_CONFIG = {
    "working-directory": ".",
    "generator": {"class": None, "arguments": None},
    "output": {"file_name": None},
}


@click.command("default-config", help="Write a default configuration file to disk.")
@click.option("--output", type=str, help="File name of the output", default="config.yaml")
@click.option("--overwrite", is_flag=True, help="If flagged, the existing file will be overwritten.")
def default_config(output: str, overwrite: bool) -> None:
    """Write the default configuration file to disk.

    Args:
        output (str): Name of the output file.
        overwrite (bool): If True, overwrite the existing file, otherwise raise an error if output already exists.

    Raises:
        FileExistsError: If file_name exists and overwrite is False, raise an error.
    """
    save_config(file_name=Path(output), config=_DEFAULT_CONFIG, overwrite=overwrite)
