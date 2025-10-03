"""
A tool to generate default configuration file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .utils.config import save_config

_DEFAULT_CONFIG = {
    "globals": {
        "working-directory": ".",
        "sampling-frequency": 16384,
        "duration": 4,
        "output-directory": "output",
        "metadata-directory": "metadata",
    },
    "simulators": {
        "example": {
            "class": "WhiteNoise",  # Resolves to gwsim.noise.WhiteNoise
            "arguments": {
                "batch_size": 1,
                "max_samples": 10,
                "loc": 0.0,
                "scale": 1.0,
                "seed": 0,
            },
            "output": {
                "file_name": "example-{{ start-time }}-{{ duration }}.gwf",
                "arguments": {
                    "channel": "STRAIN",
                },
            },
        },
    },
}


def default_config_command(
    output: Annotated[str, typer.Option("--output", help="File name of the output", prompt=True)] = "config.yaml",
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite the existing file")] = False,
) -> None:
    """Write the default configuration file to disk.

    Args:
        output (str): Name of the output file.
        overwrite (bool): If True, overwrite the existing file, otherwise raise an error if output already exists.

    Raises:
        FileExistsError: If file_name exists and overwrite is False, raise an error.
    """
    save_config(file_name=Path(output), config=_DEFAULT_CONFIG, overwrite=overwrite)
