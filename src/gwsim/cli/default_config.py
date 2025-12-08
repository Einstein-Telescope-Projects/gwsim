"""
A tool to generate and manage default configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

# The default base path for example configuration files
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "examples"

_DEFAULT_CONFIG = {
    "globals": {
        "simulator-arguments": {
            "sampling-frequency": 4096,
            "duration": 4096,
            "total-duration": 4096,
            "start-time": 1577491218,
        },
        "working-directory": ".",
        "output-directory": "output",
        "metadata-directory": "metadata",
    },
    "simulators": {
        "example": {
            "class": "WhiteNoise",  # Resolves to gwsim.noise.WhiteNoise
            "arguments": {
                "loc": 0.0,
                "scale": 1.0,
                "seed": 42,
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


def _default_config_impl(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    output: str,
    get: list[str] | None,
    init: bool,
    list_: bool,
    overwrite: bool,
):
    """Internal implementation of the config command

    Args:
        output: Path to the output directory for --init or --get operations.
            Defaults to the current directory ('.'). Created if it doesn't exist.
        get: List of example config file names to copy.
            If omitted, copies all available examples to the output directory.
        init: If True, generates a default config file in the output directory.
        list_: If True, prints the names of available example config files.
        overwrite: If True, overwrites existing files without raising an error.

    """
    import logging  # pylint: disable=import-outside-toplevel
    import shutil  # pylint: disable=import-outside-toplevel

    from gwsim.cli.utils.config import Config, save_config  # pylint: disable=import-outside-toplevel

    logger = logging.getLogger("gwsim")
    logger.setLevel(logging.DEBUG)

    # Validate and normalize output path
    output_dir = Path(output).resolve()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory: %s", output_dir)
    elif not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output}")

    # Count active flags for mutual exclusivity
    active_flags = sum([bool(init), bool(list_), bool(get)])
    if active_flags == 0:
        logger.error("Error: Exactly one of --init, --list, or --get must be provided.")
        raise typer.Exit(code=1)
    if active_flags > 1:
        logger.error("Error: Only one of --init, --list, or --get can be used at a time.")
        raise typer.Exit(code=1)

    # Assume examples are in a 'examples/' dir relative to this script; adjust path as needed
    examples_dir = DEFAULT_CONFIG_PATH
    example_relpaths = [f.relative_to(examples_dir).as_posix() for f in examples_dir.rglob("*.yaml")]
    example_files = [f.split("/")[-1] for f in example_relpaths]

    # `List` flag
    if list_:
        logger.info("Available example configuration files:")
        for name in sorted(example_files):
            logger.info("  - %s", name)
        return

    # `Init` flag
    if init:
        default_config_path = output_dir / "default_config.yaml"
        config_instance = Config.model_validate(_DEFAULT_CONFIG)
        save_config(file_name=Path(default_config_path), config=config_instance, overwrite=overwrite)
        logger.info("Generated default configuration file: %s", default_config_path)
        return

    # `Get` flag
    if get is not None:
        files_to_copy = [f for f in example_relpaths if f.split("/")[-1] in get]
        if len(files_to_copy) < len(get):
            missing = set(get) - set(example_files)
            raise ValueError(f"Invalid example names: {', '.join(missing)}.\nAvailable: {', '.join(example_files)}")

        copied_count = 0
        for src_relpath in files_to_copy:
            src_path = examples_dir / src_relpath
            src_name = src_relpath.split("/")[-1]
            dst_path = output_dir / src_name
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if dst_path.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {dst_path}")
            shutil.copy2(src_path, dst_path)
            logger.info("Copied: %s -> %s", src_relpath, dst_path)
            copied_count += 1

        logger.info("Copied %s example configuration file(s) to %s.", copied_count, output_dir)
        return


def default_config_command(
    output: Annotated[
        str,
        typer.Option(
            "--output", "-o", help="Output directory for generated/copied config files (default: current directory)"
        ),
    ] = ".",
    get: Annotated[list[str] | None, typer.Option("--get", help="Copy the example configuration files")] = None,
    init: Annotated[bool, typer.Option("--init", help="Generate a default configuration file")] = False,
    list_: Annotated[bool, typer.Option("--list", help="List the available example configuration files")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing files")] = False,
) -> None:
    """
    Manage default configuration files

    Exactly one of --init, --list, or --get must be provided.

    Args:
        output: Path to the output directory for --init or --get operations.
            Defaults to the current directory ('.'). Created if it doesn't exist.
        get: List of example config file names to copy.
        init: If True, generates a default config file in the output directory.
        list_: If True, prints the names of available example config files.
        overwrite: If True, overwrites existing files without raising an error.

    Raises:
        typer.Exit: If no flags are provided or multiple flags are used together.
        FileExistsError: If a target file already exists during copy/init (use --force if added later).
        ValueError: If invalid example names are specified in --get or output path is invalid.
    """
    _default_config_impl(
        output=output,
        get=get,
        init=init,
        list_=list_,
        overwrite=overwrite,
    )
