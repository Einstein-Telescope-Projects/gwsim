# ruff: noqa PLC0415

"""
A tool to generate and manage default configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def _config_command_impl(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    output: Path,
    get: Path | None,
    init: Path | None,
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
    import logging

    from gwsim.cli.utils.config import (
        get_examples_dir,
        load_config,
        save_config,
    )

    def copy_config_file(src_path: Path, dst_path: Path, overwrite: bool) -> None:
        """Copy a configuration file from src_path to dst_path.

        Args:
            src_path: Source file path.
            dst_path: Destination file path.
            overwrite: Whether to overwrite existing files.
        """
        # Load the configuration file
        config = load_config(src_path)

        # Create parent directories if they don't exist
        dst_path.resolve().parent.mkdir(parents=True, exist_ok=True)

        # Save the configuration file to the destination
        try:
            save_config(file_name=dst_path, config=config, overwrite=overwrite, backup=False)
        except FileExistsError as e:
            raise FileExistsError(f"File already exists: {dst_path}. Use --overwrite to overwrite.") from e

    logger = logging.getLogger("gwsim")

    examples_dir = get_examples_dir()

    # Generate default config file
    if init is not None:
        copy_config_file(src_path=examples_dir / "default_config" / "config.yaml", dst_path=init, overwrite=overwrite)
        logger.info("Generated default configuration file: %s", init)
        return

    if list_:
        available_labels = set()
        for config_file in examples_dir.rglob("config.yaml"):
            # Show the full relative path from examples_dir, excluding the filename
            relative_path = config_file.relative_to(examples_dir)
            available_labels.add(relative_path.parent)
        logger.info("[bold cyan]Available example configuration labels:[/bold cyan]")
        for label in sorted(available_labels):
            logger.info("  - %s", label)

        logger.info("To copy example configuration files, use the [bold green]--get[/bold green] option with a label.")
        logger.info(
            "For example: %s",
            f"[bold green]gwsim config --get {sorted(available_labels)[0]} --output config.yaml[/bold green]",
        )
        return

    # pylint: disable=duplicate-code
    if get is not None:
        src_path = examples_dir / get / "config.yaml"
        if not src_path.exists():
            logger.error("Example configuration '%s' does not exist in examples directory.", get)
            raise typer.Exit(1)
        dst_path = output / "config.yaml" if output.is_dir() else output
        copy_config_file(src_path=src_path, dst_path=dst_path, overwrite=overwrite)
        logger.info("Copied example configuration file: %s to %s", get / "config.yaml", dst_path)
        return

    logger.error(
        "No action specified. Please provide one of [bold green]--init[/bold green], "
        "[bold green]--list[/bold green], or [bold green]--get[/bold green]."
    )
    logger.error("Use [bold green]--help[/bold green] for more information.")
    raise typer.Exit(1)


def config_command(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output file or directory for generated/copied config files (default: current directory)",
        ),
    ] = Path("config.yaml"),
    get: Annotated[Path | None, typer.Option("--get", help="Label of the configuration file to copy")] = None,
    init: Annotated[Path | None, typer.Option("--init", help="Generate a default configuration file")] = None,
    list_: Annotated[bool, typer.Option("--list", help="List the available example configuration files")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing files")] = False,
) -> None:
    """
    Manage default and example configuration files.

    This command provides utilities to list available example configuration files,
    generate a default configuration, or copy one or more example configurations
    into a target directory.

    Exactly **one** of ``--init``, ``--list``, or ``--get`` must be provided.

    **Options:**

    - `--init`
          Generate a default configuration file in the output directory.

    - `--list`
          Print the names of all available example configuration files and exit.

    - `--get <NAME> [<NAME> ...]`
          Copy one or more example configuration files into the output directory.
            The names must match the values shown by ``--list``.

    - `--output <PATH>`
          Output directory where configuration files will be written.
            Defaults to the current directory. The directory is created if it
            does not already exist.

    - `--overwrite`
          Overwrite existing files in the output directory. If not set,
            the command fails when a target file already exists.

    Examples:
        List available example configuration files:

            gwsim config --list

        Generate a default configuration file in the current directory:

            gwsim config --init

        Generate a default configuration file with a specific name:

            gwsim config --init config.yaml

        Copy specific example configuration files:

            gwsim config --get basic.yaml

        Copy example configuration files and overwrite existing ones:

            gwsim config --get basic.yaml --overwrite

    Args:
        output: Path to the output directory for --init or --get operations.
            Defaults to the current directory ('.'). Created if it doesn't exist.
        get: List of example config file names to copy.
        init: If provided, generates a default config file in the current directory or specified path.
        list_: If True, prints the names of available example config files.
        overwrite: If True, overwrites existing files without raising an error.

    Raises:
        typer.Exit: If no flags are provided or multiple flags are used together.
        FileExistsError: If a target file already exists during copy/init (use --force if added later).
        ValueError: If invalid example names are specified in --get or output path is invalid.
    """
    _config_command_impl(
        output=output,
        get=get,
        init=init,
        list_=list_,
        overwrite=overwrite,
    )
