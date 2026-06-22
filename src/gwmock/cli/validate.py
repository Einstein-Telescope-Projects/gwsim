# ruff: noqa PLC0415
"""
Validation functions for CLI commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def validate_command(
    paths: Annotated[
        list[Path],
        typer.Argument(
            help="Files/directories to validate. Can be output files, metadata files, or directories containing either."
        ),
    ],
    metadata_paths: Annotated[
        list[str] | None, typer.Option("--metadata-paths", help="Additional metadata files or directories")
    ] = None,
    pattern: Annotated[
        str | None, typer.Option("--pattern", help="File pattern to match (e.g., '*noise*' for subset validation)")
    ] = None,
    metadata_pattern: Annotated[
        str | None, typer.Option("--metadata-pattern", help="Metadata file pattern to match")
    ] = "*metadata.*",
) -> None:
    """Validate output files against metadata hashes and other checks.

    This command verifies the integrity of generated simulation files by:

    1. Loading metadata files and extracting expected file hashes
    2. Recomputing hashes for actual output files
    3. Comparing hashes and reporting mismatches
    4. Future: Add sampling rate and continuity checks

    The command automatically detects whether provided paths are:

    - Output files (.gwf, etc.) - will find corresponding metadata
    - Metadata files (.metadata.json / legacy .metadata.yaml) - will validate their output files
    - Directories - will scan for both types of files

    Examples:
        Validate specific output files (finds metadata automatically):

        - `gwmock validate H1-NOISE-123.gwf L1-SIGNAL-456.gwf`

        Validate specific metadata files:

        - `gwmock validate --metadata signal-0.metadata.yaml noise-0.metadata.yaml`

        Validate all files in a directory:
        gwmock validate /path/to/output/

        Validate subset using pattern:
        gwmock validate /path/to/output/ --pattern "*noise*"

        Mix files and directories:
        gwmock validate H1-NOISE-123.gwf /path/to/more/files/

        Override output directory:
        gwmock validate metadata/ --output-dir /custom/output/

    Args:
        paths: Files/directories to validate. Can be output files, metadata files, or directories containing either.
        metadata_paths: Additional metadata files or directories.
        pattern: File pattern to match (e.g., '*noise*' for subset validation).
        metadata_pattern: Metadata file pattern to match.
    """
    import fnmatch
    import logging

    from rich.console import Console
    from rich.table import Table

    from gwmock.cli.utils.hash import compute_file_hash
    from gwmock.cli.utils.metadata import load_metadata_with_external_state

    logger = logging.getLogger("gwmock")

    console = Console()

    logger.info("Validating simulation files...")

    # Separate into metadata files and potential output files
    metadata_files: list[Path] = []
    output_files: list[Path] = []
    output_directories: list[Path] = []
    metadata_directories: list[Path] = []

    for path_str in paths:
        path = Path(path_str)
        if path.is_dir():
            output_directories.append(path)
        elif path.is_file():
            # Assume it's an output file
            output_files.append(path)
        else:
            console.print(f"[red]Error:[/red] Path not found: {path}")

    for path_str in metadata_paths or []:
        path = Path(path_str)
        if path.is_dir():
            metadata_directories.append(path)
        elif path.is_file():
            if path.name.endswith(".metadata.json") or path.name.endswith(".metadata.yaml"):
                metadata_files.append(path)
            else:
                console.print(f"[yellow]Warning:[/yellow] Ignoring non-metadata file: {path}")
        else:
            console.print(f"[red]Error:[/red] Metadata path not found: {path}")

    # Scan directories for files
    for directory in output_directories:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                output_files.append(file_path)

    for directory in metadata_directories:
        for file_path in directory.rglob("*"):
            if "metadata" in file_path.name and file_path.is_file():
                metadata_files.append(file_path)

    # Apply pattern filtering if specified
    if pattern:
        output_files = [f for f in output_files if fnmatch.fnmatch(f.name, pattern)]

    if metadata_pattern:
        metadata_files = [f for f in metadata_files if fnmatch.fnmatch(f.name, metadata_pattern)]

    # Build validation plan: output_file -> metadata_file
    output_to_metadata = {}

    # First, extract output files from provided metadata files
    for metadata_file in metadata_files:
        try:
            metadata = load_metadata_with_external_state(metadata_file)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error loading metadata %s: %s", metadata_file, e)
            continue

        outputs = metadata.get("outputs", [])
        # Use the full relative ``path`` recorded in ``outputs`` (e.g.
        # ``output/signal/foo.gwf``) rather than just its basename. Joining only
        # the basename to ``working-directory`` dropped the ``output/<kind>/``
        # sub-directory, so the reconstructed path never existed on disk and the
        # file was reported as "File not found" *in addition to* the real PASS
        # found by the directory scan -- double-counting every output.
        output_files_in_meta = [output["path"] for output in outputs] or metadata.get("output_files", [])
        config = metadata.get("config", {})
        globals_config = config.get("globals", metadata.get("globals_config", {}))
        if outputs:
            output_dir = Path(globals_config.get("working-directory") or globals_config.get("working_directory") or ".")
        else:
            output_dir = Path(globals_config.get("output_directory", "."))

        for filename in output_files_in_meta:
            # Apply pattern filtering
            if pattern and not fnmatch.fnmatch(filename, pattern):
                continue

            output_file = output_dir / filename
            if output_file not in output_to_metadata:
                output_to_metadata[output_file] = metadata_file

    # Then, for explicitly provided output files, find their metadata
    for output_file in output_files:
        if output_file not in output_to_metadata:
            potential_metadata = None

            # First, check if any already-identified metadata files contain this output file
            for metadata_file in metadata_files:
                try:
                    meta_data = load_metadata_with_external_state(metadata_file)
                    output_names = [Path(output["path"]).name for output in meta_data.get("outputs", [])]
                    if output_file.name in output_names or output_file.name in meta_data.get("output_files", []):
                        potential_metadata = metadata_file
                        break
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error("Error reading metadata file %s: %s", metadata_file, e)
                    continue

            # If not found in existing metadata, search in directories
            if not potential_metadata:
                # Look in the same directory or parent directories
                search_dir = output_file.parent
                metadata_dir = search_dir / "metadata"
                if metadata_dir.exists():
                    # Look for metadata files that might match
                    for meta_file in list(metadata_dir.glob("*.metadata.json")) + list(
                        metadata_dir.glob("*.metadata.yaml")
                    ):
                        try:
                            meta_data = load_metadata_with_external_state(meta_file)
                            output_names = [Path(output["path"]).name for output in meta_data.get("outputs", [])]
                            if output_file.name in output_names or output_file.name in meta_data.get(
                                "output_files", []
                            ):
                                potential_metadata = meta_file
                                break
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            logger.error("Error reading metadata file %s: %s", meta_file, e)
                            continue

            if potential_metadata:
                output_to_metadata[output_file] = potential_metadata
            else:
                logger.warning("No metadata found for output file %s", output_file)  # Combine all metadata files
    all_metadata_files = list({v for v in output_to_metadata.values() if v is not None})

    if not all_metadata_files and not metadata_files:
        logger.error("Error: No metadata files found")
        raise typer.Exit(1)

    # Create results table
    table = Table(title="Validation Results")
    table.add_column("Metadata File", style="cyan")
    table.add_column("Output File", style="magenta")
    table.add_column("Hash Match", style="green")
    table.add_column("Status", style="yellow")

    total_files = len(output_to_metadata)
    failed_files = 0

    # Order output for consistent reporting
    for output_file in sorted(output_to_metadata.keys()):
        metadata_file = output_to_metadata[output_file]
        if metadata_file is None:
            table.add_row("N/A", output_file.name, "N/A", "[red]No metadata found[/red]")
            failed_files += 1
            continue
        try:
            metadata: dict = load_metadata_with_external_state(metadata_file)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error loading metadata %s: %s", metadata_file, e)
            table.add_row(str(metadata_file.name), output_file.name, "N/A", "[red]Error loading metadata[/red]")
            continue

        file_hashes = metadata.get("file_hashes", {})
        if file_hashes:
            expected_hash = file_hashes.get(output_file.name)
        else:
            output_records = {Path(output["path"]).name: output for output in metadata.get("outputs", [])}
            expected_hash = output_records.get(output_file.name, {}).get("sha256")

        if not output_file.exists():
            table.add_row(str(metadata_file.name), output_file.name, "N/A", "[red]File not found[/red]")
            failed_files += 1
            continue

        if not expected_hash:
            table.add_row(str(metadata_file.name), output_file.name, "N/A", "[yellow]No hash in metadata[/yellow]")
            failed_files += 1
            continue

        try:
            actual_hash = compute_file_hash(output_file)
            if actual_hash == expected_hash:
                table.add_row(str(metadata_file.name), output_file.name, "[green]✓[/green]", "[green]PASS[/green]")
            else:
                table.add_row(str(metadata_file.name), output_file.name, "[red]✗[/red]", "[red]HASH MISMATCH[/red]")
                failed_files += 1
        except Exception as e:  # pylint: disable=broad-exception-caught
            table.add_row(str(metadata_file.name), output_file.name, "N/A", f"[red]Error: {e}[/red]")
            failed_files += 1

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {total_files - failed_files}/{total_files} files passed validation")

    if failed_files > 0:
        console.print(f"[red]{failed_files} files failed validation[/red]")
        raise typer.Exit(1)
    console.print("[green]All files validated successfully![/green]")
