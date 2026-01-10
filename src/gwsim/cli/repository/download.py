# ruff: noqa PLC0415

"""CLI for downloading Zenodo repository files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def download_command(
    deposition_id: Annotated[str | None, typer.Argument(help="Deposition ID")] = None,
    filename: Annotated[str | None, typer.Option("--file", help="Filename to download")] = None,
    output: Annotated[Path | None, typer.Option("--output", help="Output file path")] = None,
    sandbox: Annotated[bool, typer.Option("--sandbox", help="Use sandbox environment")] = False,
    file_size_mb: Annotated[int | None, typer.Option("--file-size-mb", help="File size in MB (for timeout)")] = None,
) -> None:
    """Download a file from a published Zenodo record.

    Examples:
        gwsim repository download 10.5281/zenodo.123456 --file data.gwf --output ./data.gwf
        gwsim repository download 10.5281/zenodo.123456 --file metadata.yaml
    """
    import logging

    from rich.console import Console

    from gwsim.cli.repository.utils import get_zenodo_client

    logger = logging.getLogger("gwsim")
    console = Console()

    if not deposition_id:
        deposition_id = typer.prompt("Deposition ID (e.g., 123456)")
    if not filename:
        filename = typer.prompt("Filename to download")
    if not output and filename:
        output = Path(filename)
    else:
        console.print("[red]Error:[/red] Output path must be specified.")
        raise typer.Exit(1)

    client = get_zenodo_client(sandbox=sandbox, token=None)  # Downloads don't require auth

    console.print(f"[bold blue]Downloading {filename} from {deposition_id}...[/bold blue]")

    try:
        client.download_file(str(deposition_id), filename, output, file_size_in_mb=file_size_mb)
        console.print("[green]✓ Downloaded successfully[/green]")
        console.print(f"  [cyan]Saved to:[/cyan] {output.resolve()}")
    except Exception as e:
        console.print(f"[red]✗ Download failed: {e}[/red]")
        logger.error("Download failed: %s", e)
        raise typer.Exit(1) from e
