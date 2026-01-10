# ruff: noqa PLC0415

"""CLI for updating Zenodo repository depositions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def update_command(
    deposition_id: Annotated[str, typer.Argument(help="Deposition ID")],
    metadata_file: Annotated[
        Path | None, typer.Option("--metadata-file", help="YAML file with metadata to update")
    ] = None,
    sandbox: Annotated[bool, typer.Option("--sandbox", help="Use sandbox environment")] = False,
    token: Annotated[str | None, typer.Option("--token", help="Zenodo access token")] = None,
) -> None:
    """Update metadata for a deposition.

    Examples:
        gwsim repository update 123456 --metadata-file metadata.yaml
    """
    import logging

    import yaml
    from rich.console import Console

    from gwsim.cli.repository.utils import get_zenodo_client

    logger = logging.getLogger("gwsim")
    console = Console()

    if not metadata_file:
        metadata_file = Path(typer.prompt("Path to metadata YAML file"))

    if not metadata_file.exists():
        console.print(f"[red]Error:[/red] File not found: {metadata_file}")
        raise typer.Exit(1)

    with metadata_file.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    if not metadata:
        console.print("[red]Error:[/red] Metadata file is empty.")
        raise typer.Exit(1)

    client = get_zenodo_client(sandbox=sandbox, token=token)

    console.print(f"[bold blue]Updating metadata for deposition {deposition_id}...[/bold blue]")
    try:
        client.update_metadata(deposition_id, metadata)
        console.print("[green]✓ Metadata updated successfully[/green]")
        console.print(f"[cyan]Next:[/cyan] gwsim repository publish {deposition_id}")
    except Exception as e:
        console.print(f"[red]✗ Failed to update metadata: {e}[/red]")
        logger.error("Update metadata failed: %s", e)
        raise typer.Exit(1) from e
