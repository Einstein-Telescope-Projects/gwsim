# ruff: noqa PLC0415

"""CLI for creating Zenodo repository depositions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def create_command(  # pylint: disable=too-many-locals
    title: Annotated[str | None, typer.Option("--title", help="Deposition title")] = None,
    description: Annotated[str | None, typer.Option("--description", help="Deposition description")] = None,
    metadata_file: Annotated[
        Path | None, typer.Option("--metadata-file", help="YAML file with additional metadata")
    ] = None,
    sandbox: Annotated[bool, typer.Option("--sandbox", help="Use sandbox environment for testing")] = False,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            help=(
                "Zenodo access token (default: ZENODO_API_TOKEN env or"
                " ZENODO_SANDBOX_API_TOKEN env for Zenodo Sandbox)"
            ),
        ),
    ] = None,
) -> None:
    """Create a new deposition on Zenodo.

    Interactive mode: Leave options blank to be prompted.

    Examples:
        # Interactive mode
        gwsim repository create

        # With all options
        gwsim repository create --title "GW Simulation Data" --description "MDC v1"

        # Using metadata file
        gwsim repository create --metadata-file metadata.yaml
    """
    import logging

    import yaml
    from rich.console import Console

    from gwsim.cli.repository.utils import get_zenodo_client

    logger = logging.getLogger("gwsim")
    console = Console()

    client = get_zenodo_client(sandbox=sandbox, token=token)

    if title is None:
        title = typer.prompt("Deposition Title")

    if description is None:
        description = typer.prompt("Deposition Description", default="")

    metadata_dict = {"title": title}
    metadata_dict["description"] = description

    if metadata_file:
        if not metadata_file.exists():
            console.print(f"[red]Error:[/red] Metadata file not found: {metadata_file}")
            raise typer.Exit(1)
        with metadata_file.open("r") as f:
            extra = yaml.safe_load(f)
            if extra:
                metadata_dict.update(extra)

    console.print("[bold blue]Creating deposition...[/bold blue]")
    try:
        result = client.create_deposition(metadata=metadata_dict)
        deposition_id = result.get("id")
        if sandbox:
            console.print("[yellow]Note:[/yellow] Created in Zenodo Sandbox environment.")
        console.print("[green]✓ Deposition created successfully![/green]")
        console.print(f"  [cyan]ID:[/cyan] {deposition_id}")
        if sandbox:
            console.print(
                "[yellow]Note:[/yellow] This is a sandbox deposition. Use [bold]--sandbox[/bold] to access it later."
            )
            console.print(f"  [cyan]Next:[/cyan] gwsim repository upload {deposition_id} --file <path> --sandbox")
        else:
            console.print(f"  [cyan]Next:[/cyan] gwsim repository upload {deposition_id} --file <path>")
    except Exception as e:
        console.print(f"[red]✗ Failed to create deposition: {e}[/red]")
        logger.error("Create deposition failed: %s", e)
        raise typer.Exit(1) from e
