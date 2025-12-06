"""CLI for deleting Zenodo repository depositions."""

from __future__ import annotations

import logging
from typing import Annotated

import typer
from rich.console import Console

from gwsim.cli.repository.utils import get_zenodo_client

logger = logging.getLogger("gwsim")
console = Console()


def delete_command(
    deposition_id: Annotated[str, typer.Argument(help="Deposition ID")],
    sandbox: Annotated[bool, typer.Option("--sandbox", help="Use sandbox environment")] = False,
    token: Annotated[str | None, typer.Option("--token", help="Zenodo access token")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation prompt")] = False,
) -> None:
    """Delete an unpublished deposition.

    Warning: Only unpublished (draft) depositions can be deleted.

    Examples:
        gwsim repository delete 123456
        gwsim repository delete 123456 --force
    """
    if not force and not typer.confirm(f"[red]Delete deposition {deposition_id}?[/red] This cannot be undone."):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(0)

    client = get_zenodo_client(sandbox=sandbox, token=token)

    console.print(f"[bold blue]Deleting deposition {deposition_id}...[/bold blue]")
    try:
        client.delete_deposition(deposition_id)
        console.print("[green]✓ Deposition deleted[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to delete: {e}[/red]")
        logger.error("Delete failed: %s", e)
        raise typer.Exit(1) from e
