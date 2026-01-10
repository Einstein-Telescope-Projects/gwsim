# ruff: noqa PLC0415

"""CLI for publishing Zenodo repository depositions."""

from __future__ import annotations

from typing import Annotated

import typer


def publish_command(
    deposition_id: Annotated[str, typer.Argument(help="Deposition ID")],
    sandbox: Annotated[bool, typer.Option("--sandbox", help="Use sandbox environment")] = False,
    token: Annotated[str | None, typer.Option("--token", help="Zenodo access token")] = None,
) -> None:
    """Publish a deposition to Zenodo.

    Warning: Publishing is permanent and cannot be undone.

    Examples:
        gwsim repository publish 123456
    """
    import logging

    from rich.console import Console

    from gwsim.cli.repository.utils import get_zenodo_client

    logger = logging.getLogger("gwsim")
    console = Console()

    if not typer.confirm(
        f"[yellow]Publish deposition {deposition_id}?[/yellow] This action is permanent and cannot be undone."
    ):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(0)

    client = get_zenodo_client(sandbox=sandbox, token=token)

    console.print(f"[bold blue]Publishing deposition {deposition_id}...[/bold blue]")
    try:
        result = client.publish_deposition(deposition_id)
        doi = result.get("doi")
        console.print("[green]✓ Published successfully![/green]")
        console.print(f"  [cyan]DOI:[/cyan] {doi}")
        if sandbox:
            console.print(
                "[yellow]Note:[/yellow] This is a sandbox record. Use [bold]--sandbox[/bold] to access it later."
            )
    except Exception as e:
        console.print(f"[red]✗ Failed to publish: {e}[/red]")
        logger.error("Publish failed: %s", e)
        raise typer.Exit(1) from e
