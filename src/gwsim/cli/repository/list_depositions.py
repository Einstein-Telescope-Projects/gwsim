# ruff: noqa PLC0415
"""CLI for managing Zenodo repositories."""

from __future__ import annotations

from typing import Annotated

import typer


def list_depositions_command(
    status: Annotated[
        str, typer.Option("--status", help="Filter by status (draft, published, unsubmitted)")
    ] = "published",
    sandbox: Annotated[bool, typer.Option("--sandbox", help="Use sandbox environment")] = False,
    token: Annotated[str | None, typer.Option("--token", help="Zenodo access token")] = None,
) -> None:
    """List depositions for the authenticated user.

    Examples:
        gwsim repository list
        gwsim repository list --status draft
        gwsim repository list --status published --sandbox
    """
    import logging

    from rich.console import Console
    from rich.table import Table

    from gwsim.cli.repository.utils import get_zenodo_client

    logger = logging.getLogger("gwsim")
    console = Console()

    client = get_zenodo_client(sandbox=sandbox, token=token)

    console.print(f"[bold blue]Listing {status} depositions...[/bold blue]")
    try:
        depositions = client.list_depositions(status=status)

        if not depositions:
            console.print(f"[yellow]No {status} depositions found.[/yellow]")
            return

        table = Table(title=f"{status.capitalize()} Depositions")
        table.add_column("ID", style="cyan", width=12)
        table.add_column("Title", style="green", width=40)
        table.add_column("DOI", style="blue", width=20)
        table.add_column("Created", style="magenta", width=12)

        max_length_of_title = 38
        for dep in depositions:
            dep_id = str(dep.get("id", "N/A"))
            title = dep.get("metadata", {}).get("title", "N/A")
            if len(title) > max_length_of_title:
                title = title[:35] + "..."
            doi = dep.get("doi", "N/A")
            created = dep.get("created", "N/A")[:10]
            table.add_row(dep_id, title, doi, created)

        console.print(table)
    except Exception as e:
        console.print(f"[red]✗ Failed to list depositions: {e}[/red]")
        logger.error("List depositions failed: %s", e)
        raise typer.Exit(1) from e
