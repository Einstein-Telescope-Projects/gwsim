# ruff: noqa PLC0415
"""CLI for managing Zenodo repositories."""

from __future__ import annotations

from typing import Annotated

import typer


def verify_command(
    sandbox: Annotated[bool, typer.Option("--sandbox", help="Verify sandbox token")] = False,
    token: Annotated[str | None, typer.Option("--token", help="Zenodo access token to verify")] = None,
) -> None:
    """Verify that your Zenodo API token is valid and has the correct permissions.

    Examples:
        # Verify production token
        gwsim repository verify

        # Verify sandbox token
        gwsim repository verify --sandbox

        # Verify with explicit token
        gwsim repository verify --token your_token_here
    """
    from rich.console import Console

    from gwsim.cli.repository.utils import get_zenodo_client

    console = Console()

    console.print("[bold blue]Verifying Zenodo API token...[/bold blue]")

    try:
        client = get_zenodo_client(sandbox=sandbox, token=token)

        # Try to list depositions as a test
        console.print("Testing API access...")
        depositions = client.list_depositions(status="draft")

        env_name = "Zenodo Sandbox" if sandbox else "Zenodo (Production)"
        console.print("[green]✓ Token is valid![/green]")
        console.print(f"  [cyan]Environment:[/cyan] {env_name}")
        console.print(f"  [cyan]Found {len(depositions)} draft deposition(s)[/cyan]")

    except Exception as e:
        env_name = "Zenodo Sandbox" if sandbox else "Zenodo (Production)"
        console.print(f"[red]✗ Token verification failed for {env_name}[/red]")
        console.print(f"  [yellow]Error:[/yellow] {e}")
        console.print("\n[bold]Troubleshooting:[/bold]")
        if sandbox:
            console.print(
                "  1. Get a new token from: https://sandbox.zenodo.org/account/settings/applications/tokens/new"
            )
            console.print("  2. Ensure the token has 'deposit:write' and 'deposit:actions' scopes")
            console.print("  3. Set: export ZENODO_SANDBOX_API_TOKEN='your_token'")
        else:
            console.print("  1. Get a new token from: https://zenodo.org/account/settings/applications/tokens/new")
            console.print("  2. Ensure the token has 'deposit:write' and 'deposit:actions' scopes")
            console.print("  3. Set: export ZENODO_API_TOKEN='your_token'")
        raise typer.Exit(1) from e
