"""Utility functions for repository CLI commands."""

from __future__ import annotations

import os

import typer
from rich.console import Console

from gwsim.repository.zenodo import ZenodoClient

console = Console()


def get_zenodo_client(sandbox: bool = False, token: str | None = None) -> ZenodoClient:
    """Get a ZenodoClient instance with token from env or argument.

    Args:
        sandbox: Use sandbox environment for testing.
        token: Access token (defaults to ZENODO_TOKEN env var).

    Returns:
        Configured ZenodoClient.

    Raises:
        typer.Exit: If no token is provided or found in environment.
    """
    if token is None:
        token = os.environ.get("ZENODO_SANDBOX_API_TOKEN") if sandbox else os.environ.get("ZENODO_API_TOKEN")
    if not token:
        if sandbox:
            console.print(
                "[red]Error:[/red] No Zenodo Sandbox access token provided.\n"
                "Set [bold]ZENODO_SANDBOX_API_TOKEN[/bold] environment variable or use [bold]--token[/bold] option.\n"
                "Get a token from: https://sandbox.zenodo.org/account/settings/applications/tokens/new"
            )
        else:
            console.print(
                "[red]Error:[/red] No Zenodo access token provided.\n"
                "Set [bold]ZENODO_API_TOKEN[/bold] environment variable or use [bold]--token[/bold] option.\n"
                "Get a token from: https://zenodo.org/account/settings/applications/tokens/new"
            )
        raise typer.Exit(1)
    return ZenodoClient(access_token=token, sandbox=sandbox)
