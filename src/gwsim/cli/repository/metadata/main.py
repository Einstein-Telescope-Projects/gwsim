# ruff: noqa PLC0415

"""CLI for managing Zenodo metadata."""

from __future__ import annotations

import typer

# Create the metadata subcommand app
metadata_app = typer.Typer(
    name="metadata",
    help="Manage Zenodo metadata for simulation data.",
    rich_markup_mode="rich",
)


# Import and register commands after app is created
def register_commands() -> None:
    """Register all CLI commands."""

    from gwsim.cli.repository.metadata.update import update_command

    metadata_app.command("update")(update_command)


register_commands()
