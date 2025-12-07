"""CLI for managing Zenodo repositories."""

from __future__ import annotations

import typer

from gwsim.cli.repository.create import create_command
from gwsim.cli.repository.delete import delete_command
from gwsim.cli.repository.download import download_command
from gwsim.cli.repository.list_depositions import list_depositions_command
from gwsim.cli.repository.update import update_command
from gwsim.cli.repository.upload import upload_command
from gwsim.cli.repository.verify import verify_command

# Create the repository subcommand app
repository_app = typer.Typer(
    name="repository",
    help="Manage Zenodo repositories for simulation data.",
    rich_markup_mode="rich",
)


# Import and register commands after app is created
def register_commands() -> None:
    """Register all CLI commands."""

    repository_app.command("create")(create_command)
    repository_app.command("upload")(upload_command)
    repository_app.command("update")(update_command)
    repository_app.command("download")(download_command)
    repository_app.command("list")(list_depositions_command)
    repository_app.command("delete")(delete_command)
    repository_app.command("verify")(verify_command)


register_commands()
