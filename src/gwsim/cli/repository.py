"""CLI for managing Zenodo repositories."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from gwsim.repository.zenodo import ZenodoClient

logger = logging.getLogger("gwsim")
console = Console()

# Create the repository subcommand app
repository_app = typer.Typer(
    name="repository",
    help="Manage Zenodo repositories for simulation data.",
    rich_markup_mode="rich",
)


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
        if sandbox:
            token = os.environ.get("ZENODO_SANDBOX_API_TOKEN")
        else:
            token = os.environ.get("ZENODO_API_TOKEN")
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


@repository_app.command()
def create(
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


@repository_app.command()
def upload(
    deposition_id: Annotated[str, typer.Argument(help="Deposition ID")],
    files: Annotated[list[str] | None, typer.Option("--file", help="Files to upload (repeat for multiple)")] = None,
    sandbox: Annotated[bool, typer.Option("--sandbox", help="Use sandbox environment")] = False,
    token: Annotated[str | None, typer.Option("--token", help="Zenodo access token")] = None,
) -> None:
    """Upload files to a deposition.

    Examples:
        # Single file
        gwsim repository upload 123456 --file data.gwf

        # Multiple files
        gwsim repository upload 123456 --file data1.gwf --file data2.gwf --file metadata.yaml
    """
    if not files:
        console.print("[red]Error:[/red] No files specified. Use [bold]--file <path>[/bold] to specify files.")
        raise typer.Exit(1)

    client = get_zenodo_client(sandbox=sandbox, token=token)

    console.print(f"[bold blue]Uploading {len(files)} file(s) to deposition {deposition_id}...[/bold blue]")

    failed_count = 0
    with Progress() as progress:
        task = progress.add_task("Uploading", total=len(files))

        for file_path_str in files:
            file_path = Path(file_path_str)
            if not file_path.exists():
                console.print(f"[red]✗ File not found:[/red] {file_path}")
                failed_count += 1
                progress.update(task, advance=1)
                continue

            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                client.upload_file(deposition_id, file_path, auto_timeout=True)
                console.print(f"[green]✓ {file_path.name}[/green] ({file_size_mb:.2f} MB)")
                progress.update(task, advance=1)
            except Exception as e:  # pylint: disable=broad-exception-caught
                console.print(f"[red]✗ Failed to upload {file_path.name}:[/red] {e}")
                logger.error("Upload failed for %s: %s", file_path, e)
                failed_count += 1
                progress.update(task, advance=1)

    if failed_count == 0:
        if sandbox:
            console.print("[cyan]Next:[/cyan] gwsim repository update <id> --metadata-file <file> --sandbox")
        else:
            console.print("[cyan]Next:[/cyan] gwsim repository update <id> --metadata-file <file>")
    else:
        console.print(f"[yellow]Warning:[/yellow] {failed_count} file(s) failed to upload.")
        raise typer.Exit(1)


@repository_app.command()
def update(
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


@repository_app.command()
def publish(
    deposition_id: Annotated[str, typer.Argument(help="Deposition ID")],
    sandbox: Annotated[bool, typer.Option("--sandbox", help="Use sandbox environment")] = False,
    token: Annotated[str | None, typer.Option("--token", help="Zenodo access token")] = None,
) -> None:
    """Publish a deposition to Zenodo.

    Warning: Publishing is permanent and cannot be undone.

    Examples:
        gwsim repository publish 123456
    """
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


@repository_app.command()
def delete(
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


@repository_app.command()
def download(
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


@repository_app.command(name="list")
def list_depositions(
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

        for dep in depositions:
            dep_id = str(dep.get("id", "N/A"))
            title = dep.get("metadata", {}).get("title", "N/A")
            if len(title) > 38:
                title = title[:35] + "..."
            doi = dep.get("doi", "N/A")
            created = dep.get("created", "N/A")[:10]
            table.add_row(dep_id, title, doi, created)

        console.print(table)
    except Exception as e:
        console.print(f"[red]✗ Failed to list depositions: {e}[/red]")
        logger.error("List depositions failed: %s", e)
        raise typer.Exit(1) from e
