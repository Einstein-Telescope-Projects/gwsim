# ruff: noqa PLC0415

"""CLI for uploading files to Zenodo depositions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def upload_command(
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
    import logging

    from rich.console import Console
    from rich.progress import Progress

    from gwsim.cli.repository.utils import get_zenodo_client

    logger = logging.getLogger("gwsim")
    console = Console()

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
