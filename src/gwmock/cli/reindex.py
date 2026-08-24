# ruff: noqa: PLC0415
"""CLI command to rebuild ``signal_index.yaml`` from the batch metadata files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def reindex_command(
    metadata_dir: Annotated[
        Path,
        typer.Option("--metadata-dir", help="Directory with batch metadata files and signal_index.yaml."),
    ],
) -> None:
    """Rebuild the signal index from the batch metadata files.

    ``signal_index.yaml`` is a cache of the injections recorded in the
    ``*.metadata.json`` files, which are the source of truth. Rebuilding it
    discards whatever the index held and derives it again from those files, so an
    index that lost entries -- concurrent runs sharing this directory on a
    filesystem without working ``flock``, or writers on different hosts -- is
    repaired without rerunning any simulation.

    The rebuild takes the same exclusive lock a running batch does, and it
    re-baselines the digest recorded beside the index, so a directory whose writes
    were being refused as stale accepts them again afterwards.

    Stop writers on other hosts first: a rebuild indexes the batch metadata files
    it can list, and a shared filesystem may not yet be listing one another host
    wrote.
    """
    from gwmock.cli.simulate_utils import SignalIndexRebuildError, rebuild_signal_index

    try:
        rebuilt = rebuild_signal_index(metadata_dir)
    except SignalIndexRebuildError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    typer.echo(
        f"Rebuilt {rebuilt.index_file} from {rebuilt.batches} batch metadata file(s): {rebuilt.events} event(s)."
    )
