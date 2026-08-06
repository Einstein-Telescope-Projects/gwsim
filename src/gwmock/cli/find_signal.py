# ruff: noqa: PLC0415
"""CLI command to look up which frame file(s) contain a given simulated signal."""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Annotated

import typer


def find_signal_command(
    metadata_dir: Annotated[
        Path,
        typer.Option("--metadata-dir", help="Directory with batch metadata files and signal_index.yaml."),
    ],
    event_id: Annotated[
        int | None,
        typer.Option("--id", help="Signal event id (its index in the coa_time-sorted population)."),
    ] = None,
    param: Annotated[
        list[str] | None,
        typer.Option("--param", help="Parameter filter, e.g. 'mass_1>=30' (repeatable; combined with AND)."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit results as JSON.")] = False,
) -> None:
    """Find the signal frame file(s) that contain a given injected signal.

    Look up by ``--id`` (fast, via ``signal_index.yaml``) and/or by one or more
    ``--param`` predicates matched against the source parameters recorded in the
    batch metadata (e.g. ``--param mass_1>=30 --param network_snr>8``).

    Reports every frame the signal reaches, which for a waveform crossing a segment
    boundary -- or for a continuous wave, which is in all of them -- is more than one.
    """
    from gwmock.cli.utils.signal_lookup import find_signals, parse_param_filter

    filters = [parse_param_filter(spec) for spec in (param or [])]
    if event_id is None and not filters:
        raise typer.BadParameter("Provide --id and/or at least one --param filter.")

    matches = find_signals(metadata_dir, event_id=event_id, param_filters=filters)

    if json_output:
        typer.echo(_json.dumps(matches, indent=2, default=str))
        return

    if not matches:
        typer.echo("No matching signals found.")
        raise typer.Exit(code=1)

    for match in matches:
        frames = ", ".join(match.get("frames") or []) or "(no frame recorded)"
        typer.echo(f"event {match['event_id']} (coa_time={match.get('coa_time')}) -> {frames}")
