"""Turn gwmock-signal's batched device output into gwmock chunks.

The batched entry point returns ``strain`` of shape ``(n_events, n_detectors, n_samples)`` --
per-event and *unsuperposed*, with placement left to the caller. gwmock already has an assembler for
exactly that: ``TimeSeries.inject_from_list`` places chunks by absolute time, caches whatever
overflows a segment, and carries it into the next one. So the device does generation and projection,
and gwmock keeps the incremental, checkpointed assembly it already owns.

That split is deliberate rather than a compromise. Generation was measured at 9.67e-3 s per second
of data against 3.66e-4 for assembly, so the expensive 96% moves to the device while the part that
owns spill-over, resumability and memory bounds stays where it works.

Placement is required to be **on the output lattice**. When ``simulate_cbc_batch`` is given an
``output_grid``, each buffer starts exactly on a sample of that grid, so injection is an
integer-offset add. Without it, buffers begin at an arbitrary time and gwmock has to interpolate
every chunk onto the segment grid -- which costs a resample per chunk and is, in gwmock-signal's own
words, "accurate only for heavily oversampled strain". This module refuses that case rather than
letting it degrade quietly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList

#: Dimensions of a batched strain array: event, detector, sample.
_BATCHED_STRAIN_DIMENSIONS = 3

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gwmock_signal import BatchedDetectorStrain


def batched_strain_to_chunks(
    batch: BatchedDetectorStrain,
    *,
    expected_detector_names: tuple[str, ...] | None = None,
) -> TimeSeriesList:
    """Convert one batched device result into per-event chunks ready for injection.

    Args:
        batch: Result of ``gwmock_signal.simulate_cbc_batch``, generated with an ``output_grid``.
        expected_detector_names: Detector order the caller intends to write. Checked rather than
            assumed, because the chunk's channel *k* becomes detector *k* downstream and a
            reordering would silently attribute each signal to the wrong interferometer.

    Returns:
        One chunk per event, each of shape ``(n_detectors, n_samples)``, in event order.

    Raises:
        ValueError: If the batch was generated without an output grid, if any buffer does not start
            on that grid, or if the detector names disagree with *expected_detector_names*.
    """
    if batch.grid is None or batch.start_index is None:
        raise ValueError(
            "The batched strain was generated without an output grid, so each buffer starts at an "
            "arbitrary time and every chunk would have to be resampled onto the segment grid. Pass "
            "output_grid=SamplingGrid(segment_start, sampling_frequency) to simulate_cbc_batch."
        )

    produced_names = tuple(batch.detector_names)
    if expected_detector_names is not None and produced_names != tuple(expected_detector_names):
        raise ValueError(
            f"The device returned detectors {produced_names} but the caller expects "
            f"{tuple(expected_detector_names)}. Channel order carries detector identity downstream, "
            f"so this would attribute each signal to the wrong interferometer."
        )

    # One host transfer for the whole batch. Converting per event would copy the same device buffer
    # repeatedly, and the device-to-host path was measured to be allocator-bound rather than
    # bus-bound, so the number of transfers is what costs.
    strain = np.asarray(batch.strain)
    if strain.ndim != _BATCHED_STRAIN_DIMENSIONS:
        raise ValueError(f"Expected strain of shape (n_events, n_detectors, n_samples), got {strain.shape}.")

    start_indices = np.atleast_1d(np.asarray(batch.start_index))
    if start_indices.size != strain.shape[0]:
        raise ValueError(
            f"Got {start_indices.size} start indices for {strain.shape[0]} events; each event needs "
            f"its own buffer position."
        )
    if strain.shape[1] != len(produced_names):
        raise ValueError(f"Strain has {strain.shape[1]} detector rows but {len(produced_names)} detector names.")

    grid = batch.grid
    start_times = np.array([float(grid.time_of(int(index))) for index in start_indices], dtype=float)

    off_lattice = ~np.atleast_1d(grid.is_on_lattice(start_times))
    if np.any(off_lattice):
        first = int(np.argmax(off_lattice))
        raise ValueError(
            f"Event {first}'s buffer starts at GPS {start_times[first]!r}, which is not on the "
            f"output grid (epoch {grid.epoch!r}, {grid.sampling_frequency!r} Hz). Injecting it would "
            f"resample the chunk instead of placing it exactly."
        )

    sampling_frequency = float(batch.sampling_frequency)
    return TimeSeriesList(
        [
            TimeSeries(
                data=np.ascontiguousarray(strain[event]),
                start_time=float(start_times[event]),
                sampling_frequency=sampling_frequency,
            )
            for event in range(strain.shape[0])
        ]
    )


def per_event_injections(parameters: dict[str, Any], event_ids: list[int]) -> list[dict[str, Any]]:
    """Return provenance records for a batch, one per event.

    The per-event path records ``{"event_id", "parameters"}`` for each injection, and the batched
    path has to produce the same thing or provenance silently thins out as soon as the device path is
    used. Chunks stay per-event precisely so this remains possible.

    Args:
        parameters: The struct-of-arrays handed to the device, one entry per parameter.
        event_ids: Population indices of the events in the batch, in the same order.

    Returns:
        One record per event, with that event's scalar parameters.

    Raises:
        ValueError: If a parameter column is shorter than the number of events.
    """
    records: list[dict[str, Any]] = []
    for position, event_id in enumerate(event_ids):
        scalars: dict[str, Any] = {}
        for name, column in parameters.items():
            if np.ndim(column) == 0:
                scalars[name] = column
                continue
            values = np.atleast_1d(np.asarray(column))
            if position >= values.size:
                raise ValueError(
                    f"Parameter '{name}' has {values.size} values but the batch has "
                    f"{len(event_ids)} events, so event {event_id} has no value for it."
                )
            scalars[name] = values[position].item() if hasattr(values[position], "item") else values[position]
        records.append({"event_id": int(event_id), "parameters": scalars})
    return records
