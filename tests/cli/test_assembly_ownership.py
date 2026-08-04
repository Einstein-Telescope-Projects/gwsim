"""Two implementations could place signals into segments; this pins that they agree.

``gwmock_signal.jax_batch.assemble_segments`` scatters a batch into a fixed tiling of segments in
memory, given every segment start up front. gwmock assembles differently by necessity: it streams,
holding one segment at a time, claiming events for it and carrying the overflow forward as a tail.

gwmock owns assembly and does not call ``assemble_segments`` -- the batched path is a different way
to *generate*, not a different way to assemble. That decision is recorded in
:meth:`~gwmock.cli.adapter_orchestration.AdapterOrchestrator._simulate_batched_segment`, and it is
the reason this file exists rather than a deduplication: two assemblers stay in the ecosystem, so
the risk is that they drift while both keep producing plausible segments. Nothing about a wrong
segment looks wrong.

So the guard is an equivalence measurement rather than a shared implementation. Same events, same
generation, both assemblers, compared sample by sample.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gwmock.cli.utils.config import Config

_FS = 1024.0
_SEGMENT = 16.0
_SEGMENTS = 3
_START = 1577491296.0
_DETECTOR = "ET-Triangle-Sardinia"

#: One event mid-segment, one just past a boundary so the claiming rule has to pull it earlier, and
#: one in the last segment. The middle event is the interesting one: gwmock claims it for the
#: *previous* segment because its waveform starts there, while ``assemble_segments`` simply scatters
#: it into both. If the two disagree anywhere, it is on an event like that.
_EVENTS: list[dict[str, Any]] = [
    {
        "detector_frame_mass_1": 30.0,
        "detector_frame_mass_2": 25.0,
        "distance": 400.0,
        "right_ascension": 1.0,
        "declination": 0.5,
        "polarization_angle": 0.2,
        "inclination": 0.3,
        "coa_time": _START + offset,
    }
    for offset in (8.0, _SEGMENT + 1.0, 2 * _SEGMENT + 8.0)
]


def _population_file(directory: Path) -> Path:
    """Write the events as a catalogue CSV, since the loader reads from disk."""
    path = directory / "population.csv"
    columns = list(_EVENTS[0])
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for event in _EVENTS:
            writer.writerow([event[column] for column in columns])
    return path


def _orchestrator(directory: Path):
    """Return an orchestrator on the batched path, which is the one that shares generation."""
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    raw = {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _FS,
                "duration": _SEGMENT,
                "total-duration": _SEGMENT * _SEGMENTS,
                "start-time": _START,
                "seed": 20260804,
            },
            "working-directory": str(directory),
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "orchestration": {
            "population": {
                "backend": "FilePopulationLoader",
                "source-type": "bbh",
                "n-samples": len(_EVENTS),
                "arguments": {"path": str(_population_file(directory))},
            },
            "signal": {
                "source-type": "bbh",
                "waveform-model": "IMRPhenomD",
                # The batched entry point is ripple-only, and both paths must generate with the
                # same library or this compares waveforms rather than assemblers.
                "waveform-backend": "ripple",
                "execution": "batched",
                "minimum-frequency": 20,
                "detectors": [_DETECTOR],
                "output": {
                    "output_directory": "signal",
                    "file_name": "signal.gwf",
                    "arguments": {"channel": "X:STRAIN"},
                },
            },
        },
    }
    config = Config.model_validate(raw)
    return AdapterOrchestrator.from_config(
        config.orchestration,
        global_simulator_arguments=dict(config.globals.simulator_arguments),
    )


def _streamed_segments(orchestrator) -> list[np.ndarray]:
    """Assemble with gwmock: one segment at a time, tails carried forward."""
    segments = []
    for _ in range(_SEGMENTS):
        segment = orchestrator.simulate().signal_segment
        segments.append(np.atleast_2d(np.asarray(segment, dtype=float)).copy())
        orchestrator.update_state()
    return segments


def _scattered_segments(orchestrator) -> list[np.ndarray]:
    """Assemble with gwmock-signal: one batch on the run grid, scattered into the tiling."""
    from gwmock_signal import SamplingGrid, simulate_cbc_batch
    from gwmock_signal.jax_batch import assemble_segments

    from gwmock.signal.adapter import SignalAdapter
    from gwmock.signal.device_chunks import canonicalise_parameters

    batch = simulate_cbc_batch(
        orchestrator.signal_adapter.device_approximant(),
        list(orchestrator._signal_network.detector_names),
        sampling_frequency=_FS,
        minimum_frequency=20.0,
        parameters=canonicalise_parameters(SignalAdapter.events_to_struct_of_arrays(_EVENTS)),
        backend=orchestrator._batched_waveform_backend(),
        earth_rotation=orchestrator.earth_rotation,
        output_grid=SamplingGrid(_START, _FS),
    )
    stacks = assemble_segments(
        batch,
        segment_duration=_SEGMENT,
        segment_start_times=[_START + index * _SEGMENT for index in range(_SEGMENTS)],
    )
    return [np.atleast_2d(np.asarray(stack.data, dtype=float)).copy() for stack in stacks]


@pytest.fixture(scope="module")
def assembled(tmp_path_factory) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Assemble the same events both ways once, since generation dominates the runtime."""
    pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
    orchestrator = _orchestrator(tmp_path_factory.mktemp("assembly"))
    return _streamed_segments(orchestrator), _scattered_segments(orchestrator)


def test_both_assemblers_produce_the_same_strain(assembled):
    """The whole point: same events in, same samples out, whichever assembler ran.

    Compared relative to the segment's own peak. ``atol=0.0`` because strain is ~1e-45 here and the
    default absolute tolerance would make any two such arrays compare equal, so the assertion could
    not fail.

    The tolerance admits float64 accumulation and nothing more. gwmock anchors each segment's
    sampling grid at that segment's start while the scatter anchors one grid at the run start, so
    later segments carry a different rounding history for the same arithmetic -- measured at 3e-13
    of peak in the last segment and exactly zero in the first two.
    """
    streamed, scattered = assembled

    assert len(streamed) == len(scattered) == _SEGMENTS
    for index, (mine, theirs) in enumerate(zip(streamed, scattered, strict=True)):
        assert mine.shape == theirs.shape, f"segment {index}: shape differs"
        peak = float(np.max(np.abs(theirs)))
        assert peak > 0.0, f"segment {index} is silent, so this comparison proves nothing"
        difference = float(np.max(np.abs(mine - theirs)))
        assert difference / peak < 1e-10, (
            f"segment {index}: assemblers differ by {difference:.3e}, {difference / peak:.2e} of "
            f"peak {peak:.3e} -- more than float64 accumulation, so one of them has drifted"
        )


def test_no_energy_is_gained_or_lost_between_the_two(assembled):
    """A weaker check than the sample comparison, and it fails differently.

    Two assemblers could match sample-for-sample inside every segment and still disagree about how
    much signal reached the data at all -- if one dropped a whole event, or double-counted one at a
    boundary, the per-segment arrays it did produce would still match. Summed energy across the run
    is what notices that.
    """
    streamed, scattered = assembled

    mine = sum(float(np.sum(segment**2)) for segment in streamed)
    theirs = sum(float(np.sum(segment**2)) for segment in scattered)

    assert mine == pytest.approx(theirs, rel=1e-9, abs=0.0), (
        f"total strain energy differs: streamed {mine:.6e} against scattered {theirs:.6e}; an event "
        f"is being dropped or counted twice by one of them"
    )


def test_gwmock_does_not_call_the_other_assembler(assembled):
    """Ownership, asserted rather than assumed.

    If a future change wires `assemble_segments` into gwmock, the equivalence tests above would keep
    passing -- they would be comparing that function against itself. This is what notices.
    """
    import gwmock.cli.adapter_orchestration as orchestration
    from gwmock.cli import simulate_utils

    for module in (orchestration, simulate_utils):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "assemble_segments" not in source, (
            f"{module.__name__} now references assemble_segments. gwmock owns assembly; if that is "
            f"changing deliberately, the equivalence tests here become circular and must be rewritten"
        )
