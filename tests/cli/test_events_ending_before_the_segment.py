"""Events whose waveform has finished before the segment starts, and the batch they inflate.

The companion to :mod:`tests.cli.test_inspiral_segment_placement`, from the other side. That
module pins the *upper* bound: a segment claims an event whose waveform starts before the segment
ends, so an inspiral is not cropped. The rule has no lower bound, and the omission is invisible in
a sequential run -- earlier segments have already consumed the earlier events, so the walk starts
past them.

It is not invisible in a run whose ``start-time`` is later than its population's first event.
Nothing has consumed those events, every one of them satisfies "starts before this segment ends",
and the first segment claims the entire back catalogue. **Measured on the shipped ET config:** a
run starting 53 ks into the catalogue batched 126 events where the segment contains 13, estimated
12.5 GiB, and was refused by the preflight; at 41 ks it batched 98 for 16; at 4 ks, 19 for 8. The
batch tracks the offset rather than the local event density. Their samples are then cropped as
out-of-segment, so the work is generated and discarded.

Resuming a campaign mid-catalogue, or slicing a long population into per-day jobs, is exactly this.

**Why the fix cannot be another clause in the same predicate.** Both loops ``break`` on the first
event that does not belong, and ``population_index`` -- checkpointed state meaning "every event
before this one is consumed" -- advances by the count of events consumed. An early event answering
"does not belong" would therefore stop the walk at position 0 and consume nothing, and it would do
so again on every following segment: the run would stall permanently rather than over-batch. An
event that has finished must be *skipped and consumed*, which is a different operation from the
one the upper bound performs, and ``test_a_finished_event_does_not_stop_the_walk`` is what pins
the difference.

``None`` from the query means unknown, and unknown must claim. The query exists to let a caller
*prove* an event is finished; on no answer the conservative branch is the one that generates the
event and lets injection crop it, because the alternative silently deletes signal.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from gwmock.cli.utils.config import Config

_POPULATION_CSV = Path(__file__).resolve().parents[2] / "examples" / "signal" / "bbh_population.csv"
_START = 1577491296.0
_SAMPLING_FREQUENCY = 1024.0
_SEGMENT_DURATION = 16.0

#: A complete BBH parameter set, so a real query could answer; these tests supply the answer
#: through the stub, but an underspecified event would exercise the unknown path by accident.
_COMPLETE_EVENT: dict[str, Any] = {
    "detector_frame_mass_1": 30.0,
    "detector_frame_mass_2": 25.0,
    "distance": 400.0,
    "right_ascension": 1.0,
    "declination": 0.5,
    "polarization_angle": 0.2,
    "inclination": 0.3,
}


def _config(working_directory: Path, execution: str) -> dict[str, Any]:
    """Return a one-segment BBH config, matching the placement module's fixture.

    No ``waveform-backend`` key, so constructing the orchestrator does not instantiate a backend
    needing the ``[jax]`` extra.
    """
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _SAMPLING_FREQUENCY,
                "duration": _SEGMENT_DURATION,
                "total-duration": _SEGMENT_DURATION,
                "start-time": _START,
                "seed": 20260807,
            },
            "working-directory": str(working_directory),
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "orchestration": {
            "population": {
                "backend": "FilePopulationLoader",
                "source-type": "bbh",
                "n-samples": 1,
                "arguments": {"path": str(_POPULATION_CSV)},
            },
            "signal": {
                "source-type": "bbh",
                "waveform-model": "IMRPhenomD",
                "execution": execution,
                "minimum-frequency": 30,
                "detectors": ["ET-Triangle-Sardinia"],
                "output": {
                    "output_directory": "signal",
                    "file_name": "sig-{{ detectors }}.gwf",
                    "arguments": {"channel": "{{ detectors }}:STRAIN"},
                },
            },
        },
    }


def _orchestrator(working_directory: Path, execution: str = "batched"):
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    working_directory.mkdir(parents=True, exist_ok=True)
    config = Config.model_validate(_config(working_directory, execution))
    return AdapterOrchestrator.from_config(
        config.orchestration,
        global_simulator_arguments=dict(config.globals.simulator_arguments),
    )


class _StubAdapter:
    """A signal adapter answering both placement queries however the test needs.

    Both sides are supplied, because a test that fixed only the tail would let the lead default to
    something the sort key also depends on, and the two together decide which events the walk even
    reaches.
    """

    def __init__(self, lead: float | None = 0.0, tail: float | None = 0.0) -> None:
        self._lead = lead
        self._tail = tail
        self.detector_names = ("E1",)
        #: Every event handed to :meth:`simulate`, so a test can assert what was *generated* rather
        #: than what a predicate said. The two are not the same thing, which is the whole reason
        #: this list exists -- see ``test_the_per_event_loop_steps_over_finished_events``.
        self.simulated: list[float] = []

    def simulate(self, parameters: Mapping[str, Any], **_: Any):
        """Return a one-sample segment-length strain, recording that the event was generated."""
        import numpy as np

        from gwmock.data.time_series.time_series import TimeSeries

        self.simulated.append(float(parameters["coa_time"]))
        return TimeSeries(
            np.zeros((1, int(_SAMPLING_FREQUENCY * _SEGMENT_DURATION))),
            start_time=float(parameters["coa_time"]),
            sampling_frequency=_SAMPLING_FREQUENCY,
        )

    def pre_coalescence_duration(self, parameters: Mapping[str, Any], **_: Any) -> float | None:
        del parameters
        return self._lead

    def post_coalescence_duration(self, parameters: Mapping[str, Any], **_: Any) -> float | None:
        del parameters
        return self._tail


def _install_events(orchestrator, coa_times: list[float]) -> None:
    """Give the orchestrator a synthetic catalogue and drop the cached placement order.

    The order is cached on first use and keyed by the events present when it was built, so a test
    replacing the catalogue afterwards would otherwise walk positions that no longer exist.
    """
    orchestrator._population_events = [{**_COMPLETE_EVENT, "coa_time": t} for t in coa_times]
    orchestrator._placement_order_cache = None
    orchestrator.population_index = 0


def _start_time(orchestrator) -> float:
    return float(getattr(orchestrator.start_time, "value", orchestrator.start_time))


class TestTheBatchDoesNotGrowWithTheOffset:
    """The defect: a run starting after its population's first event batches all of them."""

    def test_the_batch_holds_the_events_in_the_window_not_the_ones_before_it(self, tmp_path):
        """The measured failure, reduced to its shape.

        Ten events sit wholly before the segment -- each finishing, with its 0.4 s tail, well
        before the segment starts -- and three fall inside it. A run whose ``start-time`` skipped
        the first ten claims all thirteen, generates ten waveforms whose samples are then cropped
        as out-of-segment, and sizes its batch from the offset rather than from the segment.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=0.4)
        start = _start_time(orchestrator)
        before = [start - 3600.0 + 60.0 * i for i in range(10)]
        inside = [start + 1.0, start + 5.0, start + 9.0]
        _install_events(orchestrator, before + inside)

        event_ids, events = orchestrator._events_for_this_segment()

        assert len(events) == 3, f"batched {len(events)} events for a segment containing 3"
        assert sorted(event_ids) == [10, 11, 12]

    def test_the_batch_tracks_the_segment_not_the_distance_from_the_catalogue_start(self, tmp_path):
        """Doubling the offset must not change the batch.

        This is the property the measurement actually showed violated -- 126, then 98, then 19 for
        the same kind of segment -- and it is the one a caller relies on when slicing a population
        into per-day jobs. A test fixing a single offset would pass against a fix that merely
        shifted the over-claiming.
        """
        counts = []
        for offset in (600.0, 1200.0, 2400.0):
            orchestrator = _orchestrator(tmp_path / f"offset_{int(offset)}")
            orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=0.4)
            start = _start_time(orchestrator)
            n_before = int(offset / 60.0)
            before = [start - offset + 60.0 * i for i in range(n_before)]
            _install_events(orchestrator, [*before, start + 1.0, start + 5.0])
            counts.append(len(orchestrator._events_for_this_segment()[1]))

        assert counts == [2, 2, 2], f"batch grew with the offset: {counts}"

    def test_a_finished_event_does_not_stop_the_walk(self, tmp_path):
        """Skipped, not refused -- the distinction the loops' ``break`` makes load-bearing.

        If a finished event answered the boundary question with "does not belong", the walk would
        stop at position 0 and consume nothing, on this segment and on every one after it. The run
        would stall rather than over-batch, which is a worse failure than the one being fixed. So
        the events *behind* the finished one must still be claimed.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=0.4)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 1000.0, start + 2.0, start + 6.0])

        event_ids, events = orchestrator._events_for_this_segment()

        assert len(events) == 2, "the walk stopped at the finished event instead of stepping over it"
        assert sorted(event_ids) == [1, 2]

    def test_a_skipped_event_is_consumed_so_the_next_segment_does_not_reconsider_it(self, tmp_path):
        """``population_index`` means every event before it is consumed; skipping must keep that.

        Advancing only by the events *generated* would leave the finished ones in front of the
        index forever: every later segment would re-examine and re-skip them, and a resumed run
        would restart inside a prefix it had already dealt with. The count is what resume depends
        on, so it is asserted rather than the ids.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=0.4)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 900.0, start - 800.0, start + 2.0])

        event_ids, _ = orchestrator._events_for_this_segment()
        orchestrator._commit_consumed_events(event_ids)

        assert orchestrator.population_index == 3, (
            f"index advanced to {orchestrator.population_index}; the two skipped events are still "
            "in front of it and will be reconsidered by the next segment"
        )


class TestUnknownStillClaims:
    """Unknown is not zero, and on this side reading it as zero deletes signal."""

    def test_an_event_whose_tail_is_unknown_is_claimed(self, tmp_path):
        """A backend that cannot say where its buffer ends must not have events dropped for it.

        PyCBC answers ``None`` by design, as does any custom-registered model. Reading that as a
        zero-length tail would conclude every such event finished at ``coa_time`` and discard the
        ones sitting before the segment -- silently, and for a whole backend at a time.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=None)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 1000.0, start + 2.0])

        _, events = orchestrator._events_for_this_segment()

        assert len(events) == 2, "an event with an unknown tail was dropped; unknown is not zero"

    def test_an_event_still_ringing_into_the_segment_is_claimed(self, tmp_path):
        """The tail is a fixed fraction of the buffer, so for a BNS it is tens of seconds.

        Measured on the shipped backends: 0.4 s for a 4 s BBH buffer, **25.6 s** for a 256 s BNS
        buffer. An event coalescing 10 s before the segment is therefore still producing signal
        inside it, and a lower bound written against a millisecond intuition would delete it.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=100.0, tail=25.6)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 10.0, start + 2.0])

        _, events = orchestrator._events_for_this_segment()

        assert len(events) == 2, "an event whose ringdown reaches into the segment was dropped"

    def test_the_boundary_is_inclusive_the_way_the_upper_one_is(self, tmp_path):
        """An event ending exactly at the segment start contributes its final sample.

        Pinned because the off-by-one here is silent: it drops one event at one boundary, in a run
        that otherwise looks correct.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=2.0)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 2.0, start + 2.0])

        _, events = orchestrator._events_for_this_segment()

        assert len(events) == 2, "the event ending exactly at the segment start was dropped"


class TestBothLoopsAgree:
    """Per-event and batched must place identically, or resume skips or repeats events."""

    @pytest.mark.parametrize("execution", ["per-event", "batched"])
    def test_the_two_modes_claim_the_same_events(self, tmp_path, execution):
        """``population_index`` is checkpointed, so a run switched between modes must not disagree.

        Asserted through the shared predicate rather than by generating, because the per-event loop
        generates as it walks and this is a placement question.
        """
        orchestrator = _orchestrator(tmp_path / execution, execution=execution)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6, tail=0.4)
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 1000.0, start - 500.0, start + 2.0, start + 6.0])

        finished = [
            orchestrator._event_ended_before_segment_start(dict(orchestrator._population_events[i])) for i in range(4)
        ]

        assert finished == [True, True, False, False]

    def test_the_per_event_loop_steps_over_finished_events(self, tmp_path):
        """Asserted on what the loop *generated*, not on what the predicate returned.

        This test exists because the first version of this module did not have it, and a mutation
        deleting the per-event loop's lower bound outright survived the whole suite: every other
        test here reaches the rule through ``_events_for_this_segment`` or by calling the predicate
        directly, so the per-event walk -- the default execution mode -- was unpinned while looking
        covered. Both loops share the predicate but apply it separately, and ``population_index``
        is checkpointed state, so a mode that keeps generating finished events also breaks resume
        for a run switched between modes.
        """
        orchestrator = _orchestrator(tmp_path, execution="per-event")
        adapter = _StubAdapter(lead=3.6, tail=0.4)
        orchestrator.signal_adapter = adapter
        start = _start_time(orchestrator)
        _install_events(orchestrator, [start - 1000.0, start - 500.0, start + 2.0])

        orchestrator._simulate()

        assert adapter.simulated == [start + 2.0], (
            f"generated {len(adapter.simulated)} event(s); the finished ones were not stepped over"
        )
        assert orchestrator.population_index == 3
