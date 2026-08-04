"""Which segment claims a compact binary, when its waveform starts before its coalescence.

A compact binary's buffer begins seconds before ``coa_time``. A segment chosen from ``coa_time``
alone therefore starts *after* the waveform does, and injection crops the difference away with
nowhere to put it: the earlier segments are already written. The loss is not marginal -- for a 30+25
solar-mass binary at 1024 Hz the lead is 3.6 s, so an event landing 1 s into a 16-second segment
loses 2.6 s of inspiral.

The rule these tests pin is that a segment claims the events whose *waveform* starts before it ends,
taken from :meth:`SignalAdapter.pre_coalescence_duration`. Two loops apply it -- per-event and
batched -- and both are checked, because ``population_index`` is checkpointed state and a run
switched between modes must not skip or repeat an event because the two disagreed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gwmock.cli.utils.config import Config

_POPULATION_CSV = Path(__file__).resolve().parents[2] / "examples" / "signal" / "bbh_population.csv"
_START = 1577491296.0
_SAMPLING_FREQUENCY = 1024.0
_SEGMENT_DURATION = 16.0

#: A complete BBH parameter set, so the query can actually answer. Underspecified events are the
#: subject of their own test below rather than an accident of this one.
_COMPLETE_EVENT: dict[str, Any] = {
    "detector_frame_mass_1": 30.0,
    "detector_frame_mass_2": 25.0,
    "distance": 400.0,
    "right_ascension": 1.0,
    "declination": 0.5,
    "polarization_angle": 0.2,
    "inclination": 0.3,
}


def _config(working_directory: Path, execution: str, total_duration: float = _SEGMENT_DURATION) -> dict[str, Any]:
    """Return a one-segment BBH config in the given execution mode.

    No ``waveform-backend`` key, so the default LAL library is used: naming ripple would make
    *constructing* the orchestrator instantiate ``RippleBackend``, which needs the ``[jax]`` extra
    long before any generation happens, and these tests are about placement rather than a library.
    """
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _SAMPLING_FREQUENCY,
                "duration": _SEGMENT_DURATION,
                "total-duration": total_duration,
                "start-time": _START,
                "seed": 20260804,
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


def _orchestrator(working_directory: Path, execution: str = "per-event", total_duration: float = _SEGMENT_DURATION):
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    working_directory.mkdir(parents=True, exist_ok=True)
    config = Config.model_validate(_config(working_directory, execution, total_duration))
    return AdapterOrchestrator.from_config(
        config.orchestration,
        global_simulator_arguments=dict(config.globals.simulator_arguments),
    )


class _StubAdapter:
    """A signal adapter that answers the placement query however the test needs.

    Stands in for the real adapter so the boundary rule can be tested without generating a waveform
    per case, and so *unknown* and *failing* answers can be produced at all -- neither is reachable
    from the bundled backends with complete parameters.
    """

    def __init__(self, lead: float | None = None, error: Exception | None = None) -> None:
        self._lead = lead
        self._error = error
        self.detector_names = ("E1",)

    def pre_coalescence_duration(self, _parameters: Any, **_: Any) -> float | None:
        if self._error is not None:
            raise self._error
        return self._lead


def _end_time(orchestrator) -> float:
    return float(getattr(orchestrator.end_time, "value", orchestrator.end_time))


class TestTheBoundaryRule:
    """The rule itself, with the lead supplied rather than generated."""

    def test_an_event_coalescing_after_the_segment_is_claimed_when_its_waveform_starts_inside(self, tmp_path):
        """This is the defect. Its ``coa_time`` is past the boundary; 3.6 s of its inspiral is not.

        Under the previous ``coa_time`` rule this event was left to the next segment, whose start is
        after the waveform's -- so the lead was cropped and could never be placed, because by then
        this segment had been written.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6)
        end_time = _end_time(orchestrator)

        assert orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 1.0}), (
            "an event whose waveform begins 2.6 s before the boundary belongs to this segment"
        )

    def test_an_event_whose_waveform_also_starts_later_is_left_for_the_next_segment(self, tmp_path):
        """The rule must still end the segment, or every remaining event would be pulled forward."""
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6)
        end_time = _end_time(orchestrator)

        assert not orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 10.0})

    def test_the_boundary_is_the_waveform_start_and_not_the_coalescence(self, tmp_path):
        """Pins which quantity is compared, by making the two answers differ.

        With a lead of 5 s an event coalescing 4 s past the boundary is claimed and one coalescing
        6 s past it is not, so the comparison cannot be against ``coa_time``.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=5.0)
        end_time = _end_time(orchestrator)

        assert orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 4.0})
        assert not orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 6.0})

    def test_an_event_without_a_coalescence_time_always_belongs(self, tmp_path):
        """Non-coalescing sources reach the loop unchanged; there is nothing to place them by."""
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=3.6)

        assert orchestrator._event_starts_before_segment_end({"frequency": 100.0})


class TestWhenTheLeadIsNotAvailable:
    """Unknown must mean unknown. Reading it as zero would drop the whole inspiral silently."""

    def test_an_unknown_lead_falls_back_to_the_coalescence_time(self, tmp_path):
        """``None`` restores the previous behaviour rather than claiming the waveform starts at tc.

        A backend that cannot say (PyCBC) and an installed gwmock-signal predating the query both
        arrive here. Treating ``None`` as ``0.0`` would look like an answer and behave like the bug.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(lead=None)
        end_time = _end_time(orchestrator)

        assert orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time - 0.5})
        assert not orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 0.5}), (
            "without a lead the only rule available is coa_time, which is where this started"
        )

    def test_a_query_that_raises_does_not_fail_the_run(self, tmp_path, caplog):
        """Placement must not be the thing that turns an odd catalogue into a crash.

        The same parameters go to generation immediately afterwards, which raises whatever this
        raised -- with the context of the event it was generating rather than of a boundary test.
        """
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(error=ValueError("Missing required parameter: 'distance'"))
        end_time = _end_time(orchestrator)

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            claimed = orchestrator._event_starts_before_segment_end({"coa_time": end_time - 0.5})

        assert claimed, "the fallback is the coa_time rule, not a refusal to place the event"
        assert "Missing required parameter" in caplog.text, (
            "the swallowed reason must appear, or a run silently reverts to the cropping behaviour"
        )

    def test_the_fallback_warns_once_rather_than_once_per_event(self, tmp_path, caplog):
        """A catalogue the query cannot answer for cannot answer for any of its events."""
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = _StubAdapter(error=ValueError("nope"))
        end_time = _end_time(orchestrator)

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            for offset in range(5):
                orchestrator._event_starts_before_segment_end({"coa_time": end_time - offset - 1.0})

        assert caplog.text.count("Cannot establish how long before coalescence") == 1

    def test_a_run_without_a_signal_adapter_still_walks_the_catalogue(self, tmp_path):
        """Noise-only orchestration reaches this helper through shared code paths."""
        orchestrator = _orchestrator(tmp_path)
        orchestrator.signal_adapter = None
        end_time = _end_time(orchestrator)

        assert orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time - 0.5})
        assert not orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 0.5})


class TestBothLoopsAgree:
    """``population_index`` is checkpointed, so the two execution modes must claim the same events."""

    def test_the_batched_walk_claims_the_same_events_as_the_rule(self, tmp_path):
        """``_events_for_this_segment`` must apply the boundary rule, not its own copy of one."""
        orchestrator = _orchestrator(tmp_path, "batched")
        orchestrator.signal_adapter = _StubAdapter(lead=3.6)
        end_time = _end_time(orchestrator)
        orchestrator._population_events = (
            {**_COMPLETE_EVENT, "coa_time": end_time - 1.0},
            {**_COMPLETE_EVENT, "coa_time": end_time + 1.0},  # coalesces later, starts inside
            {**_COMPLETE_EVENT, "coa_time": end_time + 10.0},
        )

        event_ids, events = orchestrator._events_for_this_segment()

        assert event_ids == [0, 1], "the event coalescing just past the boundary starts inside it"
        assert len(events) == 2
        assert int(orchestrator.population_index) == 0, "reading must not advance the checkpointed index"

    def test_the_per_event_loop_claims_the_same_events(self, tmp_path):
        """The same catalogue through the other loop, so a mode switch cannot skip or repeat."""
        orchestrator = _orchestrator(tmp_path, "batched")
        orchestrator.signal_adapter = _StubAdapter(lead=3.6)
        end_time = _end_time(orchestrator)
        events = (
            {**_COMPLETE_EVENT, "coa_time": end_time - 1.0},
            {**_COMPLETE_EVENT, "coa_time": end_time + 1.0},
            {**_COMPLETE_EVENT, "coa_time": end_time + 10.0},
        )
        orchestrator._population_events = events
        batched_ids, _ = orchestrator._events_for_this_segment()

        per_event = _orchestrator(tmp_path / "per", "per-event")
        per_event.signal_adapter = _StubAdapter(lead=3.6)
        per_event._population_events = events
        claimed = [
            index for index, parameters in enumerate(events) if per_event._event_starts_before_segment_end(parameters)
        ]

        assert claimed == batched_ids, (
            "the two modes must agree on the boundary; population_index is checkpointed across them"
        )


class TestNothingIsDiscarded:
    """The point of the rule, measured on assembled segments rather than argued from the boundary."""

    def _run_two_segments(self, orchestrator, coa_time: float):
        """Assemble two consecutive segments containing one event, and return them."""
        orchestrator._population_events = ({**_COMPLETE_EVENT, "coa_time": coa_time},)
        first = orchestrator.simulate().signal_segment
        orchestrator.update_state()
        second = orchestrator.simulate().signal_segment
        return first, second

    def test_a_waveform_straddling_the_boundary_keeps_all_of_its_energy(self, tmp_path, caplog):
        """No sample is dropped, and the check is the strain itself rather than the absence of a log.

        The event coalesces 1 s into the *second* segment, so its buffer starts 2.6 s before that
        segment begins. Claimed by ``coa_time`` it would be generated during the second segment, and
        injection would crop those 2.6 s -- 65% of its samples -- because the first segment is
        already written by then.

        This offset is deliberately the *mild* case. Measured for this binary at 1024 Hz, the
        unweighted strain-squared energy lost runs 0.34% at 1 s past the boundary, 11% at 0.25 s,
        50% at 0.1 s and 99.8% at 1 ms, because a compact binary's energy is concentrated in the
        last fraction of a second before merger. Testing the mild case makes the assertion depend on
        placement rather than on that concentration: 0.34% is a loss no energy-fraction tolerance
        would forgive, and a test pinned at the catastrophic end would still pass if placement were
        only approximately right.

        Summed squares across the two assembled segments are compared against the same waveform
        generated on its own, which is the quantity that must survive placement.
        """
        orchestrator = _orchestrator(tmp_path, "per-event", total_duration=2 * _SEGMENT_DURATION)
        adapter = orchestrator.signal_adapter
        assert adapter is not None
        coa_time = _START + _SEGMENT_DURATION + 1.0
        standalone = adapter.simulate(
            {**_COMPLETE_EVENT, "coa_time": coa_time},
            sampling_frequency=_SAMPLING_FREQUENCY,
            minimum_frequency=30.0,
        )
        expected_energy = float(np.sum(np.square(np.atleast_2d(np.asarray(standalone, dtype=float)))))

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            first, second = self._run_two_segments(orchestrator, coa_time)

        placed_energy = sum(
            float(np.sum(np.square(np.atleast_2d(np.asarray(segment, dtype=float))))) for segment in (first, second)
        )

        assert "Discarding" not in caplog.text, caplog.text
        # atol=0.0 because strain-squared energies are ~1e-45: the default atol would make any two
        # such numbers compare equal and the assertion could not fail.
        assert placed_energy == pytest.approx(expected_energy, rel=1e-9, abs=0.0), (
            f"placed {placed_energy} of {expected_energy}; the boundary lost "
            f"{100.0 * (1.0 - placed_energy / expected_energy):.2f}% of the waveform"
        )

    def test_the_event_is_attributed_to_the_segment_its_waveform_starts_in(self, tmp_path):
        """Provenance follows placement: the frame listing the event is the one its data begins in.

        Stated rather than assumed, because it is a change -- an event used to be listed against the
        frame containing its coalescence, and a reader of the metadata needs to know which.
        """
        orchestrator = _orchestrator(tmp_path, "per-event", total_duration=2 * _SEGMENT_DURATION)
        coa_time = _START + _SEGMENT_DURATION + 1.0

        orchestrator._population_events = ({**_COMPLETE_EVENT, "coa_time": coa_time},)
        orchestrator.simulate()
        first_segment_injections = list(orchestrator._batch_injections)
        orchestrator.update_state()
        orchestrator.simulate()
        second_segment_injections = list(orchestrator._batch_injections)

        assert [entry["event_id"] for entry in first_segment_injections] == [0], (
            "the event belongs to the first segment, where its waveform starts"
        )
        assert second_segment_injections == [], "and it is not cross-listed against the frame its coalescence falls in"


class TestAgainstRealGeneration:
    """The predicted lead has to be the lead generation actually produces, or placement is wrong."""

    def test_the_predicted_lead_matches_the_generated_buffer(self, tmp_path):
        """Measured against the produced ``TimeSeries``, not against another prediction.

        This is what makes the rule safe: if the query said less than generation produces, a segment
        chosen from it would still crop the start.
        """
        orchestrator = _orchestrator(tmp_path)
        adapter = orchestrator.signal_adapter
        assert adapter is not None
        coa_time = _START + 8.0
        parameters = {**_COMPLETE_EVENT, "coa_time": coa_time}

        predicted = adapter.pre_coalescence_duration(
            parameters,
            sampling_frequency=_SAMPLING_FREQUENCY,
            minimum_frequency=30.0,
        )
        strain = adapter.simulate(parameters, sampling_frequency=_SAMPLING_FREQUENCY, minimum_frequency=30.0)
        measured = coa_time - float(strain.start_time.value)

        assert predicted is not None, "LAL can answer this; a None here would disable the fix silently"
        assert predicted == pytest.approx(measured, abs=0.5 / _SAMPLING_FREQUENCY), (
            f"predicted lead {predicted} s but generation starts {measured} s before coalescence"
        )

    def test_an_event_just_past_the_boundary_is_claimed_using_the_real_query(self, tmp_path):
        """End to end through the real adapter: no stub, no supplied lead."""
        orchestrator = _orchestrator(tmp_path)
        end_time = _end_time(orchestrator)

        assert orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 1.0}), (
            "a 30+25 binary at 1024 Hz leads coalescence by 3.6 s, so this event starts in this segment"
        )
        assert not orchestrator._event_starts_before_segment_end({**_COMPLETE_EVENT, "coa_time": end_time + 5.0})
