"""Selecting batched execution from a configuration file.

``execution: batched`` generates a segment's events together through gwmock-signal's batched entry
point instead of looping over them. That is what makes GPU execution reachable from a config, though
whether it *runs* on a GPU depends on the JAX device present, not on this key.

The chunks the batched path produces stay per-event, so gwmock's own assembler still handles
spill-over into later segments and provenance stays per-injection. The tests below check that the two
modes consume the catalogue identically and produce the same physics, because a difference in either
would make the choice of mode change the dataset rather than only how it was computed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from gwmock.cli.utils.config import Config, SignalConfig
from gwmock.signal.device_chunks import canonicalise_parameters

_POPULATION_CSV = Path(__file__).resolve().parents[2] / "examples" / "signal" / "bbh_population.csv"
_START = 1577491296.0
_SAMPLING_FREQUENCY = 1024.0


def _config(working_directory: Path, execution: str, waveform_backend: str | None = "ripple") -> dict[str, Any]:
    """Return a one-segment BBH config in the given execution mode.

    ``waveform_backend`` is omitted entirely when ``None``. That matters for the tests that must run
    without the ``[jax]`` extra: naming ripple makes *constructing* the orchestrator instantiate
    ``RippleBackend``, which needs ripplegw long before any generation happens.
    """
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _SAMPLING_FREQUENCY,
                "duration": 16,
                "total-duration": 16,
                "start-time": _START,
                "seed": 20260731,
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
                # Naming ripple for both modes makes the comparison isolate batching rather than
                # also changing the waveform library.
                **({"waveform-backend": waveform_backend} if waveform_backend is not None else {}),
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


def _orchestrator(
    working_directory: Path,
    execution: str,
    waveform_backend: str | None = "ripple",
    **signal_overrides: Any,
):
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    working_directory.mkdir(parents=True, exist_ok=True)
    raw = _config(working_directory, execution, waveform_backend=waveform_backend)
    raw["orchestration"]["signal"].update(signal_overrides)
    config = Config.model_validate(raw)
    return AdapterOrchestrator.from_config(
        config.orchestration,
        global_simulator_arguments=dict(config.globals.simulator_arguments),
    )


class TestConfigKey:
    """The key itself, before anything runs."""

    def test_the_default_is_per_event(self):
        """Existing configs must be unaffected."""
        assert SignalConfig(detectors=["H1"]).execution == "per-event"

    def test_batched_is_accepted(self):
        assert SignalConfig.model_validate({"detectors": ["H1"], "execution": "batched"}).execution == "batched"

    def test_an_unknown_mode_is_rejected(self):
        """A typo must not leave the run silently on the default path.

        The output of a per-event run looks entirely normal, so an author who mistyped the mode
        would have no way to notice they had not switched it.
        """
        with pytest.raises(ValueError, match="'execution' must be one of"):
            SignalConfig.model_validate({"detectors": ["H1"], "execution": "batchd"})


class TestCanonicalParameters:
    """Aliases, so switching mode does not change whether a config runs at all."""

    def test_a_known_alias_is_renamed(self):
        """The bundled BBH catalogue uses ``distance``; the batched path needs the canonical name.

        The per-event backends resolve this internally, so without the rename the same config runs
        one way and fails the other -- which is how this was found.
        """
        assert canonicalise_parameters({"distance": [400.0]}) == {"luminosity_distance": [400.0]}

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [("mass1", "detector_frame_mass_1"), ("spin1z", "spin_1z"), ("tidal_2", "lambda_2")],
    )
    def test_the_alias_table_matches_gwmock_signals(self, alias: str, canonical: str):
        """Taken from gwmock-signal's LAL backend rather than invented."""
        assert canonicalise_parameters({alias: [1.0]}) == {canonical: [1.0]}

    def test_an_unknown_key_is_left_alone(self):
        assert canonicalise_parameters({"coa_time": [1.0]}) == {"coa_time": [1.0]}

    def test_a_conflicting_alias_and_canonical_pair_is_refused(self):
        """Choosing one silently would pick which physics to simulate."""
        with pytest.raises(ValueError, match="both 'luminosity_distance' and its alias"):
            canonicalise_parameters({"distance": [400.0], "luminosity_distance": [500.0]})

    def test_an_agreeing_pair_is_accepted(self):
        """Redundant but consistent input is not an error."""
        assert canonicalise_parameters({"distance": [400.0], "luminosity_distance": [400.0]}) == {
            "luminosity_distance": [400.0]
        }


class TestWaveformConfigurationIsHonoured:
    """Batched mode must not quietly ignore the waveform settings the config asks for.

    The batched entry point takes a backend *instance*, so omitting it discards
    ``waveform-backend-arguments`` and silently uses ripple's defaults -- the same silent drop this
    codebase already fixed once for the per-event path. Output from the wrong settings looks
    entirely normal, which is what makes it worth testing rather than assuming.
    """

    def test_backend_arguments_reach_the_device(self, tmp_path):
        """``taper_fraction`` must change the generated waveform.

        It lowers the effective start frequency, which lengthens the inspiral, so the buffer's
        length changes -- an unmistakable signal that the argument was applied rather than dropped.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        default = _orchestrator(tmp_path / "default", "batched")._simulate()
        tapered = _orchestrator(
            tmp_path / "tapered", "batched", **{"waveform-backend-arguments": {"taper_fraction": 0.25}}
        )._simulate()

        assert np.asarray(default[0]).shape != np.asarray(tapered[0]).shape, (
            "taper_fraction did not change the waveform, so waveform-backend-arguments are being "
            "discarded and the run silently uses ripple's defaults"
        )

    def test_a_non_ripple_library_is_refused(self, tmp_path):
        """The batched path is ripple-only; substituting it would ignore the configuration.

        Deliberately not gated on ``ripplegw``: the guard runs before anything touches the device,
        so gating it would hide this branch from the CI job that has no extras -- which is the job
        measuring coverage.
        """
        with pytest.raises(ValueError, match="always generates with ripple"):
            _orchestrator(tmp_path, "batched", waveform_backend="lal")._simulate()

    def test_waveform_options_are_refused(self, tmp_path):
        """There is no equivalent parameter on the batched entry point, so silence would lose them."""
        with pytest.raises(ValueError, match="cannot apply waveform-options"):
            _orchestrator(
                tmp_path, "batched", waveform_backend=None, **{"waveform-options": {"ModeArray": [[2, 2]]}}
            )._simulate()


class TestSegmentEventSelection:
    """Which events a segment claims, tested without generating anything.

    ``_events_for_this_segment`` is pure catalogue walking, so it needs no device -- and testing it
    here rather than behind an ``importorskip`` is what keeps it covered in the CI job that installs
    no extras.
    """

    def test_events_beyond_the_segment_are_left_for_the_next_one(self, tmp_path):
        """The boundary rule is ``coa_time >= end_time``, and it is checkpointed state."""
        orchestrator = _orchestrator(tmp_path, "batched", waveform_backend=None)
        end_time = float(getattr(orchestrator.end_time, "value", orchestrator.end_time))
        orchestrator._population_events = (
            {"coa_time": end_time - 1.0, "detector_frame_mass_1": 30.0},
            {"coa_time": end_time, "detector_frame_mass_1": 30.0},
            {"coa_time": end_time + 5.0, "detector_frame_mass_1": 30.0},
        )

        event_ids, events = orchestrator._events_for_this_segment()

        assert event_ids == [0], "only the event before the segment end belongs to this segment"
        assert len(events) == 1
        assert int(orchestrator.population_index) == 0, (
            "reading the segment's events must not advance the checkpointed index: generation can "
            "still fail afterwards, and a caller would then see events consumed that were never "
            "produced"
        )

        orchestrator._commit_consumed_events(event_ids)

        assert int(orchestrator.population_index) == 1, (
            "committing must advance past exactly the events generated, so the next segment resumes "
            "at the boundary rather than skipping or repeating"
        )

    def test_an_empty_segment_consumes_nothing(self, tmp_path):
        """A segment with no events must not advance the index or invent provenance."""
        orchestrator = _orchestrator(tmp_path, "batched", waveform_backend=None)
        end_time = float(getattr(orchestrator.end_time, "value", orchestrator.end_time))
        orchestrator._population_events = ({"coa_time": end_time + 100.0},)

        chunks = orchestrator._simulate()

        assert len(chunks) == 0
        assert int(orchestrator.population_index) == 0
        assert orchestrator._batch_injections == []

    def test_the_execution_mode_comes_from_the_config(self, tmp_path):
        assert _orchestrator(tmp_path / "a", "batched", waveform_backend=None)._execution_mode() == "batched"
        assert _orchestrator(tmp_path / "b", "per-event", waveform_backend=None)._execution_mode() == "per-event"


class TestCatalogueConsumption:
    """Both modes must walk the catalogue the same way, or a resumed run skips or repeats events."""

    def test_both_modes_consume_the_same_events(self, tmp_path):
        """``population_index`` is checkpointed state, so the two paths must agree on it.

        A run switched between modes partway through would otherwise lose or duplicate events at the
        boundary, and nothing downstream would notice.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        per_event = _orchestrator(tmp_path / "per", "per-event")
        batched = _orchestrator(tmp_path / "bat", "batched")

        per_event._simulate()
        batched._simulate()

        assert int(batched.population_index) == int(per_event.population_index)

    def test_the_batched_path_records_provenance_per_event(self, tmp_path):
        """Segment-shaped output must not thin provenance down to one record per segment."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        batched = _orchestrator(tmp_path, "batched")

        chunks = batched._simulate()

        assert len(batched._batch_injections) == len(chunks)
        assert all("event_id" in record and "parameters" in record for record in batched._batch_injections)
        assert all("injection_parameters" in chunk.metadata for chunk in chunks)

    def test_both_modes_record_the_same_provenance(self, tmp_path):
        """An injection must be described identically whichever mode produced it.

        The batched path canonicalises names and merges fixed waveform arguments before handing
        parameters to the device. Recording *that* mapping would make the same event appear with
        ``luminosity_distance`` in one mode and ``distance`` in the other, so provenance is taken
        from the catalogue instead.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        per_event = _orchestrator(tmp_path / "per", "per-event")
        batched = _orchestrator(tmp_path / "bat", "batched")

        per_event._simulate()
        batched._simulate()

        assert batched._batch_injections == per_event._batch_injections


@pytest.mark.e2e
class TestBothModesAgree:
    """The batched path must produce the same physics, not merely run.

    Marked ``e2e`` because it drives the CLI twice and ripple compiles on first use. This is the
    assertion the whole device path rests on: if batching changed the data, the mode would be a
    choice about the dataset rather than about how it was computed.
    """

    @staticmethod
    def _generate(working_directory: Path, execution: str) -> np.ndarray:
        from gwmock.cli.simulate import _simulate_impl

        working_directory.mkdir(parents=True, exist_ok=True)
        config_path = working_directory / "config.yaml"
        config_path.write_text(yaml.safe_dump(_config(working_directory, execution)), encoding="utf-8")
        _simulate_impl(str(config_path))

        from gwpy.timeseries import TimeSeries

        written = working_directory / "output" / "signal" / "sig-ET1_SARD.gwf"
        assert written.is_file(), f"no output from the {execution} run"
        return np.asarray(TimeSeries.read(written, channel="ET1_SARD:STRAIN").value)

    def test_the_two_modes_agree_to_floating_point(self, tmp_path):
        """Same events, same library, different execution: the strain must match.

        Not bit-identical, and it should not be expected to be -- a batched ``vmap`` accumulates in a
        different order than a loop. Measured at 1.03e-12 of the peak on this configuration, which is
        floating-point reassociation rather than a difference in physics.

        Compared against the **peak**, not element-wise. ``np.allclose`` with ``atol=0`` applies a
        relative tolerance per sample, which fails wherever the strain is near zero -- an absolute
        difference of 1e-34 against a sample of 1e-30 is a relative difference of 1e-4, and most of a
        segment is near-zero. Scaling to the peak asks the question that matters: is the waveform the
        same, to a fraction of its own amplitude. The default ``atol`` is not usable either, since at
        ~1e-22 it would call any two strain arrays equal.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        per_event = self._generate(tmp_path / "per", "per-event")
        batched = self._generate(tmp_path / "bat", "batched")

        assert per_event.shape == batched.shape
        assert np.count_nonzero(per_event) == np.count_nonzero(batched), (
            "the two modes placed the signal over a different number of samples"
        )
        assert int(np.argmax(np.abs(per_event))) == int(np.argmax(np.abs(batched))), (
            "the peak moved between modes, so the signal is at a different time"
        )
        peak = float(np.max(np.abs(per_event)))
        assert peak > 0.0, "the per-event run produced no signal, so there is nothing to compare"
        largest_difference = float(np.max(np.abs(per_event - batched)))

        assert largest_difference <= 1e-9 * peak, (
            f"the batched path produced materially different strain: largest difference "
            f"{largest_difference:.3e} is {largest_difference / peak:.3e} of the peak. That makes "
            f"the execution mode a choice about the dataset rather than about how it was computed."
        )


class TestReferenceFrequencyReachesTheBackend:
    """`f_ref` is written under waveform-arguments but is a backend option.

    gwmock-signal's own reserved-argument table says so: "f_ref is configured on the backend, not
    through waveform_arguments". Left in the parameter mapping it reaches `simulate_cbc_batch` and is
    read by nobody, so the same config key would set the reference frequency on the per-event path
    and do nothing on the batched one -- two execution modes silently disagreeing about the waveform.
    """

    def test_it_is_forwarded_to_the_ripple_backend(self, tmp_path):
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        orchestrator = _orchestrator(tmp_path, "batched", **{"waveform-arguments": {"f_ref": 25.0}})

        backend = orchestrator._batched_waveform_backend()

        assert backend is not None, "f_ref alone must still produce a configured backend"
        # `_f_ref` because the backend keeps it private; there is no public reader, and the
        # alternative -- trusting that construction succeeded -- would pass without forwarding.
        assert backend._f_ref == 25.0

    def test_it_does_not_also_go_into_the_parameter_mapping(self, tmp_path):
        """Passing it both ways would hand the batch entry point a key it does not read."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        captured = {}
        orchestrator = _orchestrator(tmp_path, "batched", **{"waveform-arguments": {"f_ref": 25.0}})

        import gwmock_signal

        real = gwmock_signal.simulate_cbc_batch

        def spy(*args, **kwargs):
            captured.update(kwargs.get("parameters", {}))
            return real(*args, **kwargs)

        gwmock_signal.simulate_cbc_batch = spy
        try:
            orchestrator._simulate()
        finally:
            gwmock_signal.simulate_cbc_batch = real

        assert captured, "the spy never saw a call, so this test proves nothing"
        assert "f_ref" not in captured

    def test_disagreeing_with_waveform_backend_arguments_is_refused(self, tmp_path):
        """Two keys setting one physical quantity: pick silently and the waveform depends on which
        one a reader happened to look at.

        Deliberately not gated on ripplegw -- the conflict is caught before the backend is built, so
        gating it would hide the branch from the CI job that installs no extras.
        """
        orchestrator = _orchestrator(
            tmp_path,
            "batched",
            waveform_backend=None,
            **{"waveform-arguments": {"f_ref": 25.0}, "waveform-backend-arguments": {"f_ref": 30.0}},
        )

        with pytest.raises(ValueError, match="f_ref is set to"):
            orchestrator._batched_waveform_backend()

    def test_agreeing_with_waveform_backend_arguments_is_accepted(self, tmp_path):
        """Refusing a redundant but consistent pair would be gratuitous."""
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        orchestrator = _orchestrator(
            tmp_path,
            "batched",
            **{"waveform-arguments": {"f_ref": 25.0}, "waveform-backend-arguments": {"f_ref": 25.0}},
        )

        assert orchestrator._batched_waveform_backend()._f_ref == 25.0
