"""Wiring continuous waves through the orchestrator.

A continuous wave fits neither route the orchestrator already had. The stationary branch runs only
when there are *no* population events, because a stochastic background has no sources; the
per-event loop consumes events and expects each to carry a ``coa_time``. A continuous wave is
stationary *and* has sources -- a catalogue of pulsars, every one of which contributes to every
segment -- so it gets its own branch.

The load-bearing test here is :meth:`TestPhaseCoherence.test_segments_join_up_through_the_orchestrator`.
The simulator guarantees coherence only if ``reference_time_ssb`` reaches it as one value for the
whole run; plumbing that derived it per segment would produce frames that look entirely normal and
are useless to a coherent search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")

from gwmock.cli.adapter_orchestration import AdapterOrchestrator
from gwmock.cli.utils.config import Config

_EPOCH = 1577491218.0
_FS = 64.0
_DETECTORS = ["H1", "L1"]
_EPHEMERIS = Path.home() / ".cache/ripplegw/ephemeris"

_PULSARS = (
    "right_ascension,declination,frequency,initial_phase,amplitude_plus,amplitude_cross,polarization_angle\n"
    "1.1,0.3,20.0,0.4,1.0e-24,7.0e-25,0.2\n"
    "2.7,-0.6,15.0,1.9,6.0e-25,4.0e-25,1.1\n"
)


def _ephemeris_available() -> bool:
    return (_EPHEMERIS / "earth00-40-DE405.dat.gz").is_file() and (_EPHEMERIS / "sun00-40-DE405.dat.gz").is_file()


pytestmark = pytest.mark.skipif(not _ephemeris_available(), reason="LALPulsar ephemeris tables are not cached locally")


def _config(tmp_path: Path, *, duration: float, total: float, n_pulsars: int = 2) -> dict[str, Any]:
    catalogue = tmp_path / "pulsars.csv"
    rows = _PULSARS.splitlines()
    catalogue.write_text("\n".join([rows[0], *rows[1 : 1 + n_pulsars]]) + "\n")
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _FS,
                "duration": duration,
                "total-duration": total,
                "start-time": _EPOCH,
                "seed": 7,
            },
            "working-directory": str(tmp_path),
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "orchestration": {
            "population": {
                "backend": "FilePopulationLoader",
                "source-type": "cw",
                "n-samples": n_pulsars,
                "arguments": {"path": str(catalogue)},
            },
            "signal": {
                "source-type": "cw",
                "detectors": list(_DETECTORS),
                "minimum-frequency": 5,
                "earth-rotation": True,
                "arguments": {
                    "earth_ephemeris": str(_EPHEMERIS / "earth00-40-DE405.dat.gz"),
                    "sun_ephemeris": str(_EPHEMERIS / "sun00-40-DE405.dat.gz"),
                    "reference_time_ssb": _EPOCH,
                },
                "output": {
                    "output_directory": "signal",
                    "file_name": "cw.gwf",
                    "arguments": {"channel": "{{ detectors }}:STRAIN"},
                },
            },
        },
    }


def _orchestrator(tmp_path: Path, **kwargs: Any) -> AdapterOrchestrator:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = Config.model_validate(_config(tmp_path, **kwargs))
    return AdapterOrchestrator.from_config(
        config.orchestration, global_simulator_arguments=dict(config.globals.simulator_arguments)
    )


def _segment(orchestrator: AdapterOrchestrator) -> np.ndarray:
    chunks = orchestrator._simulate()
    assert len(chunks) == 1, f"expected one chunk spanning the segment, got {len(chunks)}"
    return np.asarray(chunks[0][0], dtype=float)


class TestPhaseCoherence:
    """The property the whole design protects, checked through the orchestrator."""

    def test_segments_join_up_through_the_orchestrator(self, tmp_path):
        """Two consecutive segments must reconstruct the same span generated in one go.

        The simulator only guarantees this if ``reference_time_ssb`` arrives as a single value for
        the run. Plumbing that derived it per segment -- from the segment start, say -- would leave
        each segment individually perfect and the join incoherent, with nothing in the frames
        showing it. The tolerance clears the reassociation floor between differently-sized FFTs and
        is nine orders below the ~1 of peak a phase reset produces.
        """
        stitched_source = _orchestrator(tmp_path / "segmented", duration=64, total=128)
        first = _segment(stitched_source)
        stitched_source.update_state()
        second = _segment(stitched_source)
        stitched = np.concatenate([first, second])

        whole = _segment(_orchestrator(tmp_path / "whole", duration=128, total=128))

        assert stitched.shape == whole.shape
        peak = float(np.max(np.abs(whole)))
        worst = float(np.max(np.abs(stitched - whole))) / peak
        assert worst < 1e-9, f"segments disagree with the continuous run by {worst:.3e} of peak"


class TestTheOrderingExemption:
    """Skipping the `coa_time` sort is for continuous waves only, and that gate is load-bearing."""

    def test_a_compact_binary_catalogue_without_coa_time_still_raises(self, tmp_path):
        """Without the source-type gate this regresses into silent, plausible corruption.

        The per-event loop breaks on ``coa_time >= end_time``; when the key is absent that test is
        never true, so every event lands in the first segment. A `bbh` catalogue that lost the
        column -- a header typo, a dropped field -- would produce a full-looking run with every
        coalescence time wrong. It used to raise, and must keep raising.
        """
        catalogue = tmp_path / "no_coa_time.csv"
        catalogue.write_text("detector_frame_mass_1,detector_frame_mass_2,luminosity_distance\n30.0,25.0,400.0\n")
        config = _config(tmp_path, duration=64, total=64)
        config["orchestration"]["population"].update(
            {"source-type": "bbh", "n-samples": 1, "arguments": {"path": str(catalogue)}}
        )
        config["orchestration"]["signal"]["source-type"] = "bbh"
        config["orchestration"]["signal"]["waveform-model"] = "IMRPhenomD"
        config["orchestration"]["signal"].pop("arguments")

        parsed = Config.model_validate(config)
        global_arguments = dict(parsed.globals.simulator_arguments)

        with pytest.raises(ValueError, match="ordering key 'coa_time' is missing"):
            AdapterOrchestrator.from_config(parsed.orchestration, global_simulator_arguments=global_arguments)


class TestUnsupportedCombinations:
    """What the branch refuses, and why refusing beats ignoring."""

    def test_the_batched_execution_mode_is_refused(self, tmp_path):
        """The CW branch dispatches before the batched check, so silence would mean substitution.

        A configuration asking for `execution: batched` would otherwise get the per-source loop --
        output produced through a different execution mode than the one requested, with nothing
        anywhere saying so.
        """
        config = _config(tmp_path, duration=64, total=64)
        config["orchestration"]["signal"]["execution"] = "batched"
        parsed = Config.model_validate(config)
        orchestrator = AdapterOrchestrator.from_config(
            parsed.orchestration, global_simulator_arguments=dict(parsed.globals.simulator_arguments)
        )

        with pytest.raises(ValueError, match="execution: batched is not available for continuous waves"):
            orchestrator._simulate()


class TestTheCatalogueReachesEverySegment:
    """Every pulsar contributes to every segment, unlike an event that is consumed once."""

    def test_the_population_index_never_advances(self, tmp_path):
        """Advancing it would silently drop sources from later segments.

        The per-event loop consumes events by advancing ``population_index``; for a catalogue whose
        sources are all permanently present, that would leave the second segment short and the
        output still plausible.
        """
        orchestrator = _orchestrator(tmp_path, duration=64, total=128)

        _segment(orchestrator)
        assert int(orchestrator.population_index) == 0

        orchestrator.update_state()
        _segment(orchestrator)
        assert int(orchestrator.population_index) == 0

    def test_every_source_is_recorded_for_every_segment(self, tmp_path):
        """Provenance lists all pulsars each time, because all of them are present each time."""
        orchestrator = _orchestrator(tmp_path, duration=64, total=128, n_pulsars=2)

        _segment(orchestrator)
        assert len(orchestrator._batch_injections) == 2

        orchestrator.update_state()
        _segment(orchestrator)
        assert len(orchestrator._batch_injections) == 2

    def test_two_pulsars_give_the_sum_of_the_two_alone(self, tmp_path):
        """The catalogue must be *summed*, and this checks the sum rather than merely a difference.

        Detecting that a second source changed the output would also pass for an implementation
        that overwrote, scaled, or otherwise distorted the total. Comparing against the two sources
        generated separately and added pins the actual composition.
        """
        first_only = _segment(_orchestrator(tmp_path / "first", duration=64, total=64, n_pulsars=1))
        both = _segment(_orchestrator(tmp_path / "both", duration=64, total=64, n_pulsars=2))

        # The second pulsar alone, obtained by removing the first from the catalogue.
        second_dir = tmp_path / "second"
        second_dir.mkdir(parents=True, exist_ok=True)
        rows = _PULSARS.splitlines()
        (second_dir / "only.csv").write_text(rows[0] + "\n" + rows[2] + "\n")
        config = _config(second_dir, duration=64, total=64, n_pulsars=1)
        config["orchestration"]["population"]["arguments"]["path"] = str(second_dir / "only.csv")
        parsed = Config.model_validate(config)
        second_only = _segment(
            AdapterOrchestrator.from_config(
                parsed.orchestration, global_simulator_arguments=dict(parsed.globals.simulator_arguments)
            )
        )

        expected = first_only + second_only
        peak = float(np.max(np.abs(expected)))
        worst = float(np.max(np.abs(both - expected))) / peak
        assert worst < 1e-9, f"two pulsars together differ from their sum by {worst:.3e} of peak"
