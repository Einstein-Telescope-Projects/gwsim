#
# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
"""A checkpoint must say which configuration wrote it, and a resume must check.

Measured before this existed: two configs run from one working directory, the first interrupted
after two of three batches. The second **loaded the first's checkpoint, skipped those batches as
already complete, and wrote one frame where a clean run writes three** — exit code 0, no warning.
Silent data loss from an ordinary action, since every shipped example puts `.gwmock_checkpoints` and
`metadata/` relative to the working directory.

`_batch_outputs_present` does not catch it, and cannot: it looks up
``{simulator_name}-{batch_index}.metadata.json`` — a name with nothing config-specific in it — and
then verifies the outputs *that file* records. The first run's metadata records the first run's
frames, which exist, so the check passes on one run's evidence while a different run is executing.

The refusal is deliberate rather than a silent fresh start: ignoring the foreign checkpoint would fix
the data loss and hide its cause, leaving a stale checkpoint in place and an unexplained full re-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gwmock.cli.utils.checkpoint import CheckpointManager, ForeignCheckpointError, require_matching_config

#: Only ever formatted into an error message -- nothing here opens it -- so any path serves. Named
#: rather than written inline so it does not read as a real location a test might depend on.
_ANY_CHECKPOINT = Path("checkpoints") / "simulation.checkpoint.json"


class TestTheGuardItself:
    """`require_matching_config`, isolated from everything that has to call it."""

    def test_a_different_configuration_is_refused(self):
        with pytest.raises(ForeignCheckpointError, match="different configuration"):
            require_matching_config("a" * 64, "b" * 64, _ANY_CHECKPOINT)

    def test_the_message_names_both_hashes_and_the_file(self):
        """The user has to be able to act on it: which two collided, and what to move."""
        with pytest.raises(ForeignCheckpointError) as raised:
            require_matching_config("a" * 64, "b" * 64, Path("/runs/x/.gwmock_checkpoints/simulation.checkpoint.json"))

        message = str(raised.value)
        assert "aaaaaaaaaaaa" in message
        assert "bbbbbbbbbbbb" in message
        assert "/runs/x/.gwmock_checkpoints/simulation.checkpoint.json" in message

    def test_the_same_configuration_is_allowed(self):
        """The case that must not break: an ordinary resume of the run that wrote the checkpoint."""
        require_matching_config("a" * 64, "a" * 64, _ANY_CHECKPOINT)

    def test_a_checkpoint_without_a_hash_is_allowed_with_a_warning(self, caplog):
        """Refusing would break a legitimate resume for anyone who upgrades mid-run.

        A certain cost against an uncertain one, so it warns instead -- but it does warn, because the
        checkpoint genuinely cannot be checked and the consequence if it is foreign is silent.
        """
        import logging

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            require_matching_config(None, "b" * 64, _ANY_CHECKPOINT)

        assert "predates configuration fingerprinting" in caplog.text

    def test_an_unknown_plan_hash_is_allowed(self):
        """Nothing to compare against is not evidence of a mismatch."""
        require_matching_config("a" * 64, None, _ANY_CHECKPOINT)


class TestTheFingerprintSurvivesTheCheckpoint:
    """It is only a guard if it is written, and written checkpoints are what a resume reads."""

    def test_the_hash_is_saved_and_read_back(self, tmp_path):
        manager = CheckpointManager(tmp_path)
        manager.save_checkpoint([0], "orchestration", 0, {"counter": 1}, config_sha256="c" * 64)

        assert (manager.load_checkpoint() or {}).get("config_sha256") == "c" * 64

    def test_a_checkpoint_saved_without_one_reads_as_none(self, tmp_path):
        """Not as a missing key: the guard distinguishes "unknown" from "known and different"."""
        manager = CheckpointManager(tmp_path)
        manager.save_checkpoint([0], "orchestration", 0, {"counter": 1})

        assert (manager.load_checkpoint() or {}).get("config_sha256") is None


@pytest.mark.integration
def test_a_second_configuration_in_the_same_directory_is_refused_end_to_end(tmp_path):
    """The whole failure, through the CLI, because the unit tests above cannot reach the wiring.

    Two configs differing only in their population and output names, run from one directory. Before
    the guard the second wrote 1 frame of 3 and exited 0; it must now refuse and write none.

    Deterministic: the interrupt waits for the checkpoint file to appear rather than for a delay, so
    it lands after a batch has committed however slow the machine is.
    """
    pytest.importorskip("ripplegw", reason="ripplegw not installed")
    import json
    import shutil
    import signal as signal_module
    import subprocess
    import time

    executable = shutil.which("gwmock")
    if executable is None:
        pytest.skip("the gwmock console script is not on PATH")

    (tmp_path / "pop_a.csv").write_text(_POPULATION.format(mass_1="1.6", mass_2="1.4"))
    (tmp_path / "pop_b.csv").write_text(_POPULATION.format(mass_1="30.0", mass_2="25.0"))
    (tmp_path / "a.yaml").write_text(_CONFIG.format(population="pop_a.csv", tag="A"))
    (tmp_path / "b.yaml").write_text(_CONFIG.format(population="pop_b.csv", tag="B"))

    checkpoint = tmp_path / ".gwmock_checkpoints" / "simulation.checkpoint.json"
    first = subprocess.Popen(  # noqa: S603 - absolute path from `shutil.which`
        [executable, "simulate", "a.yaml"], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 600.0
        while not checkpoint.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert checkpoint.exists(), "no checkpoint was written, so there is nothing for the second run to find"
        completed = set(json.loads(checkpoint.read_text()).get("completed_batch_indices") or [])
        assert completed, "the checkpoint records no completed batch, so the second run would skip nothing"
        first.send_signal(signal_module.SIGINT)
        first.wait(timeout=120)
    finally:
        if first.poll() is None:  # pragma: no cover - only on an unexpected hang
            first.kill()

    second = subprocess.run(  # noqa: S603 - absolute path from `shutil.which`
        [executable, "simulate", "b.yaml"], cwd=tmp_path, capture_output=True, text=True, timeout=900, check=False
    )

    assert second.returncode != 0, "the second configuration resumed from the first's checkpoint instead of refusing"
    assert "different configuration" in (second.stdout + second.stderr)
    written = list((tmp_path / "output" / "signal").glob("*B-*.gwf"))
    assert not written, f"the refused run still wrote {[p.name for p in written]}"


_POPULATION = (
    "detector_frame_mass_1,detector_frame_mass_2,coa_time,distance,"
    "declination,right_ascension,polarization_angle,inclination\n"
    "{mass_1},{mass_2},1000000610.0,100.0,0.3,1.1,0.2,0.5\n"
)

_CONFIG = """
globals:
    simulator-arguments:
        sampling-frequency: 1024
        duration: 32
        total-duration: 96
        start-time: 1000000540
    working-directory: .
    output-directory: output
    metadata-directory: metadata

orchestration:
    population:
        backend: FilePopulationLoader
        source-type: bns
        arguments:
            path: {population}
    signal:
        waveform-model: IMRPhenomXPHM
        minimum-frequency: 30
        earth-rotation: true
        detectors:
            - ET-Triangle-Sardinia
        output:
            output_directory: signal
            file_name: 'E-{{{{ detectors }}}}_STRAIN_{tag}-{{{{ start_time }}}}-{{{{ duration }}}}.gwf'
            arguments:
                channel: '{{{{ detectors }}}}:STRAIN'
"""
