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

from gwmock.cli.utils.checkpoint import (
    CheckpointManager,
    ForeignCheckpointError,
    require_matching_config,
    run_fingerprint,
)

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


class TestTheFingerprintIsTheRunNotTheConfigFile:
    """The config hash alone is not run identity, and treating it as one leaves the bug reachable.

    Found in review after the first version of this fix shipped the narrower identity: the same
    config file with a different ``--output-dir`` hashes identically, so the checkpoint is accepted
    and its batches skipped -- measured at 2 frames where a clean run writes 3.
    """

    def test_the_output_directory_changes_it(self, tmp_path):
        first = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta")
        second = run_fingerprint(["a" * 64], tmp_path / "out2", tmp_path / "meta")
        assert first != second

    def test_the_metadata_directory_changes_it(self, tmp_path):
        first = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta")
        second = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta2")
        assert first != second

    def test_an_equivalent_path_does_not(self, tmp_path):
        """Resolved, so `./out` and its absolute form are one run -- a false refusal is still a bug."""
        (tmp_path / "out").mkdir()
        (tmp_path / "meta").mkdir()
        first = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta")
        second = run_fingerprint(["a" * 64], tmp_path / "out" / ".", tmp_path / "meta")
        assert first == second

    def test_every_batch_hash_counts_not_just_the_first(self, tmp_path):
        """A plan assembled from several metadata records can mix configs; one of them is not enough."""
        first = run_fingerprint(["a" * 64, "b" * 64], tmp_path / "out", tmp_path / "meta")
        second = run_fingerprint(["a" * 64, "c" * 64], tmp_path / "out", tmp_path / "meta")
        assert first != second

    def test_an_unhashed_batch_is_not_the_same_as_no_batch(self, tmp_path):
        """`None` is kept as a marker; dropping it would collapse two different plans onto one id."""
        first = run_fingerprint(["a" * 64, None], tmp_path / "out", tmp_path / "meta")
        second = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta")
        assert first != second

    def test_the_same_run_fingerprints_the_same(self, tmp_path):
        """The property every resume depends on: unchanged inputs give an unchanged identity."""
        assert run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta") == run_fingerprint(
            ["a" * 64], tmp_path / "out", tmp_path / "meta"
        )


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


@pytest.mark.integration
def test_ignore_checkpoint_gets_past_the_refusal(tmp_path):
    """The refusal must have a way out that is not "delete the file by hand".

    Without one it is a dead end for anything that cannot answer a prompt: an automated campaign
    that knows the checkpoint is stale would fail on it with no forward path. Raised in review, and
    the reason the default stays a refusal rather than a warning -- the escape is explicit, so
    nothing is skipped by accident.
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
    (tmp_path / "a.yaml").write_text(_CONFIG.format(population="pop_a.csv", tag="A"))

    checkpoint = tmp_path / ".gwmock_checkpoints" / "simulation.checkpoint.json"
    first = subprocess.Popen(  # noqa: S603 - absolute path from `shutil.which`
        [executable, "simulate", "a.yaml"], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 600.0
        while not checkpoint.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert checkpoint.exists()
        assert json.loads(checkpoint.read_text()).get("completed_batch_indices")
        first.send_signal(signal_module.SIGINT)
        first.wait(timeout=120)
    finally:
        if first.poll() is None:  # pragma: no cover - only on an unexpected hang
            first.kill()

    # A different output directory is a different run, so this would be refused without the flag --
    # which the test above pins. Here it must go through and produce a complete dataset.
    completed = subprocess.run(  # noqa: S603 - absolute path from `shutil.which`
        [executable, "simulate", "a.yaml", "--output-dir", "elsewhere", "--ignore-checkpoint"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "different configuration" not in (completed.stdout + completed.stderr)
    frames = sorted((tmp_path / "elsewhere" / "signal").glob("*ET1*.gwf"))
    assert len(frames) == 3, f"ignoring the checkpoint should give a complete run, got {[f.name for f in frames]}"


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
