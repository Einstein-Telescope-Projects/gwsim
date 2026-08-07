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

"""``index.yaml`` must survive concurrent writers and a failed write.

The sibling of ``signal_index.yaml``, written by the same `save_batch_metadata` into the same
metadata directory, and carrying the same two faults that file had before #323: an unlocked
read-modify-write, and an in-place truncating write whose failure the loader then treats as
"create a new index" -- discarding every entry rather than one.

Found in review of the signal-index repair, by asking what a whole series of reviews had missed:
nobody had looked at the file written on the line above.

**Deliberately weaker than its sibling.** No staleness digest here: that guard's job is to refuse
a write, and nothing in production reads ``index.yaml`` today, so aborting a run over it would
cost more than the entry it protects. Cross-host staleness can still lose an entry here. If a
real consumer appears, that decision should be revisited.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from multiprocessing.synchronize import Barrier
from pathlib import Path
from typing import Any

import pytest
import yaml

from gwmock.cli.simulate_utils import update_metadata_index

pytestmark = pytest.mark.unit

_BARRIER_TIMEOUT_SECONDS = 2.0
_LOCK_WAIT_EVIDENCE_SECONDS = 0.5
_SCENARIO_ATTEMPTS = 3


def _writer(directory: str, name: str, batch: int, ready: Barrier, barrier: Barrier, evidence: Any) -> None:
    """Add one distinct entry, with the two writers rendezvousing inside the critical section.

    The same shape as the signal-index race test, and for the same reason: a barrier placed
    before the call only starts the processes together and leaves the overlap to the scheduler.
    This one is armed on the read, and timed, because under the fix the second process is locked
    out and can never arrive.
    """
    import gwmock.cli.simulate_utils as module

    real_load = module.yaml.safe_load

    def _rendezvous_then_load(stream: Any) -> Any:
        loaded = real_load(stream)
        try:
            barrier.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
            evidence["rendezvous"] = True
        except Exception:  # noqa: BLE001 - a broken barrier is the expected outcome under the fix
            pass
        return loaded

    module.yaml.safe_load = _rendezvous_then_load

    if module.fcntl is not None:
        real_flock = module.fcntl.flock

        def _timed_flock(descriptor: int, operation: int) -> Any:
            if operation != module.fcntl.LOCK_EX:
                return real_flock(descriptor, operation)
            started = time.perf_counter()
            result = real_flock(descriptor, operation)
            evidence["lock_wait"] = max(evidence["lock_wait"], time.perf_counter() - started)
            return result

        module.fcntl.flock = _timed_flock

    ready.wait(timeout=30)
    update_metadata_index(Path(directory), [Path(name)], f"orchestration-{batch}.metadata.json")


def test_two_processes_adding_entries_keep_both(tmp_path: Path) -> None:
    """Neither writer's entry may be lost when both update the index at once.

    Fails on the unfixed code: the read-modify-write is unlocked, so the second writer builds on
    a pre-race read and its dump drops the first's entry. Measured at 4 losses in 20 trials
    before the fix.
    """
    context = mp.get_context("fork")
    manager = context.Manager()

    for attempt in range(_SCENARIO_ATTEMPTS):
        directory = tmp_path / f"attempt-{attempt}"
        directory.mkdir()
        ready = context.Barrier(2)
        barrier = context.Barrier(2)
        evidence = [manager.dict(rendezvous=False, lock_wait=0.0) for _ in range(2)]
        processes = [
            context.Process(target=_writer, args=(str(directory), name, batch, ready, barrier, evidence[batch]))
            for batch, name in enumerate(("data-A.gwf", "data-B.gwf"))
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
        assert all(p.exitcode == 0 for p in processes), [p.exitcode for p in processes]

        met = any(record["rendezvous"] for record in evidence)
        waited = max(record["lock_wait"] for record in evidence)
        if met or waited > _LOCK_WAIT_EVIDENCE_SECONDS:
            break
    else:
        pytest.fail(
            f"scenario did not run in {_SCENARIO_ATTEMPTS} attempts: rendezvous={met}, longest "
            f"lock wait={waited:.3f}s -- the writers were serialised by scheduling rather than by "
            "the lock, so nothing was exercised either way."
        )

    index = yaml.safe_load((directory / "index.yaml").read_text())
    assert set(index) == {"data-A.gwf", "data-B.gwf"}, index


def test_a_failed_write_leaves_the_previous_index_intact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write that fails part-way must not discard the entries already recorded.

    The in-place ``open("w")`` truncated first, and the loader treats an unparsable index as
    "create a new index" -- so one interrupted write silently discarded every entry, not one.
    """
    update_metadata_index(tmp_path, [Path("data-0.gwf")], "orchestration-0.metadata.json")
    index_file = tmp_path / "index.yaml"
    assert set(yaml.safe_load(index_file.read_text())) == {"data-0.gwf"}

    def _fail_to_publish(source: Any, destination: Any) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr("gwmock.cli.simulate_utils.os.replace", _fail_to_publish)
    with pytest.raises(OSError, match="no space left on device"):
        update_metadata_index(tmp_path, [Path("data-1.gwf")], "orchestration-1.metadata.json")
    monkeypatch.undo()

    recovered = yaml.safe_load(index_file.read_text())
    assert recovered is not None, "the index was truncated by the failed write"
    assert set(recovered) == {"data-0.gwf"}, recovered
    assert not list(tmp_path.glob("*.tmp")), list(tmp_path.glob("*.tmp"))


def test_the_index_keeps_its_permissions(tmp_path: Path) -> None:
    """Replacing by rename must not tighten the index, as it did for its sibling."""
    previous = os.umask(0o022)
    try:
        update_metadata_index(tmp_path, [Path("data-0.gwf")], "orchestration-0.metadata.json")
        index_file = tmp_path / "index.yaml"
        assert oct(index_file.stat().st_mode & 0o777) == "0o644"
        index_file.chmod(0o664)
        update_metadata_index(tmp_path, [Path("data-1.gwf")], "orchestration-1.metadata.json")
        assert oct(index_file.stat().st_mode & 0o777) == "0o664"
    finally:
        os.umask(previous)
