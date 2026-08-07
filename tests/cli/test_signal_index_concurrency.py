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

"""Two writers must not lose each other's entries in ``signal_index.yaml``.

The index is written by a read-modify-write. Two runs sharing a metadata directory therefore
race, and the loser's events vanish from the id lookup while their samples sit in the frames --
``gwmock find-signal`` then reports nothing for an event that was injected.

**Processes, not threads.** The contended state is a file, and the fix is an OS-level file lock,
which is per-process. A threaded version of this test can pass on the unfixed code for the wrong
reason (the GIL serialising the small critical section) and would not discriminate. Each writer
here is a real forked process.

**Where the barrier goes, and why it has a timeout.** An earlier version of this test put the
barrier before ``update_signal_index``, which only starts the two processes together and leaves
the race to the scheduler: measured against the unfixed code it passed **3 trials in 60**, so it
green-lit the bug 5% of the time and the mutation testing built on it was equally probabilistic.
The barrier now sits *inside* the read-modify-write, patched over ``_withdraw_batch`` -- the
first call after the index is read and before it is written.

It must be a *timed* barrier, and that is not a detail. Under the fix the two processes cannot
both be inside the critical section, so waiting for each other unconditionally would deadlock
the correct code. Timing out means: unfixed, both arrive and proceed together from stale reads,
and one write is lost; fixed, the first holder waits, times out, finishes and releases, and the
second then reads a file that already has the first event. Deterministic in both directions, at
the cost of one timeout per process.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import stat
import threading
from multiprocessing.synchronize import Barrier
from pathlib import Path

import pytest
import yaml

from gwmock.cli.simulate_utils import update_signal_index

pytestmark = pytest.mark.unit

# Long enough that a slow forked child still arrives when the code is unfixed, short enough that
# the fixed path (where the second process is locked out and can never arrive) stays quick.
_BARRIER_TIMEOUT_SECONDS = 2.0


def _metadata(event_id: int, batch: int) -> dict:
    """One batch's metadata injecting a single distinct event."""
    return {
        "signal": {"injections": [{"event_id": event_id, "parameters": {"coa_time": 100.0 + event_id}}]},
        "outputs": [{"kind": "signal", "path": f"signal/signal-{batch}.gwf"}],
    }


def _writer(directory: str, event_id: int, batch: int, ready: Barrier, barrier: Barrier) -> None:
    """Update the index with the barrier armed inside the read-modify-write.

    ``_withdraw_batch`` is the first call after the index is read and before it is written, so
    patching it here puts the rendezvous exactly in the window the lock is supposed to close.
    Patching happens in the forked child, so the parent and the other child are unaffected.

    Two barriers, not one. ``ready`` is untimed and makes both children enter
    ``update_signal_index`` together; without it a slow child can arrive after the other has
    already finished, read the completed index, and merge cleanly -- the assertion then passes
    on unfixed code without the two stale reads ever overlapping. ``barrier`` is the timed
    in-critical-section rendezvous.
    """
    import gwmock.cli.simulate_utils as module

    original = module._withdraw_batch

    def _rendezvous_then_withdraw(index: dict, metadata_file_name: str) -> dict:
        # A broken barrier is expected under the fix: the other process is blocked on the lock
        # and cannot arrive, which is the whole point -- the critical section is exclusive.
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
        return original(index, metadata_file_name)

    module._withdraw_batch = _rendezvous_then_withdraw
    ready.wait(timeout=30)
    update_signal_index(Path(directory), _metadata(event_id, batch), f"orchestration-{batch}.metadata.json")


def test_two_processes_updating_the_index_keep_both_events(tmp_path: Path) -> None:
    """Neither writer's event may be lost when both update the index at once.

    Fails on the unfixed code: the read-modify-write is unlocked, so both processes reach the
    in-critical-section barrier together, and whichever writes second overwrites the first's
    contribution with an index built from its own pre-race read.
    """
    context = mp.get_context("fork")
    ready = context.Barrier(2)
    barrier = context.Barrier(2)
    processes = [
        context.Process(target=_writer, args=(str(tmp_path), event_id, batch, ready, barrier))
        for batch, event_id in enumerate((11, 22))
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)

    assert all(process.exitcode == 0 for process in processes), [p.exitcode for p in processes]

    index = yaml.safe_load((tmp_path / "signal_index.yaml").read_text())
    # The whole point: both survive. Asserting only on the count would pass an index holding one
    # event twice, which a broken merge could produce.
    assert set(index) == {"11", "22"}, index
    assert index["11"]["batches"][0]["frames"] == ["signal/signal-0.gwf"]
    assert index["22"]["batches"][0]["frames"] == ["signal/signal-1.gwf"]


def test_atomic_write_keeps_the_index_readable_by_others(tmp_path: Path) -> None:
    """Replacing the index must not tighten its permissions.

    ``mkstemp`` creates 0600 and ignores the umask, and ``os.replace`` carries that mode over, so
    an atomic write is a silent way to lock a second account out of the shared metadata directory
    that the locking exists to support. Both reviewers found this by probing rather than from a
    test, which is why one exists now.
    """
    previous = os.umask(0o022)
    try:
        update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
        index_file = tmp_path / "signal_index.yaml"
        created = stat.S_IMODE(index_file.stat().st_mode)
        assert created == 0o644, oct(created)

        # An existing index's own mode wins over the default, so a deliberately group-writable
        # index stays that way across an update.
        index_file.chmod(0o664)
        update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")
        preserved = stat.S_IMODE(index_file.stat().st_mode)
        assert preserved == 0o664, oct(preserved)
    finally:
        os.umask(previous)


def test_index_survives_a_write_that_fails_partway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed write must not destroy the entries already in the index.

    Separate defect from the race, and the more damaging one. The final write truncates the file
    in place, and the loader treats an unparsable index as "create new index" -- so a crash or a
    full disk mid-dump silently discards *every* prior entry on the next update, rather than
    losing one race's worth.
    """
    update_signal_index(tmp_path, _metadata(7, 0), "orchestration-0.metadata.json")
    index_file = tmp_path / "signal_index.yaml"
    before = index_file.read_text()
    assert "7" in yaml.safe_load(before)

    real_safe_dump = yaml.safe_dump

    def _fail_after_writing_some(data, stream=None, **kwargs):  # type: ignore[no-untyped-def]
        if stream is not None:
            stream.write("partial: [")  # a prefix that is not valid YAML on its own
            raise OSError("no space left on device")
        return real_safe_dump(data, stream, **kwargs)

    monkeypatch.setattr("gwmock.cli.simulate_utils.yaml.safe_dump", _fail_after_writing_some)
    with pytest.raises(OSError, match="no space left on device"):
        update_signal_index(tmp_path, _metadata(8, 1), "orchestration-1.metadata.json")
    monkeypatch.undo()

    # The pre-existing entry must still be there and still parse.
    recovered = yaml.safe_load(index_file.read_text())
    assert recovered is not None, "index was truncated by the failed write"
    assert "7" in recovered, recovered
