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
here is a real forked process, and a barrier holds both inside the window between read and write
so the race is deterministic rather than probabilistic.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.synchronize import Barrier
from pathlib import Path

import pytest
import yaml

from gwmock.cli.simulate_utils import update_signal_index

pytestmark = pytest.mark.unit


def _metadata(event_id: int, batch: int) -> dict:
    """One batch's metadata injecting a single distinct event."""
    return {
        "signal": {"injections": [{"event_id": event_id, "parameters": {"coa_time": 100.0 + event_id}}]},
        "outputs": [{"kind": "signal", "path": f"signal/signal-{batch}.gwf"}],
    }


def _writer(directory: str, event_id: int, batch: int, barrier: Barrier) -> None:
    """Update the index, both processes held at the barrier before the write lands.

    The barrier sits *before* the call rather than inside it, because the goal is to start both
    read-modify-write cycles together; holding them mid-cycle would need to reach into the
    function under test and would stop being a test of its contract.
    """
    barrier.wait(timeout=30)
    update_signal_index(Path(directory), _metadata(event_id, batch), f"orchestration-{batch}.metadata.json")


def test_two_processes_updating_the_index_keep_both_events(tmp_path: Path) -> None:
    """Neither writer's event may be lost when both update the index at once.

    Fails on the unfixed code: the read-modify-write is unlocked, so whichever process writes
    second overwrites the first's contribution with an index built from the pre-race read.
    """
    context = mp.get_context("fork")
    barrier = context.Barrier(2)
    processes = [
        context.Process(target=_writer, args=(str(tmp_path), event_id, batch, barrier))
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
