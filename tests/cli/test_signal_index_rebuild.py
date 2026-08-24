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

"""``signal_index.yaml`` must be rebuildable from the batch metadata files.

The index is a cache; the ``*.metadata.json`` records are the source of truth. Locking the
read-modify-write closes the race between two runs sharing a metadata directory on one host, but
it does not close every way an entry can go missing: the lock degrades to an unsynchronised write
where ``fcntl`` is absent or the filesystem refuses ``flock``, and a writer on a second host is
refused rather than merged. Each of those leaves the frames correct and the id lookup wrong, and
nothing before this could put it right short of rerunning the simulation.

The tests that matter here are the two that pin what a repair has to mean: the rebuilt index is
the same index the incremental path would have produced (otherwise the repair is a second,
divergent implementation of the schema), and rebuilding leaves the staleness guard satisfied
(otherwise the repair wedges the directory it just fixed).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import re
import time
from multiprocessing.synchronize import Barrier
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from gwmock.cli.main import app
from gwmock.cli.simulate_utils import (
    SignalIndexRebuildError,
    StaleIndexReadError,
    rebuild_signal_index,
    update_signal_index,
)
from gwmock.cli.utils.signal_lookup import find_signals

pytestmark = pytest.mark.unit

runner = CliRunner()

# How long the lock-holding child keeps the lock. Long enough that a parent which waited for it is
# unmistakable against scheduler noise, short enough not to pad the suite.
_LOCK_HOLD_SECONDS = 0.5

# Below the hold above by a margin that absorbs fork and startup skew, and far above the ~0
# a rebuild that never took the lock would show.
_LOCK_WAIT_EVIDENCE_SECONDS = 0.3


def _batch(index: int, events: list[tuple[int, float]], frames: list[str]) -> dict[str, Any]:
    """One batch metadata record injecting *events* into *frames*."""
    return {
        "signal": {
            "injections": [
                {"event_id": event_id, "parameters": {"coa_time": coa_time, "mass_1": 30.0 + event_id}}
                for event_id, coa_time in events
            ]
        },
        "outputs": [{"kind": "signal", "path": frame} for frame in frames]
        + [{"kind": "noise", "path": f"noise-{index}.gwf"}],
    }


def _write_batch(directory: Path, index: int, metadata: dict[str, Any], *, update_index: bool = True) -> str:
    """Write a batch metadata file, optionally folding it into the index as a run would."""
    name = f"orchestration-{index}.metadata.json"
    (directory / name).write_text(json.dumps(metadata), encoding="utf-8")
    if update_index:
        update_signal_index(directory, metadata, name)
    return name


def _populate(directory: Path, *, update_index: bool = True) -> None:
    """Write a directory whose batches exercise the shapes an entry can take.

    Event 22 spans batches 0 and 1, which is the case the index stores per batch for: a signal
    long enough to cross a segment boundary is written by every batch whose frames it reaches.
    Batch 2 injects nothing, so it must not contribute an entry -- or a rebuild would invent one.
    """
    _write_batch(
        directory, 0, _batch(0, [(11, 111.0), (22, 122.0)], ["signal/signal-0.gwf"]), update_index=update_index
    )
    _write_batch(
        directory, 1, _batch(1, [(22, 122.0), (33, 133.0)], ["signal/signal-1.gwf"]), update_index=update_index
    )
    _write_batch(directory, 2, _batch(2, [], ["signal/signal-2.gwf"]), update_index=update_index)


def test_rebuild_reproduces_the_index_the_incremental_path_wrote(tmp_path: Path) -> None:
    """A rebuild must produce the same mapping an ordinary run built batch by batch.

    The point of the assertion is not that the rebuild "works" but that there is one schema. The
    rebuild and the incremental update share :func:`_record_batch_in_index` precisely so this
    holds; were the entry shape written out twice, a later change to one of them would leave
    ``find-signal`` reading rebuilt entries it does not understand, and no test of the rebuild
    alone would notice.

    Batches are written in metadata-file-name order here, so the two paths also agree on the order
    within an entry and the comparison can be exact rather than order-insensitive.
    """
    _populate(tmp_path)
    index_file = tmp_path / "signal_index.yaml"
    incremental = yaml.safe_load(index_file.read_text())
    # Guard the guard: an empty or single-entry `incremental` would make the comparison vacuous.
    assert set(incremental) == {"11", "22", "33"}, incremental
    assert len(incremental["22"]["batches"]) == 2, incremental["22"]

    index_file.unlink()
    rebuilt = rebuild_signal_index(tmp_path)

    assert rebuilt.batches == 2, "the batch injecting nothing must not count as a contribution"
    assert rebuilt.events == 3
    assert yaml.safe_load(index_file.read_text()) == incremental


def test_rebuild_restores_an_entry_a_lost_write_dropped(tmp_path: Path) -> None:
    """The repair itself: an index missing a batch's events is made whole from the metadata.

    The state set up here is exactly what an unsynchronised concurrent write leaves behind -- the
    frames and the batch metadata hold every event, while the index holds only the writer that
    happened to dump last -- and it is the state ``find-signal --id`` answers "not found" in while
    the samples sit in the frames.
    """
    _populate(tmp_path, update_index=False)
    index_file = tmp_path / "signal_index.yaml"
    # Only batch 1 ever reached the index; batch 0's write was lost.
    update_signal_index(
        tmp_path, _batch(1, [(22, 122.0), (33, 133.0)], ["signal/signal-1.gwf"]), "orchestration-1.metadata.json"
    )
    assert set(yaml.safe_load(index_file.read_text())) == {"22", "33"}
    assert find_signals(tmp_path, event_id=11) == [], "precondition: the lost event is unfindable by id"

    rebuild_signal_index(tmp_path)

    recovered = find_signals(tmp_path, event_id=11)
    assert [match["frames"] for match in recovered] == [["signal/signal-0.gwf"]], recovered
    # The event that spans both batches must come back naming both frames, not just the surviving
    # batch's -- a rebuild that merely re-added the missing key would pass a weaker assertion.
    spanning = find_signals(tmp_path, event_id=22)
    assert [match["frames"] for match in spanning] == [["signal/signal-0.gwf", "signal/signal-1.gwf"]], spanning


def test_rebuild_rebaselines_the_digest_so_the_next_write_is_accepted(tmp_path: Path) -> None:
    """Rebuilding must leave the staleness guard satisfied, not merely leave a correct file.

    Writing the index without recording its digest would fix the lookup and wedge the directory:
    every later batch would refuse as stale, and the operator would be told to delete the sidecar
    by hand -- the manual step this command exists to replace.

    The rebuild has to *change* the index for the assertion to mean anything. Batch 3's metadata
    file is on disk with no index entry -- a lost write, the fault being repaired -- so the rebuilt
    index differs from the one the sidecar last described. An earlier version of this test wedged a
    directory whose rebuild reproduced the previous bytes exactly; the digest then matched whether
    or not it had been re-recorded, and dropping the call passed.
    """
    _populate(tmp_path)
    index_file = tmp_path / "signal_index.yaml"
    lock_file = tmp_path / "signal_index.yaml.lock"
    _write_batch(tmp_path, 3, _batch(3, [(44, 144.0)], ["signal/signal-3.gwf"]), update_index=False)

    # A hand-repaired index: digest nobody updated. This is the wedged state the sidecar refuses in.
    index_file.write_text(index_file.read_text() + "\n", encoding="utf-8")
    stale = lock_file.read_text(encoding="utf-8").strip()
    with pytest.raises(StaleIndexReadError):
        update_signal_index(
            tmp_path, _batch(4, [(55, 155.0)], ["signal/signal-4.gwf"]), "orchestration-4.metadata.json"
        )

    rebuild_signal_index(tmp_path)

    recorded = lock_file.read_text(encoding="utf-8").strip()
    assert recorded != stale, "the rebuild produced the previous bytes, so this proves nothing"
    assert recorded == hashlib.sha256(index_file.read_bytes()).hexdigest()
    # And the directory takes writes again.
    _write_batch(tmp_path, 4, _batch(4, [(55, 155.0)], ["signal/signal-4.gwf"]))
    assert set(yaml.safe_load(index_file.read_text())) == {"11", "22", "33", "44", "55"}


def test_the_stale_read_refusal_names_the_rebuild_command(tmp_path: Path) -> None:
    """The refusal an operator actually meets must point at the command that repairs it.

    A repair nobody is told about is a repair nobody runs. The message is where this is
    discovered -- it is raised at the moment a run stops -- and it previously offered only
    "delete the sidecar", which re-baselines the digest against whatever index is on disk and so
    blesses one that has already lost entries.
    """
    _populate(tmp_path)
    index_file = tmp_path / "signal_index.yaml"
    index_file.write_text(index_file.read_text() + "\n", encoding="utf-8")

    with pytest.raises(StaleIndexReadError) as raised:
        update_signal_index(
            tmp_path, _batch(3, [(44, 144.0)], ["signal/signal-3.gwf"]), "orchestration-3.metadata.json"
        )

    assert f"gwmock reindex --metadata-dir {tmp_path}" in str(raised.value), str(raised.value)


def test_rebuild_refuses_an_unreadable_metadata_file_and_leaves_the_index(tmp_path: Path) -> None:
    """A source file it cannot read must stop the rebuild, not shrink its result.

    ``find_signals`` skips an unparsable metadata file and answers with the rest, which is right
    for a query. It is wrong here: this command replaces the index, so a skipped file would delete
    that batch's events from the lookup -- and it would do it under the name of a repair, leaving
    an index that looks complete and is not.
    """
    _populate(tmp_path)
    index_file = tmp_path / "signal_index.yaml"
    before = index_file.read_text()
    (tmp_path / "orchestration-0.metadata.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(SignalIndexRebuildError, match=re.escape("orchestration-0.metadata.json")):
        rebuild_signal_index(tmp_path)

    assert index_file.read_text() == before, "a refused rebuild must not touch the index"


# Shapes that decode as JSON without being a batch metadata record. Each one previously reached a
# `.get` on a value that has none and raised `AttributeError` -- a traceback where the docstring,
# the CLI help and the user guide all promise a refusal naming the file. Found in review, with the
# first of these; the rest are the same defect at the depths the builder also indexes into, since
# validating only the top level would have moved the traceback rather than removed it.
_MALFORMED_RECORDS: list[tuple[str, Any]] = [
    ("a top-level list", []),
    ("a top-level null", None),
    ("a top-level scalar", "orchestration-0"),
    ("a non-object signal", {"signal": "injected"}),
    ("non-list injections", {"signal": {"injections": {"event_id": 1}}}),
    ("a non-object injection", {"signal": {"injections": [42]}}),
    ("an object event_id", {"signal": {"injections": [{"event_id": {"population": 1}}]}}),
    ("a float event_id", {"signal": {"injections": [{"event_id": 3.0}]}}),
    ("a boolean event_id", {"signal": {"injections": [{"event_id": True}]}}),
    ("non-object parameters", {"signal": {"injections": [{"event_id": 1, "parameters": [122.0]}]}}),
    ("non-list outputs", {"signal": {"injections": [{"event_id": 1}]}, "outputs": "signal-0.gwf"}),
    ("a non-object output", {"signal": {"injections": [{"event_id": 1}]}, "outputs": ["signal-0.gwf"]}),
    (
        "a non-string output path",
        {"signal": {"injections": [{"event_id": 1}]}, "outputs": [{"kind": "signal", "path": 7}]},
    ),
]


@pytest.mark.parametrize(("description", "record"), _MALFORMED_RECORDS, ids=[case[0] for case in _MALFORMED_RECORDS])
def test_rebuild_refuses_valid_json_that_is_not_a_batch_metadata_record(
    tmp_path: Path, description: str, record: Any
) -> None:
    """Parsing as JSON is not the same as being a batch metadata record.

    The decode guard catches a file that is not JSON. It says nothing about a file that *is* JSON
    and is not a record, which is the case a directory in trouble actually produces -- a truncated
    write that happened to land on a valid document, a placeholder, a file from another tool.

    The refusal has to name the file, because the operator's next move is to look at it, and it
    has to be a refusal rather than a traceback, because the promise made in three places is that
    a batch file it cannot use stops the rebuild.
    """
    _populate(tmp_path)
    index_file = tmp_path / "signal_index.yaml"
    before = index_file.read_bytes()
    # Sorts before `orchestration-*`, so it is read first: the refusal must come before any write,
    # not merely before the write of the batch that follows it.
    (tmp_path / "bad.metadata.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(SignalIndexRebuildError, match=re.escape("bad.metadata.json")) as raised:
        rebuild_signal_index(tmp_path)

    assert "not a usable batch metadata record" in str(raised.value), str(raised.value)
    assert index_file.read_bytes() == before, f"the index was rewritten despite {description}"


def test_the_malformed_record_refusal_reaches_the_shell(tmp_path: Path) -> None:
    """Through the CLI it must be a message and an exit code, not an unhandled AttributeError.

    The library refusal is only half of it: this defect was found through ``gwmock reindex``, where
    it surfaced as a traceback. A caught :class:`SignalIndexRebuildError` is what turns it into
    something an operator can act on.
    """
    (tmp_path / "bad.metadata.json").write_text("[]", encoding="utf-8")

    result = runner.invoke(app, ["reindex", "--metadata-dir", str(tmp_path)])

    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    assert "bad.metadata.json" in result.output
    assert "not a usable batch metadata record" in result.output


def test_rebuild_still_accepts_the_records_a_run_legitimately_writes(tmp_path: Path) -> None:
    """The validation must not refuse shapes gwmock itself produces, or it breaks the repair.

    A batch that injected nothing writes ``signal`` and ``outputs`` that may be absent or null, and
    an id is an integer. Refusing any of those would make the rebuild reject a healthy directory --
    the failure mode a shape check is most likely to introduce, and invisible to the tests above,
    which only ever assert that something is rejected.
    """
    (tmp_path / "orchestration-0.metadata.json").write_text(json.dumps({"signal": None}), encoding="utf-8")
    (tmp_path / "orchestration-1.metadata.json").write_text(
        json.dumps({"signal": {"injections": None}, "outputs": None}), encoding="utf-8"
    )
    (tmp_path / "orchestration-2.metadata.json").write_text(json.dumps({}), encoding="utf-8")
    # An injection with no outputs at all: a real entry, and no frames to name.
    (tmp_path / "orchestration-3.metadata.json").write_text(
        json.dumps({"signal": {"injections": [{"event_id": 77, "parameters": {"coa_time": 177.0}}]}, "outputs": None}),
        encoding="utf-8",
    )

    rebuilt = rebuild_signal_index(tmp_path)

    assert rebuilt.batches == 1
    index = yaml.safe_load((tmp_path / "signal_index.yaml").read_text())
    assert index == {
        "77": {"batches": [{"metadata": "orchestration-3.metadata.json", "frames": []}], "coa_time": 177.0}
    }


def test_rebuild_refuses_a_directory_holding_no_batch_metadata(tmp_path: Path) -> None:
    """A directory with no batch records is a mistyped path, not an empty run.

    Treating it as an empty run would make ``gwmock reindex --metadata-dir <typo>`` destroy a good
    index, which is the worst thing a repair command can do.
    """
    with pytest.raises(SignalIndexRebuildError, match="No batch metadata files"):
        rebuild_signal_index(tmp_path)

    # And it left nothing behind -- not even the sidecar the lock would have created.
    assert not list(tmp_path.iterdir()), list(tmp_path.iterdir())


def _hold_the_lock(directory: str, taken: Barrier) -> None:
    """Hold the index lock for a fixed time, writing a new batch's metadata part-way through.

    The write lands *after* the parent has been released to start its rebuild, and before this
    lock is dropped. That ordering is the whole experiment: a rebuild that lists the directory
    before waiting for the lock does so while the file does not yet exist, so it cannot report it;
    one that lists after acquiring the lock cannot miss it.
    """
    import gwmock.cli.simulate_utils as module

    with module._exclusive_index_lock(Path(directory) / "signal_index.yaml"):
        taken.wait(timeout=30)
        time.sleep(_LOCK_HOLD_SECONDS / 2)
        (Path(directory) / "orchestration-9.metadata.json").write_text(
            json.dumps(_batch(9, [(99, 199.0)], ["signal/signal-9.gwf"])), encoding="utf-8"
        )
        time.sleep(_LOCK_HOLD_SECONDS / 2)


def test_rebuild_waits_for_the_lock_and_then_reads_the_directory(tmp_path: Path) -> None:
    """The rebuild must run inside the same lock a batch takes, and scan after acquiring it.

    Two properties in one scenario, because one process holding the lock demonstrates both.

    *Waiting* is what stops a rebuild and a running batch interleaving; measured, because a
    rebuild that never took the lock returns immediately and no assertion on the resulting file
    would tell the difference.

    *Scanning afterwards* is the subtler half. A batch writes its metadata file first and takes
    the lock second, so a rebuild that listed the directory before waiting would miss a batch
    caught mid-update and then overwrite the entry that batch went on to write -- the lock held,
    and an entry lost anyway. The child writes ``orchestration-9`` half-way through its hold, so
    the file does not exist at the moment a rebuild scanning before the lock would look, and does
    exist by the time one scanning after it does.
    """
    _populate(tmp_path)
    context = mp.get_context("fork")
    taken = context.Barrier(2)
    holder = context.Process(target=_hold_the_lock, args=(str(tmp_path), taken))
    holder.start()
    try:
        taken.wait(timeout=30)
        started = time.perf_counter()
        rebuild_signal_index(tmp_path)
        waited = time.perf_counter() - started
    finally:
        holder.join(timeout=30)

    assert holder.exitcode == 0
    assert waited > _LOCK_WAIT_EVIDENCE_SECONDS, (
        f"the rebuild returned in {waited:.3f}s while another process held the index lock, so it did not take the lock"
    )
    index = yaml.safe_load((tmp_path / "signal_index.yaml").read_text())
    assert "99" in index, "the batch written while the lock was held was not in the listing"


def test_reindex_command_rebuilds_and_reports(tmp_path: Path) -> None:
    """The CLI wiring: the command rebuilds the index and says what went into it."""
    _populate(tmp_path, update_index=False)
    result = runner.invoke(app, ["reindex", "--metadata-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "2 batch metadata file(s)" in result.output
    assert "3 event(s)" in result.output
    assert set(yaml.safe_load((tmp_path / "signal_index.yaml").read_text())) == {"11", "22", "33"}


def test_reindex_command_exits_non_zero_on_a_directory_it_cannot_rebuild_from(tmp_path: Path) -> None:
    """A refusal must reach the shell as a failure, not a zero exit with a message."""
    result = runner.invoke(app, ["reindex", "--metadata-dir", str(tmp_path)])

    assert result.exit_code == 1, result.output
    assert "No batch metadata files" in result.output
