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

"""Making the index's rename durable before its digest describes it.

``_atomically_write_index`` fsyncs the temporary's contents and then ``os.replace``s it into place.
That is atomic with respect to readers, but atomicity is not durability: the directory entry can sit
in the page cache while ``_record_digest`` writes the sidecar. A crash in that window leaves the
**old** index with the **new** digest, and every later write refuses as stale against a good file --
loud, recoverable by deleting the sidecar, and needing an operator.

Flushing is only half of it, and the half a reviewer had to point out. When the flush cannot be done,
the digest must not be recorded either -- a digest describing a name that may not survive is the same
wedge with an extra step. Nor may the *previous* digest simply be left in place: it then describes the
old index while the new one is installed, wedged in the other direction. Of the three states a crash
can leave -- old digest, new digest, no digest -- only the last is accepted whichever index survives,
so a refused flush clears it. A platform with no directory-flush primitive at all is treated
differently, and `test_a_platform_without_directory_flushing_still_records_its_digest` says why.

**What these tests pin, and what they cannot.** They pin the mechanism: that the directory is fsynced,
after the rename, before the digest is recorded, and that the digest is withheld when it is not. They
do **not** demonstrate durability -- that needs power loss or a crash-consistency harness, and no test
here can produce one. The rollback test replays the resulting *state*, not the crash. Found by a
reviewer during R24; the ordering is the part a future change can silently break.
"""

from __future__ import annotations

import hashlib
import logging
import os
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from gwmock.cli import simulate_utils
from gwmock.cli.simulate_utils import update_signal_index

pytestmark = pytest.mark.unit


def _metadata(event_id: int, batch: int) -> dict[str, Any]:
    """One batch's metadata injecting a single distinct event."""
    return {
        "signal": {"injections": [{"event_id": event_id, "parameters": {"coa_time": 100.0 + event_id}}]},
        "outputs": [{"kind": "signal", "path": f"signal/signal-{batch}.gwf"}],
    }


def _is(descriptor: int, path: Path) -> bool:
    """Whether *descriptor* is open on *path*, by the file's identity on disk.

    Three identifications were tried here and two were wrong. The ``O_DIRECTORY`` flag was wrong in
    both directions: flushing the **wrong** directory -- ``index_file.parent.parent`` -- still produced
    an ``O_DIRECTORY`` open and passed every assertion, while a correct refactor to
    ``os.open(directory, os.O_RDONLY)`` would have failed them. Spying on ``os.open`` to record paths
    was wrong too, and quietly: ``Path.open`` reaches the ``open()`` syscall through ``io.FileIO`` in C
    and never calls the ``os.open`` *Python* function, so the sidecar's descriptor was invisible and the
    assertion about it could not fail.

    ``st_dev``/``st_ino`` is neither: it survives any refactor of how the file is opened, and it cannot
    confuse two different files. Recycled descriptors stop mattering, because the identity is read from
    the descriptor at the moment it is used.
    """
    try:
        opened, target = os.fstat(descriptor), path.stat()
    except OSError:
        return False
    return (opened.st_dev, opened.st_ino) == (target.st_dev, target.st_ino)


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_the_directory_is_flushed_after_the_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Order is the assertion: rename, then directory fsync, then the digest is returned.

    Flushing before the rename would be useless -- the entry it needs to persist does not exist yet --
    and flushing after ``_record_digest`` would leave exactly the window this closes. A spy records
    the sequence rather than the outcome, because the outcome is indistinguishable without a crash.
    """
    events: list[str] = []
    real_replace, real_fsync = os.replace, os.fsync
    sidecar = tmp_path / "signal_index.yaml.lock"

    def _spy_replace(src, dst, *args, **kwargs):
        result = real_replace(src, dst, *args, **kwargs)
        events.append("replace")
        return result

    def _spy_fsync(fd):
        # Recorded *after* the call, and only if it returns. Recording first meant a host whose
        # directory fsync fails -- a write-only directory, a refusing mount -- logged the flush as
        # having happened, because the production code swallows the error. The test then passed while
        # nothing was flushed.
        result = real_fsync(fd)
        if _is(fd, tmp_path):
            events.append("fsync-directory")
        elif _is(fd, sidecar):
            events.append("fsync-sidecar")
        else:
            events.append("fsync-file")
        return result

    monkeypatch.setattr(os, "replace", _spy_replace)
    monkeypatch.setattr(os, "fsync", _spy_fsync)

    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    assert "replace" in events, "the index was never renamed into place"
    assert "fsync-directory" in events, (
        f"the index's own directory was not flushed, so the rename is not durable: {events}"
    )
    assert events.index("replace") < events.index("fsync-directory"), (
        f"the directory was flushed before the rename it must persist: {events}"
    )
    # Named explicitly rather than "something followed the flush", which was the weaker assertion an
    # earlier version made while its comment claimed this one.
    assert "fsync-sidecar" in events, f"the digest was never fsynced, so the order is untested: {events}"
    assert events.index("fsync-directory") < events.index("fsync-sidecar"), (
        f"the digest reached the disk before the rename it describes: {events}"
    )


def _refuse_directory_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every fsync of a *directory* fail with ``EINVAL``, as some network mounts do."""
    real_fsync = os.fsync

    def _fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(22, "Invalid argument")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fsync)
    simulate_utils._warn_flush_refused_once.cache_clear()


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_a_filesystem_refusing_the_flush_does_not_fail_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index is already committed when the flush runs, so refusing it must not raise.

    Some network mounts reject an fsync on a directory. Turning that into an exception would report a
    failed update to a caller whose data had in fact landed -- strictly worse than the durability gap
    this closes, because the caller would then retry or abort on a false negative.
    """
    _refuse_directory_fsync(monkeypatch)

    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
    update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")

    index = yaml.safe_load((tmp_path / "signal_index.yaml").read_text())
    assert sorted(index) == ["1", "2"], f"a refused directory flush cost an event: {sorted(index)}"


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_a_refused_flush_clears_the_digest_instead_of_recording_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest may only be recorded for a rename known to be durable, and the old one must go.

    Surviving the write is not the property that matters -- recording the digest anyway also survives,
    which is why the assertion here is on the sidecar and not on the index.

    The first write is flushed normally so a digest is actually **there** to be cleared. Without it,
    this test starts from the empty sidecar the lock creates and cannot tell clearing from doing
    nothing: a mutation that dropped the clear and kept only the warning passed.
    """
    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
    sidecar = tmp_path / "signal_index.yaml.lock"
    assert sidecar.read_text().strip(), "the first write recorded no digest, so there is nothing to clear"

    _refuse_directory_fsync(monkeypatch)
    update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")

    assert sidecar.read_text().strip() == "", (
        "the sidecar still holds a digest after an unflushed write. Either it describes the index this "
        "update replaced -- wedging the next write against a good file -- or it describes a name that "
        f"may not survive a crash: {sidecar.read_text()!r}"
    )


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_the_next_write_after_an_unflushed_one_is_not_refused_as_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keeping the previous digest wedges the directory with no crash involved at all.

    This is the ordering that makes "just don't record it" wrong, and it needs no power loss: the
    unflushed rename *does* land, so the sidecar's older digest now describes an index that has been
    replaced. The next write reads a good index, finds it does not match, and raises
    ``StaleIndexReadError`` -- permanently, since nothing advances the digest but a successful write.
    """
    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    _refuse_directory_fsync(monkeypatch)
    update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")
    monkeypatch.undo()

    update_signal_index(tmp_path, _metadata(3, 2), "orchestration-2.metadata.json")

    index = yaml.safe_load((tmp_path / "signal_index.yaml").read_text())
    assert sorted(index) == ["1", "2", "3"], f"the write after an unflushed one did not land: {sorted(index)}"


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_a_directory_that_cannot_be_opened_for_reading_is_a_refusal(tmp_path: Path) -> None:
    """The ``os.open`` failure is a real input, not a defensive branch.

    A write-only metadata directory -- mode ``0o333`` -- accepts the ``os.replace`` that installs the
    index but refuses ``os.open(directory, O_RDONLY | O_DIRECTORY)`` with ``EACCES``. A reviewer
    supplied this case; the branch had no test and was the one still marked unreachable.
    """
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()
    metadata_directory.chmod(0o333)
    try:
        # The real permission, not a patched `os.open`: root ignores the mode, and a mocked refusal
        # would pass on a host where the branch cannot actually be reached.
        try:
            os.close(os.open(metadata_directory, os.O_RDONLY | os.O_DIRECTORY))
        except PermissionError:
            pass
        else:
            pytest.skip("this user can read a write-only directory, so the refusal cannot be produced")

        assert simulate_utils._fsync_directory(metadata_directory) is simulate_utils._DirectoryFlush.REFUSED
    finally:
        metadata_directory.chmod(0o755)


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_the_degradation_warns_where_an_operator_will_see_it_but_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A degradation with no signal is indistinguishable from a guarantee.

    ``_fsync_directory`` logs at ``DEBUG`` and the CLI runs at ``INFO``, so the flush failure alone
    produces nothing an operator sees. The warning has to be at ``WARNING`` -- and once per index, not
    once per batch, because a mount that refuses the flush refuses it for every batch in the run and a
    per-batch warning is how the message gets filtered out.
    """
    _refuse_directory_fsync(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="gwmock.cli.simulate_utils"):
        update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
        update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")

    refusals = [record for record in caplog.records if "refused to flush the directory" in record.message]
    assert len(refusals) == 1, f"expected exactly one warning across two batches, got {len(refusals)}"
    assert refusals[0].levelno == logging.WARNING


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_the_directory_does_not_wedge_when_an_unflushed_rename_is_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The crash ordering itself, replayed: an unflushed rename lost, the sidecar surviving.

    A crash after ``os.replace`` but before its directory entry reaches the disk can reboot to the
    *previous* index -- while the sidecar, written in place, keeps whatever digest it holds. Rolling the
    index bytes back by hand is the closest a test without power loss gets to that state.

    With a digest recorded, the next write reads an index that does not match it and raises
    ``StaleIndexReadError`` against a perfectly good file, and the directory stays wedged until an
    operator deletes the sidecar. With the digest cleared, the same state is the permissive legacy path:
    it warns and proceeds.
    """
    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
    index_file = tmp_path / "signal_index.yaml"
    before_the_lost_write = index_file.read_bytes()

    _refuse_directory_fsync(monkeypatch)
    update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")
    monkeypatch.undo()

    # The crash: the rename that installed event 2's index never reached stable storage.
    index_file.write_bytes(before_the_lost_write)

    update_signal_index(tmp_path, _metadata(3, 2), "orchestration-2.metadata.json")

    index = yaml.safe_load(index_file.read_text())
    assert sorted(index) == ["1", "3"], f"the write after the rolled-back rename did not land: {sorted(index)}"


def test_the_flush_is_skipped_where_the_platform_has_no_directory_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows cannot open a directory with ``os.open``, and no host here can test the alternative.

    A reviewer proposed a ``FlushFileBuffers`` path through ``ctypes``. It is deliberately absent:
    untested platform code that looks like protection is worse than a documented gap. What is pinned
    here is that its absence is *handled* -- the write still succeeds -- rather than raising an
    ``AttributeError`` on the platform the branch exists for.
    """
    monkeypatch.delattr(os, "O_DIRECTORY", raising=False)

    assert simulate_utils._fsync_directory(tmp_path) is simulate_utils._DirectoryFlush.UNSUPPORTED
    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    assert (tmp_path / "signal_index.yaml").exists()


def test_a_platform_without_directory_flushing_still_records_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staleness guard stays on where durability cannot be verified at all.

    Distinct from a *refused* flush, which clears the digest. A platform with no directory-flush
    primitive can never satisfy the check, so clearing there would disable the cross-host guard on
    every write for good -- a certain loss against the uncertain one of a crash in a sub-millisecond
    window. Without this test, clearing unconditionally passes every other test in this file.
    """
    monkeypatch.delattr(os, "O_DIRECTORY", raising=False)

    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    recorded = (tmp_path / "signal_index.yaml.lock").read_text().strip()
    assert recorded, "the digest was cleared on a platform that cannot flush directories at all"
    assert recorded == hashlib.sha256((tmp_path / "signal_index.yaml").read_bytes()).hexdigest(), (
        "the recorded digest does not describe the index that was committed"
    )
