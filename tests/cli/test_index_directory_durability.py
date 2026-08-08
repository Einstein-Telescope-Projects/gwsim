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

**What these tests pin, and what they cannot.** They pin the mechanism: that the directory is fsynced,
after the rename, before the function returns its digest. They do **not** demonstrate durability --
that needs power loss or a crash-consistency harness, and no test here can produce one. Found by a
reviewer during R24; the ordering is the part a future change can silently break.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from gwmock.cli import simulate_utils
from gwmock.cli.simulate_utils import update_signal_index

pytestmark = pytest.mark.unit


def _metadata(event_id: int, batch: int) -> dict[str, Any]:
    """One batch's metadata injecting a single distinct event."""
    return {
        "signal": {"injections": [{"event_id": event_id, "parameters": {"coa_time": 100.0 + event_id}}]},
        "outputs": [{"kind": "signal", "path": f"signal/signal-{batch}.gwf"}],
    }


def _track_directory_fds(monkeypatch: pytest.MonkeyPatch) -> set[int]:
    """Return a live set of open directory descriptors.

    Descriptors are **recycled**: the first version of this helper added directory fds to a set and
    never removed them, so the sidecar reused a closed directory's number and the fault injection
    below hit the digest write instead. Closes are tracked for that reason.
    """
    directory_fds: set[int] = set()
    real_open, real_close = os.open, os.close

    def _spy_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if flags & getattr(os, "O_DIRECTORY", 0):
            directory_fds.add(descriptor)
        return descriptor

    def _spy_close(descriptor):
        directory_fds.discard(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(os, "open", _spy_open)
    monkeypatch.setattr(os, "close", _spy_close)
    return directory_fds


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_the_directory_is_flushed_after_the_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Order is the assertion: rename, then directory fsync, then the digest is returned.

    Flushing before the rename would be useless -- the entry it needs to persist does not exist yet --
    and flushing after ``_record_digest`` would leave exactly the window this closes. A spy records
    the sequence rather than the outcome, because the outcome is indistinguishable without a crash.
    """
    events: list[str] = []
    real_replace, real_fsync = os.replace, os.fsync
    directory_fds = _track_directory_fds(monkeypatch)

    def _spy_replace(src, dst, *args, **kwargs):
        events.append("replace")
        return real_replace(src, dst, *args, **kwargs)

    def _spy_fsync(fd):
        events.append("fsync-directory" if fd in directory_fds else "fsync-file")
        return real_fsync(fd)

    monkeypatch.setattr(os, "replace", _spy_replace)
    monkeypatch.setattr(os, "fsync", _spy_fsync)

    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    assert "replace" in events, "the index was never renamed into place"
    assert "fsync-directory" in events, "the directory was not flushed, so the rename is not durable"
    assert events.index("replace") < events.index("fsync-directory"), (
        f"the directory was flushed before the rename it must persist: {events}"
    )
    # The sidecar's digest write is the last fsync, so the directory flush must precede it.
    assert events.index("fsync-directory") < len(events) - 1, (
        f"nothing followed the directory flush, so the digest write was not after it: {events}"
    )


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_a_filesystem_refusing_the_flush_does_not_fail_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index is already committed when the flush runs, so refusing it must not raise.

    Some network mounts reject an fsync on a directory. Turning that into an exception would report a
    failed update to a caller whose data had in fact landed -- strictly worse than the durability gap
    this closes, because the caller would then retry or abort on a false negative.
    """
    real_fsync = os.fsync
    directory_fds = _track_directory_fds(monkeypatch)

    def _refuse_directory_fsync(fd):
        if fd in directory_fds:
            raise OSError(22, "Invalid argument")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _refuse_directory_fsync)

    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
    update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")

    import yaml

    index = yaml.safe_load((tmp_path / "signal_index.yaml").read_text())
    assert sorted(index) == ["1", "2"], f"a refused directory flush cost an event: {sorted(index)}"


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

    simulate_utils._fsync_directory(tmp_path)  # must not raise
    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    assert (tmp_path / "signal_index.yaml").exists()
