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

**What happens when the flush cannot be done is the interesting half, and two rounds of review got it
backwards before landing here.** The digest is recorded anyway. Withholding it looks safer -- no digest
describing a rename that might vanish -- but an empty sidecar *is* the permissive legacy path, so a
writer on another host with a stale cached view is accepted and silently discards the entries it could
not see, with no crash needed. Withholding also cannot deliver its own invariant, since clearing happens
after the rename and a crash between them leaves the sidecar's old digest against a possibly-new index,
which refuses exactly as the state it was avoiding does.

So the guard is kept and the durability gap is carried. A crash in that window leaves index and sidecar
disagreeing one way or the other, and either way the next write refuses: loud, and repaired by deleting
the sidecar. Not "a rare window" -- an NFS client holds the rename until it sends the RPC, so on the very
mounts that refuse the flush the exposure is seconds. A loud recoverable failure beats a silent one
whatever the probability, and both failure modes have a test here, because the argument is only as good
as the ordering it is checked against.

**What these tests pin, and what they cannot.** They pin the mechanism: the directory is fsynced, after
the rename, before the digest is recorded. They do **not** demonstrate durability -- that needs power
loss or a crash-consistency harness, and no test here can produce one. The crash tests replay the
resulting *state*, not the crash. Found by a reviewer during R24; the ordering is the part a future
change can silently break.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from gwmock.cli import simulate_utils
from gwmock.cli.simulate_utils import StaleIndexReadError, update_signal_index

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _forget_degraded_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test its own degraded-target set.

    Process state, so without this a test's first refusal is suppressed by an earlier test's. It is a
    fixture rather than a call inside the refusal helper: doing it there is what hid the case where a
    mount recovers and degrades again, since every episode looked like the first.
    """
    monkeypatch.setattr(simulate_utils, "_DEGRADED_TARGETS", set())


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


def _refuse_directory_fsync(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Make fsync of a *directory* fail with ``EINVAL``, as some network mounts do.

    Returns a one-element switch so a test can let the mount recover and degrade again. Flipping the
    switch rather than calling ``monkeypatch.undo()`` is deliberate: ``undo`` reverts *every* patch,
    including the autouse fixture's replacement of the degraded-mount set, which restored the real
    module-level set mid-test and made the toggle test pass whatever the suppression did.
    """
    refusing = [True]
    real_fsync = os.fsync

    def _fsync(fd):
        if refusing[0] and stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(22, "Invalid argument")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fsync)
    return refusing


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
def test_a_refused_flush_still_records_the_digest_of_what_it_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staleness guard survives a filesystem that cannot flush directories.

    An earlier version of this branch withheld the digest here, reasoning that a digest for a rename of
    unknown durability can wedge the directory after a crash. That is the wrong way round, and the test
    below shows what it costs. The durability gap is carried instead, and warned about.
    """
    _refuse_directory_fsync(monkeypatch)

    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
    update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")

    recorded = (tmp_path / "signal_index.yaml.lock").read_text().strip()
    assert recorded == hashlib.sha256((tmp_path / "signal_index.yaml").read_bytes()).hexdigest(), (
        "the sidecar does not describe the index on disk, so the next writer cannot tell a stale read "
        f"from a current one: {recorded!r}"
    )


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_a_stale_reader_is_still_refused_on_a_filesystem_that_cannot_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard still refuses a stale read when the flush fails -- which is what withholding would cost.

    A reviewer supplied this ordering. Writer A's flush is refused; if that withheld the digest, the
    sidecar is empty, and a writer on another host whose cached view predates A's update takes the
    *permissive legacy path* -- it is accepted, and its write discards the entries it could not see. No
    crash, no error, both runs exit 0. Refusing a stale read is loud and costs one batch; believing one
    costs the entries.

    Note which mounts these are. A filesystem that refuses ``fsync`` on a directory is the same class
    that serves stale cached reads, so the guard is being disabled exactly where it is load-bearing.
    """
    _refuse_directory_fsync(monkeypatch)
    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
    update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")

    index_file = tmp_path / "signal_index.yaml"
    assert sorted(yaml.safe_load(index_file.read_text())) == ["1", "2"]

    # The other host's stale view: the index as it was one update ago. Only the index's bytes are
    # patched -- blanking the sidecar too would send the guard down its "no digest" branch and pass for
    # the wrong reason.
    one_update_ago = yaml.safe_dump({"1": {"batches": [], "coa_time": 101.0}}).encode()
    real_read_bytes = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: one_update_ago if self == index_file else real_read_bytes(self),
    )

    with pytest.raises(StaleIndexReadError, match="stale"):
        update_signal_index(tmp_path, _metadata(3, 2), "orchestration-2.metadata.json")

    monkeypatch.undo()
    assert sorted(yaml.safe_load(index_file.read_text())) == ["1", "2"], (
        "the refusal itself damaged the index, which is no better than believing the stale read"
    )


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_a_crash_losing_an_unflushed_rename_refuses_loudly_and_names_the_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap this branch knowingly carries, and the shape of it when it fires.

    Recording the digest for a rename that may not be durable means a crash in that window reboots to
    the *previous* index while the sidecar describes the new one. That state is a refusal, not a silent
    acceptance -- and the message has to carry the repair, because nothing advances the digest by
    itself: only deleting the sidecar clears it.

    Rolling the index bytes back by hand is the closest a test without power loss gets to that state.
    """
    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
    index_file = tmp_path / "signal_index.yaml"
    before_the_lost_write = index_file.read_bytes()

    refusing = _refuse_directory_fsync(monkeypatch)
    update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")
    refusing[0] = False

    # The crash: the rename that installed event 2's index never reached stable storage.
    index_file.write_bytes(before_the_lost_write)

    # The repair, not merely the sidecar's name: a refusal that does not say what to delete leaves the
    # operator with a permanently wedged directory and no instruction.
    with pytest.raises(StaleIndexReadError, match=re.escape("delete signal_index.yaml.lock")):
        update_signal_index(tmp_path, _metadata(3, 2), "orchestration-2.metadata.json")


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_a_directory_that_cannot_be_opened_for_reading_is_unreadable(tmp_path: Path) -> None:
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
            # Root ignores the mode, so the branch genuinely cannot be reached and skipping is honest.
            # The project's CI runs as a non-root user on ubuntu-latest and macos-latest, so this is
            # exercised there -- but a root cell added to that matrix would make these tests vanish
            # silently rather than fail.
            pytest.skip("this user can read a write-only directory, so the refusal cannot be produced")

        assert simulate_utils._fsync_directory(metadata_directory) is simulate_utils._DirectoryFlush.UNREADABLE
    finally:
        metadata_directory.chmod(0o755)


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_an_open_failure_that_is_not_about_permissions_prescribes_no_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``EMFILE`` is not a permission problem, and must not be answered with a permission repair.

    Every ``OSError`` from the open used to be classified as unreadable, whose warning tells the operator
    to widen the directory's mode -- which repairs nothing when the real cause is an exhausted descriptor
    table, a directory removed under the run, or a device error. An automated reviewer caught it: the same
    wrong-repair defect as an earlier round, in the branch that round added.

    Patching ``os.open`` is safe here where patching ``Path.stat`` was not -- nothing else in this path
    calls it with ``O_DIRECTORY``.
    """
    real_open = os.open

    def _refuse_to_open(path, flags, *args, **kwargs):
        if flags & os.O_DIRECTORY:
            raise OSError(errno.EMFILE, "Too many open files")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _refuse_to_open)

    assert simulate_utils._fsync_directory(tmp_path) is simulate_utils._DirectoryFlush.UNAVAILABLE

    with caplog.at_level(logging.WARNING, logger="gwmock.cli.simulate_utils"):
        update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected one warning, got {warnings}"
    assert "could not be opened" in warnings[0]
    assert "permissions" not in warnings[0], (
        f"a descriptor exhaustion was answered with a permissions repair: {warnings[0]}"
    )
    assert "No repair can be named" in warnings[0]


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_the_degradation_warns_where_an_operator_will_see_it_but_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A degradation with no signal is indistinguishable from a guarantee.

    ``_fsync_directory`` logs at ``DEBUG`` and the CLI runs at ``INFO``, so the flush failure alone
    produces nothing an operator sees. The warning has to be at ``WARNING`` -- and once per *filesystem*
    rather than once per batch, because a mount that refuses the flush refuses it for every batch in the
    run and a per-batch warning is how the message gets filtered out. One index is written here, so this
    test cannot tell per-filesystem from per-index; the scope itself is pinned by
    `test_many_directories_on_one_refusing_filesystem_warn_once_between_them`.
    """
    _refuse_directory_fsync(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="gwmock.cli.simulate_utils"):
        update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
        update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")

    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    refusals = [record for record in warnings if "refused to flush the directory" in record.message]
    assert len(refusals) == 1, f"expected exactly one refusal warning across two batches, got {len(refusals)}"
    # Every warning, not only this one. The earlier version counted its own message and called that
    # "only once" while the second batch emitted a *different* warning claiming the index "predates the
    # staleness guard" -- a lie about an index written seconds earlier, and per batch. It came from
    # withholding the digest; counting all warnings is what would have caught it.
    assert warnings == refusals, f"the degradation emitted other warnings too: {[r.message for r in warnings]}"


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_a_mount_that_recovers_and_degrades_again_warns_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An episode that ends and returns is a second episode, not a repeat of the first.

    The suppression was a permanent cache before a reviewer probed this: refuse, recover, refuse again in
    one process produced a single warning, so an intermittently-refusing mount -- a server restart, a
    remount, a flaky FUSE layer -- went silent for every episode after the first. The tests hid it too,
    by resetting the cache inside the refusal helper.
    """
    refusing = _refuse_directory_fsync(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="gwmock.cli.simulate_utils"):
        update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

        refusing[0] = False  # the mount recovers
        update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")

        refusing[0] = True  # and degrades again
        update_signal_index(tmp_path, _metadata(3, 2), "orchestration-2.metadata.json")

    refusals = [r for r in caplog.records if "refused to flush the directory" in r.message]
    assert len(refusals) == 2, f"expected one warning per episode, got {len(refusals)}"


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_many_directories_on_one_refusing_filesystem_warn_once_between_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Refusing to flush is a property of the filesystem, so it is reported once for the filesystem.

    Keyed per index, a run over twenty metadata directories on one refusing mount emitted twenty copies
    of a multi-line warning and kept twenty cache entries for the life of the process. A reviewer
    measured both. Keying on ``st_dev`` collapses them. These four directories share one device, so
    what is demonstrated here is the collapse and nothing more -- the path-spelling collisions the other
    reviewer found are asserted for the directory scope by
    `test_one_relative_spelling_in_two_working_directories_is_two_directories`, not here.
    """
    directories = [tmp_path / f"run-{index}" for index in range(4)]
    for directory in directories:
        directory.mkdir()

    _refuse_directory_fsync(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="gwmock.cli.simulate_utils"):
        for index, directory in enumerate(directories):
            update_signal_index(directory, _metadata(index + 1, index), f"orchestration-{index}.metadata.json")

    refusals = [r for r in caplog.records if "refused to flush the directory" in r.message]
    assert len(refusals) == 1, (
        f"one refusing filesystem produced {len(refusals)} warnings across {len(directories)} directories"
    )


def test_an_unidentifiable_filesystem_still_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Losing the device number must not lose the warning.

    Suppression is keyed on ``st_dev``, which is itself a syscall that can fail -- a metadata directory
    removed under a running job is the ordinary way. Skipping the warning when the key cannot be read
    would silence exactly the degraded setups this reports on, so an unreadable device becomes its own
    key instead. A mutation that returned early there passed every other test in this file.

    Calls the helper directly on a directory that does not exist, rather than patching ``Path.stat``: the
    patch also caught ``mkdir(exist_ok=True)``, which stats to decide whether the existing path is a
    directory, and the test failed in its own setup.
    """
    missing = tmp_path / "unmounted"

    with caplog.at_level(logging.WARNING, logger="gwmock.cli.simulate_utils"):
        simulate_utils._note_flush_outcome(
            missing,
            simulate_utils._DirectoryFlush.REFUSED,
            missing / "signal_index.yaml",
            "signal_index.yaml.lock",
        )

        # Twice, deliberately. One call warns under any suppression at all, so a single warning cannot
        # tell "keyed as its own scope" from "warns every time"; the second call distinguishes them, and
        # it does so through the log rather than by reading the private set.
        simulate_utils._note_flush_outcome(
            missing,
            simulate_utils._DirectoryFlush.REFUSED,
            missing / "signal_index.yaml",
            "signal_index.yaml.lock",
        )

    refusals = [r for r in caplog.records if "refused to flush the directory" in r.message]
    assert len(refusals) == 1, f"expected one warning across two calls, got {len(refusals)}"


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_two_unreadable_directories_on_one_filesystem_each_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A directory that cannot be opened is its own fault, not the filesystem's.

    Both share ``st_dev``, so keying this by device reported the first and silently swallowed the
    second -- and since the cause is that directory's own permissions, the operator never saw the repair
    for it. A reviewer supplied the input. The mode is real rather than patched: root ignores it, so the
    test skips instead of faking a refusal it cannot produce.
    """
    directories = [tmp_path / "run-a", tmp_path / "run-b"]
    for directory in directories:
        directory.mkdir()
        directory.chmod(0o333)
    try:
        try:
            os.close(os.open(directories[0], os.O_RDONLY | os.O_DIRECTORY))
        except PermissionError:
            pass
        else:
            # Root ignores the mode, so the branch genuinely cannot be reached and skipping is honest.
            # The project's CI runs as a non-root user on ubuntu-latest and macos-latest, so this is
            # exercised there -- but a root cell added to that matrix would make these tests vanish
            # silently rather than fail.
            pytest.skip("this user can read a write-only directory, so the refusal cannot be produced")

        assert directories[0].stat().st_dev == directories[1].stat().st_dev, "the input needs one filesystem"

        with caplog.at_level(logging.WARNING, logger="gwmock.cli.simulate_utils"):
            for directory in directories:
                simulate_utils._note_flush_outcome(
                    directory,
                    simulate_utils._fsync_directory(directory),
                    directory / "signal_index.yaml",
                    "signal_index.yaml.lock",
                )
    finally:
        for directory in directories:
            directory.chmod(0o755)

    unreadable = [r for r in caplog.records if "could not be opened to flush" in r.message]
    assert len(unreadable) == 2, (
        f"two directories with their own permission problem produced {len(unreadable)} warnings"
    )


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_a_directory_whose_permissions_are_fixed_and_broken_again_warns_again(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The directory scope needs its own clear-on-success, exactly as the filesystem scope does.

    A mutation that cleared only the filesystem scope survived every other test: an operator who fixes
    the directory's mode, and later breaks it again, would then never be told a second time. Same defect
    as the permanent cache this branch already replaced once, in the scope added last.
    """
    directory = tmp_path / "run"
    directory.mkdir()

    def note() -> None:
        simulate_utils._note_flush_outcome(
            directory,
            simulate_utils._fsync_directory(directory),
            directory / "signal_index.yaml",
            "signal_index.yaml.lock",
        )

    directory.chmod(0o333)
    try:
        try:
            os.close(os.open(directory, os.O_RDONLY | os.O_DIRECTORY))
        except PermissionError:
            pass
        else:
            # Root ignores the mode, so the branch genuinely cannot be reached and skipping is honest.
            # The project's CI runs as a non-root user on ubuntu-latest and macos-latest, so this is
            # exercised there -- but a root cell added to that matrix would make these tests vanish
            # silently rather than fail.
            pytest.skip("this user can read a write-only directory, so the refusal cannot be produced")

        with caplog.at_level(logging.WARNING, logger="gwmock.cli.simulate_utils"):
            note()
            directory.chmod(0o755)  # the operator fixes it
            note()
            directory.chmod(0o333)  # and it breaks again
            note()
    finally:
        directory.chmod(0o755)

    unreadable = [r for r in caplog.records if "could not be opened to flush" in r.message]
    assert len(unreadable) == 2, f"expected one warning per episode, got {len(unreadable)}"


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="directory fsync is POSIX-only")
def test_one_relative_spelling_in_two_working_directories_is_two_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The directory key is the absolute path, because the same spelling can name different directories.

    A reviewer supplied this: keyed on the string it was handed, ``Path("metadata")`` reached from two
    working directories is one key, so the second directory's permission problem is suppressed and its
    repair never shown. The reviewers disagreed on whether the CLI can reach it -- the other argued the
    path passed in is stable, making duplication the only live direction -- and the disagreement is
    resolved by fixing the key, since suppressing a warning is the direction that costs a repair.
    """
    for name in ("a", "b"):
        (tmp_path / name / "metadata").mkdir(parents=True)
        (tmp_path / name / "metadata").chmod(0o333)
    try:
        try:
            os.close(os.open(tmp_path / "a" / "metadata", os.O_RDONLY | os.O_DIRECTORY))
        except PermissionError:
            pass
        else:
            pytest.skip("this user can read a write-only directory, so the refusal cannot be produced")

        with caplog.at_level(logging.WARNING, logger="gwmock.cli.simulate_utils"):
            for name in ("a", "b"):
                monkeypatch.chdir(tmp_path / name)
                relative = Path("metadata")
                simulate_utils._note_flush_outcome(
                    relative,
                    simulate_utils._fsync_directory(relative),
                    relative / "signal_index.yaml",
                    "signal_index.yaml.lock",
                )
    finally:
        for name in ("a", "b"):
            (tmp_path / name / "metadata").chmod(0o755)

    unreadable = [r for r in caplog.records if "could not be opened to flush" in r.message]
    assert len(unreadable) == 2, f"two directories sharing one relative spelling produced {len(unreadable)} warnings"


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


def test_a_platform_without_directory_flushing_records_its_digest_and_stays_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The staleness guard stays on where durability cannot be verified at all, and says nothing.

    Every flush outcome records now, so the digest half is the same policy as a refused flush rather
    than an exception to it. It is pinned separately because the platform reaches the decision by a
    different branch, and because a change that withholds the digest on either would have to break this
    test explicitly rather than by omission.

    The silence is the other half, and it needs its own assertion: a mutation that warned on *every*
    non-flushed outcome passed every other test in this file. Warning here would fire on every run of
    every batch on a platform whose gap the operator cannot close, which is how the warnings that can
    be acted on get filtered out.
    """
    monkeypatch.delattr(os, "O_DIRECTORY", raising=False)

    with caplog.at_level(logging.WARNING, logger="gwmock.cli.simulate_utils"):
        update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    recorded = (tmp_path / "signal_index.yaml.lock").read_text().strip()
    assert recorded, "the digest was withheld on a platform that cannot flush directories at all"
    assert recorded == hashlib.sha256((tmp_path / "signal_index.yaml").read_bytes()).hexdigest(), (
        "the recorded digest does not describe the index that was committed"
    )
    warnings = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
    assert warnings == [], f"a platform gap the operator cannot close warned anyway: {warnings}"
