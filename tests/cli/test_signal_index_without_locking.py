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

"""Updating the signal index where no lock can be taken.

Two branches degrade to unsynchronised writes: no ``fcntl`` module at all (Windows), and a
filesystem that rejects ``flock`` (some network mounts). Both are deliberate -- failing there would
break a write that succeeded before locking existed, on exactly the shared filesystems the locking
was aimed at.

**Both were unreachable from the test suite, and it showed.** The ``fcntl is None`` branch broke
three times in a single day, each time while something else was being fixed, and each time it was
caught by a reviewer reading the diff rather than by anything failing:

1. ``os.fchmod`` -- Unix-only, so the second write failed.
2. the digest re-read.
3. ``_record_digest`` opening ``"r+"`` on a file that branch never creates, which would have failed
   *every* update after the index was committed, with a recovery message naming a file that does
   not exist.

Three defects, zero test failures, because the branch carried ``# pragma: no cover`` and the CI
matrix is Linux and macOS. What is pinned here is the behaviour the hand-probe checked on
2026-08-07: both events are written, the recorded digest matches what is on disk, and the warning
arrives once.

The lock *itself* cannot be tested here -- there is nothing to exclude against without a second
writer, and the concurrency tests cover that where locking works. These tests pin the part that
kept breaking: that the unlocked path still writes a correct index.
"""

from __future__ import annotations

import errno
import hashlib
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


@pytest.fixture(autouse=True)
def _reset_the_once_only_warning() -> None:
    """Clear the cached warning so each test sees its own.

    ``_warn_unlocked_once`` is ``@cache``d, which is how the "warns once" claim is enforced -- but
    the cache is process-global, so without this the second test to run would observe no warning and
    a "warns once" assertion would pass or fail on test *order* rather than on behaviour.
    """
    simulate_utils._warn_unlocked_once.cache_clear()


def _without_fcntl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present Windows semantics for the calls this path actually makes.

    **What this does and does not claim.** An earlier version of this docstring said "any Unix-only
    call on this branch now raises the ``AttributeError`` it would raise there". That was false, and
    a reviewer showed why: deleting ``os.fchmod`` and ``os.fchown`` is name-removal, not a platform.
    Worse, both are *off-path* -- breakage 1's own fix replaced ``fchmod`` with ``os.chmod``, so the
    deletions guard two names nothing executes. They are kept as tripwires against those exact names
    returning, and labelled as such rather than as a simulation.

    What is simulated is the one live difference on this path: ``os.chmod`` accepts a *descriptor* on
    POSIX (it forwards to ``fchmod``) and requires a *path* on Windows, where an fd raises. The code
    holds a descriptor in scope while chmod'ing by path, and the comment there frames path-versus-
    descriptor as a considered trade -- so the natural cleanup to ``os.chmod(descriptor, ...)`` would
    break every write on Windows and, before this, passed every test here.

    Still not simulated, and known: open-handle sharing semantics, which need a real second writer
    that no POSIX temporary directory can produce.
    """
    monkeypatch.setattr(simulate_utils, "fcntl", None)
    monkeypatch.delattr(simulate_utils.os, "fchmod", raising=False)
    monkeypatch.delattr(simulate_utils.os, "fchown", raising=False)

    real_chmod = simulate_utils.os.chmod

    def _path_only_chmod(target, mode, *args, **kwargs):
        if isinstance(target, int):
            raise NotImplementedError("chmod: file descriptors are not supported on this platform")
        return real_chmod(target, mode, *args, **kwargs)

    monkeypatch.setattr(simulate_utils.os, "chmod", _path_only_chmod)


def _with_flock_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present a filesystem whose ``flock`` fails the way a network mount's does."""

    class _RejectingFcntl:
        LOCK_EX = 2
        LOCK_UN = 8

        @staticmethod
        def flock(_fileno: int, _operation: int) -> None:
            raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(simulate_utils, "fcntl", _RejectingFcntl)


@pytest.mark.parametrize("degrade", [_without_fcntl, _with_flock_unsupported], ids=["no-fcntl", "flock-rejected"])
def test_both_events_are_written_without_a_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, degrade) -> None:
    """Two sequential updates must both survive, which is what breakage 1 and 3 destroyed.

    ``os.fchmod`` failing made the *second* write fail; ``_record_digest`` opening ``"r+"`` made
    every update after the first fail. One update would have passed either way, so two are used.
    """
    degrade(monkeypatch)

    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
    update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")

    index = yaml.safe_load((tmp_path / "signal_index.yaml").read_text())
    # Keys as the index stores them -- strings, since YAML round-trips them that way and
    # `find_signals` reads them as such. Comparing against ints passed nothing and proved nothing.
    assert sorted(index) == ["1", "2"], f"the unlocked path lost an event: {sorted(index)}"


@pytest.mark.parametrize("degrade", [_without_fcntl, _with_flock_unsupported], ids=["no-fcntl", "flock-rejected"])
def test_the_recorded_digest_matches_what_is_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, degrade) -> None:
    """The digest guard has to stay true here too, or the next run refuses a good index.

    This is breakage 2. A digest recorded from anything other than the bytes actually written makes
    the *next* update raise ``StaleIndexReadError`` against an index that is perfectly current --
    the guard turning on its own side.
    """
    degrade(monkeypatch)

    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
    update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")

    index_file = tmp_path / "signal_index.yaml"
    sidecar = index_file.with_name(index_file.name + ".lock")
    on_disk = hashlib.sha256(index_file.read_bytes()).hexdigest()

    # The digest lives in the *sidecar*, not the index -- consumers read the index's keys as event
    # ids, so a bookkeeping key there would be read as an event.
    assert simulate_utils._recorded_digest(sidecar) == on_disk, (
        "the recorded digest does not describe the index on disk, so the next run will refuse it"
    )


@pytest.mark.parametrize("degrade", [_without_fcntl, _with_flock_unsupported], ids=["no-fcntl", "flock-rejected"])
def test_the_unlocked_warning_arrives_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, degrade
) -> None:
    """Once per process, not once per update: a per-write warning would bury the run's output."""
    degrade(monkeypatch)

    with caplog.at_level("WARNING"):
        for batch in range(3):
            update_signal_index(tmp_path, _metadata(batch + 1, batch), f"orchestration-{batch}.metadata.json")

    unlocked = [record for record in caplog.records if "without a lock" in record.message]
    assert len(unlocked) == 1, f"three updates produced {len(unlocked)} warnings"


@pytest.mark.parametrize("degrade", [_without_fcntl, _with_flock_unsupported], ids=["no-fcntl", "flock-rejected"])
def test_the_warning_is_once_per_process_not_once_per_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, degrade
) -> None:
    """Two directories in one process still warn once, which the previous test cannot show.

    A reviewer named the gap precisely: the test above gives each scenario one directory and one call
    site, so a regression that keyed the cache per-directory -- or per-call-site, which a real run
    updating both the signal and metadata indexes would hit -- would warn twice in production and
    still pass. Two directories in a single test is what distinguishes "once per process" from "once
    per whatever this test happened to use".
    """
    degrade(monkeypatch)
    first, second = tmp_path / "run-a", tmp_path / "run-b"
    first.mkdir()
    second.mkdir()

    with caplog.at_level("WARNING"):
        update_signal_index(first, _metadata(1, 0), "orchestration-0.metadata.json")
        update_signal_index(second, _metadata(2, 0), "orchestration-0.metadata.json")

    unlocked = [record for record in caplog.records if "without a lock" in record.message]
    assert len(unlocked) == 1, f"two directories produced {len(unlocked)} warnings; the cache is not per-process"


def test_the_digest_sidecar_is_created_even_though_the_lock_path_never_opens_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Breakage 3, pinned from the side that actually matters.

    The no-fcntl branch yields *before* opening the lock file, so when ``_record_digest`` ran it the
    sidecar did not exist -- and it opened ``"r+"``, which requires one. That failed every update
    after the index was committed, with a recovery message naming a file that was not there. The fix
    was a ``"w"`` fallback, so the sidecar must end up present, holding a digest, on a branch that
    never opens it for locking.

    Asserted here rather than in the digest test above because it is a different claim: that test
    checks the *value*, this one checks that the file exists at all. The first version of this test
    asserted the opposite -- that no sidecar is created -- which a probe disproved before it was
    committed.
    """
    _without_fcntl(monkeypatch)

    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    sidecar = tmp_path / "signal_index.yaml.lock"
    assert sidecar.exists(), "the digest sidecar was not created, so the next update cannot verify"
    assert simulate_utils._recorded_digest(sidecar) is not None, "the sidecar exists but records nothing"


def test_the_index_is_opened_in_binary_mode_where_the_platform_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The atomic write must request ``O_BINARY``, which POSIX cannot notice on its own.

    A reviewer found this by asking the question this file's fixture was built to answer and getting
    the wrong answer: removing ``fcntl``, ``fchmod`` and ``fchown`` still permitted a Unix-only
    assumption. ``os.open`` was called without ``O_BINARY``, so on Windows the descriptor opens in
    the CRT's text mode -- and that translation sits *beneath* ``os.fdopen(..., "wb")``. The bytes on
    disk would be CRLF while the digest describes LF, so every update after the first would refuse a
    correct index as stale. Exactly the failure the code's own comment claimed to have prevented.

    POSIX has no ``O_BINARY``, so this presents a platform that does -- as Windows does -- and
    asserts the flag is requested. The spy strips it again before the real call, since the value
    means something else here.
    """
    fake_o_binary = 0x8000
    monkeypatch.setattr(simulate_utils.os, "O_BINARY", fake_o_binary, raising=False)
    seen: list[int] = []
    real_open = simulate_utils.os.open

    def _spy(path, flags, *args, **kwargs):
        seen.append(flags)
        return real_open(path, flags & ~fake_o_binary, *args, **kwargs)

    monkeypatch.setattr(simulate_utils.os, "open", _spy)

    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    assert seen, "the atomic write did not go through os.open, so this test proves nothing"
    assert all(flags & fake_o_binary for flags in seen), (
        "the index was opened without O_BINARY, so on Windows the bytes written would not be the "
        "bytes hashed and the next update would refuse the index as stale"
    )
