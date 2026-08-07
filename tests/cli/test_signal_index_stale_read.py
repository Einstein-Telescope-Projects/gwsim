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

"""A writer must not build on an index read it cannot trust.

Holding the lock is not enough when the writers are on different hosts. Measured on a shared NFS
home: the lock excluded correctly -- the second writer blocked 26.7 s -- and the update was lost
anyway, because acquiring the lock revalidates the *sidecar*, not ``signal_index.yaml``. The
second writer read its client's cached view, found nothing, and wrote a file holding only its own
event.

The repair records, in the sidecar, a digest of the index that sidecar corresponds to. A writer
that reads an index whose digest disagrees knows its read is stale before it acts on it. The
digest lives in the sidecar rather than in the index because consumers iterate the index's keys
as event ids (``find_signals``, ``_withdraw_batch``), so a bookkeeping key there would be read as
an event.

**These tests simulate the stale read locally** by feeding the reader an out-of-date view. The
real fault needs two hosts and a cache, which no single-host test can produce -- that is the
whole reason it survived #323's review. The cross-host harness lives outside the repo; what is
pinned here is that a stale read is *detected* rather than silently believed.
"""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from gwmock.cli import simulate_utils
from gwmock.cli.simulate_utils import (
    IndexDigestNotRecordedError,
    StaleIndexReadError,
    retry_with_backoff,
    update_signal_index,
)

pytestmark = pytest.mark.unit


def _metadata(event_id: int, batch: int) -> dict[str, Any]:
    """One batch's metadata injecting a single distinct event."""
    return {
        "signal": {"injections": [{"event_id": event_id, "parameters": {"coa_time": 100.0 + event_id}}]},
        "outputs": [{"kind": "signal", "path": f"signal/signal-{batch}.gwf"}],
    }


def test_a_stale_index_read_is_refused_not_believed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader whose index view predates the recorded digest must refuse to write.

    This is the cross-host bug in miniature: the writer's entries are on the server, this reader
    cannot see them, and writing anyway is what discards them.
    """
    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")
    update_signal_index(tmp_path, _metadata(2, 1), "orchestration-1.metadata.json")
    index_file = tmp_path / "signal_index.yaml"
    assert set(yaml.safe_load(index_file.read_text())) == {"1", "2"}

    # Simulate the stale client: this process reads the index as it was one update ago. Patch
    # ONLY the index's bytes -- an earlier version of this test patched Path.read_text globally
    # and so blanked the sidecar too, which sent the guard down its "no digest recorded" branch
    # and made the test pass for the wrong reason.
    one_update_ago = yaml.safe_dump({"1": {"batches": [], "coa_time": 101.0}}).encode()
    real_read_bytes = Path.read_bytes

    def _stale_index_bytes(self: Path) -> bytes:
        return one_update_ago if self == index_file else real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _stale_index_bytes)

    with pytest.raises(StaleIndexReadError, match="stale"):
        update_signal_index(tmp_path, _metadata(3, 2), "orchestration-2.metadata.json")

    monkeypatch.undo()
    # The refusal must leave the index exactly as it was -- refusing and corrupting is no better
    # than believing the stale read.
    assert set(yaml.safe_load(index_file.read_text())) == {"1", "2"}


def test_an_index_that_matches_its_digest_is_accepted(tmp_path: Path) -> None:
    """The guard must not fire on the ordinary path, or it is just an outage."""
    for batch, event in enumerate((10, 11, 12)):
        update_signal_index(tmp_path, _metadata(event, batch), f"orchestration-{batch}.metadata.json")
    assert set(yaml.safe_load((tmp_path / "signal_index.yaml").read_text())) == {"10", "11", "12"}


def test_an_index_predating_the_digest_is_accepted_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An index written before this feature has no digest to check, and must still be usable.

    Refusing would break every in-flight run on upgrade -- a certain cost against an uncertain
    one, the same trade #321 made for checkpoints without a config hash. It expires the same way:
    once no index predates the release carrying this, the acceptance can become a refusal.
    """
    update_signal_index(tmp_path, _metadata(20, 0), "orchestration-0.metadata.json")
    sidecar = tmp_path / "signal_index.yaml.lock"
    sidecar.write_text("")  # as an older version left it: present, empty, no digest

    with caplog.at_level("WARNING"):
        update_signal_index(tmp_path, _metadata(21, 1), "orchestration-1.metadata.json")

    assert set(yaml.safe_load((tmp_path / "signal_index.yaml").read_text())) == {"20", "21"}
    assert any("no digest" in record.message.lower() for record in caplog.records), [r.message for r in caplog.records]


def test_the_digest_survives_a_missing_index(tmp_path: Path) -> None:
    """A sidecar promising an index that is gone is a divergence, not a fresh start.

    Deleting the index by hand while the sidecar records a digest for it is indistinguishable, to
    this client, from not being able to see it -- which is exactly the fault being guarded. Refuse
    rather than silently starting over, and say how to proceed deliberately.
    """
    update_signal_index(tmp_path, _metadata(30, 0), "orchestration-0.metadata.json")
    (tmp_path / "signal_index.yaml").unlink()

    with pytest.raises(StaleIndexReadError, match="stale"):
        update_signal_index(tmp_path, _metadata(31, 1), "orchestration-1.metadata.json")


def test_the_recorded_digest_is_of_the_bytes_written_not_of_a_re_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sidecar must record what was committed, not what this client can read back.

    Re-reading the index to digest it runs through the very path that can be stale, so a client
    with a stale view would record a digest *of that stale view* -- and the next client with the
    same view would match it and overwrite silently. The guard would then certify the bug.
    """
    update_signal_index(tmp_path, _metadata(40, 0), "orchestration-0.metadata.json")
    index_file = tmp_path / "signal_index.yaml"
    sidecar = tmp_path / "signal_index.yaml.lock"
    committed = index_file.read_bytes()

    # From here this client reads the index stale -- as it was before the next update lands.
    real_read_bytes = Path.read_bytes

    def _stale_after_write(self: Path) -> bytes:
        return committed if self == index_file else real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _stale_after_write)
    update_signal_index(tmp_path, _metadata(41, 1), "orchestration-1.metadata.json")
    monkeypatch.undo()

    truth = hashlib.sha256(index_file.read_bytes()).hexdigest()
    stale = hashlib.sha256(committed).hexdigest()
    recorded = sidecar.read_text().strip()
    assert recorded != stale, "the sidecar recorded the digest of a stale re-read"
    assert recorded == truth, f"recorded {recorded[:12]} but the committed index is {truth[:12]}"


def test_a_withdrawing_batch_is_not_skipped_by_a_stale_existence_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch with no injections must still reach the guard when an index exists.

    The early return asked ``index_file.exists()``, which a stale negative dentry answers without
    contacting the server -- so a rerun that must *withdraw* its previous entries returned having
    withdrawn nothing, leaving rows ``find-signal --id`` still trusts.
    """
    update_signal_index(tmp_path, _metadata(50, 0), "orchestration-0.metadata.json")
    index_file = tmp_path / "signal_index.yaml"

    real_exists = Path.exists

    def _index_looks_absent(self: Path, **kwargs: Any) -> bool:
        return False if self == index_file else real_exists(self, **kwargs)

    monkeypatch.setattr(Path, "exists", _index_looks_absent)
    empty = {"signal": {"injections": []}, "outputs": []}
    update_signal_index(tmp_path, empty, "orchestration-0.metadata.json")
    monkeypatch.undo()

    # The claim is that the batch reached the locked section instead of returning early: its
    # previous row is withdrawn. Asserting a raise here would be wrong -- this client can read the
    # index perfectly well, only `exists()` was made to lie, so the guard is right not to fire.
    assert yaml.safe_load(index_file.read_text()) == {}, "the withdrawing batch was skipped"


def test_a_stale_read_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """`retry_with_backoff` must let a stale read straight through.

    Waiting changes nothing: a stale index read is a cache state, not a transient fault. Worse,
    it is raised *after* the batch's frames and metadata are written, so each retry re-simulates
    the whole batch and the second attempt trips over the first's outputs. Three attempts, three
    full simulations, same failure.
    """
    calls = {"n": 0}

    def _always_stale() -> None:
        calls["n"] += 1
        raise StaleIndexReadError("stale")

    with pytest.raises(StaleIndexReadError):
        retry_with_backoff(_always_stale, max_retries=3, initial_delay=0.0)
    assert calls["n"] == 1, f"the stale read was attempted {calls['n']} times; it must not be retried"


def test_an_ordinary_failure_is_still_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """The no-retry rule must be narrow, or it silently disables the retry loop."""
    calls = {"n": 0}

    def _transient() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("transient")
        return "ok"

    assert retry_with_backoff(_transient, max_retries=3, initial_delay=0.0) == "ok"
    assert calls["n"] == 3, calls["n"]


def test_the_recorded_digest_matches_the_bytes_on_disk(tmp_path: Path) -> None:
    """The digest must describe the file as stored, byte for byte.

    Hashing the serialised payload and then writing it through a text-mode handle would on any
    platform that translates newlines: the digest would never match the file again, and every
    update after the first would refuse. Comparing the recorded digest against a fresh hash of
    the file on disk catches that regardless of platform.
    """
    for batch, event in enumerate((60, 61)):
        update_signal_index(tmp_path, _metadata(event, batch), f"orchestration-{batch}.metadata.json")
    on_disk = hashlib.sha256((tmp_path / "signal_index.yaml").read_bytes()).hexdigest()
    recorded = (tmp_path / "signal_index.yaml.lock").read_text().strip()
    assert recorded == on_disk, f"recorded {recorded[:12]} but the file hashes to {on_disk[:12]}"


def test_an_unreadable_sidecar_does_not_disable_the_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sidecar that exists but cannot be read must refuse, not fall through as "no digest".

    Mapping every read error to "no digest recorded" would take the permissive legacy path on a
    corrupt or unreadable sidecar -- silently turning the guard off exactly when something is
    already wrong.
    """
    update_signal_index(tmp_path, _metadata(70, 0), "orchestration-0.metadata.json")
    sidecar = tmp_path / "signal_index.yaml.lock"
    real_read_text = Path.read_text

    def _unreadable(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == sidecar:
            raise OSError("input/output error")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _unreadable)
    with pytest.raises(StaleIndexReadError, match="could not be read"):
        update_signal_index(tmp_path, _metadata(71, 1), "orchestration-1.metadata.json")


def test_a_failure_to_record_the_digest_is_raised_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blocker fix needs its own test: swallowing here wedges the directory silently.

    Being able to take the lock does not mean the write lands -- fsync can still fail with EIO or
    ENOSPC. Dropping that leaves the index committed and the sidecar behind, and every later
    write refuses as stale with a message blaming a cache. Reverting to the old swallow passed
    every other test in this file, which is why this one exists.
    """
    update_signal_index(tmp_path, _metadata(80, 0), "orchestration-0.metadata.json")
    index_file = tmp_path / "signal_index.yaml"
    committed = index_file.read_bytes()

    real_open = Path.open
    sidecar = tmp_path / "signal_index.yaml.lock"

    # Only the digest write opens the sidecar for writing at a position; the lock opens it "a+".
    # Keying on that distinction failed the lock itself in an earlier version, which broke seven
    # unrelated tests -- an injection wider than its target.
    def _sidecar_write_fails(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if self == sidecar and ("r+" in mode or mode == "w"):
            raise OSError("input/output error")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _sidecar_write_fails)
    with pytest.raises(IndexDigestNotRecordedError, match="could not be recorded"):
        update_signal_index(tmp_path, _metadata(81, 1), "orchestration-1.metadata.json")
    monkeypatch.undo()

    # The index itself is committed and correct -- that is what makes the silent version so bad.
    assert index_file.read_bytes() != committed
    assert set(yaml.safe_load(index_file.read_text())) == {"80", "81"}


def test_neither_guard_error_is_retried_even_when_wrapped() -> None:
    """The no-retry rule must survive being re-raised from something else."""

    def _attempts_before_giving_up(error: Exception) -> int:
        calls = {"n": 0}

        def _wrapped() -> None:
            calls["n"] += 1
            raise RuntimeError("wrapped") from error

        with pytest.raises(RuntimeError):
            retry_with_backoff(_wrapped, max_retries=3, initial_delay=0.0)
        return calls["n"]

    for error in (StaleIndexReadError("stale"), IndexDigestNotRecordedError("unrecorded")):
        attempts = _attempts_before_giving_up(error)
        assert attempts == 1, f"{type(error).__name__} wrapped was attempted {attempts} times"


def test_a_withdrawing_batch_survives_both_dentries_being_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-op decision must not trust a sidecar read taken before the lock.

    The sidecar is created by the same first write as the index, so a client that probed the
    directory early caches a negative entry for *both*. Deciding to return early on that reading
    skipped the withdrawal exactly as the original bug did -- reached through the sidecar instead
    of the index.
    """
    update_signal_index(tmp_path, _metadata(90, 0), "orchestration-0.metadata.json")
    index_file = tmp_path / "signal_index.yaml"
    sidecar = tmp_path / "signal_index.yaml.lock"

    real_exists = Path.exists
    real_read_text = Path.read_text
    unlocked = {"active": True}

    def _both_look_absent(self: Path, **kwargs: Any) -> bool:
        return False if (unlocked["active"] and self == index_file) else real_exists(self, **kwargs)

    def _sidecar_looks_absent(self: Path, *args: Any, **kwargs: Any) -> str:
        if unlocked["active"] and self == sidecar:
            raise FileNotFoundError(sidecar)
        return real_read_text(self, *args, **kwargs)

    # The lock's own open("a+") revalidates the sidecar, so the staleness ends there -- which is
    # precisely why the decision has to be made after it, not before.
    real_lock = simulate_utils._exclusive_index_lock

    @contextlib.contextmanager
    def _lock_then_revalidate(path: Path):  # type: ignore[no-untyped-def]
        with real_lock(path) as value:
            unlocked["active"] = False
            yield value

    monkeypatch.setattr(Path, "exists", _both_look_absent)
    monkeypatch.setattr(Path, "read_text", _sidecar_looks_absent)
    monkeypatch.setattr(simulate_utils, "_exclusive_index_lock", _lock_then_revalidate)

    empty = {"signal": {"injections": []}, "outputs": []}
    update_signal_index(tmp_path, empty, "orchestration-0.metadata.json")
    monkeypatch.undo()

    assert yaml.safe_load(index_file.read_text()) == {}, "the withdrawal was skipped on a stale sidecar read"


def test_an_unrelated_failure_during_handling_is_still_retried() -> None:
    """Implicit context must not suppress retries.

    ``__context__`` is set by any exception raised while another is being handled -- a cleanup or
    logging failure, where the *new* exception is the real problem. Following it made an OSError
    raised during handling of a stale read non-retryable, which is a different bug from the one
    the chain walk prevents.
    """
    calls = {"n": 0}

    def _fails_during_handling() -> str:
        calls["n"] += 1
        try:
            raise StaleIndexReadError("stale")
        except StaleIndexReadError:
            if calls["n"] < 3:
                raise OSError("transient cleanup failure") from None
            return "ok"

    # `from None` severs the cause but pytest still records context; the retry must proceed.
    assert retry_with_backoff(_fails_during_handling, max_retries=3, initial_delay=0.0) == "ok"
    assert calls["n"] == 3, calls["n"]


def test_a_fresh_directory_does_not_warn_about_predating_the_guard(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The legacy warning must not fire where there is no legacy.

    The lock creates the sidecar empty, and empty means "no digest recorded" -- so without this
    the first write into every new metadata directory warns that an index which does not exist
    predates the guard. A warning on every clean run is how the one that matters gets ignored.
    """
    with caplog.at_level("WARNING"):
        update_signal_index(tmp_path, _metadata(100, 0), "orchestration-0.metadata.json")
    assert not [r for r in caplog.records if "predates" in r.message.lower()], [r.message for r in caplog.records]
    # ...and the digest is still recorded, so the guard is armed from the second write on.
    assert (tmp_path / "signal_index.yaml.lock").read_text().strip()
