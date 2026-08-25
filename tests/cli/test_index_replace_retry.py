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

"""Renaming the new index into place when the destination is held open.

Windows refuses ``MoveFileEx`` onto a file another process has open, and a **reader** is enough:
``find_signals`` and the CLI's own lookups open ``signal_index.yaml``, so on Windows every index
update failed for as long as any consumer had it open. POSIX permits the rename regardless, which is
why nothing in this suite saw it and why the retry is driven here by making ``os.replace`` refuse
rather than by a platform.

**These tests pin the retry, not Windows.** No host in this project runs Windows, so what they can
show is that a refusal is waited out, that the wait is bounded, and that nothing else is waited out
at all. Whether the refusal a real Windows host raises is a ``PermissionError`` is a property of
CPython's errno mapping (``ERROR_SHARING_VIOLATION`` and ``ERROR_ACCESS_DENIED`` both map to
``EACCES``), asserted nowhere below because nothing here can execute it.

The three cases are the three ways this can be got wrong: not retrying at all (the defect), retrying
without a bound (a batch loop that stalls under an exclusive lock instead of failing), and retrying
errors that are not transient (a real fault delayed and dressed as resilience).
"""

from __future__ import annotations

import errno
import hashlib
import os
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


def _refuse_the_index_rename(
    monkeypatch: pytest.MonkeyPatch,
    index_file: Path,
    refusals: int | None,
    error: OSError | None = None,
) -> list[Path]:
    """Make renames **onto the index** fail, and record every attempt at one.

    Args:
        monkeypatch: Patch fixture.
        index_file: The destination whose renames are refused. Renames onto anything else are
            delegated untouched -- the point of failing this one is to leave every other write in the
            update path working, so a test that passes has exercised the real thing around it.
        refusals: How many attempts to refuse before letting one through, or ``None`` to refuse
            every attempt.
        error: What to raise instead of the sharing violation a Windows host would.

    Returns:
        The destination of every rename attempted, appended to as they happen, so a test can count
        the attempts at the index rather than infer them from the outcome.
    """
    attempts: list[Path] = []
    real_replace = os.replace
    # The message a Windows host puts on ERROR_SHARING_VIOLATION, so a failure in these tests reads
    # like the condition being simulated rather than like a mangled permission bit.
    refusal = error or PermissionError(
        errno.EACCES, "The process cannot access the file because it is being used by another process"
    )

    def _replace(source, destination, *args, **kwargs):
        if Path(destination) != index_file:
            return real_replace(source, destination, *args, **kwargs)
        attempts.append(Path(destination))
        if refusals is None or len(attempts) <= refusals:
            raise refusal
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", _replace)
    return attempts


def _record_the_backoff(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record what the retry would have slept for, without sleeping it.

    The schedule is asserted against rather than waited out: a test that really slept would pass just
    as well against a schedule of hours, which is the bound this is here to pin.
    """
    slept: list[float] = []
    monkeypatch.setattr(simulate_utils.time, "sleep", slept.append)
    return slept


def test_a_rename_refused_while_a_reader_holds_the_index_lands_on_a_later_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two refusals then success: the update completes, rather than failing as it did before.

    This is the defect. A Windows consumer that has the index open makes the first attempts refuse and
    a later one succeed, once it closes; without the retry the first refusal is the end of the update
    and the batch's entry never reaches the index.
    """
    index_file = tmp_path / "signal_index.yaml"
    attempts = _refuse_the_index_rename(monkeypatch, index_file, refusals=2)
    slept = _record_the_backoff(monkeypatch)

    update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    assert len(attempts) == 3, f"the rename was not retried until it succeeded: {len(attempts)} attempts"
    assert slept == list(simulate_utils._INDEX_RENAME_BACKOFF_SECONDS[:2]), (
        f"the retries did not back off on the schedule: {slept}"
    )
    index = yaml.safe_load(index_file.read_text())
    assert sorted(index) == ["1"], f"the update did not land the batch's event: {sorted(index)}"
    # The digest describes what is on disk. A retry that recorded the sha of the payload it *meant* to
    # write while a different rename won would wedge the next update as stale, so the guard is checked
    # against the file rather than against the return of the write.
    sidecar = (tmp_path / "signal_index.yaml.lock").read_text(encoding="utf-8")
    assert sidecar == hashlib.sha256(index_file.read_bytes()).hexdigest(), (
        "the recorded digest does not describe the index that was installed"
    )


def test_a_rename_that_stays_refused_fails_after_a_bounded_number_of_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consumer that never lets go is an error, not a wait: bounded attempts, bounded backoff.

    The failure has to stay loud and stay soon. The caller is a per-batch loop holding the index's
    exclusive lock, so retrying without a bound would trade a reported failure -- naming a repair the
    operator can carry out -- for a run that stalls every batch and blocks every other writer.
    """
    index_file = tmp_path / "signal_index.yaml"
    attempts = _refuse_the_index_rename(monkeypatch, index_file, refusals=None)
    slept = _record_the_backoff(monkeypatch)

    with pytest.raises(PermissionError):
        update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    # Bounded independently of the schedule first. Deriving every assertion from the module's own
    # constant would pass an implementation that widened it to minutes, which is the failure this test
    # is named for -- the numbers below are what "soon" has to mean for a per-batch loop under a lock.
    assert 1 < len(attempts) <= 10, f"the rename was attempted {len(attempts)} times, which is not a retry with a bound"
    assert sum(slept) < 5.0, f"the backoff is no longer bounded to seconds: {sum(slept)} s"
    expected_attempts = len(simulate_utils._INDEX_RENAME_BACKOFF_SECONDS) + 1
    assert len(attempts) == expected_attempts, (
        f"the rename was attempted {len(attempts)} times, not the {expected_attempts} the schedule bounds it to"
    )
    assert slept == list(simulate_utils._INDEX_RENAME_BACKOFF_SECONDS), (
        f"the backoff between attempts was not the bounded schedule: {slept}"
    )
    # The failure is the same one as before the retry existed, which is the half of this that is not
    # about attempt counts: a rename that never happened leaves the previous index (here, none), no
    # digest describing bytes that were not installed, and no temporary behind.
    assert not index_file.exists(), "an index appeared for a rename that never succeeded"
    assert (tmp_path / "signal_index.yaml.lock").read_text(encoding="utf-8") == "", (
        "a digest was recorded for an index that was never installed"
    )
    assert not list(tmp_path.glob("signal_index.yaml.*.tmp")), (
        f"the temporary was left behind: {sorted(p.name for p in tmp_path.glob('*.tmp'))}"
    )


def test_a_rename_refused_for_a_reason_no_wait_can_clear_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the transient case is retried. Everything else fails on the first attempt, undelayed.

    ``ENOSPC`` does not become possible by being asked again, and neither does ``EXDEV`` or a
    directory removed under the run. Retrying them spends the whole backoff window before reporting a
    fault that was already certain, and buries the one condition the retry is for in a mechanism that
    fires for everything. This case cannot fail against the pre-retry code -- which retried nothing --
    so what it discriminates is the over-broad fix: catching ``OSError`` instead of
    ``PermissionError`` turns it red.
    """
    index_file = tmp_path / "signal_index.yaml"
    attempts = _refuse_the_index_rename(
        monkeypatch, index_file, refusals=None, error=OSError(errno.ENOSPC, "No space left on device")
    )
    slept = _record_the_backoff(monkeypatch)

    with pytest.raises(OSError, match="No space left on device") as raised:
        update_signal_index(tmp_path, _metadata(1, 0), "orchestration-0.metadata.json")

    assert raised.value.errno == errno.ENOSPC, f"a different error surfaced: {raised.value!r}"
    assert not isinstance(raised.value, PermissionError), (
        "the test's own error was a PermissionError, so it proves nothing about what is retried"
    )
    assert len(attempts) == 1, f"a non-transient error was retried {len(attempts)} times"
    assert slept == [], f"a non-transient error was slept on: {slept}"
    assert not list(tmp_path.glob("signal_index.yaml.*.tmp")), (
        f"the temporary was left behind: {sorted(p.name for p in tmp_path.glob('*.tmp'))}"
    )
