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

from pathlib import Path
from typing import Any

import pytest
import yaml

from gwmock.cli.simulate_utils import StaleIndexReadError, update_signal_index

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

    # Simulate the stale client: the reader sees the index as it was one update ago.
    one_update_ago = yaml.safe_dump({"1": {"batches": [], "coa_time": 101.0}})
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: one_update_ago if self == index_file else "")

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
