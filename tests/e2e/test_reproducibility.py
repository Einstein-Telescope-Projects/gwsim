"""Does running the same configuration twice produce the same data?

This is a prerequisite, not a nicety. Every plan for comparing output against stored reference
values assumes a run is reproducible; if a path is not, a stored value for it would fail
intermittently and teach people to regenerate references reflexively, at which point the
references stop meaning anything.

So each matrix entry is run twice in two directories with an identical configuration, and the
two are compared by *content* hash -- decoded samples plus their timing metadata, not file bytes.
That distinction matters for GWF, whose container records a write-time stamp: hashing the bytes
would report a difference on every rerun regardless of the data.

A failure here is information about the pipeline, not necessarily a bug in it. Some paths may
legitimately need a seed that the configuration does not currently supply, in which case the
finding is that the seed has to be threaded, before any reference value is stored for it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gwmock.cli.utils.hash import compute_content_hash

from .matrix import E2E_MATRIX, MatrixEntry
from .overlay import NOT_HERMETIC
from .test_examples_end_to_end import _run, _written_files

pytestmark = pytest.mark.e2e

#: Files that are expected to differ between two runs and are not part of the data. Provenance
#: records timings and host details by design, so comparing them would be comparing clocks.
_NOT_DATA = ("resource_usage_summary.json", "config.yaml")


def _content_hashes(working_directory: Path) -> dict[str, str | None]:
    """Return ``{path relative to the run directory: content hash}`` for the data written."""
    return {
        str(path.relative_to(working_directory)): compute_content_hash(path)
        for path in _written_files(working_directory)
        if path.name not in _NOT_DATA
    }


@pytest.mark.parametrize("entry", E2E_MATRIX, ids=lambda entry: entry.label)
def test_running_an_example_twice_produces_identical_data(entry: MatrixEntry, tmp_path: Path):
    """Two runs of one configuration must agree file for file.

    Compared by content hash rather than by array equality so that a mismatch names the file it
    is in, and so that the GWF container's write-time stamp does not count as a difference.
    """
    for module in entry.requires:
        pytest.importorskip(module, reason=f"'{entry.label}' needs {module}")
    if entry.label in NOT_HERMETIC:
        pytest.skip(f"'{entry.label}' {NOT_HERMETIC[entry.label]}")

    _run(entry, tmp_path / "first")
    _run(entry, tmp_path / "second")

    first = _content_hashes(tmp_path / "first")
    second = _content_hashes(tmp_path / "second")

    assert first, f"'{entry.label}' wrote no data files to hash"
    assert sorted(first) == sorted(second), (
        f"'{entry.label}' wrote different file names on the second run: "
        f"only in first {sorted(set(first) - set(second))}, only in second {sorted(set(second) - set(first))}"
    )

    differing = [name for name in first if first[name] != second[name]]
    assert not differing, (
        f"'{entry.label}' is not reproducible: {differing} differ between two runs of an identical "
        f"configuration. Until this is resolved, no reference value can be stored for it -- a "
        f"stored value would fail intermittently."
    )
