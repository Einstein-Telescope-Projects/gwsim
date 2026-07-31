"""Does running the same configuration twice produce the same data?

This is a prerequisite, not a nicety. Every plan for comparing output against stored reference
values assumes a run is reproducible; if a path is not, a stored value for it would fail
intermittently and teach people to regenerate references reflexively, at which point the
references stop meaning anything.

So each matrix entry is run twice in two directories with an identical configuration, and the
two are compared by *content* hash -- decoded samples plus their timing metadata, not file bytes.
That distinction matters for GWF, whose container records a write-time stamp: hashing the bytes
would report a difference on every rerun regardless of the data.

Each run happens in a **separate subprocess**. Running both in one interpreter would let shared
state -- a cached RNG, JAX's compilation cache, LAL's process-global detector registry -- make two
runs agree for reasons that do not hold across processes. Since stored reference values will be
compared between different CI runs on different machines, process-independent reproducibility is
the property that actually matters, and same-process agreement would overstate it.

A failure here is information about the pipeline, not necessarily a bug in it. Some paths may
legitimately need a seed that the configuration does not currently supply, in which case the
finding is that the seed has to be threaded, before any reference value is stored for it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gwmock.cli.utils.hash import compute_content_hash

from .matrix import E2E_MATRIX, MatrixEntry
from .overlay import EXAMPLES_DIR, NOT_HERMETIC, apply_overlay
from .test_examples_end_to_end import _written_files

pytestmark = pytest.mark.e2e

#: Files that are expected to differ between two runs and are not part of the data. Provenance
#: records timings and host details by design, so comparing them would be comparing clocks.
_NOT_DATA = ("resource_usage_summary.json", "config.yaml")


def _run_in_subprocess(entry: MatrixEntry, working_directory: Path) -> None:
    """Run one matrix entry in a fresh interpreter.

    A subprocess rather than an in-process call, so no state is shared between the two runs being
    compared. ``sys.executable`` rather than the ``gwmock`` console script, so this does not
    depend on what is on PATH.
    """
    source = EXAMPLES_DIR / entry.label / "config.yaml"
    config = apply_overlay(yaml.safe_load(source.read_text(encoding="utf-8")), entry.label, working_directory)

    working_directory.mkdir(parents=True, exist_ok=True)
    config_path = working_directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import sys; from gwmock.cli.simulate import _simulate_impl; _simulate_impl(sys.argv[1])",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"'{entry.label}' failed in a subprocess (exit {completed.returncode}):\n{completed.stderr[-2000:]}"
    )


def _content_hashes(working_directory: Path) -> dict[str, str | None]:
    """Return ``{path relative to the run directory: content hash}`` for the data written."""
    return {
        str(path.relative_to(working_directory)): compute_content_hash(path)
        for path in _written_files(working_directory)
        if path.name not in _NOT_DATA
    }


@pytest.mark.parametrize("entry", E2E_MATRIX, ids=lambda entry: entry.label)
def test_running_an_example_twice_produces_identical_data(entry: MatrixEntry, tmp_path: Path):
    """Two runs of one configuration, in two fresh interpreters, must agree file for file.

    Compared by content hash rather than by array equality so that a mismatch names the file it
    is in, and so that the GWF container's write-time stamp does not count as a difference.

    Separate processes so that a cached RNG, JAX's compilation cache or LAL's process-global
    registry cannot make the two agree for a reason that would not survive a fresh run.
    """
    for module in entry.requires:
        pytest.importorskip(module, reason=f"'{entry.label}' needs {module}")
    if entry.label in NOT_HERMETIC:
        pytest.skip(f"'{entry.label}' {NOT_HERMETIC[entry.label]}")

    _run_in_subprocess(entry, tmp_path / "first")
    _run_in_subprocess(entry, tmp_path / "second")

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
