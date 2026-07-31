"""Drive each matrix example through the real CLI and check what it produced.

These are the tests the matrix in :mod:`tests.e2e.matrix` exists for. Each runs one example
configuration -- shortened by :mod:`tests.e2e.overlay`, never edited in place -- and asserts the
run completed and wrote data of the expected shape.

Marked ``e2e`` and therefore excluded from the default run: they generate data and take seconds
to minutes each. The ``e2e`` CI job runs them with every extra installed.

Scope of what is checked here is deliberately structural. Whether the output *matches a stored
reference* comes later, and depends on first establishing that each path is reproducible at all.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from gwmock.cli.utils.hash import compute_content_hash

from .matrix import E2E_MATRIX, MatrixEntry
from .overlay import CONTAINS_SIGNAL, EXAMPLES_DIR, NOT_HERMETIC, apply_overlay

pytestmark = pytest.mark.e2e


def _skip_if_unavailable(entry: MatrixEntry) -> None:
    """Skip when the entry cannot run here, saying which reason applies."""
    for module in entry.requires:
        pytest.importorskip(module, reason=f"'{entry.label}' needs {module}")
    if entry.label in NOT_HERMETIC:
        pytest.skip(f"'{entry.label}' {NOT_HERMETIC[entry.label]}")


def _gwmock_executable() -> str:
    """Return the installed ``gwmock`` console script.

    These tests exercise what a user actually runs, so they go through the entry point declared
    in ``[project.scripts]`` rather than calling an internal function. Reaching past the console
    script would skip argument parsing and the CLI's own error handling -- the parts an
    end-to-end test exists to cover.
    """
    executable = shutil.which("gwmock")
    assert executable, (
        "the 'gwmock' console script is not on PATH, so the end-to-end suite cannot invoke the "
        "CLI the way a user would. Install the project (`uv sync`) before running these tests."
    )
    return executable


def _write_config(entry: MatrixEntry, working_directory: Path) -> tuple[Path, dict[str, Any]]:
    """Write the overlaid configuration for *entry* and return its path and contents."""
    source = EXAMPLES_DIR / entry.label / "config.yaml"
    config = apply_overlay(yaml.safe_load(source.read_text(encoding="utf-8")), entry.label, working_directory)

    working_directory.mkdir(parents=True, exist_ok=True)
    config_path = working_directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, config


def _run(entry: MatrixEntry, working_directory: Path) -> dict[str, Any]:
    """Run one matrix entry through the real CLI and return the configuration used."""
    config_path, config = _write_config(entry, working_directory)

    completed = subprocess.run(  # noqa: S603
        [_gwmock_executable(), "simulate", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"'{entry.label}' failed via the CLI (exit {completed.returncode}):\n{completed.stderr[-2000:]}"
    )
    return config


@pytest.fixture(scope="session")
def completed_run(tmp_path_factory: pytest.TempPathFactory):
    """Return a callable giving the output directory for an entry, running it at most once.

    Every assertion below inspects the same run, so re-running per test would multiply the cost
    of the whole suite by the number of assertions -- and each run is now a real CLI subprocess.
    Caching per label keeps adding assertions cheap, which matters because the reference-value
    comparisons still to come will add several more.

    Session-scoped and shared, so the assertions must only *read* the output. They do.
    """
    cache: dict[str, Path] = {}

    def _get(entry: MatrixEntry) -> Path:
        if entry.label not in cache:
            directory = tmp_path_factory.mktemp(entry.label.replace("/", "_"))
            _run(entry, directory)
            cache[entry.label] = directory
        return cache[entry.label]

    return _get


def _config_of(entry: MatrixEntry, working_directory: Path) -> dict[str, Any]:
    """Return the configuration the run actually used, read back from its directory.

    Read from disk rather than recomputed, so a test cannot end up asserting against a
    configuration that differs from the one the run was given.
    """
    _ = entry
    return yaml.safe_load((working_directory / "config.yaml").read_text(encoding="utf-8"))


def _written_files(working_directory: Path) -> list[Path]:
    """Return every data file the run produced, in a stable order."""
    output = working_directory / "output"
    return sorted(path for path in output.rglob("*") if path.is_file())


def _samples(path: Path) -> np.ndarray:
    """Return the concatenated samples in *path*, whatever container it uses.

    Channel and dataset names are discovered from the file rather than assumed, so these tests
    do not restate the naming templates the configurations already define. Asserting on names
    duplicated from the config would only check that the test and the config agree.
    """
    if path.suffix == ".gwf":
        from gwpy.io.gwf import get_channel_names
        from gwpy.timeseries import TimeSeriesDict

        names = get_channel_names(str(path))
        if not names:
            return np.array([])
        series = TimeSeriesDict.read(str(path), names)
        return np.concatenate([np.asarray(item.value) for item in series.values()])
    if path.suffix in {".hdf5", ".h5"}:
        import h5py

        collected: list[np.ndarray] = []
        with h5py.File(path, "r") as handle:
            handle.visititems(
                lambda _name, item: collected.append(np.ravel(item[()])) if isinstance(item, h5py.Dataset) else None
            )
        return np.concatenate(collected) if collected else np.array([])
    if path.suffix == ".npy":
        return np.ravel(np.load(path))
    raise AssertionError(
        f"'{path.name}' has no reader here, so its contents would go unchecked. Add one rather "
        f"than letting an unrecognised output format be skipped silently."
    )


def _manifest(working_directory: Path) -> list[dict[str, Any]]:
    """Return the ``outputs`` entries the run recorded in its metadata.

    The run declares every file it wrote, with the channels and content hash for each. Checking
    the directory against that declaration is stronger than counting files against a number
    restated in the test, and it catches the case a count would miss in the other direction --
    a file written that the run does not know about.
    """
    metadata_files = sorted((working_directory / "metadata").glob("*.metadata.json"))
    assert metadata_files, f"no run metadata was written to {working_directory / 'metadata'}"

    outputs: list[dict[str, Any]] = []
    for path in metadata_files:
        outputs.extend(json.loads(path.read_text(encoding="utf-8")).get("outputs", []))
    return outputs


@pytest.mark.parametrize("entry", E2E_MATRIX, ids=lambda entry: entry.label)
class TestExampleRuns:
    """One run per matrix entry, with the checks that do not need a stored reference."""

    def test_the_run_completes_and_writes_data(self, entry: MatrixEntry, completed_run):
        """The example must run to completion and leave non-empty output behind.

        The weakest useful assertion, and the one that catches most breakage: an exception
        anywhere in orchestration, or a run that silently writes nothing.
        """
        _skip_if_unavailable(entry)
        tmp_path = completed_run(entry)

        written = _written_files(tmp_path)
        assert written, f"'{entry.label}' completed but wrote no output files"
        empty = [path.name for path in written if path.stat().st_size == 0]
        assert not empty, f"'{entry.label}' wrote empty files: {empty}"

    def test_every_declared_output_was_written(self, entry: MatrixEntry, completed_run):
        """The files on disk must be exactly the ones the run says it wrote.

        Checked both ways on purpose. Requiring only that *some* output exists lets a run pass
        having written one detector's frame and dropped the rest -- and a run that writes a file
        its own metadata does not mention is equally wrong, in a way a file count would miss.
        """
        _skip_if_unavailable(entry)
        tmp_path = completed_run(entry)

        declared = {item["path"] for item in _manifest(tmp_path)}
        assert declared, f"'{entry.label}' recorded no outputs in its metadata"

        found = {str(path.relative_to(tmp_path)) for path in _written_files(tmp_path)}
        assert declared == found, (
            f"'{entry.label}' output does not match its own manifest: "
            f"declared but absent {sorted(declared - found)}, "
            f"present but undeclared {sorted(found - declared)}"
        )

    def test_every_output_decodes_and_is_finite(self, entry: MatrixEntry, completed_run):
        """*Every* written file must decode, carry samples, and contain no NaN or infinity.

        Previously this counted the files that decoded and required at least one, which let an
        unreadable or empty file pass as long as a sibling was fine. An unrecognised format now
        fails in ``_samples`` rather than being skipped.

        The channel list is cross-checked against the manifest as well, since a frame can decode
        while holding fewer channels than the run claims to have put in it.
        """
        _skip_if_unavailable(entry)
        tmp_path = completed_run(entry)

        for item in _manifest(tmp_path):
            path = tmp_path / item["path"]
            samples = _samples(path)
            assert samples.size, f"'{item['path']}' decoded to no samples"
            assert np.all(np.isfinite(samples)), f"'{item['path']}' contains non-finite samples"

            declared_hash = item.get("content_sha256")
            if declared_hash:
                # The run records a content hash of what it wrote. Recomputing it here checks the
                # pipeline's own bookkeeping against an independent calculation -- a recorded hash
                # that does not match the data would make every downstream provenance claim wrong,
                # and nothing else would notice.
                assert compute_content_hash(path) == declared_hash, (
                    f"'{item['path']}' does not match the content hash recorded for it in the run "
                    f"metadata, so the run's own provenance is wrong about its output"
                )

            declared_channels = item.get("channels")
            if declared_channels and path.suffix == ".gwf":
                from gwpy.io.gwf import get_channel_names

                assert sorted(get_channel_names(str(path))) == sorted(declared_channels), (
                    f"'{item['path']}' does not hold the channels its metadata declares"
                )

    def test_the_output_contains_signal_where_expected(self, entry: MatrixEntry, completed_run):
        """A run whose span covers the population must not be all zeros.

        This is the assertion that distinguishes "the pipeline ran" from "the pipeline produced
        data". A configuration whose segment misses its events completes, writes
        correctly-shaped files, and contains nothing at all -- a trap hit for real while
        developing these tests, not a hypothetical one.
        """
        if entry.label not in CONTAINS_SIGNAL:
            pytest.skip(f"'{entry.label}' is not expected to contain a located signal")
        _skip_if_unavailable(entry)
        tmp_path = completed_run(entry)
        config = _config_of(entry, tmp_path)

        # Only the signal outputs. Counting every file instead lets a signal+noise
        # configuration pass on its noise alone -- verified: with the start time deliberately
        # misaligned, `signal/bbh` failed but `quick_start` still passed, because its noise is
        # non-zero whatever the signal did.
        signal_directory = tmp_path / "output" / config["orchestration"]["signal"]["output"]["output_directory"]
        signal_files = [path for path in _written_files(tmp_path) if signal_directory in path.parents]
        assert signal_files, f"'{entry.label}' wrote nothing under {signal_directory}"

        occupied = sum(int(np.count_nonzero(_samples(path))) for path in signal_files)
        assert occupied > 0, (
            f"'{entry.label}' wrote only zeros to {signal_directory.name}/, so no signal reached the "
            f"output -- most likely the segment does not cover the population's events."
        )
