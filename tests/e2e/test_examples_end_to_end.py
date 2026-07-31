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
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from .matrix import E2E_MATRIX, MatrixEntry
from .overlay import CONTAINS_SIGNAL, EXAMPLES_DIR, NOT_HERMETIC, apply_overlay

pytestmark = pytest.mark.e2e


def _skip_if_unavailable(entry: MatrixEntry) -> None:
    """Skip when the entry cannot run here, saying which reason applies."""
    for module in entry.requires:
        pytest.importorskip(module, reason=f"'{entry.label}' needs {module}")
    if entry.label in NOT_HERMETIC:
        pytest.skip(f"'{entry.label}' {NOT_HERMETIC[entry.label]}")


def _run(entry: MatrixEntry, working_directory: Path) -> dict[str, Any]:
    """Run one matrix entry through the CLI and return the configuration used."""
    from gwmock.cli.simulate import _simulate_impl

    source = EXAMPLES_DIR / entry.label / "config.yaml"
    config = apply_overlay(yaml.safe_load(source.read_text(encoding="utf-8")), entry.label, working_directory)

    working_directory.mkdir(parents=True, exist_ok=True)
    config_path = working_directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _simulate_impl(str(config_path))
    return config


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

    def test_the_run_completes_and_writes_data(self, entry: MatrixEntry, tmp_path: Path):
        """The example must run to completion and leave non-empty output behind.

        The weakest useful assertion, and the one that catches most breakage: an exception
        anywhere in orchestration, or a run that silently writes nothing.
        """
        _skip_if_unavailable(entry)
        _run(entry, tmp_path)

        written = _written_files(tmp_path)
        assert written, f"'{entry.label}' completed but wrote no output files"
        empty = [path.name for path in written if path.stat().st_size == 0]
        assert not empty, f"'{entry.label}' wrote empty files: {empty}"

    def test_every_declared_output_was_written(self, entry: MatrixEntry, tmp_path: Path):
        """The files on disk must be exactly the ones the run says it wrote.

        Checked both ways on purpose. Requiring only that *some* output exists lets a run pass
        having written one detector's frame and dropped the rest -- and a run that writes a file
        its own metadata does not mention is equally wrong, in a way a file count would miss.
        """
        _skip_if_unavailable(entry)
        _run(entry, tmp_path)

        declared = {item["path"] for item in _manifest(tmp_path)}
        assert declared, f"'{entry.label}' recorded no outputs in its metadata"

        found = {str(path.relative_to(tmp_path)) for path in _written_files(tmp_path)}
        assert declared == found, (
            f"'{entry.label}' output does not match its own manifest: "
            f"declared but absent {sorted(declared - found)}, "
            f"present but undeclared {sorted(found - declared)}"
        )

    def test_every_output_decodes_and_is_finite(self, entry: MatrixEntry, tmp_path: Path):
        """*Every* written file must decode, carry samples, and contain no NaN or infinity.

        Previously this counted the files that decoded and required at least one, which let an
        unreadable or empty file pass as long as a sibling was fine. An unrecognised format now
        fails in ``_samples`` rather than being skipped.

        The channel list is cross-checked against the manifest as well, since a frame can decode
        while holding fewer channels than the run claims to have put in it.
        """
        _skip_if_unavailable(entry)
        _run(entry, tmp_path)

        for item in _manifest(tmp_path):
            path = tmp_path / item["path"]
            samples = _samples(path)
            assert samples.size, f"'{item['path']}' decoded to no samples"
            assert np.all(np.isfinite(samples)), f"'{item['path']}' contains non-finite samples"

            declared_channels = item.get("channels")
            if declared_channels and path.suffix == ".gwf":
                from gwpy.io.gwf import get_channel_names

                assert sorted(get_channel_names(str(path))) == sorted(declared_channels), (
                    f"'{item['path']}' does not hold the channels its metadata declares"
                )

    def test_the_output_contains_signal_where_expected(self, entry: MatrixEntry, tmp_path: Path):
        """A run whose span covers the population must not be all zeros.

        This is the assertion that distinguishes "the pipeline ran" from "the pipeline produced
        data". A configuration whose segment misses its events completes, writes
        correctly-shaped files, and contains nothing at all -- a trap hit for real while
        developing these tests, not a hypothetical one.
        """
        if entry.label not in CONTAINS_SIGNAL:
            pytest.skip(f"'{entry.label}' is not expected to contain a located signal")
        _skip_if_unavailable(entry)
        config = _run(entry, tmp_path)

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
