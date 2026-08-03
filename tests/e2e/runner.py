"""Running a matrix entry and inspecting what it wrote.

Separated from the test modules so that several of them -- structural checks, reproducibility,
reference values -- can share one definition, and so that ``conftest.py`` can build the
run-once-per-entry fixture without importing a test module.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from .matrix import MatrixEntry
from .overlay import EXAMPLES_DIR, NOT_HERMETIC, apply_overlay


def skip_if_unavailable(entry: MatrixEntry) -> None:
    """Skip when the entry cannot run here, saying which reason applies."""
    for module in entry.requires:
        pytest.importorskip(module, reason=f"'{entry.label}' needs {module}")
    if entry.label in NOT_HERMETIC:
        pytest.skip(f"'{entry.label}' {NOT_HERMETIC[entry.label]}")


def gwmock_executable() -> str:
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


def write_config(entry: MatrixEntry, working_directory: Path) -> tuple[Path, dict[str, Any]]:
    """Write the overlaid configuration for *entry* and return its path and contents."""
    source = EXAMPLES_DIR / entry.label / "config.yaml"
    config = apply_overlay(yaml.safe_load(source.read_text(encoding="utf-8")), entry.label, working_directory)

    working_directory.mkdir(parents=True, exist_ok=True)
    config_path = working_directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, config


@functools.lru_cache(maxsize=1)
def _deterministic_iers_environment() -> dict[str, str]:
    """Return an environment that pins the Earth-orientation table the run will use.

    A package version does not identify which IERS table Astropy loaded. `auto_download` defaults
    to ``True`` with an `auto_max_age` of 30 days, so once the installed `astropy-iers-data` is
    older than that, Astropy fetches the current table from the IERS server instead of using the
    one the package ships. Two runs with byte-identical dependency sets can then produce different
    strain, because sidereal time reaches the projection through that table.

    That matters here more than anywhere else: a reference is only meaningful if the inputs are
    determined by things the reference records, and the fetched table is recorded nowhere. Left
    alone, this suite would eventually compare against references generated from a table it can no
    longer obtain, and report a regression.

    Disabled through a `sitecustomize` shim on `PYTHONPATH` rather than an Astropy config file,
    because the run is a subprocess and the config-file route did not take effect (the env var is
    honoured, the section is not). The cache is redirected too, so a `finals2000A` downloaded by
    something else cannot be picked up.
    """
    directory = Path(tempfile.mkdtemp(prefix="gwmock-e2e-iers-"))
    (directory / "sitecustomize.py").write_text(
        "\n".join(
            (
                "# Written by tests/e2e/runner.py; see _deterministic_iers_environment.",
                "from astropy.utils import iers",
                "",
                "iers.conf.auto_download = False",
                "",
            )
        ),
        encoding="utf-8",
    )
    cache = directory / "astropy-cache"
    cache.mkdir()
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{directory}{os.pathsep}{existing}" if existing else str(directory)
    environment["ASTROPY_CACHE_DIR"] = str(cache)
    return environment


def run_entry(entry: MatrixEntry, working_directory: Path) -> dict[str, Any]:
    """Run one matrix entry through the real CLI and return the configuration used."""
    config_path, config = write_config(entry, working_directory)

    completed = subprocess.run(  # noqa: S603
        [gwmock_executable(), "simulate", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
        env=_deterministic_iers_environment(),
    )
    assert completed.returncode == 0, (
        f"'{entry.label}' failed via the CLI (exit {completed.returncode}):\n{completed.stderr[-2000:]}"
    )
    return config


def config_of(entry: MatrixEntry, working_directory: Path) -> dict[str, Any]:
    """Return the configuration the run actually used, read back from its directory.

    Read from disk rather than recomputed, so a test cannot end up asserting against a
    configuration that differs from the one the run was given.
    """
    _ = entry
    return yaml.safe_load((working_directory / "config.yaml").read_text(encoding="utf-8"))


def written_files(working_directory: Path) -> list[Path]:
    """Return every data file the run produced, in a stable order."""
    output = working_directory / "output"
    return sorted(path for path in output.rglob("*") if path.is_file())


def samples(path: Path) -> np.ndarray:
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


def manifest(working_directory: Path) -> list[dict[str, Any]]:
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
