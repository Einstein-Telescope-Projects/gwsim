"""Fixtures shared by the end-to-end test modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from .matrix import MatrixEntry
from .runner import run_entry


@pytest.fixture(scope="session")
def completed_run(tmp_path_factory: pytest.TempPathFactory):
    """Return a callable giving the output directory for an entry, running it at most once.

    Several modules assert against the same run -- structural checks, reference values -- so
    running per test would multiply the cost of the suite by the number of assertions, and each
    run is a real CLI subprocess. Caching per label keeps adding assertions cheap.

    Session-scoped and shared, so every assertion must only *read* the output. They do. The
    reproducibility tests deliberately do not use this fixture: they need two independent runs.
    """
    cache: dict[str, Path] = {}

    def _get(entry: MatrixEntry) -> Path:
        if entry.label not in cache:
            directory = tmp_path_factory.mktemp(entry.label.replace("/", "_"))
            run_entry(entry, directory)
            cache[entry.label] = directory
        return cache[entry.label]

    return _get
