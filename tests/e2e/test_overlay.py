"""Invariants the test-time overlay has to hold.

Not marked ``e2e``: these run no simulation, so they belong in the default suite where they guard
the overlay on every pull request. What they protect is subtle and was got wrong once already --
an overlay that quietly fails to apply, or that changes the code path it was meant to leave
alone, makes the end-to-end suite report on something other than what its matrix claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from .matrix import E2E_MATRIX
from .overlay import (
    _ALIGNED_START,
    _FIXTURE_EVENT_GPS,
    _OVERLAYS,
    _TEST_SEED,
    CONTAINS_SIGNAL,
    EXAMPLES_DIR,
    NOT_HERMETIC,
    POPULATION_FIXTURE,
    apply_overlay,
)

#: Settings that select a code path. An overlay may shorten a run or repoint an input; changing
#: any of these would make the entry stop covering what it claims to.
_PATH_DEFINING_KEYS = ("waveform-model", "waveform-backend", "detectors", "source-type", "backend")

#: Matrix entries that can actually be run here. A non-hermetic entry has no overlay yet on
#: purpose: it cannot be exercised, so an overlay for it would be untested guesswork. The
#: KeyError in ``apply_overlay`` is what tells whoever adds its fixture to write one.
_RUNNABLE = tuple(entry for entry in E2E_MATRIX if entry.label not in NOT_HERMETIC)


def _example(label: str) -> dict[str, Any]:
    """Load one example configuration."""
    return yaml.safe_load((EXAMPLES_DIR / label / "config.yaml").read_text(encoding="utf-8"))


def test_every_matrix_entry_has_an_overlay():
    """An entry with no overlay would run at full example size -- a day of data for most."""
    missing = [entry.label for entry in _RUNNABLE if entry.label not in _OVERLAYS]
    assert not missing, f"these runnable matrix entries have no test overlay: {missing}"


def test_the_population_fixture_exists():
    """Every remote population is repointed here, so its absence breaks every signal entry."""
    assert POPULATION_FIXTURE.is_file(), f"the population fixture is missing: {POPULATION_FIXTURE}"


def test_the_aligned_start_brackets_the_fixture_event():
    """The signal entries' segment must actually contain the population's only event.

    A segment that misses it still runs and writes correctly-shaped files full of zeros, so this
    arithmetic is the difference between testing the pipeline and testing nothing. Checked
    against the shortest duration any entry uses, since that is the tightest case.
    """
    shortest = min(
        overlay["globals"]["simulator-arguments"]["duration"]
        for label, overlay in _OVERLAYS.items()
        if label in CONTAINS_SIGNAL
    )

    assert _ALIGNED_START <= _FIXTURE_EVENT_GPS < _ALIGNED_START + shortest, (
        f"the fixture event at GPS {_FIXTURE_EVENT_GPS} falls outside "
        f"[{_ALIGNED_START}, {_ALIGNED_START + shortest}), so signal entries would write zeros"
    )


@pytest.mark.parametrize("entry", _RUNNABLE, ids=lambda entry: entry.label)
def test_the_overlay_pins_the_seed_where_the_orchestrator_reads_it(entry, tmp_path: Path):
    """The seed must land in the noise arguments, not only in the globals.

    The orchestrator copies the global seed into the noise arguments with ``setdefault``, so an
    example that pins its own -- several pin 42 -- keeps it and a global-only override does
    nothing. That was measured, not guessed: changing the global left all nine output files
    identical, while changing the noise seed changed all nine.
    """
    merged = apply_overlay(_example(entry.label), entry.label, tmp_path)

    assert merged["globals"]["simulator-arguments"]["seed"] == _TEST_SEED
    noise = merged.get("orchestration", {}).get("noise")
    if isinstance(noise, dict):
        assert noise["arguments"]["seed"] == _TEST_SEED, (
            "the example's own noise seed survived the overlay, so runs are pinned by the example "
            "rather than by the test"
        )


@pytest.mark.parametrize("entry", _RUNNABLE, ids=lambda entry: entry.label)
def test_the_overlay_does_not_change_the_code_path(entry, tmp_path: Path):
    """Shortening a run must not alter what it exercises.

    Compares every path-defining setting before and after the overlay. Without this an overlay
    could, say, swap a waveform model for a faster one and the entry would keep claiming
    coverage it no longer has.
    """
    original = _example(entry.label)
    merged = apply_overlay(original, entry.label, tmp_path)

    for section in ("signal", "population", "noise"):
        before = original.get("orchestration", {}).get(section)
        after = merged.get("orchestration", {}).get(section)
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        for key in _PATH_DEFINING_KEYS:
            if key in before:
                assert after.get(key) == before[key], (
                    f"the overlay changed orchestration.{section}.{key} for '{entry.label}', "
                    f"which changes the code path the entry is meant to cover"
                )


def test_contains_signal_only_names_matrix_entries():
    """A stale label here would silently stop asserting that a signal was produced."""
    labels = {entry.label for entry in E2E_MATRIX}
    unknown = sorted(CONTAINS_SIGNAL - labels)
    assert not unknown, f"CONTAINS_SIGNAL names entries that are not in the matrix: {unknown}"
