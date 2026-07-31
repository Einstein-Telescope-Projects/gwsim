"""Test-time overrides that make an example configuration cheap to run.

The examples are sized to be realistic -- most generate a day of data in 4096-second segments.
The suite needs seconds. Rather than shorten the examples themselves, which would make them
worse at their actual job, each is loaded and a small declarative overlay is merged over it.

Two consequences worth knowing:

* Changing an example changes what the suite runs. That is deliberate -- it is how an example
  that stops working becomes a test failure.
* An overlay is *not* allowed to change which code path the config takes. Shortening a duration
  or repointing an input is fine; switching the waveform model or the detector network would
  make the test stop covering the thing its matrix entry claims.

Inputs are repointed at in-repo files because most example configs fetch their population over
the network -- two of them from ``sandbox.zenodo.org``, whose records are purged. A suite that
depends on those is a suite that fails for reasons unrelated to the code.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = _REPO_ROOT / "examples"

#: In-repo CBC population, used in place of every remote population file. Small and fixed, so a
#: run is reproducible and needs no network.
POPULATION_FIXTURE = EXAMPLES_DIR / "signal" / "bbh_population.csv"

#: The single event in :data:`POPULATION_FIXTURE`.
_FIXTURE_EVENT_GPS = 1577491300.0

#: Start time placing that event inside a short segment. A run whose span misses the population
#: still succeeds and writes only zeros, so this is not a cosmetic choice -- see the
#: ``contains_signal`` assertions in the end-to-end tests.
_ALIGNED_START = 1577491296.0

#: Entries that cannot be run without fetching from the network, with the reason. These are
#: skipped rather than run against a remote URL. Removing an entry from here requires adding a
#: local fixture for whatever it downloads.
NOT_HERMETIC: dict[str, str] = {
    "noise/glitches/gengli/et_triangle_sardinia/e1": (
        "needs a local blip-glitch population fixture; the example reads one from "
        "sandbox.zenodo.org, whose records are purged"
    ),
}

#: Per-example overrides, deep-merged over the example's own configuration.
#:
#: Only durations, rates, sample counts, seeds and input paths appear here. Anything that would
#: change the code path under test does not belong in an overlay.
_OVERLAYS: dict[str, dict[str, Any]] = {
    "default_config": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 8,
                "total-duration": 8,
                "seed": 20260731,
            }
        }
    },
    "noise/uncorrelated_gaussian/quick_start": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 16,
                "total-duration": 16,
                "start-time": _ALIGNED_START,
                "seed": 20260731,
            }
        },
        "orchestration": {"population": {"arguments": {"path": str(POPULATION_FIXTURE)}}},
    },
    # Kept multi-segment on purpose: this entry exists to cover chunking and per-segment seeds,
    # so collapsing it to one segment would silently retire what it tests.
    "noise/uncorrelated_gaussian/et_triangle_sardinia": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 4,
                "total-duration": 12,
                "seed": 20260731,
            }
        }
    },
    "signal/bbh/et_triangle_sardinia": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 16,
                "total-duration": 16,
                "start-time": _ALIGNED_START,
                "seed": 20260731,
            }
        },
        "orchestration": {"population": {"arguments": {"path": str(POPULATION_FIXTURE)}}},
    },
    # Already 16 s at 512 Hz, so only the seed is pinned.
    "signal/sgwb/et_triangle_sardinia": {"globals": {"simulator-arguments": {"seed": 20260731}}},
}

#: Entries whose span is expected to contain a gravitational-wave signal, so an all-zero output
#: is a failure rather than a valid result. Noise-only runs are excluded because their content
#: is noise, and the SGWB run produces a background rather than a located event.
CONTAINS_SIGNAL: frozenset[str] = frozenset(
    {
        "noise/uncorrelated_gaussian/quick_start",
        "signal/bbh/et_triangle_sardinia",
    }
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return *base* with *override* merged into it, recursing into nested mappings."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_overlay(config: dict[str, Any], label: str, working_directory: Path) -> dict[str, Any]:
    """Return *config* shortened for test use and rooted at *working_directory*.

    Args:
        config: The example configuration, as loaded from its ``config.yaml``.
        label: The example label, used to select the overlay.
        working_directory: Directory the run should write into.

    Returns:
        A new configuration; *config* is not modified.

    Raises:
        KeyError: If no overlay is defined for *label*. Deliberate -- a new matrix entry has to
            state how it is shortened, rather than silently running at full example size.
    """
    if label not in _OVERLAYS:
        raise KeyError(
            f"no test overlay is defined for '{label}'. Add one to tests/e2e/overlay.py; running "
            f"an example at its full size would take minutes to hours."
        )
    merged = _deep_merge(config, _OVERLAYS[label])
    merged.setdefault("globals", {})["working-directory"] = str(working_directory)
    return merged
