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

#: Seed pinned for every run, so two runs of one entry are comparable.
#:
#: Written to ``noise.arguments.seed`` as well as the global, because the orchestrator copies the
#: global seed with ``setdefault`` -- an example that pins its own (several do, at 42) keeps it
#: and the global is ignored. Setting only the global looks like it controls the seed while the
#: example silently does; verified by changing each and seeing which moved the output.
_TEST_SEED = 20260731

#: In-repo CBC population, used in place of every remote population file. Small and fixed, so a
#: run is reproducible and needs no network.
POPULATION_FIXTURE = EXAMPLES_DIR / "signal" / "bbh_population.csv"

#: In-repo pulsar catalogue for the continuous-wave entry.
#:
#: The example points at this file with a path relative to the examples tree, which does not
#: survive being copied into a test working directory -- the run failed with
#: `FileNotFoundError: ../../cw_population.csv`. Repointed absolutely for the same reason every
#: other entry's population is: the overlay owns where inputs come from.
CW_POPULATION_FIXTURE = EXAMPLES_DIR / "signal" / "cw_population.csv"

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
            }
        },
        "orchestration": {"population": {"arguments": {"path": str(POPULATION_FIXTURE)}}},
    },
    # Already 16 s at 512 Hz; the seed is pinned centrally, so there is nothing to override.
    "signal/sgwb/et_triangle_sardinia": {},
    # The example pins its population to a commit, so the URL is stable -- but it is still a
    # network fetch, and the local fixture is the same data. `waveform-backend: ripple` is left
    # alone: it is the entire point of this entry.
    #
    # ripple JIT-compiles on first use, so this entry costs seconds where the others cost
    # fractions of one. IMRPhenomD is what the example already selects, and is the cheap case --
    # the precessing models take roughly seven times longer to compile.
    "signal/waveform_backend/ripple": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 16,
                "total-duration": 16,
                "start-time": _ALIGNED_START,
            }
        },
        "orchestration": {"population": {"arguments": {"path": str(POPULATION_FIXTURE)}}},
    },
    # A continuous wave is on for the whole run, so there is no event to bracket and nothing to
    # align a start time to -- the only thing to shorten is the span. Kept multi-segment on
    # purpose: this entry exists to cover the branch where a population is never consumed and
    # every source contributes to every segment, and one segment could not distinguish that from
    # the per-event path.
    #
    # The ephemeris the example names is fetched by ripple on first use; CI caches it and verifies
    # it against `tests/data/ephemeris.sha256`. That is why this entry is no longer in
    # NOT_HERMETIC: the tables are obtained once and their content is pinned, rather than being
    # re-downloaded per run or trusted unchecked.
    "signal/cw/et_triangle_sardinia": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 256,
                "duration": 8,
                "total-duration": 16,
            }
        },
        "orchestration": {"population": {"arguments": {"path": str(CW_POPULATION_FIXTURE)}}},
    },
    # Deliberately the same overlay as the ripple entry above: the two configs differ only by
    # `execution`, and holding the span, rate and population identical is what makes their stored
    # references comparable. The batched path JIT-compiles like the per-event ripple one.
    "signal/execution/batched": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 16,
                "total-duration": 16,
                "start-time": _ALIGNED_START,
            }
        },
        "orchestration": {"population": {"arguments": {"path": str(POPULATION_FIXTURE)}}},
    },
}

#: Entries whose span is expected to contain a gravitational-wave signal, so an all-zero output
#: is a failure rather than a valid result. Noise-only runs are excluded because their content
#: is noise, and the SGWB run produces a background rather than a located event.
CONTAINS_SIGNAL: frozenset[str] = frozenset(
    {
        "noise/uncorrelated_gaussian/quick_start",
        "signal/bbh/et_triangle_sardinia",
        "signal/waveform_backend/ripple",
        "signal/execution/batched",
        # Every pulsar is present in every segment, so any span contains signal -- unlike the
        # transient entries, whose span has to be aligned to an event.
        "signal/cw/et_triangle_sardinia",
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
    merged["globals"].setdefault("simulator-arguments", {})["seed"] = _TEST_SEED

    # Override the noise seed rather than defaulting it: the orchestrator's own copy of the
    # global seed is a `setdefault`, so an example that pins its own would otherwise keep it and
    # this would have no effect. See _TEST_SEED.
    noise = merged.get("orchestration", {}).get("noise")
    if isinstance(noise, dict):
        noise.setdefault("arguments", {})["seed"] = _TEST_SEED
    return merged
