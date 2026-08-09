"""Reference values for the end-to-end matrix: building them, and comparing against them.

Each matrix entry has a small JSON file under ``references/`` recording, per output file, a content
hash and a handful of summary statistics. **The statistics are what the tests assert on**; the hash
is recorded and reported but not compared, for the reason set out under portability below.

The statistics also make a failure *diagnosable* rather than merely red. A bare hash tells you
something changed and nothing about what, so every failure becomes a bisect. With the statistics
stored alongside, a mismatch reports which files moved and by how much, and a drift from a library
upgrade looks obviously different from a waveform that shifted in time or changed amplitude.

What this does **not** detect: a change that preserves every gated statistic. Reordering samples, or
altering the shape between the peak and the edges, can leave ``n``, ``nonzero``, ``argmax``,
``peak``, ``rms`` and ``signed_peak`` all intact. Sign inversion is covered, by ``signed_peak``;
arbitrary shape change is not. Closing that needs a quantised digest of the samples, which is future
work rather than something this claims today.

A mismatch is not automatically a bug. When a dependency bump changes the last bits of a
waveform, the right response is to look at the reported differences, decide the change is
harmless, and regenerate. When it changes a peak by a percent, it is not harmless. The stored
environment makes the first case quick to recognise -- it says which versions produced the
reference.

Regenerate with::

    GWMOCK_WRITE_E2E_REFERENCES=1 uv run pytest -m e2e --no-cov

Review the resulting diff before committing it. The references are small JSON in the repository
rather than an opaque archive precisely so that an update shows up as a reviewable change.

**Portability: measured, and it does not hold.** An earlier run suggested these hashes reproduced
bit-for-bit between a local Linux/x86_64 machine and GitHub's ubuntu-latest runner. That was wrong.
With identical package versions -- numpy, scipy, lalsuite, jax and ripplegw all unchanged -- CI and
local now disagree in the last bits. The observable difference is confined to ``mean``, at 1e-16 to
1e-12 relative, while ``peak``, ``rms``, ``argmax`` and ``nonzero`` are identical: floating-point
reassociation, not a change in the data.

The fingerprint cannot separate those two environments, because both are Linux/x86_64 on the same
Python. Widening it to CPU or BLAS identity is not something this can do reliably.

So the exact hash is no longer the assertion. It is recorded and reported, but what *fails* a run is
a statistic moving beyond :data:`STATISTIC_TOLERANCE` -- integer statistics exactly, float
magnitudes within tolerance. That is weaker than a bit-for-bit check and the weakening is real: a
change of one sample in one channel now passes. What it buys is a check that means the same thing on
every machine, rather than one that is green only where the references happened to be generated.

``mean`` is excluded from the gate. It is a sum, so its last bits depend on summation order rather
than on the data -- the reason sums were left out of the statistics in the first place, before it was
added back to catch sign inversions. It stays as a diagnostic, where being sensitive is useful.

**One reference, both devices -- and how that was established.** These references are written on a CPU,
and the entries that generate through JAX (`execution: batched`, the ripple backend, the CW branch) run
on a GPU wherever one is present. That makes "does the GPU still match?" a real question, and it is
answered by replaying the matrix on a CUDA host rather than by argument:

    # on a host with a CUDA-capable GPU and the `cuda` extra installed
    uv run pytest -m e2e --no-cov tests/e2e/test_reference_values.py

Done on 2026-08-09 against an RTX 5060 Ti (compute capability 12.0): **all eight stored references
matched**, `argmax` included -- which is compared exactly, so the device difference moved no peak off its
sample. (The run reports "9 passed, 1 skipped": eight reference comparisons, plus the separate test that
every runnable entry has a reference, with the gengli entry skipped. An earlier draft of this paragraph
read that as nine entries, which a reviewer corrected.) An earlier measurement on an RTX 2080 Ti agreed. The delta
itself was characterised separately as a global time shift of 2.3e-16 s, 3.16e-13 relative end to end.

**Where the gate actually sits.** Two effects were measured against it, and it falls between them:

===========================================================  ==========  =====================
difference                                                   relative    against a 1e-06 gate
===========================================================  ==========  =====================
CPU to GPU, one Linux host                                   3.16e-13    passes, 3e6 to spare
Linux/x86_64/py3.12 to Darwin/arm64/py3.13                   4.17e-06    **fails, by 4x**
===========================================================  ==========  =====================

So the tolerance is not simply loose: it admits the device difference and rejects a platform change. What
it does *not* do is catch a GPU-specific regression smaller than about 1e-06, which is seven orders above
the device difference -- passing the GPU replay says these references survive the device, not that the
device is tightly watched.

The macOS row is the reason this suite cannot be run green on an Apple machine against Linux-written
references; it is a known gap rather than a fault in a particular entry.

This check also cannot run in CI: GitHub-hosted runners have no GPU. Repeat it when the waveform path or
the JAX dependency changes, rather than expecting it to guard every pull request.

Note what the fingerprint deliberately does *not* include: package versions. A dependency bump
changing a waveform is precisely what these references exist to surface, so putting versions in the
gate would silently downgrade the very comparison that is wanted. Versions are recorded, and
reported when a comparison fails, but they do not excuse a difference.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from gwmock.cli.utils.hash import compute_content_hash

REFERENCES_DIR = Path(__file__).resolve().parent / "references"

#: Environment variable that switches the tests from comparing to writing.
WRITE_ENVIRONMENT_VARIABLE = "GWMOCK_WRITE_E2E_REFERENCES"

#: Packages whose versions are recorded with a reference. These are the ones whose output could
#: plausibly change a waveform or a noise realisation, so they are the first things to look at
#: when a comparison fails.
_RECORDED_PACKAGES = (
    "gwmock",
    "gwmock-signal",
    "gwmock-noise",
    "gwmock-pop",
    "numpy",
    "scipy",
    "lalsuite",
    "ripplegw",
    "jax",
    # Astropy supplies sidereal time to the projection, and `astropy-iers-data` is the Earth
    # orientation table behind it. These are the *most frequent* reason this comparison moves, so
    # they belong here: the table is republished weekly.
    #
    # Added after a lock-file bump moved a BBH peak by 1.569e-06 and the report said "no recorded
    # package version changed, so a dependency bump does not explain this" -- pointing the next
    # reader at a code regression that did not exist.
    #
    # What used to move, and why it no longer does. The shipped examples run at GPS 1577491296 --
    # 2030-01-01, about 885 days *beyond* the end of the packaged IERS table, where Astropy clamps
    # UT1-UTC to the final tabulated value. What changed each week was not a revised measurement
    # for 2030 but the clamped edge following the table's new
    # end date: -0.044955 s to -0.047695 s, a step of 2.740 ms. That rotates GMST by 1.9976e-07
    # rad, matching 2.740e-3 s x 7.2921e-5 rad/s to 0.02%.
    #
    # Because the epoch rides the table edge, this churn is structural rather than incidental: a
    # test epoch *inside* the table would move only when a finalised value is revised. That is a
    # design question, not something to fix by widening a tolerance.
    "astropy",
    "astropy-iers-data",
    # The C library behind Astropy's time and coordinate transforms; a change here moves sidereal
    # time for the same reason astropy itself does.
    "pyerfa",
    # The implementation actually executing JAX computations on the device path.
    "jaxlib",
)

#: Why this is a curated list rather than the full environment.
#:
#: `gwmock.cli.utils.environment.capture_environment` records *every* installed distribution, and a
#: reference could store that instead -- which would make this diagnostic exhaustive by
#: construction and remove the class of false negative that motivated adding astropy above.
#:
#: The cost is churn: every reference file would be rewritten whenever any development tool moved,
#: which buries a real numerical change in noise and trains readers to skim the diff. So the list
#: is deliberately confined to packages whose output can reach generated data. When a comparison
#: fails and nothing here explains it, the run's own metadata carries the complete environment.
_RECORDED_PACKAGES_ARE_CURATED = True


#: Relative tolerance applied to ``peak`` and ``rms`` when the exact comparison is not available.
#:
#: **This number is not measured.** It is chosen to sit far below any physically meaningful change
#: -- a different waveform, a rescaled PSD, a lost taper all move these by percent or more -- and
#: far above the last-bit differences a different BLAS or libm produces for the same algorithm.
#: Nobody has measured how far apart two platforms actually land, so treat it as a bound on what
#: this check can detect rather than as a statement about numerical agreement.
#:
#: A weekly `astropy-iers-data` release used to exceed this bound, reaching 1.569e-06 on one
#: detector, because the examples ran at a 2030 epoch beyond the end of the Earth-orientation
#: table where Astropy clamps UT1-UTC to the final tabulated value -- so every release moved that
#: clamp and the generated strain with it. That is fixed at the cause: the suite now runs inside
#: the finalised part of the table (see ``_ALIGNED_START`` in ``overlay.py``), where the value is
#: bit-identical across releases.
#:
#: Left at 1e-06 rather than widened, and the distinction matters. Raising it to sit above routine
#: Earth-orientation movement would also have stopped it detecting a projection regression of the
#: same size, which is what it exists to catch.
#:
#: What can still move output through this route is a *revised finalised* value, which IERS does
#: occasionally publish. That is rare and small rather than weekly, and it is why
#: `astropy-iers-data` stays in :data:`_RECORDED_PACKAGES`: the churn is gone, the dependency is
#: not.
STATISTIC_TOLERANCE = 1e-6


def writing_references() -> bool:
    """Return whether this run should write references instead of comparing against them.

    Raises:
        RuntimeError: If write mode is requested while ``CI`` is set. Regenerating references is a
            deliberate local act; a continuous-integration run that quietly rewrote them would
            overwrite the very thing it is supposed to be checking against and then report
            success.
    """
    requested = os.environ.get(WRITE_ENVIRONMENT_VARIABLE, "").strip().lower() in {"1", "true", "yes"}
    # Presence, not value: CI systems spell it CI=true, CI=1, and occasionally CI=. Failing closed
    # on any setting is right for a mode whose accidental use would overwrite the comparison.
    if requested and "CI" in os.environ:
        raise RuntimeError(
            f"{WRITE_ENVIRONMENT_VARIABLE} is set while CI is set. Reference values must not be "
            f"regenerated by a CI run: it would overwrite the comparison and then pass. Generate "
            f"them locally, review the diff, and commit it."
        )
    return requested


def fingerprint() -> dict[str, str]:
    """Return what distinguishes this numerical environment from another.

    Exact content hashes are only a fair comparison between runs on the *same* numerical stack.
    Package versions do not capture that: the same numpy and LAL built against a different BLAS,
    or on a different architecture, can produce different last bits for identical inputs. So the
    reference records a fingerprint, and the comparison uses it to decide whether an exact match
    is a reasonable thing to demand.
    """
    import platform
    import sys

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": ".".join(str(part) for part in sys.version_info[:2]),
        # The compute device, because two runs on one machine can differ by it alone. `execution:
        # batched` generates through JAX, so the same host produces different last bits on its CPU and
        # its GPU -- measured end to end at 3.16e-13 relative, a global time shift of 2.3e-16 s, which is
        # benign and far inside `STATISTIC_TOLERANCE`. Without this the two runs fingerprint identically
        # and the bit-mismatch note below blames "references written somewhere subtly different" for what
        # is simply the other device.
        "device": _jax_device(),
    }


def _jax_device() -> str:
    """Return the JAX backend this run will generate on, or why there is none.

    ``"none"`` when JAX is absent, which is a real configuration here: the matrix entries that do not
    need `ripplegw` run without it, and CI has a cell with no `jax` extra at all.
    """
    try:
        import jax
    except ImportError:
        return "none"
    try:
        return str(jax.default_backend())
    except Exception:  # pragma: no cover - a broken backend must not fail the comparison
        # A CUDA plugin that cannot see a supported GPU raises rather than falling back, and that is
        # worth recording as its own state instead of crashing a test run that would otherwise pass.
        return "unavailable"


def same_environment(stored: dict[str, str] | None, produced: dict[str, str] | None) -> bool:
    """Whether two fingerprints describe the same numerical environment.

    Every key, in both records. An earlier version compared only the keys the *stored* record carried, so
    that references written before a key existed would keep behaving as they had -- and that defeated the
    point of adding one: a CPU reference replayed on a GPU still counted as the same environment, and the
    bit-mismatch note still blamed "references written somewhere subtly different" for the device. Both
    reviewers caught it. A note that misattributes is worse than a note that is missing, and the
    references now carry `device` anyway, so there is nothing to be compatible with.

    Args:
        stored: The fingerprint recorded with the reference, if any.
        produced: The fingerprint of this run, if any.

    Returns:
        Whether the two describe the same environment. A reference with no fingerprint at all is not the
        same environment as anything, since there is nothing to compare.
    """
    if not stored or not produced:
        return False
    return stored == produced


def _environment() -> dict[str, str]:
    """Return the installed versions of the packages that could change the data."""
    from importlib import metadata

    versions: dict[str, str] = {}
    for name in _RECORDED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


def summarise(samples: np.ndarray) -> dict[str, Any]:
    """Return diagnostic statistics for one file's samples.

    Chosen so that the *kind* of a change is often readable from the numbers: ``peak`` and ``rms``
    move when amplitudes change, ``argmax`` when something shifts in time, ``nonzero`` when a
    signal appears or vanishes, and ``signed_peak`` when signs invert.

    These triage a difference; they do not explain every one. A phase change that preserves the
    amplitude envelope, or a permutation of channels within a file, can leave all of them
    untouched -- the hash still catches such a change, but the report will say only that the
    recorded statistics are unchanged. That is a limit of the explanation, not of the detection.
    """
    finite = np.asarray(samples, dtype=float)
    return {
        "n": int(finite.size),
        "nonzero": int(np.count_nonzero(finite)),
        "peak": float(np.max(np.abs(finite))) if finite.size else 0.0,
        "rms": float(np.sqrt(np.mean(finite**2))) if finite.size else 0.0,
        "argmax": int(np.argmax(np.abs(finite))) if finite.size else 0,
        # Signed, so an inversion shows up: `peak` and `rms` are magnitudes and would not move at
        # all if every sample changed sign. Taken at the peak rather than as a mean because it is
        # then O(peak) instead of near zero -- an inversion moves it by twice the peak -- and
        # because it is a single sample rather than a sum, so it carries no summation-order noise.
        "signed_peak": float(finite[int(np.argmax(np.abs(finite)))]) if finite.size else 0.0,
        # Kept for diagnosis only. It is a sum, so its last bits track summation order rather than
        # the data, which is why it is not part of the gate below.
        "mean": float(np.mean(finite)) if finite.size else 0.0,
    }


def build_reference(label: str, working_directory: Path, files: list[Path]) -> dict[str, Any]:
    """Return a reference document describing what a run produced.

    Args:
        label: The matrix entry's label.
        working_directory: The run directory, used to make paths relative.
        files: The data files the run wrote.

    Returns:
        A JSON-serialisable reference document.
    """
    from .runner import samples

    outputs: dict[str, Any] = {}
    for path in sorted(files):
        outputs[str(path.relative_to(working_directory))] = {
            "content_hash": compute_content_hash(path),
            **summarise(samples(path)),
        }
    return {
        "label": label,
        "environment": _environment(),
        "fingerprint": fingerprint(),
        "outputs": outputs,
    }


def reference_path(label: str) -> Path:
    """Return the file storing *label*'s reference, with slashes flattened for a filename."""
    return REFERENCES_DIR / f"{label.replace('/', '__')}.json"


def write_reference(document: dict[str, Any]) -> Path:
    """Write *document* and return where it went."""
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    path = reference_path(document["label"])
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_reference(label: str) -> dict[str, Any] | None:
    """Return *label*'s stored reference, or ``None`` if there is not one yet."""
    path = reference_path(label)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def statistic_differences(stored: dict[str, Any], produced: dict[str, Any]) -> list[str]:
    """Return the statistics that differ beyond what floating-point drift explains.

    Used instead of the hash when the two runs are not on the same numerical stack, where an exact
    match is not a fair thing to require. Integer statistics -- sample count, occupancy, the peak's
    position -- must match exactly, because no amount of last-bit drift moves them. The float
    magnitudes are compared with :data:`STATISTIC_TOLERANCE`.
    """
    differences: list[str] = []
    for name in sorted(set(stored) & set(produced)):
        before, after = stored[name], produced[name]
        for key in ("n", "nonzero", "argmax"):
            if before.get(key) != after.get(key):
                differences.append(f"{name}: {key} {before.get(key)!r} -> {after.get(key)!r}")
        # `mean` is deliberately absent: it is a sum, so its last bits track summation order rather
        # than the data, and it is the statistic that moved between machines. `signed_peak` covers
        # what `mean` was added for -- a full sign inversion -- without that noise, since it is a
        # single sample of magnitude `peak` rather than a near-zero sum.
        for key in ("peak", "rms", "signed_peak"):
            old, new = float(before.get(key, 0.0)), float(after.get(key, 0.0))
            # A NaN makes every comparison below false, so it would be read as "no difference".
            # Non-finite values are the difference.
            if not (math.isfinite(old) and math.isfinite(new)):
                differences.append(f"{name}: {key} {old!r} -> {new!r} (not finite)")
                continue
            scale = max(abs(old), abs(new))
            if scale and abs(new - old) / scale > STATISTIC_TOLERANCE:
                differences.append(f"{name}: {key} {old!r} -> {new!r} (relative {abs(new - old) / scale:.3e})")
    return differences


def describe_difference(stored: dict[str, Any], produced: dict[str, Any]) -> str:
    """Return a human-readable account of how a run differs from its reference.

    Written to answer the question a failure actually raises -- *is this harmless?* -- so it
    reports the statistics side by side with a relative change, and the environment differences,
    rather than only saying that two hashes are unequal.
    """
    lines: list[str] = []

    stored_outputs, produced_outputs = stored.get("outputs", {}), produced.get("outputs", {})
    missing = sorted(set(stored_outputs) - set(produced_outputs))
    extra = sorted(set(produced_outputs) - set(stored_outputs))
    if missing:
        lines.append(f"  files in the reference but not produced: {missing}")
    if extra:
        lines.append(f"  files produced but not in the reference: {extra}")

    for name in sorted(set(stored_outputs) & set(produced_outputs)):
        before, after = stored_outputs[name], produced_outputs[name]
        if before.get("content_hash") == after.get("content_hash"):
            continue
        lines.append(f"  {name}:")
        for key in ("n", "nonzero", "argmax", "peak", "rms", "signed_peak", "mean"):
            old, new = before.get(key), after.get(key)
            if old == new:
                continue
            if isinstance(old, float) and isinstance(new, float) and old != 0.0:
                relative = abs(new - old) / abs(old)
                lines.append(f"    {key}: {old!r} -> {new!r}  (relative change {relative:.3e})")
            else:
                lines.append(f"    {key}: {old!r} -> {new!r}")
        if all(
            before.get(key) == after.get(key)
            for key in ("n", "nonzero", "argmax", "peak", "rms", "signed_peak", "mean")
        ):
            lines.append("    every recorded statistic is unchanged; the difference is below them")

    changed_environment = {
        name: (stored.get("environment", {}).get(name), produced.get("environment", {}).get(name))
        for name in sorted(set(stored.get("environment", {})) | set(produced.get("environment", {})))
        if stored.get("environment", {}).get(name) != produced.get("environment", {}).get(name)
    }
    if changed_environment:
        lines.append("  versions that differ from when the reference was written:")
        lines.extend(f"    {name}: {old} -> {new}" for name, (old, new) in changed_environment.items())
    else:
        lines.append("  no recorded package version changed, so a dependency bump does not explain this")

    return "\n".join(lines) if lines else "  no differences found"
