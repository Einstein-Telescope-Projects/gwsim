"""Reference values for the end-to-end matrix: building them, and comparing against them.

Each matrix entry has a small JSON file under ``references/`` recording, per output file, the
content hash and a handful of summary statistics. The hash is what the tests assert on; the
statistics exist to make a failure *diagnosable* rather than merely red.

That split is the point. A bare hash tells you something changed and nothing about what, so
every failure becomes a bisect. With the statistics stored alongside, a mismatch reports which
files moved and by how much, and a one-part-in-10^15 drift from a library upgrade looks obviously
different from a waveform that shifted by samples or changed amplitude.

A mismatch is not automatically a bug. When a dependency bump changes the last bits of a
waveform, the right response is to look at the reported differences, decide the change is
harmless, and regenerate. When it changes a peak by a percent, it is not harmless. The stored
environment makes the first case quick to recognise -- it says which versions produced the
reference.

Regenerate with::

    GWMOCK_WRITE_E2E_REFERENCES=1 uv run pytest -m e2e --no-cov

Review the resulting diff before committing it. The references are small JSON in the repository
rather than an opaque archive precisely so that an update shows up as a reviewable change.

**Portability is the open question.** Reproducibility has been established on one machine across
separate processes; whether these hashes hold across machines, platforms and BLAS builds has not.
If they do not, the fix is to compare the statistics with tolerances instead -- the stored files
already contain everything that would need, so no regeneration would be required.
"""

from __future__ import annotations

import json
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
)


def writing_references() -> bool:
    """Return whether this run should write references instead of comparing against them."""
    return os.environ.get(WRITE_ENVIRONMENT_VARIABLE, "").strip().lower() in {"1", "true", "yes"}


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

    Chosen so that the *kind* of a change is readable from the numbers: ``peak`` and ``rms`` move
    when amplitudes change, ``argmax`` moves when something shifts in time, and ``nonzero`` moves
    when a signal appears or vanishes. Floating-point sums are deliberately not included, since
    their last bits depend on summation order rather than on the data.
    """
    finite = np.asarray(samples, dtype=float)
    return {
        "n": int(finite.size),
        "nonzero": int(np.count_nonzero(finite)),
        "peak": float(np.max(np.abs(finite))) if finite.size else 0.0,
        "rms": float(np.sqrt(np.mean(finite**2))) if finite.size else 0.0,
        "argmax": int(np.argmax(np.abs(finite))) if finite.size else 0,
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
    return {"label": label, "environment": _environment(), "outputs": outputs}


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
        for key in ("n", "nonzero", "argmax", "peak", "rms"):
            old, new = before.get(key), after.get(key)
            if old == new:
                continue
            if isinstance(old, float) and isinstance(new, float) and old != 0.0:
                relative = abs(new - old) / abs(old)
                lines.append(f"    {key}: {old!r} -> {new!r}  (relative change {relative:.3e})")
            else:
                lines.append(f"    {key}: {old!r} -> {new!r}")
        if all(before.get(key) == after.get(key) for key in ("n", "nonzero", "argmax", "peak", "rms")):
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
