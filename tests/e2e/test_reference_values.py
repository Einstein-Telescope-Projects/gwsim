"""Compare each matrix entry's output against stored reference values.

This is the assertion the rest of the harness was built to support: not just that a run completes
and looks structurally sane, but that it produces *the same data as before*. It is what turns an
unintended change in a waveform, a PSD, or a projection into a failed check on a pull request
rather than something noticed months later.

A failure is a prompt to look, not a verdict. Dependency upgrades will change the last bits of a
waveform, and when they do the correct response is to read the reported differences, judge them
harmless, and regenerate. :func:`~tests.e2e.reference_values.describe_difference` exists so that
judgement takes a minute rather than a bisect.

Reusing ``completed_run`` means these add no simulation cost -- they inspect the same run the
structural assertions already used.
"""

from __future__ import annotations

import pytest

from .matrix import E2E_MATRIX, MatrixEntry
from .reference_values import (
    WRITE_ENVIRONMENT_VARIABLE,
    build_reference,
    describe_difference,
    load_reference,
    statistic_differences,
    write_reference,
    writing_references,
)
from .runner import skip_if_unavailable, written_files

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("entry", E2E_MATRIX, ids=lambda entry: entry.label)
def test_the_output_matches_its_stored_reference(entry: MatrixEntry, completed_run):
    """Every output file must match the statistics recorded for it.

    Integer statistics -- sample count, occupancy, the peak's position -- must match exactly, since
    no amount of floating-point drift moves them. ``peak``, ``rms`` and ``signed_peak`` may differ
    within :data:`STATISTIC_TOLERANCE`.

    The content hash is recorded and reported but is **not** the assertion. It was, until it turned
    out not to reproduce between this project's CI runner and a local machine with identical package
    versions. That means the check is no longer bit-for-bit: a change of one sample in one channel
    passes. See the module docstring for the measurement and what was ruled out.
    """
    skip_if_unavailable(entry)
    directory = completed_run(entry)
    produced = build_reference(entry.label, directory, written_files(directory))

    if writing_references():
        destination = write_reference(produced)
        pytest.skip(f"wrote {destination.relative_to(destination.parents[3])}")

    stored = load_reference(entry.label)
    assert stored is not None, (
        f"'{entry.label}' has no stored reference. Generate one with "
        f"`{WRITE_ENVIRONMENT_VARIABLE}=1 uv run pytest -m e2e --no-cov` and review the diff "
        f"before committing it."
    )

    stored_outputs, produced_outputs = stored["outputs"], produced["outputs"]
    assert sorted(produced_outputs) == sorted(stored_outputs), (
        f"'{entry.label}' produced a different set of files than its reference:\n"
        f"{describe_difference(stored, produced)}"
    )

    # Both sides must actually have a hash. `compute_content_hash` returns None for a format it
    # cannot decode, and two Nones compare equal -- so without this the comparison could pass by
    # having nothing to compare.
    unhashed = [
        name
        for name in stored_outputs
        if not isinstance(stored_outputs[name].get("content_hash"), str)
        or not isinstance(produced_outputs[name].get("content_hash"), str)
    ]
    assert not unhashed, (
        f"'{entry.label}' has files with no content hash on one or both sides, so comparing them "
        f"would prove nothing: {unhashed}"
    )

    # The statistics are the gate, not the hash. Identical bits turned out not to be reproducible
    # between this project's CI runner and a local machine with the same package versions -- the
    # difference is confined to summation order -- so a bit-for-bit assertion is green only where
    # the references were generated. See the module docstring.
    differences = statistic_differences(stored_outputs, produced_outputs)
    assert not differences, (
        f"'{entry.label}' differs from its reference by more than floating-point drift. This is a "
        f"prompt to look, not necessarily a bug -- if the change is harmless, regenerate with "
        f"`{WRITE_ENVIRONMENT_VARIABLE}=1 uv run pytest -m e2e --no-cov`.\n"
        f"  reference environment: {stored.get('fingerprint')}\n"
        f"  this environment:      {produced.get('fingerprint')}\n  " + "\n  ".join(differences)
    )

    identical = [
        name
        for name in stored_outputs
        if stored_outputs[name]["content_hash"] == produced_outputs[name]["content_hash"]
    ]
    if len(identical) != len(stored_outputs) and produced.get("fingerprint") == stored.get("fingerprint"):
        # Same environment, statistics agree, bits do not. Worth saying rather than passing in
        # silence: it is the signal that the references were written somewhere subtly different.
        print(
            f"note: {entry.label} matches its reference statistically but not bit-for-bit "
            f"({len(stored_outputs) - len(identical)} of {len(stored_outputs)} files differ)"
        )


def test_every_runnable_entry_has_a_reference():
    """A matrix entry with no reference is one whose data nothing is watching.

    Separate from the comparison above because that test skips when its entry cannot run, which
    would also hide a missing reference. Entries that cannot run here are excluded, since there is
    no way to generate a reference for them.
    """
    from .overlay import NOT_HERMETIC

    missing = [
        entry.label for entry in E2E_MATRIX if entry.label not in NOT_HERMETIC and load_reference(entry.label) is None
    ]
    assert not missing, (
        f"these entries have no stored reference, so a change in their output would go unnoticed: {missing}"
    )
