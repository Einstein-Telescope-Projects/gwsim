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
    STATISTIC_TOLERANCE,
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
    """Every output file must have the content hash recorded for it.

    Compared by content hash, so the check is exact: a change of one sample in one channel fails.
    That sensitivity is deliberate -- the alternative is a tolerance, and a tolerance chosen
    without a reason to justify it silently accepts whatever falls inside it.
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

    if produced.get("fingerprint") != stored.get("fingerprint"):
        # A different numerical stack. Requiring identical bits here would fail for reasons that
        # say nothing about gwmock, so the statistics are compared instead -- and the weaker check
        # is stated rather than passed off as the strong one.
        differences = statistic_differences(stored_outputs, produced_outputs)
        assert not differences, (
            f"'{entry.label}' differs from its reference by more than floating-point drift.\n"
            f"  reference environment: {stored.get('fingerprint')}\n"
            f"  this environment:      {produced.get('fingerprint')}\n  " + "\n  ".join(differences)
        )
        pytest.skip(
            f"exact comparison not applicable: reference generated on {stored.get('fingerprint')}, "
            f"this is {produced.get('fingerprint')}; statistics agree to within "
            f"{STATISTIC_TOLERANCE:g} relative"
        )

    differing = [
        name
        for name in stored_outputs
        if stored_outputs[name]["content_hash"] != produced_outputs[name]["content_hash"]
    ]
    assert not differing, (
        f"'{entry.label}' no longer matches its stored reference. This is a prompt to look, not "
        f"necessarily a bug -- if the change is harmless, regenerate with "
        f"`{WRITE_ENVIRONMENT_VARIABLE}=1 uv run pytest -m e2e --no-cov`.\n"
        f"{describe_difference(stored, produced)}"
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
