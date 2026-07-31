"""``examples/README.md`` must describe the examples that actually exist, and the matrix.

Documentation that lists coverage is read as a statement *about* coverage, so it is worth
enforcing rather than trusting. Three ways it can lie, one test each: an example missing
from the index, an index row naming an example that is gone, and the documented matrix
disagreeing with the matrix declared in ``matrix.py``.

Note what is *not* checked here: that running a matrix entry reaches the code path claimed
for it. No runner exists yet, so every claim in ``covers`` is intent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from .matrix import E2E_MATRIX

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _REPO_ROOT / "examples"
_README = _EXAMPLES / "README.md"

#: Heading that introduces the matrix table. Named once because the parser splits on it, so a
#: reworded heading silently empties the comparison rather than failing it.
_MATRIX_HEADING = "## The end-to-end test matrix"

#: Placeholders used in the index table so it stays browsable at ~26 examples. Expanding
#: them here rather than writing every row out keeps the README readable *and* checkable.
_PLACEHOLDERS = {
    "<network>": ("et_triangle_sardinia", "et_triangle_emr", "et_2l_aligned", "et_2l_misaligned"),
    "<e1|e2|e3>": ("e1", "e2", "e3"),
}


def _actual_labels() -> set[str]:
    """Return every example label, derived exactly as ``gwmock config --list`` derives it."""
    return {str(path.parent.relative_to(_EXAMPLES)) for path in _EXAMPLES.rglob("config.yaml")}


def _documented_patterns() -> list[str]:
    """Return the label patterns in the README, from the leading code span of each table row."""
    patterns = []
    for line in _README.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        match = re.match(r"\| `([^`]+)`", line)
        if match:
            # A literal '|' has to be escaped inside a markdown table, so the alternation
            # placeholders read as '<e1\|e2\|e3>' in the source. Undo that before matching.
            patterns.append(match.group(1).replace(r"\|", "|"))
    return patterns


def _expand(pattern: str) -> set[str]:
    """Expand one README pattern into the concrete labels it stands for."""
    expanded = {pattern}
    for placeholder, options in _PLACEHOLDERS.items():
        if not any(placeholder in candidate for candidate in expanded):
            continue
        expanded = {candidate.replace(placeholder, option) for candidate in expanded for option in options}
    return expanded


def _documented_labels() -> set[str]:
    """Return every concrete label the README index accounts for."""
    return {label for pattern in _documented_patterns() for label in _expand(pattern)}


def test_every_example_appears_in_the_index():
    """An example absent from the README is one users cannot discover by browsing."""
    undocumented = _actual_labels() - _documented_labels()
    assert not undocumented, (
        f"these examples exist but no README row covers them: {sorted(undocumented)}. "
        f"Add a row, or extend an existing pattern."
    )


def test_the_index_does_not_name_examples_that_are_gone():
    """A row for a deleted example sends users to a label the CLI will reject.

    Only concrete rows are checked. A pattern row expands to every combination, and a
    partially-populated family is legitimate -- the 2L networks have no ``e3`` detector.
    """
    concrete = {pattern for pattern in _documented_patterns() if "<" not in pattern}
    stale = concrete - _actual_labels()
    assert not stale, f"the README indexes examples that no longer exist: {sorted(stale)}"


def test_the_documented_matrix_matches_the_declared_matrix():
    """The README's matrix table and ``E2E_MATRIX`` must be the same set, in the same order.

    The *declared* matrix -- no runner executes these configs yet, so this compares two
    statements of intent and nothing here shows that running them reaches the claimed paths.

    Order too, not just membership: the table is read top to bottom as the coverage
    argument, and a silently reordered one invites the two to drift apart in content next.
    """
    text = _README.read_text(encoding="utf-8")
    assert _MATRIX_HEADING in text, (
        f"the README no longer has a '{_MATRIX_HEADING}' heading, so this test cannot locate "
        f"the table it is meant to compare."
    )
    body = text.split(_MATRIX_HEADING)[-1]
    documented = [re.match(r"\| `([^`]+)`", line).group(1) for line in body.splitlines() if line.startswith("| `")]
    assert documented == [entry.label for entry in E2E_MATRIX], (
        "examples/README.md and tests/e2e/matrix.py disagree about the end-to-end matrix."
    )


@pytest.mark.parametrize("entry", E2E_MATRIX, ids=lambda entry: entry.label)
def test_every_matrix_entry_is_a_real_example(entry):
    """The matrix must not reference a label that has been renamed or removed."""
    assert (_EXAMPLES / entry.label / "config.yaml").is_file(), (
        f"matrix entry '{entry.label}' has no config.yaml; the example was renamed or removed."
    )


def test_each_matrix_entry_states_a_distinct_code_path():
    """Two entries claiming the same coverage means one of them is not earning its runtime."""
    claims = [entry.covers for entry in E2E_MATRIX]
    assert len(set(claims)) == len(claims), "two matrix entries claim to cover the same code path"
