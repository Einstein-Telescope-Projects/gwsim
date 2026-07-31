"""The subset of ``examples/`` that the end-to-end suite runs, and why each one is in it.

This module is the single source of truth for the matrix. ``examples/README.md`` documents
the same set for users, and ``test_examples_index.py`` fails if the two disagree -- a
README that can drift from what the suite actually runs is worse than no README, because
it is read as a statement of coverage.

The selection principle is one entry per distinct **code path**, not per configuration.
Examples that differ only in values flowing through the same code (four detector networks
resolved by one ``Network`` class; ``bbh`` and ``bns`` both resolving to ``CBCSimulator``)
are represented once. Unit tests are what guarantee the unrepresented ones follow, so an
entry earns its place only by reaching code no other entry does.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixEntry:
    """One example configuration in the end-to-end matrix.

    Attributes:
        label: Path-derived example label, exactly as ``gwmock config --get`` takes it.
        covers: The code path this entry exists to exercise. Written as a claim that can be
            checked against the code, not as a restatement of the label.
        requires: Import names that must be present for this entry to run, if any. An entry
            whose dependency is missing is skipped rather than failed.
    """

    label: str
    covers: str
    requires: tuple[str, ...] = ()


#: The matrix. Keep in step with the "Which of these the test suite runs" table in
#: ``examples/README.md`` -- a test enforces it.
E2E_MATRIX: tuple[MatrixEntry, ...] = (
    MatrixEntry(
        label="default_config",
        covers="The blank template must run unedited; noise-only, single segment",
    ),
    MatrixEntry(
        label="noise/uncorrelated_gaussian/quick_start",
        covers="Signal **and** noise in one run; CBC; GWF output",
    ),
    MatrixEntry(
        label="noise/uncorrelated_gaussian/et_triangle_sardinia",
        covers="Noise-only across **many** segments (chunking, per-segment seeds)",
    ),
    MatrixEntry(
        label="signal/bbh/et_triangle_sardinia",
        covers="Signal-only CBC; Earth rotation; population loaded from file",
    ),
    MatrixEntry(
        label="signal/sgwb/et_triangle_sardinia",
        covers="`StochasticBackgroundSimulator` -- a different simulator class; **HDF5** output",
    ),
    MatrixEntry(
        label="noise/glitches/gengli/et_triangle_sardinia/e1",
        covers="Glitch injection. Skipped when `gengli` is absent",
        requires=("gengli",),
    ),
)
