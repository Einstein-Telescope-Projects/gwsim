"""The subset of ``examples/`` that the end-to-end suite runs, and why each one is in it.

``test_examples_end_to_end.py`` drives each entry through the CLI and
``test_reproducibility.py`` runs each twice in separate processes. One entry is an exception and
says so in its own description: it cannot be executed here, and a test enforces that it is
labelled ``not run`` rather than being counted as coverage that happens.

This module is the single source of truth for the set. ``examples/README.md`` documents the
same set for users, and ``test_examples_index.py`` fails if the two disagree.

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
        covers: The code path this entry exercises. Written as a claim that can be checked
            against the code, not as a restatement of the label. The entry is run, but nothing
            asserts that the run *reaches* the named path -- that much is still a human claim,
            kept honest by review.
        requires: Import names that must be present for this entry to run, if any. An entry
            whose dependency is missing is skipped rather than failed.
    """

    label: str
    covers: str
    requires: tuple[str, ...] = ()


#: The declared matrix. Keep in step with the corresponding table in ``examples/README.md``
#: -- a test enforces it.
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
        label="signal/waveform_backend/ripple",
        covers="A non-default waveform library resolved from config. Needs `ripplegw`",
        requires=("ripplegw",),
    ),
    MatrixEntry(
        label="signal/execution/batched",
        covers=(
            "`execution: batched` -- a segment's events generated in one call through "
            "gwmock-signal's batched entry point, then converted back to per-event chunks. "
            "Runs on whatever JAX backend is present, so this entry is CPU here; measured on an "
            "RTX 2080 Ti, GPU and CPU agree far inside the gate. Needs `ripplegw`"
        ),
        requires=("ripplegw",),
    ),
    MatrixEntry(
        label="signal/cw/et_triangle_sardinia",
        covers=(
            "The continuous-wave branch of `_simulate` -- the one path where a population is never "
            "consumed and every source contributes to every segment. Multi-segment on purpose: a "
            "single segment cannot distinguish that from the per-event path. Needs `ripplegw` and "
            "the LALPulsar ephemeris tables, which CI caches and verifies against "
            "`tests/data/ephemeris.sha256`"
        ),
        requires=("ripplegw",),
    ),
    MatrixEntry(
        label="noise/glitches/gengli/et_triangle_sardinia/e1",
        covers="**Not run** -- glitch injection, blocked on `gengli` and a local glitch fixture",
        requires=("gengli",),
    ),
)
