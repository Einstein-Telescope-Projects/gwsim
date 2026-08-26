# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""The one entry-and-platform relaxation in the reference gate, and its limits.

An entry that generates through LALSimulation does not reproduce across processor
architectures: replayed on Darwin/arm64 against these Linux/x86_64 references, the two such
entries move by 3.90e-06 and 4.17e-06 of peak, while every other entry stays at 8.0e-13 or
below and the Python version moves nothing at all. The difference is present in LAL's own
frequency-domain output before any code in this project runs, so the comparison relaxes for
those entries rather than pretending the run is wrong.

Not marked ``e2e``: none of this needs a generation run, and the relaxation is exactly the kind
of thing that should be checked on every pull request rather than in the job that runs for
minutes. What it guards is a gate getting quietly wider -- the flag spreading to entries that
never touch LAL, the loose bound leaking onto the platform that wrote the reference, or the
integer statistics being swept along with the floats.
"""

from __future__ import annotations

import pytest

from .matrix import E2E_MATRIX
from .reference_values import (
    FOREIGN_PLATFORM_TOLERANCE,
    STATISTIC_TOLERANCE,
    same_platform,
    statistic_differences,
)

pytestmark = pytest.mark.unit

LINUX = {"system": "Linux", "machine": "x86_64", "python": "3.12", "device": "cpu"}
DARWIN = {"system": "Darwin", "machine": "arm64", "python": "3.13", "device": "cpu"}

#: The entries measured to move across architectures. Restated here rather than read from the
#: matrix, so that flagging a new entry has to be a deliberate edit in two places instead of
#: silently widening the gate for it.
LAL_ENTRIES = {"noise/uncorrelated_gaussian/quick_start", "signal/bbh/et_triangle_sardinia"}

#: The largest relative move measured between Linux/x86_64 and Darwin/arm64 on a LAL entry.
MEASURED_PLATFORM_DIFFERENCE = 4.17e-6


def _outputs(peak: float, *, n: int = 16384, nonzero: int = 5734, argmax: int = 4080) -> dict:
    """Return a one-file statistics block of the shape a reference stores."""
    return {
        "signal.hdf5": {
            "n": n,
            "nonzero": nonzero,
            "argmax": argmax,
            "peak": peak,
            "rms": peak / 11.4,
            "signed_peak": peak,
            "mean": 3.5e-26,
        }
    }


class TestSamePlatform:
    """Which fingerprint keys decide that two runs are on the same machine architecture."""

    def test_a_run_matches_itself(self) -> None:
        assert same_platform(LINUX, dict(LINUX))

    @pytest.mark.parametrize("key", ["python", "device"])
    def test_the_keys_measured_not_to_matter_do_not_separate_platforms(self, key: str) -> None:
        """The narrowing against ``same_environment`` is the measurement, not a convenience.

        Both Python versions produce bit-identical output on either platform -- 37 of 37 files --
        and the device difference is 3.16e-13. Letting either key answer this question would hand
        a LAL entry the loose tolerance for a reason known not to move it, which is the one thing
        a relaxed gate must not do.
        """
        assert same_platform(LINUX, dict(LINUX, **{key: "something-else"}))

    @pytest.mark.parametrize("key", ["system", "machine"])
    def test_the_keys_measured_to_matter_do_separate_platforms(self, key: str) -> None:
        assert not same_platform(LINUX, dict(LINUX, **{key: "something-else"}))

    def test_linux_and_darwin_are_different_platforms(self) -> None:
        assert not same_platform(LINUX, DARWIN)

    def test_a_missing_fingerprint_matches_nothing(self) -> None:
        """There is nothing to compare, so the honest answer is "not the same platform".

        It has to fall this way round rather than the other: a reference with no fingerprint
        that counted as *every* platform would hand every LAL entry the loose bound everywhere,
        including on the machine that wrote it.
        """
        assert not same_platform(None, LINUX)
        assert not same_platform({}, LINUX)
        assert not same_platform(LINUX, None)


class TestTheToleranceIsHonoured:
    """``statistic_differences`` compares floats against the bound it is given."""

    def test_the_measured_platform_difference_fails_the_tight_gate(self) -> None:
        """The default is unchanged: this is still a failure everywhere it used to be."""
        stored = _outputs(8.942912893531663e-22)
        produced = _outputs(8.942912893531663e-22 * (1 + MEASURED_PLATFORM_DIFFERENCE))
        assert statistic_differences(stored, produced)

    def test_the_measured_platform_difference_passes_the_foreign_bound(self) -> None:
        stored = _outputs(8.942912893531663e-22)
        produced = _outputs(8.942912893531663e-22 * (1 + MEASURED_PLATFORM_DIFFERENCE))
        assert not statistic_differences(stored, produced, FOREIGN_PLATFORM_TOLERANCE)

    def test_a_percent_level_regression_still_fails_the_foreign_bound(self) -> None:
        """What the relaxation is sized to keep catching.

        A different waveform, a rescaled PSD or a lost taper moves these by percent. If the loose
        bound admitted that too it would not be a gate at all, just a comment.
        """
        stored = _outputs(8.942912893531663e-22)
        produced = _outputs(8.942912893531663e-22 * 1.01)
        assert statistic_differences(stored, produced, FOREIGN_PLATFORM_TOLERANCE)

    @pytest.mark.parametrize(("key", "moved"), [("n", 16383), ("nonzero", 5733), ("argmax", 4081)])
    def test_the_integer_statistics_stay_exact_at_the_loose_bound(self, key: str, moved: int) -> None:
        """A relaxed float tolerance must not carry the integers with it.

        No amount of last-bit drift moves a sample count or a peak's position, and the
        measurement confirms it: all 37 files kept every integer statistic across the platform
        change. So these stay exact whatever bound the floats are held to -- otherwise the
        loose comparison would stop noticing a waveform that shifted off its sample.
        """
        stored = _outputs(8.942912893531663e-22)
        produced = _outputs(8.942912893531663e-22, **{key: moved})
        assert statistic_differences(stored, produced, FOREIGN_PLATFORM_TOLERANCE)


class TestWhichEntriesAreFlagged:
    """The flag names the entries whose numbers come out of LAL, and only those."""

    def test_exactly_the_measured_entries_are_flagged(self) -> None:
        """Pinned as a set, so both directions are caught.

        Adding a flag hands an entry a bound 100x looser than the rest of the suite, and dropping
        one puts an entry back under a gate it is measured to fail. Neither should be reachable by
        editing ``matrix.py`` alone.
        """
        assert {entry.label for entry in E2E_MATRIX if entry.lal_waveform} == LAL_ENTRIES

    def test_the_entries_that_reproduce_across_platforms_keep_the_tight_gate(self) -> None:
        """Six of the eight reproduce at 8.0e-13 or better; they are not part of this at all.

        This is the reason the relaxation is per entry rather than per platform: an Apple machine
        still holds most of the matrix to 1e-06, which a suite-wide skip or a widened
        ``STATISTIC_TOLERANCE`` would both have thrown away.
        """
        unflagged = {entry.label for entry in E2E_MATRIX if not entry.lal_waveform}
        assert unflagged, "every entry is flagged, so nothing is left under the tight gate"
        assert not (unflagged & LAL_ENTRIES)


class TestTheBoundsThemselves:
    """The two numbers, and the ordering between them the whole scheme relies on."""

    def test_the_foreign_bound_is_looser_than_the_tight_one(self) -> None:
        assert FOREIGN_PLATFORM_TOLERANCE > STATISTIC_TOLERANCE

    def test_the_foreign_bound_admits_the_measured_difference_with_margin(self) -> None:
        """Margin for LAL builds nobody has measured, not for a spread that was observed.

        The 4.17e-06 comes from one Mac. Whether an x86_64 Mac or a Linux/aarch64 host lands in
        the same place is unmeasured, which is what the margin is for -- so it is worth pinning
        that the margin exists rather than leaving the bound sitting just above the one number
        that happens to have been taken.
        """
        assert FOREIGN_PLATFORM_TOLERANCE > 10 * MEASURED_PLATFORM_DIFFERENCE

    def test_the_tight_gate_still_rejects_the_measured_difference(self) -> None:
        """The relaxation is a second bound, not a replacement for the first."""
        assert MEASURED_PLATFORM_DIFFERENCE > STATISTIC_TOLERANCE
