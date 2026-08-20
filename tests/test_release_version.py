"""Which version the next release ships under.

The scheduled release asks `git-cliff --bumped-version` for the tag, and git-cliff's default takes a
breaking change straight to the next major. At `v0.13.2`, with one breaking commit in the history since
that tag, its answer is `v1.0.0` -- measured against this repository's tags, not predicted -- and `v1.x`
declares a stable API this package is not ready to promise.

So the rule, and what these tests pin:

    while the current version is 0.x.y, a breaking change bumps the *minor*, not the major, unless the
    release is run with `allow_major_bump` -- and once the current major is >= 1, git-cliff's answer
    stands untouched.

This is decided in a module rather than in workflow shell because a wrong answer here is expensive and
silent: a version cannot be unpublished, and nothing downstream would flag `1.0.0` as unintended.
"""

from __future__ import annotations

import pytest
from release_version import choose_version

pytestmark = pytest.mark.unit


class TestZeroVerHoldsTheMajorBack:
    """0.x.y is the case the rule exists for."""

    @pytest.mark.parametrize(
        ("current", "proposed", "expected"),
        [
            ("v0.13.2", "v1.0.0", "v0.14.0"),
            ("v0.1.0", "v1.0.0", "v0.2.0"),
            ("v0.0.1", "v1.0.0", "v0.1.0"),
            ("v0.99.99", "v1.0.0", "v0.100.0"),
        ],
    )
    def test_a_proposed_major_bump_becomes_a_minor_bump(self, current: str, proposed: str, expected: str) -> None:
        """The live case: `v0.13.2` plus a breaking change gives `v0.14.0`, not `v1.0.0`.

        The patch component resets, because this is a minor bump and not a re-labelled major one.
        """
        assert choose_version(current=current, proposed=proposed) == expected

    @pytest.mark.parametrize(
        ("current", "proposed"),
        [
            ("v0.13.2", "v0.13.3"),
            ("v0.13.2", "v0.14.0"),
            ("v0.1.0", "v0.1.1"),
        ],
    )
    def test_a_proposal_that_is_not_a_major_bump_is_untouched(self, current: str, proposed: str) -> None:
        """Only the major bump is demoted. Feature and fix bumps are git-cliff's business.

        The first row is this repository's own non-breaking case, measured: at `v0.13.2` with no breaking
        commit in the history since that tag, git-cliff proposes `v0.13.3` and the guard hands it
        straight back.
        """
        assert choose_version(current=current, proposed=proposed) == proposed

    def test_the_input_can_still_ask_for_the_major(self) -> None:
        """`allow_major_bump` is the negotiation the rule leaves open -- 1.0.0 when it is meant."""
        assert choose_version(current="v0.13.2", proposed="v1.0.0", allow_major_bump=True) == "v1.0.0"


class TestOnceStableTheMajorIsEnforced:
    """At `>= 1` the rule does not apply, and a breaking change bumps the major without negotiation."""

    @pytest.mark.parametrize(
        ("current", "proposed"),
        [
            ("v1.0.0", "v2.0.0"),
            ("v1.4.2", "v2.0.0"),
            ("v9.9.9", "v10.0.0"),
        ],
    )
    def test_a_major_bump_stands(self, current: str, proposed: str) -> None:
        """git-cliff's answer is taken unchanged, `allow_major_bump` or not."""
        assert choose_version(current=current, proposed=proposed) == proposed
        assert choose_version(current=current, proposed=proposed, allow_major_bump=False) == proposed

    def test_a_minor_bump_at_one_point_x_stands(self) -> None:
        """The guard must not reach past the case it is for."""
        assert choose_version(current="v1.4.2", proposed="v1.5.0") == "v1.5.0"


class TestTheShapesItRefuses:
    """A release tag is not a place to be lenient about what it was handed."""

    @pytest.mark.parametrize("current", ["", "v", "0.10", "vx.y.z", "v0.10.1.2", "1.0.0-rc1+build"])
    def test_an_unparsable_current_version_is_refused(self, current: str) -> None:
        """Guessing here would silently ship a wrong version, which cannot be withdrawn."""
        with pytest.raises(ValueError, match="could not read"):
            choose_version(current=current, proposed="v1.0.0")

    @pytest.mark.parametrize("proposed", ["", "v", "not-a-version"])
    def test_an_unparsable_proposal_is_refused(self, proposed: str) -> None:
        """git-cliff printing something unexpected must stop the release, not be pattern-matched past."""
        with pytest.raises(ValueError, match="could not read"):
            choose_version(current="v0.13.2", proposed=proposed)

    def test_a_proposal_that_goes_backwards_is_refused(self) -> None:
        """Not the rule's business, but it is the one place that sees both numbers.

        A proposal below the current version means git-cliff was run against the wrong history, and the
        release would either fail on an existing tag or publish a version that reads as older.
        """
        with pytest.raises(ValueError, match="not ahead of"):
            choose_version(current="v0.13.2", proposed="v0.12.0")

    def test_the_same_version_is_refused(self) -> None:
        """A release that does not move the version has nothing to release."""
        with pytest.raises(ValueError, match="not ahead of"):
            choose_version(current="v0.13.2", proposed="v0.13.2")


class TestTheFormatItReturns:
    """The tag is used verbatim by `git tag`, so its shape is part of the contract."""

    def test_the_v_prefix_is_preserved(self) -> None:
        """Every tag in this repo carries it, and the workflow does not add one."""
        assert choose_version(current="v0.13.2", proposed="v1.0.0").startswith("v")

    def test_a_prefixless_current_is_accepted_and_the_proposal_s_shape_is_kept(self) -> None:
        """`git describe` and git-cliff both emit the `v`, but neither is guaranteed to."""
        assert choose_version(current="0.13.2", proposed="1.0.0") == "0.14.0"


class TestWhitespaceNeverReachesTheTag:
    """The returned string goes straight to `git tag`, so it has to be a tag.

    The parser strips before matching, so validating a stripped copy while returning the original would
    let `" v2.0.0 "` be accepted as `v2.0.0` and emitted with its spaces intact --
    `git check-ref-format 'refs/tags/ v2.0.0 '` rejects that, and the release would fail at the tag with
    a message about ref format. The demoted path is safe either way, because it rebuilds the string from
    parsed numbers rather than passing one through; the untouched path is the one that needs pinning.
    """

    @pytest.mark.parametrize(
        ("current", "proposed", "expected"),
        [
            ("v1.0.0", " v2.0.0 ", "v2.0.0"),
            ("v1.0.0", "v2.0.0\n", "v2.0.0"),
            ("v0.13.2", "  v0.13.3", "v0.13.3"),
        ],
    )
    def test_the_untouched_path_returns_a_clean_tag(self, current: str, proposed: str, expected: str) -> None:
        """A proposal taken unchanged is still normalised -- `git-cliff` output can carry a newline."""
        assert choose_version(current=current, proposed=proposed) == expected

    def test_the_demoted_path_is_clean_too(self) -> None:
        """It always was, since it composes the string; asserted so both paths are pinned together."""
        assert choose_version(current="v0.13.2", proposed=" v1.0.0 ") == "v0.14.0"
