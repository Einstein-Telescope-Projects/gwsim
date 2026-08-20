"""Decide the version the next release ships under, given git-cliff's proposal.

The scheduled release asks `git-cliff --bumped-version` for a tag. git-cliff's default sends a breaking
change to the next major, which is right for a package that has promised an API and wrong for one that
has not: at `v0.13.2`, a single `feat!:`/`fix!:`/`BREAKING CHANGE:` commit makes its answer `v1.0.0` --
measured against this repository's tag history, not predicted. That would publish a stable-API claim as
a side effect of an ordinary change, and a published version cannot be withdrawn.

The rule this applies:

    while the current version is `0.x.y`, a proposed major bump is demoted to a minor bump, unless the
    release was run with `allow_major_bump`; once the current major is `>= 1`, git-cliff's answer stands
    untouched.

`cliff.toml` cannot express that. `[bump] breaking_always_bump_major = false` would also disable the
enforced major bump at `>= 1`, which is the half worth keeping -- so the decision is post-processing on
git-cliff's answer rather than configuration of it.

Run as a script it takes the two versions and prints the answer, which is what the workflow uses:

    python scripts/release_version.py --current v0.13.2 --proposed "$(git-cliff --bumped-version)"

It lives in `scripts/` rather than in `src/`: it is release tooling and has no business in the wheel.
`pyproject.toml` puts `scripts` on the test path so it is covered like anything else, because a wrong
answer here is both expensive and silent -- a version cannot be unpublished, and nothing downstream
would flag `1.0.0` as unintended.
"""

from __future__ import annotations

import argparse
import re
import sys

#: A release tag: an optional `v`, then exactly three dot-separated numbers. Deliberately strict --
#: pre-release and build metadata are not used by this project's tags, and a tag shape nobody expected
#: is a reason to stop the release rather than to guess at it.
_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def _parse(version: str, *, described_as: str) -> tuple[int, int, int]:
    """Return the numeric components of *version*.

    Args:
        version: The tag, with or without a leading `v`.
        described_as: What to call it in the message, so a failure names which of the two was wrong.

    Returns:
        The major, minor and patch components.

    Raises:
        ValueError: If the version is not `<major>.<minor>.<patch>`, optionally `v`-prefixed.
    """
    matched = _VERSION.match(version.strip())
    if matched is None:
        raise ValueError(
            f"could not read {described_as} {version!r} as a version: expected `<major>.<minor>.<patch>`, "
            f"optionally prefixed with `v`. Refusing to guess, because the answer becomes a published tag."
        )
    major, minor, patch = matched.groups()
    return int(major), int(minor), int(patch)


def choose_version(*, current: str, proposed: str, allow_major_bump: bool = False) -> str:
    """Return the version to release, holding the major back while the package is pre-1.0.

    Args:
        current: The latest released tag, e.g. from `git describe --tags --abbrev=0`.
        proposed: What `git-cliff --bumped-version` suggests.
        allow_major_bump: Take the proposal even if it crosses into `1.x`. This is the explicit
            negotiation the rule leaves open; it does nothing once the current major is `>= 1`, where a
            major bump is enforced anyway.

    Returns:
        The tag to create, carrying the `v` prefix if *proposed* had one.

    Raises:
        ValueError: If either version is unreadable, or the proposal is not ahead of the current version.
    """
    current_parts = _parse(current, described_as="the current version")
    proposed_parts = _parse(proposed, described_as="the proposed version")
    current_major, current_minor, _ = current_parts
    proposed_major = proposed_parts[0]

    # Checked before the rule is applied, because both numbers are only visible here. A proposal that is
    # not ahead means git-cliff ran against the wrong history, and the release would either collide with
    # an existing tag or publish a version that reads as older than the one before it.
    if proposed_parts <= current_parts:
        raise ValueError(
            f"the proposed version {proposed!r} is not ahead of the current version {current!r}. "
            f"git-cliff was run against unexpected history; releasing this would either collide with an "
            f"existing tag or publish a version that reads as older."
        )

    # Normalised once, and it is the normalised form that is returned. Validating a stripped copy while
    # returning the original would let `" v2.0.0 "` pass the parser and come back with its spaces, and
    # `git check-ref-format 'refs/tags/ v2.0.0 '` refuses that -- so the release would fail at the tag,
    # with a message about ref format rather than about where the whitespace came from. The demoted path
    # is not exposed to it, since it builds its answer out of parsed numbers. The general shape:
    # validating a copy is not validating the thing you return.
    #
    # Not cosmetic in practice: the proposal arrives from `$(git-cliff ...)`, and command substitution
    # strips trailing newlines but nothing else.
    proposed = proposed.strip()
    prefix = "v" if proposed.startswith("v") else ""

    # The rule applies to exactly one situation: a 0.x package being pushed to 1.0 by a breaking change.
    # Everything else -- 0.x staying in 0.x, and any bump at 1.x or above -- is git-cliff's answer.
    demote = current_major == 0 and proposed_major > current_major and not allow_major_bump
    if not demote:
        return proposed

    # A minor bump, with the patch reset: this is a 0.x minor release, not a major one wearing a smaller
    # number. `0.13.2` -> `0.14.0`.
    return f"{prefix}{current_major}.{current_minor + 1}.0"


def main(argv: list[str] | None = None) -> int:
    """Print the version to release.

    Args:
        argv: Command-line arguments, for testing; defaults to `sys.argv[1:]`.

    Returns:
        The process exit status: 0 on success, 2 if the versions could not be used.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", required=True, help="the latest released tag")
    parser.add_argument("--proposed", required=True, help="the tag git-cliff proposes")
    parser.add_argument(
        "--allow-major-bump",
        action="store_true",
        help="accept a bump into the next major while the current major is 0",
    )
    arguments = parser.parse_args(argv)
    try:
        print(
            choose_version(
                current=arguments.current,
                proposed=arguments.proposed,
                allow_major_bump=arguments.allow_major_bump,
            )
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
