"""Check the cached LALPulsar ephemeris tables against their recorded hashes.

Run by CI after fetching them, and usable by hand:

    uv run python tests/data/verify_ephemeris.py

These are physics inputs, not build artefacts: they define the Earth and Sun trajectories the
barycentring uses, so their content changes generated strain. Ripple takes them from lalsuite
``master`` -- an unpinned moving target -- and its fetch is satisfied by the file merely existing,
so without this a changed upstream table or a corrupt cached copy is used silently and surfaces as
a parse error deep inside a test, attributed to nothing.

Exits non-zero and names the offending file with expected and actual digests. The message
distinguishes the two causes, because "the cache is corrupt" and "the reference trajectories moved"
want very different responses.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

MANIFEST = pathlib.Path(__file__).with_name("ephemeris.sha256")


def cache_directory() -> pathlib.Path:
    """Return where ripple keeps its cached tables, asking ripple rather than guessing.

    Ripple's default is platform-dependent -- ``~/Library/Caches`` on macOS against
    ``$XDG_CACHE_HOME`` or ``~/.cache`` elsewhere -- and ``RIPPLEGW_CACHE_DIR`` overrides both.
    Reproducing that logic here would be one more place to drift from it.
    """
    from ripplegw.waveforms.cw.ephemeris import _cache_dir

    return pathlib.Path(_cache_dir())


def main() -> int:
    """Return 0 when every table matches, 1 otherwise."""
    cache = cache_directory()
    expected_by_name: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split()
        expected_by_name[name] = digest

    if not expected_by_name:
        print(f"::error::{MANIFEST} lists no tables, so this check would pass vacuously")
        return 1

    failures: list[str] = []
    for name, expected in sorted(expected_by_name.items()):
        path = cache / name
        if not path.is_file():
            failures.append(f"{name}: missing from {cache}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")

    if failures:
        print(f"::error::ephemeris tables do not match {MANIFEST}")
        for failure in failures:
            print(f"::error::{failure}")
        print(
            "::error::either the cache holds a corrupt copy, or lalsuite master changed the "
            "tables -- the second means the reference trajectories moved and is worth "
            "understanding before updating the manifest"
        )
        return 1

    print(f"verified {len(expected_by_name)} ephemeris tables under {cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
