"""Unit tests for content/file hashing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gwmock.cli.utils.hash import compute_content_hash, compute_file_hash


def test_content_hash_identical_data_matches(tmp_path: Path):
    """Two files holding identical data share a content hash."""
    a = tmp_path / "a.npy"
    b = tmp_path / "b.npy"
    data = np.arange(32, dtype="float64")
    np.save(a, data)
    np.save(b, data)
    assert compute_content_hash(a) == compute_content_hash(b)


def test_content_hash_changes_with_data(tmp_path: Path):
    """Different data yields a different content hash."""
    a = tmp_path / "a.npy"
    b = tmp_path / "b.npy"
    np.save(a, np.arange(32, dtype="float64"))
    np.save(b, np.arange(32, dtype="float64") + 1.0)
    assert compute_content_hash(a) != compute_content_hash(b)


def test_content_hash_is_byteorder_independent(tmp_path: Path):
    """Content hash ignores on-disk byte order (canonical little-endian)."""
    le = tmp_path / "le.npy"
    be = tmp_path / "be.npy"
    data = np.linspace(0, 1, 16)
    np.save(le, data.astype("<f8"))
    np.save(be, data.astype(">f8"))
    assert compute_content_hash(le) == compute_content_hash(be)


def test_content_hash_unsupported_format_returns_none(tmp_path: Path):
    """Unsupported/undecodable formats return None (caller falls back to bytes)."""
    txt = tmp_path / "note.txt"
    txt.write_text("not an array")
    assert compute_content_hash(txt) is None


def test_content_hash_prefixed_like_file_hash(tmp_path: Path):
    """Content hash carries the ``<algorithm>:`` prefix, like compute_file_hash."""
    a = tmp_path / "a.npy"
    np.save(a, np.zeros(4))
    assert compute_content_hash(a).startswith("sha256:")
    assert compute_file_hash(a).startswith("sha256:")
