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

"""The content hash must notice a change of grid, in every format a run can write.

The digest folds each channel's samples together with the scalar metadata that says *where* those
samples sit -- epoch, sample interval, length. The GWF branch always recorded those; the HDF5 branch
recorded none of them, so two HDF5 files holding identical samples at different GPS times, or at
different sample rates, produced the same hash. Measured, both collided.

That asymmetry was survivable while HDF5 was the secondary format. It is not now that HDF5 is what a run
writes by default, because the weaker check would become the one every user gets.

These tests are parametrised over both formats deliberately: the point is not that HDF5 works, it is that
the two answer the same question.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock.cli.utils.hash import compute_content_hash

pytestmark = pytest.mark.unit

FORMATS = [".gwf", ".hdf5"]
SAMPLES = np.linspace(0.0, 1.0, 1024)


def _written(
    directory: Path,
    suffix: str,
    *,
    t0: float = 1000000000.0,
    rate: float = 256.0,
    samples: np.ndarray | None = None,
) -> Path:
    """Write one channel, however it differs from the baseline.

    `name` is passed as well as `channel`: gwpy's HDF5 writer names the dataset from it and raises
    without one, so a series built for a variation has to be constructed the same way as the baseline
    rather than with the minimum that GWF happens to accept.
    """
    series = TimeSeries(
        SAMPLES if samples is None else samples,
        t0=t0,
        sample_rate=rate,
        channel="H1:STRAIN",
        unit="strain",
        name="H1:STRAIN",
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"probe{suffix}"
    series.write(str(path), overwrite=True)
    return path


@pytest.mark.parametrize("suffix", FORMATS)
def test_the_same_file_hashes_the_same(tmp_path: Path, suffix: str) -> None:
    """The baseline the others are read against: nothing moved, nothing changes."""
    first = compute_content_hash(_written(tmp_path / "a", suffix))
    second = compute_content_hash(_written(tmp_path / "b", suffix))
    assert first is not None
    assert first == second


@pytest.mark.parametrize("suffix", FORMATS)
def test_moving_the_epoch_changes_the_hash(tmp_path: Path, suffix: str) -> None:
    """Identical samples at a different GPS time are different data.

    This is the case that collided for HDF5: a segment written for the wrong epoch was indistinguishable
    from the right one.
    """
    original = compute_content_hash(_written(tmp_path / "a", suffix))
    moved = compute_content_hash(_written(tmp_path / "b", suffix, t0=1000000512.0))
    assert original != moved


@pytest.mark.parametrize("suffix", FORMATS)
def test_changing_the_sample_rate_changes_the_hash(tmp_path: Path, suffix: str) -> None:
    """Same samples, half the span. Also collided for HDF5."""
    original = compute_content_hash(_written(tmp_path / "a", suffix))
    resampled = compute_content_hash(_written(tmp_path / "b", suffix, rate=512.0))
    assert original != resampled


@pytest.mark.parametrize("suffix", FORMATS)
def test_changing_a_sample_changes_the_hash(tmp_path: Path, suffix: str) -> None:
    """The property that already held, kept explicit so a metadata-only digest cannot pass this file."""
    original = compute_content_hash(_written(tmp_path / "a", suffix))
    altered = SAMPLES.copy()
    altered[17] += 1e-3
    assert original != compute_content_hash(_written(tmp_path / "b", suffix, samples=altered))


def test_an_hdf5_dataset_without_grid_attributes_still_hashes(tmp_path: Path) -> None:
    """A file written by something other than gwpy has no `x0`/`dx`, and must not raise.

    The check is only as strong as what the writer recorded. Reading the attributes when they are there
    and hashing the samples alone when they are not is the honest behaviour -- the alternative, demanding
    them, would refuse to hash a perfectly good file somebody else produced.
    """
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "bare.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("strain", data=SAMPLES)

    assert compute_content_hash(path) is not None
