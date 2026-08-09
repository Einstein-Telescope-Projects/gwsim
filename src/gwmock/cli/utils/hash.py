# ruff: noqa: PLC0415
"""Contains utility functions for hashing operations.

``numpy``/``gwpy``/``h5py`` are imported lazily inside the content-hash helpers
so importing this module stays cheap; the heavy readers are only pulled in when
a file is actually decoded for content hashing.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger("gwmock")


def compute_file_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
    """Compute the hash of a file using the specified algorithm.

    Args:
        file_path: Path to the file.
        algorithm: Hashing algorithm to use (default is 'sha256').

    Returns:
        The computed hash as a hexadecimal string.
    """
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return f"{algorithm}:{hash_func.hexdigest()}"


def _hash_named_arrays(hash_func: hashlib._Hash, items: list[tuple[str, dict, object]]) -> None:
    """Fold a list of ``(label, scalar-meta, array)`` into *hash_func* canonically.

    Channels/datasets are hashed in the order given (callers sort first). Arrays
    are normalised to little-endian, C-contiguous bytes so the digest is stable
    across platforms and write order. Scalar metadata is packed as raw bytes (not
    ``repr``) to avoid float-formatting ambiguity.
    """
    import numpy as np

    for label, meta, array in items:
        hash_func.update(b"\x00CHAN\x00")
        hash_func.update(label.encode("utf-8"))
        for key in sorted(meta):
            value = meta[key]
            hash_func.update(f"\x00{key}=".encode())
            if isinstance(value, (int, float)):
                hash_func.update(np.asarray([value], dtype="<f8").tobytes())
            else:
                hash_func.update(str(value).encode("utf-8"))
        arr = np.ascontiguousarray(array)
        arr = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
        hash_func.update(f"\x00dtype={arr.dtype.str}\x00shape={tuple(arr.shape)}\x00".encode())
        hash_func.update(arr.tobytes())


def _read_content_items(file_path: Path) -> list[tuple[str, dict, object]] | None:
    """Decode *file_path* into ``(label, meta, array)`` items, or ``None`` if the
    format is not understood (caller then falls back to the raw-file hash)."""
    suffix = file_path.suffix.lower()

    if suffix == ".gwf":
        from gwpy.io.gwf import iter_channel_names
        from gwpy.timeseries import TimeSeriesDict

        # The framel reader does not auto-discover channels, so enumerate them.
        channels = sorted(iter_channel_names(str(file_path)))
        if not channels:
            return []
        tsd = TimeSeriesDict.read(str(file_path), channels=channels)
        items = []
        for name in channels:
            series = tsd[name]
            meta = {"t0": float(series.t0.value), "dt": float(series.dt.value), "n": int(series.size)}
            items.append((name, meta, series.value))
        return items

    if suffix in (".hdf5", ".h5"):
        import h5py

        names: list[str] = []
        with h5py.File(file_path, "r") as handle:
            handle.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
            # The epoch, the sample interval and the length, exactly as the GWF branch above records
            # them. Without these the digest saw only the samples, so an HDF5 file moved to a different
            # GPS time -- or written at a different rate -- hashed identically to the original. Measured:
            # both cases collided for `.hdf5` while `.gwf` distinguished them. That asymmetry did not
            # matter while HDF5 was the secondary format; it does now that it is the one a run writes by
            # default.
            #
            # gwpy stores them as the `x0` and `dx` dataset attributes. A dataset written by something
            # else may carry neither, and then the entry is hashed on its samples alone as before -- the
            # check is only as strong as what the writer recorded, which is why this reads attributes
            # rather than asserting them.
            items = []
            for name in sorted(names):
                dataset = handle[name]
                meta: dict[str, object] = {"n": int(dataset.shape[0]) if dataset.shape else 0}
                if "x0" in dataset.attrs:
                    meta["t0"] = float(dataset.attrs["x0"])
                if "dx" in dataset.attrs:
                    meta["dt"] = float(dataset.attrs["dx"])
                items.append((name, meta, dataset[()]))
            return items

    if suffix == ".npy":
        import numpy as np

        return [("array", {}, np.load(file_path))]

    if suffix == ".npz":
        import numpy as np

        with np.load(file_path) as data:
            return [(key, {}, data[key]) for key in sorted(data.files)]

    return None


def compute_content_hash(file_path: str | Path, algorithm: str = "sha256") -> str | None:
    """Compute a hash of the *scientific content* of an output file.

    Unlike :func:`compute_file_hash`, which hashes the raw container bytes, this
    decodes the file and hashes the decoded data (channel/dataset names plus the
    sample arrays in a canonical little-endian layout). Two files holding
    identical data therefore share a content hash even when their container bytes
    differ -- e.g. GWF frames embed a write-time timestamp and the frame-library
    version string, so byte-identical GWF output is not achievable across
    machines, but the content hash is.

    Args:
        file_path: Path to the output file (``.gwf``, ``.hdf5``/``.h5``, ``.npy``,
            ``.npz``).
        algorithm: Hashing algorithm to use (default ``"sha256"``).

    Returns:
        ``"<algorithm>:<hexdigest>"``, or ``None`` if the format is unsupported or
        the file could not be decoded (the caller should fall back to the
        raw-file hash in that case).
    """
    path = Path(file_path)
    try:
        items = _read_content_items(path)
    except Exception as exc:
        logger.debug("Could not compute content hash for %s: %s", path, exc)
        return None
    if items is None:
        return None
    hash_func = hashlib.new(algorithm)
    _hash_named_arrays(hash_func, items)
    return f"{algorithm}:{hash_func.hexdigest()}"
