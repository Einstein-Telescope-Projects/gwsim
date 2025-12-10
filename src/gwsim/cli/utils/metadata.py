"""Utility functions for handling metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def save_metadata_with_external_state(
    metadata: dict[str, Any],
    metadata_file: Path | str,
    metadata_dir: Path | str | None = None,
    encoding: str = "utf-8",
) -> None:
    """Save metadata to a YAML file, extracting large numpy arrays to external .npy files.

    Args:
        metadata: Metadata dictionary to save.
        metadata_file: Path to the metadata YAML file.
        metadata_dir: Directory to save external numpy array files. If None, uses the directory of metadata_file.
        encoding: File encoding for the YAML file. Default is 'utf-8'.
    """
    metadata_file = Path(metadata_file)
    if metadata_dir is None:
        metadata_dir = metadata_file.parent
    else:
        metadata_dir = Path(metadata_dir)

    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Process pre_batch_state to extract all numpy arrays
    metadata_copy = metadata.copy()
    if "pre_batch_state" in metadata_copy:
        external_state = {}
        for key, value in metadata_copy["pre_batch_state"].items():
            if isinstance(value, np.ndarray):
                # Save all arrays to external files
                state_file = metadata_dir / f"{metadata_file.stem}_state_{key}.npy"
                np.save(state_file, value)
                external_state[key] = {
                    "_external_file": True,
                    "dtype": str(value.dtype),
                    "shape": value.shape,
                    "size_bytes": value.nbytes,
                    "file": state_file.name,
                }
            else:
                external_state[key] = value
        metadata_copy["pre_batch_state"] = external_state

    # Write metadata YAML
    with metadata_file.open("w", encoding=encoding) as f:
        yaml.safe_dump(metadata_copy, f, default_flow_style=False, sort_keys=False)
