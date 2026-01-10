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
    metadata_dir = metadata_file.parent if metadata_dir is None else Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    # Process pre_batch_state to extract all numpy arrays
    metadata_copy = metadata.copy()
    if "pre_batch_state" in metadata_copy:
        external_state = {}
        for key, value in metadata_copy["pre_batch_state"].items():
            # Use type() to avoid issues with subclasses of np.ndarray
            if type(value) is np.ndarray:  # pylint: disable=unidiomatic-typecheck
                # Save all arrays to external files
                state_file = f"{metadata_file.stem}_state_{key}.npy"
                np.save(metadata_dir / state_file, value)
                external_state[key] = {
                    "_external_file": True,
                    "dtype": str(value.dtype),
                    "shape": value.shape,
                    "size_bytes": value.nbytes,
                    "file": state_file,
                }
            else:
                external_state[key] = value
        metadata_copy["pre_batch_state"] = external_state

    # Write metadata YAML
    with metadata_file.open("w", encoding=encoding) as f:
        yaml.safe_dump(metadata_copy, f, default_flow_style=False, sort_keys=False)


def load_metadata_with_external_state(
    metadata_file: Path | str,
    metadata_dir: Path | str | None = None,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """Load metadata from a YAML file, reconstructing large numpy arrays from external .npy files.

    Args:
        metadata_file: Path to the metadata YAML file.
        metadata_dir: Directory where external numpy array files are stored.
            If None, uses the directory of metadata_file.
        encoding: File encoding for the YAML file. Default is 'utf-8'.

    Returns:
        Loaded metadata dictionary.
    """
    metadata_file = Path(metadata_file)
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata_dir = metadata_file.parent if metadata_dir is None else Path(metadata_dir)

    # Load YAML metadata
    try:
        with metadata_file.open("r", encoding=encoding) as f:
            metadata = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse metadata YAML: {e}") from e

    # Reconstruct pre_batch_state from external files
    if "pre_batch_state" in metadata:
        reconstructed_state = {}
        for key, value in metadata["pre_batch_state"].items():
            if isinstance(value, dict) and value.get("_external_file", False):
                # Load external numpy array
                state_file = metadata_dir / value["file"]
                if not state_file.exists():
                    raise FileNotFoundError(f"External state file not found: {state_file}")
                array = np.load(state_file)
                reconstructed_state[key] = array
            else:
                reconstructed_state[key] = value
        metadata["pre_batch_state"] = reconstructed_state
    return metadata
