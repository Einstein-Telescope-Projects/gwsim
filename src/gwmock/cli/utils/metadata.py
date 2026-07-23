"""Utilities for reading and writing versioned metadata records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.2.0"
_SCHEMA_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class SubpackageVersions(BaseModel):
    """Pinned public subpackage versions used to generate a run."""

    gwmock_signal: str | None = None
    gwmock_noise: str | None = None
    gwmock_pop: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class PopulationSection(BaseModel):
    """Population provenance stored in a metadata record."""

    backend: str
    source_type: str | None = None
    n_events: int | None = None
    parameter_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalSection(BaseModel):
    """Signal provenance stored in a metadata record."""

    backend: str
    waveform_model: str | None = None
    detector_network: list[str] = Field(default_factory=list)
    # Source parameters of the signals that merge in this batch's frame(s), in
    # injection order: [{"event_id": int, "parameters": {...}}]. An event is
    # attributed to the frame its coa_time falls in; content extending into
    # adjacent frames is not cross-listed. Empty for stationary/SGWB segments.
    injections: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NoiseSection(BaseModel):
    """Noise provenance stored in a metadata record."""

    backend: str
    psd: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutputRecord(BaseModel):
    """One generated artifact tracked by the metadata record."""

    kind: str
    path: str
    channels: list[str] = Field(default_factory=list)
    t0: float | int | None = None
    duration: float | int | None = None
    sha256: str | None = None
    content_sha256: str | None = None


class HostRecord(BaseModel):
    """Host fingerprint for provenance reporting."""

    platform: str
    python: str
    cpu: str
    git_sha: str | None = None


class MetadataRecord(BaseModel):
    """Versioned metadata record written for each simulation batch."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    gwmock_version: str | None = None
    subpackage_versions: SubpackageVersions
    config: dict[str, Any]
    config_sha256: str
    seed: int | None = None
    segment_seeds: list[int] = Field(default_factory=list)
    population: PopulationSection | None = None
    signal: SignalSection | None = None
    noise: NoiseSection | None = None
    outputs: list[OutputRecord] = Field(default_factory=list)
    host: HostRecord

    model_config = ConfigDict(extra="allow")

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        """Reject malformed versions and unknown major schema revisions."""
        match = _SCHEMA_VERSION_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError("schema_version must follow semantic versioning (MAJOR.MINOR.PATCH).")

        current_major = SCHEMA_VERSION.split(".", maxsplit=1)[0]
        if match.group(1) != current_major:
            raise ValueError(f"Unsupported metadata schema major version {match.group(1)}; expected {current_major}.")
        return value


def _normalize_json_value(value: Any) -> Any:
    """Convert Python objects to JSON-safe values."""
    normalized = value
    if isinstance(value, Path):
        normalized = str(value)
    elif hasattr(value, "unit") and hasattr(value, "value"):
        normalized = _normalize_json_value(value.value)
    elif isinstance(value, np.ndarray):
        normalized = value.tolist()
    elif isinstance(value, np.generic):
        normalized = value.item()
    elif isinstance(value, dict):
        normalized = {str(key): _normalize_json_value(val) for key, val in value.items()}
    elif isinstance(value, (list, tuple, set)):
        normalized = [_normalize_json_value(item) for item in value]
    return normalized


def _extract_external_state(metadata: dict[str, Any], metadata_file: Path, metadata_dir: Path) -> dict[str, Any]:
    """Replace raw numpy arrays in pre_batch_state with external file references."""
    metadata_copy = _normalize_json_value(metadata)
    if "pre_batch_state" not in metadata_copy:
        return metadata_copy

    external_state: dict[str, Any] = {}
    raw_state = metadata.get("pre_batch_state", {})
    if not isinstance(raw_state, dict):
        metadata_copy["pre_batch_state"] = _normalize_json_value(raw_state)
        return metadata_copy

    for key, value in raw_state.items():
        if type(value) is np.ndarray:  # pylint: disable=unidiomatic-typecheck
            state_file = f"{metadata_file.stem}_state_{key}.npy"
            np.save(metadata_dir / state_file, value)
            external_state[key] = {
                "_external_file": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "size_bytes": value.nbytes,
                "file": state_file,
            }
        else:
            external_state[key] = _normalize_json_value(value)

    metadata_copy["pre_batch_state"] = external_state
    return metadata_copy


def save_metadata_with_external_state(
    metadata: dict[str, Any],
    metadata_file: Path | str,
    metadata_dir: Path | str | None = None,
    encoding: str = "utf-8",
) -> None:
    """Save metadata, extracting large numpy arrays in ``pre_batch_state`` to external ``.npy`` files."""
    metadata_file = Path(metadata_file)
    metadata_dir = metadata_file.parent if metadata_dir is None else Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_copy = _extract_external_state(metadata, metadata_file, metadata_dir)

    with metadata_file.open("w", encoding=encoding) as handle:
        if metadata_file.suffix.lower() == ".json":
            json.dump(metadata_copy, handle, indent=2, sort_keys=True)
            handle.write("\n")
        else:
            yaml.safe_dump(metadata_copy, handle, default_flow_style=False, sort_keys=False)


def load_metadata_with_external_state(
    metadata_file: Path | str,
    metadata_dir: Path | str | None = None,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """Load metadata and reconstruct external numpy arrays from ``.npy`` files."""
    metadata_file = Path(metadata_file)
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata_dir = metadata_file.parent if metadata_dir is None else Path(metadata_dir)

    try:
        with metadata_file.open("r", encoding=encoding) as handle:
            metadata = json.load(handle) if metadata_file.suffix.lower() == ".json" else yaml.safe_load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse metadata JSON: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse metadata YAML: {exc}") from exc

    if "pre_batch_state" in metadata:
        reconstructed_state = {}
        for key, value in metadata["pre_batch_state"].items():
            if isinstance(value, dict) and value.get("_external_file", False):
                state_file = metadata_dir / value["file"]
                if not state_file.exists():
                    raise FileNotFoundError(f"External state file not found: {state_file}")
                reconstructed_state[key] = np.load(state_file)
            else:
                reconstructed_state[key] = value
        metadata["pre_batch_state"] = reconstructed_state

    return metadata


def save_metadata_record(
    metadata: MetadataRecord | dict[str, Any],
    metadata_file: Path | str,
    metadata_dir: Path | str | None = None,
    encoding: str = "utf-8",
) -> None:
    """Validate and persist a versioned metadata record."""
    record = metadata if isinstance(metadata, MetadataRecord) else MetadataRecord.model_validate(metadata)
    save_metadata_with_external_state(
        metadata=_normalize_json_value(record.model_dump(mode="python", by_alias=True, exclude_none=True)),
        metadata_file=metadata_file,
        metadata_dir=metadata_dir,
        encoding=encoding,
    )


def load_metadata_record(
    metadata_file: Path | str,
    metadata_dir: Path | str | None = None,
    encoding: str = "utf-8",
) -> MetadataRecord:
    """Load and validate a versioned metadata record."""
    metadata = load_metadata_with_external_state(
        metadata_file=metadata_file, metadata_dir=metadata_dir, encoding=encoding
    )
    return MetadataRecord.model_validate(metadata)
