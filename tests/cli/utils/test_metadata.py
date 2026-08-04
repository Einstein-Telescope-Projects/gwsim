"""Integration tests for the versioned metadata schema."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from gwmock.cli.simulate import _simulate_impl
from gwmock.cli.utils.hash import compute_file_hash
from gwmock.cli.utils.metadata import MetadataRecord, SubpackageVersions, load_metadata_record


def _write_mock_config(tmp_path: Path) -> Path:
    """Write a deterministic one-batch config used by metadata schema tests."""
    config = {
        "globals": {
            "working-directory": str(tmp_path),
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "simulators": {
            "mock": {
                "class": "tests.cli.test_cli_simulate.MockSimulator",
                "arguments": {
                    "seed": 42,
                    "max_samples": 1,
                },
                "output": {
                    "file_name": "data.json",
                },
            }
        },
    }

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_file


def _strip_nondeterministic_fields(metadata: dict) -> dict:
    """Remove fields intentionally excluded from reproducibility comparisons."""
    stripped = dict(metadata)
    stripped.pop("host", None)
    stripped.pop("timestamp", None)
    return stripped


@pytest.mark.skip(reason="Legacy simulator configs removed; orchestration tests cover metadata generation.")
def test_metadata_record_uses_versioned_json_schema(tmp_path: Path) -> None:
    """A small run should emit a validated JSON metadata record."""
    config_file = _write_mock_config(tmp_path)

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    metadata_path = tmp_path / "metadata" / "mock-0.metadata.json"
    record = load_metadata_record(metadata_path)

    assert record.schema_version == "1.4.0"
    assert record.config_sha256 == compute_file_hash(config_file)
    assert record.seed == 42
    assert record.outputs[0].path == "output/data.json"
    assert record.outputs[0].sha256 is not None


@pytest.mark.skip(reason="Legacy simulator configs removed; orchestration tests cover metadata generation.")
def test_metadata_record_is_stable_for_same_config_and_seed(tmp_path: Path) -> None:
    """Repeated runs with the same config and seed should reproduce the same metadata."""
    config_file = _write_mock_config(tmp_path)

    _simulate_impl(str(config_file), overwrite=True, metadata=True)
    first = json.loads((tmp_path / "metadata" / "mock-0.metadata.json").read_text())

    shutil.rmtree(tmp_path / "metadata")
    shutil.rmtree(tmp_path / "output")

    _simulate_impl(str(config_file), overwrite=True, metadata=True)
    second = json.loads((tmp_path / "metadata" / "mock-0.metadata.json").read_text())

    assert _strip_nondeterministic_fields(first) == _strip_nondeterministic_fields(second)


def test_subpackage_versions_accepts_partial_dict() -> None:
    """Omitted subpackage keys should validate as optional (Pydantic v2)."""
    empty = SubpackageVersions.model_validate({})
    assert empty.gwmock_signal is None
    assert empty.gwmock_noise is None
    assert empty.gwmock_pop is None

    partial = SubpackageVersions.model_validate({"gwmock_signal": "0.1.0"})
    assert partial.gwmock_signal == "0.1.0"
    assert partial.gwmock_noise is None
    assert partial.gwmock_pop is None


def test_metadata_record_accepts_missing_gwmock_version() -> None:
    """``gwmock_version`` may be unknown when dependency resolution returns None."""
    minimal = {
        "subpackage_versions": {},
        "config": {},
        "config_sha256": "0" * 64,
        "host": {"platform": "x", "python": "3.12", "cpu": "cpu"},
    }
    record = MetadataRecord.model_validate(minimal)
    assert record.gwmock_version is None

    with_version = {**minimal, "gwmock_version": None}
    assert MetadataRecord.model_validate(with_version).gwmock_version is None
