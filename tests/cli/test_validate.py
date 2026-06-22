"""Unit tests for the validate command."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
import yaml

from gwmock.cli.utils.hash import compute_file_hash
from gwmock.cli.validate import validate_command


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_metadata(temp_dir):
    """Create sample metadata and output files for testing."""
    # Create a sample output file
    output_file = temp_dir / "test_output.gwf"
    output_file.write_text("dummy data")

    # Compute hash
    hash_value = compute_file_hash(output_file)

    # Create metadata with proper structure
    metadata = {
        "author": "test_user",
        "email": "test@example.com",
        "timestamp": "2023-01-01T00:00:00Z",
        "output_files": ["test_output.gwf"],
        "file_hashes": {"test_output.gwf": hash_value},
        "globals_config": {"output_directory": str(temp_dir)},
    }

    metadata_file = temp_dir / "test.metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f)

    return metadata_file, output_file


def test_validate_command_success(sample_metadata, temp_dir, capsys):
    """Test successful validation."""
    metadata_file, _output_file = sample_metadata

    # Change to temp_dir so relative paths work
    with patch("os.getcwd", return_value=str(temp_dir)):
        validate_command([], metadata_paths=[str(metadata_file)])

    captured = capsys.readouterr()
    assert "test_output.gwf" in captured.out
    assert "1/1 files passed validation" in captured.out


def test_validate_command_failure(sample_metadata, temp_dir, capsys):
    """Test validation failure when file is modified."""
    metadata_file, output_file = sample_metadata

    # Modify the file
    output_file.write_text("modified data")

    with patch("os.getcwd", return_value=str(temp_dir)), pytest.raises(typer.Exit):
        validate_command([], metadata_paths=[str(metadata_file)])

    captured = capsys.readouterr()
    assert "test_output.gwf" in captured.out
    assert "0/1 files passed validation" in captured.out


def test_validate_command_with_pattern(sample_metadata, temp_dir, capsys):
    """Test validation with pattern filtering."""
    metadata_file, _output_file = sample_metadata

    # Create another file that doesn't match pattern
    other_file = temp_dir / "other_output.gwf"
    other_file.write_text("other data")

    # Load existing metadata and add the other file
    with open(metadata_file) as f:
        metadata = yaml.safe_load(f)
    metadata["output_files"].append("other_output.gwf")
    metadata["file_hashes"]["other_output.gwf"] = compute_file_hash(other_file)
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f)

    with patch("os.getcwd", return_value=str(temp_dir)):
        validate_command([], metadata_paths=[str(metadata_file)], pattern="*test*")

    captured = capsys.readouterr()
    assert "test_output.gwf" in captured.out
    assert "other_output.gwf" not in captured.out
    assert "1/1 files passed validation" in captured.out


def test_validate_command_directory(temp_dir, capsys):
    """Test validation with directory input."""
    # Create subdirectory with files
    sub_dir = temp_dir / "subdir"
    sub_dir.mkdir()

    output_file = sub_dir / "test_output.gwf"
    output_file.write_text("dummy data")
    hash_value = compute_file_hash(output_file)

    metadata = {
        "author": "test_user",
        "email": "test@example.com",
        "timestamp": "2023-01-01T00:00:00Z",
        "output_files": ["test_output.gwf"],
        "file_hashes": {"test_output.gwf": hash_value},
        "globals_config": {"output_directory": str(sub_dir)},
    }

    metadata_file = sub_dir / "test.metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f)

    with patch("os.getcwd", return_value=str(temp_dir)):
        validate_command([str(sub_dir)], metadata_paths=[str(sub_dir)])

    captured = capsys.readouterr()
    assert "test_output.gwf" in captured.out
    assert "1/1 files passed validation" in captured.out


def test_validate_command_missing_file(sample_metadata, temp_dir, capsys):
    """Test validation when output file is missing."""
    metadata_file, output_file = sample_metadata

    # Remove the output file
    output_file.unlink()

    with patch("os.getcwd", return_value=str(temp_dir)), pytest.raises(typer.Exit):
        validate_command([], metadata_paths=[str(metadata_file)])

    captured = capsys.readouterr()
    assert "test_output.gwf" in captured.out
    assert "File not found" in captured.out


def test_validate_command_no_metadata_files(temp_dir):
    """Test validation with no metadata files found."""
    with patch("os.getcwd", return_value=str(temp_dir)), pytest.raises(typer.Exit) as exc_info:
        validate_command([str(temp_dir / "nonexistent")], metadata_paths=[])

    # The command should exit with code 1 when no metadata files are found
    assert exc_info.value.exit_code == 1


def test_validate_command_metadata_discovery_priority(temp_dir, capsys):
    """Test that metadata discovery checks existing metadata files first."""
    # Create output file
    output_file = temp_dir / "test_output.gwf"
    output_file.write_text("dummy data")
    hash_value = compute_file_hash(output_file)

    # Create metadata file that contains the output file
    metadata = {
        "author": "test_user",
        "email": "test@example.com",
        "timestamp": "2023-01-01T00:00:00Z",
        "output_files": ["test_output.gwf"],
        "file_hashes": {"test_output.gwf": hash_value},
        "globals_config": {"output_directory": str(temp_dir)},
    }

    metadata_file = temp_dir / "test.metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f)

    # Test 1: Provide both metadata file and output file - should use the provided metadata
    with patch("os.getcwd", return_value=str(temp_dir)):
        validate_command([str(output_file)], metadata_paths=[str(metadata_file)])

    captured = capsys.readouterr()
    assert "test_output.gwf" in captured.out
    assert "1/1 files passed validation" in captured.out


def test_validate_command_outputs_in_subdirectory(temp_dir, capsys):
    """Outputs recorded with a sub-directory ``path`` must not be double-counted.

    Regression test: metadata produced by ``simulate`` records each output under
    ``outputs`` as a path relative to ``working-directory`` (e.g.
    ``output/signal/foo.gwf``). Reconstructing the expected location from only
    the basename dropped the sub-directory, so every file was reported once as
    PASS (from the directory scan) and once as "File not found" (from the
    metadata reconstruction) -- e.g. "3/6 files passed" on a clean run.
    """
    # Lay out files the way ``simulate`` does: <working-dir>/output/<kind>/<file>
    signal_dir = temp_dir / "output" / "signal"
    noise_dir = temp_dir / "output" / "noise"
    signal_dir.mkdir(parents=True)
    noise_dir.mkdir(parents=True)

    signal_file = signal_dir / "sig.gwf"
    noise_file = noise_dir / "noi.gwf"
    signal_file.write_text("signal data")
    noise_file.write_text("noise data")

    metadata = {
        "author": "test_user",
        "email": "test@example.com",
        "timestamp": "2023-01-01T00:00:00Z",
        # New-style ``outputs`` records with sub-directory paths.
        "outputs": [
            {"path": "output/signal/sig.gwf", "sha256": compute_file_hash(signal_file)},
            {"path": "output/noise/noi.gwf", "sha256": compute_file_hash(noise_file)},
        ],
        "file_hashes": {
            "sig.gwf": compute_file_hash(signal_file),
            "noi.gwf": compute_file_hash(noise_file),
        },
        "globals_config": {"working-directory": str(temp_dir)},
    }

    metadata_dir = temp_dir / "metadata"
    metadata_dir.mkdir()
    metadata_file = metadata_dir / "orchestration-0.metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f)

    with patch("os.getcwd", return_value=str(temp_dir)):
        validate_command([str(temp_dir / "output")], metadata_paths=[str(metadata_dir)])

    captured = capsys.readouterr()
    # Exactly the two real files pass; nothing is reported as missing.
    assert "2/2 files passed validation" in captured.out
    assert "File not found" not in captured.out


def test_validate_command_output_file_discovery(temp_dir, capsys):
    """Test metadata discovery for output files when no metadata provided."""
    # Create output file
    output_file = temp_dir / "test_output.gwf"
    output_file.write_text("dummy data")
    hash_value = compute_file_hash(output_file)

    # Create metadata directory and file
    metadata_dir = temp_dir / "metadata"
    metadata_dir.mkdir()

    metadata = {
        "author": "test_user",
        "email": "test@example.com",
        "timestamp": "2023-01-01T00:00:00Z",
        "output_files": ["test_output.gwf"],
        "file_hashes": {"test_output.gwf": hash_value},
        "globals_config": {"output_directory": str(temp_dir)},
    }

    metadata_file = metadata_dir / "test.metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f)

    # Test: Provide only output file - should discover metadata in directory
    with patch("os.getcwd", return_value=str(temp_dir)):
        validate_command([str(output_file)])

    captured = capsys.readouterr()
    assert "test_output.gwf" in captured.out
    assert "1/1 files passed validation" in captured.out
