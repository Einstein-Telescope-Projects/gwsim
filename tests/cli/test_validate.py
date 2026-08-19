"""Unit tests for the validate command."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import typer
import yaml

from gwmock.cli.utils.hash import compute_content_hash, compute_file_hash
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


def _npy_output_with_metadata(temp_dir, *, file_hash: str, content_hash: str):
    """Write an ``output/data.npy`` and a metadata file carrying the given hashes."""
    out_dir = temp_dir / "output"
    out_dir.mkdir(exist_ok=True)
    data_file = out_dir / "data.npy"
    np.save(data_file, np.arange(16, dtype="float64"))

    metadata = {
        "author": "test_user",
        "email": "test@example.com",
        "timestamp": "2023-01-01T00:00:00Z",
        "outputs": [{"kind": "signal", "path": "output/data.npy", "sha256": file_hash, "content_sha256": content_hash}],
        "file_hashes": {"data.npy": file_hash},
        "content_hashes": {"data.npy": content_hash},
        "globals_config": {"working-directory": str(temp_dir)},
    }
    meta_dir = temp_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)
    meta_file = meta_dir / "orchestration-0.metadata.yaml"
    with open(meta_file, "w") as f:
        yaml.dump(metadata, f)
    return data_file, meta_dir


def test_validate_content_match_but_bytes_differ_is_repackaged(temp_dir, capsys):
    """Content hash matches but the recorded file hash does not -> PASS (repackaged).

    This is the normal reproduce-from-metadata case for GWF frames, whose
    container bytes carry a write-time timestamp.
    """
    # Write the data file first, compute its real content hash, then record it
    # alongside a deliberately wrong byte hash.
    data_file, _ = _npy_output_with_metadata(temp_dir, file_hash="sha256:" + "0" * 64, content_hash="")
    content_hash = compute_content_hash(data_file)
    _, meta_dir = _npy_output_with_metadata(temp_dir, file_hash="sha256:" + "0" * 64, content_hash=content_hash)

    with patch("os.getcwd", return_value=str(temp_dir)):
        validate_command([str(temp_dir / "output")], metadata_paths=[str(meta_dir)])

    captured = capsys.readouterr()
    assert "repackaged" in captured.out
    assert "1/1 files passed validation" in captured.out


def test_validate_content_mismatch_fails(temp_dir, capsys):
    """A wrong content hash fails validation regardless of the byte hash."""
    data_file = temp_dir / "output" / "data.npy"
    temp_dir.joinpath("output").mkdir(exist_ok=True)
    np.save(data_file, np.arange(16, dtype="float64"))
    real_file_hash = compute_file_hash(data_file)
    _, meta_dir = _npy_output_with_metadata(
        temp_dir,
        file_hash=real_file_hash,  # bytes are fine...
        content_hash="sha256:" + "f" * 64,  # ...but content hash is wrong
    )

    with patch("os.getcwd", return_value=str(temp_dir)), pytest.raises(typer.Exit):
        validate_command([str(temp_dir / "output")], metadata_paths=[str(meta_dir)])

    captured = capsys.readouterr()
    assert "CONTENT MISMATCH" in captured.out
    assert "0/1 files passed validation" in captured.out


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


@pytest.fixture
def wide_console(monkeypatch):
    """Render the results table wide enough that its status cells are not wrapped."""
    monkeypatch.setenv("COLUMNS", "200")


def _metadata_with(temp_dir: Path, payload: dict, name: str = "run-0.metadata.yaml") -> Path:
    """Write one metadata file into ``temp_dir`` and return its path."""
    metadata_file = temp_dir / name
    with open(metadata_file, "w") as f:
        yaml.dump(payload, f)
    return metadata_file


@pytest.mark.usefixtures("wide_console")
class TestValidateReportedVerdicts:
    """The verdict the table reports for each way a file can disagree with its record.

    A user reads this table to decide whether a run's outputs are trustworthy, and
    the exit status decides whether a pipeline continues. Both are asserted per
    verdict: "bytes differ" and "content differs" mean different things and must
    not collapse into one another.
    """

    def test_a_byte_hash_mismatch_in_legacy_metadata_is_a_hash_mismatch(self, sample_metadata, temp_dir, capsys):
        """Metadata with no content hash falls back to the byte hash as the verdict."""
        _metadata_file, output_file = sample_metadata
        output_file.write_text("modified data")

        with patch("os.getcwd", return_value=str(temp_dir)), pytest.raises(typer.Exit):
            validate_command([], metadata_paths=[str(_metadata_file)])

        captured = capsys.readouterr()
        assert "HASH MISMATCH" in captured.out
        assert "1 files failed validation" in captured.out

    def test_metadata_without_any_hash_cannot_validate(self, temp_dir, capsys):
        """A record with no hash for the file is a failure, not a pass by default."""
        output_file = temp_dir / "data.gwf"
        output_file.write_text("data")
        metadata_file = _metadata_with(
            temp_dir,
            {
                "output_files": ["data.gwf"],
                "globals_config": {"output_directory": str(temp_dir)},
            },
        )

        with patch("os.getcwd", return_value=str(temp_dir)), pytest.raises(typer.Exit):
            validate_command([], metadata_paths=[str(metadata_file)])

        assert "No hash in metadata" in capsys.readouterr().out

    def test_a_recorded_output_that_is_absent_is_reported_missing(self, temp_dir, capsys):
        """A recorded output that never landed is a distinct verdict from a wrong hash."""
        metadata_file = _metadata_with(
            temp_dir,
            {
                "output_files": ["gone.gwf"],
                "file_hashes": {"gone.gwf": "sha256:" + "0" * 64},
                "globals_config": {"output_directory": str(temp_dir)},
            },
        )

        with patch("os.getcwd", return_value=str(temp_dir)), pytest.raises(typer.Exit):
            validate_command([], metadata_paths=[str(metadata_file)])

        assert "File not found" in capsys.readouterr().out

    def test_unreadable_metadata_is_reported_not_silently_skipped(self, temp_dir, capsys, caplog):
        """Metadata that cannot be parsed is logged, so a run is never "validated" by omission."""
        output_file = temp_dir / "data.gwf"
        output_file.write_text("data")
        good_metadata = _metadata_with(
            temp_dir,
            {
                "output_files": ["data.gwf"],
                "file_hashes": {"data.gwf": compute_file_hash(output_file)},
                "globals_config": {"output_directory": str(temp_dir)},
            },
        )
        broken_metadata = temp_dir / "broken.metadata.yaml"
        broken_metadata.write_text("{ not: valid: yaml:")

        with patch("os.getcwd", return_value=str(temp_dir)):
            validate_command([], metadata_paths=[str(good_metadata), str(broken_metadata)])

        assert "Error loading metadata" in caplog.text
        assert str(broken_metadata) in caplog.text
        # The readable record still validates; one bad file does not void the run.
        assert "1/1 files passed validation" in capsys.readouterr().out

    def test_the_summary_counts_passes_against_the_total(self, temp_dir, capsys):
        """The summary line is the number a user quotes, so both halves are checked."""
        good = temp_dir / "good.gwf"
        good.write_text("good data")
        bad = temp_dir / "bad.gwf"
        bad.write_text("bad data")
        metadata_file = _metadata_with(
            temp_dir,
            {
                "output_files": ["good.gwf", "bad.gwf"],
                "file_hashes": {
                    "good.gwf": compute_file_hash(good),
                    "bad.gwf": "sha256:" + "0" * 64,
                },
                "globals_config": {"output_directory": str(temp_dir)},
            },
        )

        with patch("os.getcwd", return_value=str(temp_dir)), pytest.raises(typer.Exit):
            validate_command([], metadata_paths=[str(metadata_file)])

        captured = capsys.readouterr()
        assert "1/2 files passed validation" in captured.out
        assert "1 files failed validation" in captured.out


@pytest.mark.usefixtures("wide_console")
class TestValidateInputHandling:
    """What the command says about the paths it was given.

    Every one of these is a user mistake -- a typo, a wrong flag, a path that
    moved -- and each has to name the offending path rather than fail as a
    generic "no metadata found".
    """

    def test_a_missing_output_path_is_named(self, sample_metadata, temp_dir, capsys):
        """A path that does not exist is reported, and does not stop the rest of the run."""
        metadata_file, _output_file = sample_metadata
        missing = temp_dir / "not-here.gwf"

        with patch("os.getcwd", return_value=str(temp_dir)):
            validate_command([missing], metadata_paths=[str(metadata_file)])

        captured = capsys.readouterr()
        assert "Path not found" in captured.out
        assert "not-here.gwf" in captured.out

    def test_a_missing_metadata_path_is_named(self, sample_metadata, temp_dir, capsys):
        """The metadata-side message is distinct from the output-side one."""
        metadata_file, _output_file = sample_metadata
        missing = temp_dir / "not-here.metadata.json"

        with patch("os.getcwd", return_value=str(temp_dir)):
            validate_command([], metadata_paths=[str(metadata_file), str(missing)])

        captured = capsys.readouterr()
        assert "Metadata path not found" in captured.out

    def test_a_non_metadata_file_passed_as_metadata_is_ignored_with_a_warning(self, sample_metadata, temp_dir, capsys):
        """Silently ignoring the file would look like a clean validation of nothing."""
        metadata_file, _output_file = sample_metadata
        not_metadata = temp_dir / "notes.txt"
        not_metadata.write_text("notes")

        with patch("os.getcwd", return_value=str(temp_dir)):
            validate_command([], metadata_paths=[str(metadata_file), str(not_metadata)])

        captured = capsys.readouterr()
        assert "Ignoring non-metadata file" in captured.out
        assert "notes.txt" in captured.out

    def test_json_metadata_is_accepted_by_name(self, temp_dir, capsys):
        """``.metadata.json`` is the current format and must be recognised as metadata."""
        output_file = temp_dir / "data.gwf"
        output_file.write_text("data")
        metadata_file = temp_dir / "run-0.metadata.json"
        metadata_file.write_text(
            json.dumps(
                {
                    "output_files": ["data.gwf"],
                    "file_hashes": {"data.gwf": compute_file_hash(output_file)},
                    "globals_config": {"output_directory": str(temp_dir)},
                }
            )
        )

        with patch("os.getcwd", return_value=str(temp_dir)):
            validate_command([], metadata_paths=[str(metadata_file)])

        captured = capsys.readouterr()
        assert "Ignoring non-metadata file" not in captured.out
        assert "1/1 files passed validation" in captured.out

    def test_a_directory_inside_the_metadata_directory_is_not_read_as_metadata(self, temp_dir, capsys):
        """Only files are metadata; a directory whose name contains "metadata" is not."""
        output_file = temp_dir / "data.gwf"
        output_file.write_text("data")
        metadata_dir = temp_dir / "metadata"
        metadata_dir.mkdir()
        (metadata_dir / "old-metadata").mkdir()
        with open(metadata_dir / "run-0.metadata.yaml", "w") as f:
            yaml.dump(
                {
                    "output_files": ["data.gwf"],
                    "file_hashes": {"data.gwf": compute_file_hash(output_file)},
                    "globals_config": {"output_directory": str(temp_dir)},
                },
                f,
            )

        with patch("os.getcwd", return_value=str(temp_dir)):
            validate_command([], metadata_paths=[str(metadata_dir)])

        captured = capsys.readouterr()
        assert "Error loading metadata" not in captured.out
        assert "1/1 files passed validation" in captured.out
