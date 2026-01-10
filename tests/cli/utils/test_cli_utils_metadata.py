"""Unit tests for metadata utilities."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from gwsim.cli.utils.metadata import load_metadata_with_external_state, save_metadata_with_external_state


class TestSaveMetadataWithExternalState:
    """Test save_metadata_with_external_state function."""

    def test_save_without_numpy_arrays(self, tmp_path):
        """Test saving metadata without numpy arrays."""
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {"rng_seed": 42, "counter": 10},
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "metadata"

        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)

        # Check YAML file exists
        assert metadata_file.exists()

        # Check no external files created
        assert not any(metadata_dir.glob("*.npy"))

        # Check YAML content
        with metadata_file.open() as f:
            loaded = yaml.safe_load(f)
        assert loaded == metadata

    def test_save_with_numpy_arrays(self, tmp_path):
        """Test saving metadata with numpy arrays."""
        array1 = np.array([1, 2, 3, 4, 5])
        array2 = np.random.rand(10, 10)
        rng_seed = 42
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {
                "rng_seed": rng_seed,
                "rng_state": array1,
                "filter_state": array2,
            },
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "metadata"

        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)

        # Check YAML file exists
        assert metadata_file.exists()

        # Check external files created
        npy_files = list(metadata_dir.glob("*.npy"))
        expected_num_files = 2
        assert len(npy_files) == expected_num_files

        # Check YAML content has references
        with metadata_file.open() as f:
            loaded = yaml.safe_load(f)
        assert loaded["simulator_name"] == "test_sim"
        assert loaded["batch_index"] == 0
        assert loaded["pre_batch_state"]["rng_seed"] == rng_seed
        assert loaded["pre_batch_state"]["rng_state"]["_external_file"] is True
        assert loaded["pre_batch_state"]["filter_state"]["_external_file"] is True

    def test_save_with_empty_pre_batch_state(self, tmp_path):
        """Test saving metadata with empty pre_batch_state."""
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {},
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "metadata"

        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)

        # Check YAML file exists
        assert metadata_file.exists()

        # Check no external files
        assert not any(metadata_dir.glob("*.npy"))

    def test_save_creates_metadata_dir(self, tmp_path):
        """Test that metadata directory is created if it doesn't exist."""
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {"rng_state": np.array([1, 2, 3])},
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "new_metadata_dir"

        assert not metadata_dir.exists()
        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)
        assert metadata_dir.exists()
        assert metadata_file.exists()


class TestLoadMetadataWithExternalState:
    """Test load_metadata_with_external_state function."""

    def test_load_without_numpy_arrays(self, tmp_path):
        """Test loading metadata without numpy arrays."""
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {"rng_seed": 42, "counter": 10},
        }
        metadata_file = tmp_path / "metadata.yaml"

        # Save directly
        with metadata_file.open("w") as f:
            yaml.safe_dump(metadata, f)

        # Load
        loaded = load_metadata_with_external_state(metadata_file)

        assert loaded == metadata

    def test_load_with_numpy_arrays(self, tmp_path):
        """Test loading metadata with numpy arrays."""
        array1 = np.array([1, 2, 3, 4, 5])
        array2 = np.random.rand(3, 3)
        rng_seed = 42
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {
                "rng_seed": rng_seed,
                "rng_state": array1,
                "filter_state": array2,
            },
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "metadata"

        # Save with external state
        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)

        # Load
        loaded = load_metadata_with_external_state(metadata_file, metadata_dir)

        assert loaded["simulator_name"] == "test_sim"
        assert loaded["batch_index"] == 0
        assert loaded["pre_batch_state"]["rng_seed"] == rng_seed
        np.testing.assert_array_equal(loaded["pre_batch_state"]["rng_state"], array1)
        np.testing.assert_array_equal(loaded["pre_batch_state"]["filter_state"], array2)

    def test_load_missing_metadata_file(self, tmp_path):
        """Test loading with missing metadata file raises FileNotFoundError."""
        metadata_file = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            load_metadata_with_external_state(metadata_file)

    def test_load_missing_external_file(self, tmp_path):
        """Test loading with missing external file raises FileNotFoundError."""
        # Create metadata with external reference
        metadata = {
            "simulator_name": "test_sim",
            "pre_batch_state": {
                "rng_state": {
                    "_external_file": True,
                    "dtype": "int64",
                    "shape": [5],
                    "size_bytes": 40,
                    "file": "missing_state_rng_state.npy",
                }
            },
        }
        metadata_file = tmp_path / "metadata.yaml"
        with metadata_file.open("w") as f:
            yaml.safe_dump(metadata, f)

        with pytest.raises(FileNotFoundError, match="External state file not found"):
            load_metadata_with_external_state(metadata_file)

    def test_round_trip_save_load(self, tmp_path):
        """Test round-trip: save then load should match original."""
        original_array = np.random.rand(5, 5)
        original_metadata = {
            "simulator_name": "test_sim",
            "batch_index": 1,
            "pre_batch_state": {
                "rng_seed": 123,
                "rng_state": original_array,
                "counter": 5,
            },
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "metadata"

        # Save
        save_metadata_with_external_state(original_metadata, metadata_file, metadata_dir)

        # Load
        loaded_metadata = load_metadata_with_external_state(metadata_file, metadata_dir)

        # Check non-array fields
        assert loaded_metadata["simulator_name"] == original_metadata["simulator_name"]
        assert loaded_metadata["batch_index"] == original_metadata["batch_index"]
        assert loaded_metadata["pre_batch_state"]["rng_seed"] == original_metadata["pre_batch_state"]["rng_seed"]
        assert loaded_metadata["pre_batch_state"]["counter"] == original_metadata["pre_batch_state"]["counter"]

        # Check array
        np.testing.assert_array_equal(
            loaded_metadata["pre_batch_state"]["rng_state"], original_metadata["pre_batch_state"]["rng_state"]
        )

    def test_load_with_custom_metadata_dir(self, tmp_path):
        """Test loading with custom metadata directory."""
        array = np.array([10, 20, 30])
        metadata = {
            "simulator_name": "test_sim",
            "pre_batch_state": {"array": array},
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "custom_metadata"

        # Save
        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)

        # Load with custom dir
        loaded = load_metadata_with_external_state(metadata_file, metadata_dir)

        np.testing.assert_array_equal(loaded["pre_batch_state"]["array"], array)
