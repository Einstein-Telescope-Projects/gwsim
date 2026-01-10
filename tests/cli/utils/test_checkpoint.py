"""Unit tests for checkpoint management."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from gwsim.cli.utils.checkpoint import CheckpointManager

# Constants for test assertions
COMPLETED_BATCH_INDEX_2 = 2
COUNTER_3 = 3


class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    def test_init_creates_directory(self):
        """Test that checkpoint manager creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "nonexistent" / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)

            assert checkpoint_dir.exists()
            assert manager.checkpoint_directory == checkpoint_dir

    def test_save_and_load_checkpoint(self):
        """Test saving and loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # Save a checkpoint
            checkpoint_data = {
                "completed_batch_indices": [0, 1, 2],
                "last_simulator_name": "signal",
                "last_completed_batch_index": 2,
                "last_simulator_state": {"counter": 3, "rng_state": [1, 2, 3]},
            }

            manager.save_checkpoint(
                completed_batch_indices=checkpoint_data["completed_batch_indices"],
                last_simulator_name=checkpoint_data["last_simulator_name"],
                last_completed_batch_index=checkpoint_data["last_completed_batch_index"],
                last_simulator_state=checkpoint_data["last_simulator_state"],
            )

            # Load and verify
            loaded = manager.load_checkpoint()
            assert loaded is not None
            assert loaded["completed_batch_indices"] == [0, 1, 2]
            assert loaded["last_simulator_name"] == "signal"
            assert loaded["last_completed_batch_index"] == COMPLETED_BATCH_INDEX_2
            assert loaded["last_simulator_state"]["counter"] == COUNTER_3

    def test_load_nonexistent_checkpoint(self):
        """Test loading when no checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))
            checkpoint = manager.load_checkpoint()
            assert checkpoint is None

    def test_checkpoint_atomic_write(self):
        """Test that checkpoint is written atomically.

        Verifies that:
        1. Temp file is used during write
        2. Existing checkpoint is backed up
        3. Temp file is moved to final location
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # First checkpoint
            manager.save_checkpoint(
                completed_batch_indices=[0],
                last_simulator_name="signal",
                last_completed_batch_index=0,
                last_simulator_state={"counter": 1},
            )

            assert manager.checkpoint_file.exists()
            assert not manager.checkpoint_tmp.exists()  # Temp should be cleaned up
            assert not manager.checkpoint_backup.exists()  # No backup on first write

            # Second checkpoint
            manager.save_checkpoint(
                completed_batch_indices=[0, 1],
                last_simulator_name="signal",
                last_completed_batch_index=1,
                last_simulator_state={"counter": 2},
            )

            # Now backup should exist (previous checkpoint backed up)
            assert manager.checkpoint_file.exists()
            assert manager.checkpoint_backup.exists()
            assert not manager.checkpoint_tmp.exists()

            # Verify backup contains first checkpoint
            with manager.checkpoint_backup.open() as f:
                backup_data = json.load(f)
            assert backup_data["last_completed_batch_index"] == 0

    def test_restore_from_backup_if_checkpoint_missing(self):
        """Test that checkpoint can be restored from backup.

        Simulates scenario where checkpoint file was deleted but backup exists.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # Create first checkpoint
            manager.save_checkpoint(
                completed_batch_indices=[0],
                last_simulator_name="signal",
                last_completed_batch_index=0,
                last_simulator_state={"counter": 1},
            )

            # Create second checkpoint - this moves first checkpoint to .bak
            manager.save_checkpoint(
                completed_batch_indices=[0, 1],
                last_simulator_name="signal",
                last_completed_batch_index=1,
                last_simulator_state={"counter": 2},
            )

            # Verify backup exists after second save
            assert manager.checkpoint_backup.exists()

            # Simulate corruption: delete checkpoint file but keep backup
            manager.checkpoint_file.unlink()
            assert not manager.checkpoint_file.exists()
            assert manager.checkpoint_backup.exists()

            # Load should restore from backup
            checkpoint = manager.load_checkpoint()
            assert checkpoint is not None
            assert checkpoint["last_completed_batch_index"] == 0

            # After restoration, checkpoint file should exist again
            assert manager.checkpoint_file.exists()
            # And backup should be gone (it was renamed to checkpoint)
            assert not manager.checkpoint_backup.exists()

    def test_get_completed_batch_indices(self):
        """Test retrieving completed batch indices from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # No checkpoint initially
            completed = manager.get_completed_batch_indices()
            assert completed == set()

            # After saving checkpoint
            manager.save_checkpoint(
                completed_batch_indices=[0, 2, 4, 5],
                last_simulator_name="signal",
                last_completed_batch_index=5,
                last_simulator_state={"counter": 6},
            )

            completed = manager.get_completed_batch_indices()
            assert completed == {0, 2, 4, 5}

    def test_should_skip_batch(self):
        """Test batch skip logic based on checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # Save checkpoint with completed batches
            manager.save_checkpoint(
                completed_batch_indices=[0, 1, 2],
                last_simulator_name="signal",
                last_completed_batch_index=2,
                last_simulator_state={"counter": 3},
            )

            # Check various batch indices
            assert manager.should_skip_batch(0)  # Completed
            assert manager.should_skip_batch(1)  # Completed
            assert manager.should_skip_batch(2)  # Completed
            assert not manager.should_skip_batch(3)  # Not completed
            assert not manager.should_skip_batch(4)  # Not completed

    def test_cleanup_removes_checkpoint_files(self):
        """Test that cleanup removes checkpoint and backup files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # Create checkpoint with backup
            manager.save_checkpoint(
                completed_batch_indices=[0],
                last_simulator_name="signal",
                last_completed_batch_index=0,
                last_simulator_state={"counter": 1},
            )
            manager.save_checkpoint(
                completed_batch_indices=[0, 1],
                last_simulator_name="signal",
                last_completed_batch_index=1,
                last_simulator_state={"counter": 2},
            )

            # Both checkpoint and backup should exist
            assert manager.checkpoint_file.exists()
            assert manager.checkpoint_backup.exists()

            # Cleanup
            manager.cleanup()

            # Both should be deleted
            assert not manager.checkpoint_file.exists()
            assert not manager.checkpoint_backup.exists()

    def test_checkpoint_with_complex_state(self):
        """Test checkpoint with complex simulator state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            complex_state = {
                "counter": 42,
                "rng_state": [1.5, 2.5, 3.5],
                "filter_state": {
                    "history": [1, 2, 3, 4, 5],
                    "config": {"order": 4, "type": "butterworth"},
                },
                "metadata": {"detector": "H1", "sample_rate": 16384},
            }

            manager.save_checkpoint(
                completed_batch_indices=[0, 1, 2],
                last_simulator_name="noise",
                last_completed_batch_index=2,
                last_simulator_state=complex_state,
            )

            loaded = manager.load_checkpoint()
            assert loaded["last_simulator_state"] == complex_state

    def test_save_checkpoint_handles_io_errors(self, mocker):
        """Test that save handles IO errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # Mock Path.open to raise OSError when writing checkpoint temp file
            original_path_open = Path.open

            def mock_path_open(self, mode="r", **kwargs):
                if "simulation.checkpoint.json.tmp" in str(self) and "w" in mode:
                    raise OSError("Permission denied")
                return original_path_open(self, mode, **kwargs)

            mocker.patch.object(Path, "open", mock_path_open)

            with pytest.raises(OSError, match="Permission denied"):
                manager.save_checkpoint(
                    completed_batch_indices=[0],
                    last_simulator_name="signal",
                    last_completed_batch_index=0,
                    last_simulator_state={"counter": 1},
                )

    def test_multiple_simulators_in_checkpoint(self):
        """Test checkpoint tracking across multiple simulators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # Simulate execution of two simulators
            manager.save_checkpoint(
                completed_batch_indices=[0],
                last_simulator_name="noise",
                last_completed_batch_index=0,
                last_simulator_state={"counter": 1},
            )

            manager.save_checkpoint(
                completed_batch_indices=[0, 1],
                last_simulator_name="signal",
                last_completed_batch_index=1,
                last_simulator_state={"counter": 2},
            )

            # Last checkpoint should have both batches
            loaded = manager.load_checkpoint()
            assert loaded["completed_batch_indices"] == [0, 1]
            assert loaded["last_simulator_name"] == "signal"
