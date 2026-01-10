"""Checkpoint management for simulation recovery."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from gwsim.data.serialize.decoder import Decoder
from gwsim.data.serialize.encoder import Encoder

logger = logging.getLogger("gwsim")


class CheckpointManager:
    """Manages checkpoint files for simulation recovery.

    A checkpoint is created after each successfully completed batch,
    allowing resumption from that point if the simulation is interrupted.

    Checkpoint file format:
    {
        "completed_batch_indices": [0, 1, 2, ...],
        "last_simulator_name": "signal",
        "last_completed_batch_index": 2,
        "last_simulator_state": {...}
    }

    The checkpoint is written atomically:
    1. Write to .tmp file
    2. Backup existing checkpoint to .bak
    3. Rename .tmp to checkpoint file
    This ensures we never have a corrupted checkpoint.
    """

    def __init__(self, checkpoint_directory: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_directory: Directory to store checkpoint files
        """
        self.checkpoint_directory = Path(checkpoint_directory)
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_directory / "simulation.checkpoint.json"
        self.checkpoint_tmp = self.checkpoint_directory / "simulation.checkpoint.json.tmp"
        self.checkpoint_backup = self.checkpoint_directory / "simulation.checkpoint.json.bak"

    def load_checkpoint(self) -> dict[str, Any] | None:
        """Load checkpoint from file if it exists.

        Returns:
            Checkpoint dict with keys:
            - completed_batch_indices: List of completed batch indices
            - last_simulator_name: Name of last simulator
            - last_completed_batch_index: Index of last completed batch
            - last_simulator_state: State dict of last simulator
            None if no checkpoint exists or checkpoint is corrupted
        """
        # Try to restore from backup if checkpoint doesn't exist but backup does
        if not self.checkpoint_file.exists() and self.checkpoint_backup.exists():
            logger.warning("Checkpoint file missing but backup exists. Restoring from backup...")
            try:
                self.checkpoint_backup.rename(self.checkpoint_file)
                logger.info("Checkpoint restored from backup")
            except OSError as e:
                logger.error("Failed to restore checkpoint from backup: %s", e)
                return None

        if not self.checkpoint_file.exists():
            logger.debug("No checkpoint file found")
            return None

        try:
            with self.checkpoint_file.open("r") as f:
                checkpoint = json.load(f, cls=Decoder)
            logger.debug(
                "Loaded checkpoint: last_batch=%s, completed=%d batches",
                checkpoint.get("last_completed_batch_index"),
                len(checkpoint.get("completed_batch_indices", [])),
            )
            return checkpoint
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load checkpoint: %s", e)
            return None

    def save_checkpoint(
        self,
        completed_batch_indices: list[int],
        last_simulator_name: str,
        last_completed_batch_index: int,
        last_simulator_state: dict[str, Any],
    ) -> None:
        """Save checkpoint after completing a batch.

        Args:
            completed_batch_indices: List of all completed batch indices so far
            last_simulator_name: Name of the simulator that completed the batch
            last_completed_batch_index: Index of the batch that just completed
            last_simulator_state: State dict of the simulator after completion

        Raises:
            OSError: If checkpoint cannot be written
        """
        checkpoint = {
            "completed_batch_indices": completed_batch_indices,
            "last_simulator_name": last_simulator_name,
            "last_completed_batch_index": last_completed_batch_index,
            "last_simulator_state": last_simulator_state,
        }

        # Write to temp file first (atomic write pattern)
        try:
            with self.checkpoint_tmp.open("w") as f:
                json.dump(checkpoint, f, indent=2, cls=Encoder)

            # Backup existing checkpoint if it exists
            if self.checkpoint_file.exists():
                try:
                    # Remove old backup if it exists (to avoid rename conflicts)
                    if self.checkpoint_backup.exists():
                        self.checkpoint_backup.unlink()
                    self.checkpoint_file.rename(self.checkpoint_backup)
                except OSError as e:
                    logger.warning("Failed to backup previous checkpoint: %s", e)

            # Move temp to final checkpoint
            self.checkpoint_tmp.rename(self.checkpoint_file)

            logger.debug(
                "Checkpoint saved: batch_index=%d, completed=%d batches",
                last_completed_batch_index,
                len(completed_batch_indices),
            )
        except OSError as e:
            logger.error("Failed to save checkpoint: %s", e)
            # Clean up temp file if it exists
            if self.checkpoint_tmp.exists():
                with contextlib.suppress(OSError):
                    self.checkpoint_tmp.unlink()
            raise

    def cleanup(self) -> None:
        """Clean up checkpoint files after successful completion."""
        # Remove both checkpoint and backup after successful completion
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.debug("Cleaned up checkpoint file")
            if self.checkpoint_backup.exists():
                self.checkpoint_backup.unlink()
                logger.debug("Cleaned up checkpoint backup file")
        except OSError as e:
            logger.warning("Failed to clean up checkpoint files: %s", e)

    def get_completed_batch_indices(self) -> set[int]:
        """Get set of completed batch indices from checkpoint.

        Returns:
            Set of batch indices that have already been completed
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return set()
        return set(checkpoint.get("completed_batch_indices", []))

    def should_skip_batch(self, batch_index: int) -> bool:
        """Check if a batch has already been completed.

        Args:
            batch_index: Index of batch to check

        Returns:
            True if batch was already completed, False otherwise
        """
        return batch_index in self.get_completed_batch_indices()
