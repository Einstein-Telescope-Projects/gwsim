"""Checkpoint management for simulation recovery."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from gwmock.data.serialize.decoder import Decoder
from gwmock.data.serialize.encoder import Encoder

logger = logging.getLogger("gwmock")


class ForeignCheckpointError(RuntimeError):
    """A checkpoint in this directory was written by a different configuration."""


def require_matching_config(saved_sha256: str | None, plan_sha256: str | None, checkpoint_file: Path) -> None:
    """Refuse to resume from a checkpoint another configuration wrote.

    **Why this refuses instead of quietly starting fresh.** Silently ignoring the checkpoint would
    fix the data loss and hide its cause: the user would be left with a stale checkpoint in the
    directory, no idea it was there, and a full re-run they did not ask for. Refusing says which two
    things collided and lets them choose. It is also the safer default -- deleting someone's
    checkpoint on their behalf is not recoverable, and stopping is.

    **What went wrong without it,** measured: two configs run from one working directory, the first
    interrupted after two batches. The second loaded the first's checkpoint, skipped those batches as
    already complete, and wrote one frame where a clean run writes three. Exit code 0, no warning.
    `_batch_outputs_present` does not catch it because it looks up
    ``{simulator_name}-{batch_index}.metadata.json`` -- a name with nothing config-specific in it --
    and then verifies the outputs *that file* records, so the first run's metadata satisfies the
    check on the second run's behalf.

    A checkpoint written before this field existed has ``None`` and is accepted, with a warning. The
    alternative -- refusing -- would break a legitimate resume for anyone who upgrades mid-run, which
    is a certain cost against an uncertain one.

    Args:
        saved_sha256: ``config_sha256`` from the checkpoint, or ``None`` if it predates the field.
        plan_sha256: Hash of the configuration about to run, or ``None`` if unavailable.
        checkpoint_file: Path named in the error, so the message says what to delete or move.

    Raises:
        ForeignCheckpointError: If both hashes are known and they differ.
    """
    if saved_sha256 is None:
        logger.warning(
            "Checkpoint %s predates configuration fingerprinting, so it cannot be checked against "
            "the configuration being run. If it was written by a different config, batches it "
            "records as complete will be skipped and their outputs never produced.",
            checkpoint_file,
        )
        return
    if plan_sha256 is None or saved_sha256 == plan_sha256:
        return
    raise ForeignCheckpointError(
        f"The checkpoint at {checkpoint_file} was written by a different configuration "
        f"(checkpoint {saved_sha256[:12]}, this run {plan_sha256[:12]}). Resuming from it would "
        f"skip the batches it records as complete and never produce their outputs. Delete or move "
        f"the checkpoint to start this configuration fresh, or run it from its own directory."
    )


def spillover_applies(
    saved_simulator_name: str | None,
    saved_batch_index: int | None,
    simulator_name: str | None,
    batch_index: int | None,
) -> bool:
    """Whether spillover saved by one batch may be given to another.

    One function rather than the same two comparisons at each call site: the checkpoint getter and
    ``execute_plan`` both need this, and the reason `execute_plan` cannot simply call the getter is
    cost -- each load decodes the whole file, spillover included. Two copies of a predicate this
    consequential would drift.

    Both conditions guard against a *wrongly accepted* tail, which is worse than a rejected one: it
    is real strain of the right shape placed at the wrong time, in the wrong simulator's segment,
    and nothing downstream would flag it.

    ``None`` for either caller-supplied value skips that check, for callers that have already
    established it.

    **Not checked, and it cannot be from here:** whether the checkpoint belongs to *this plan* at
    all. Nothing in a checkpoint identifies the config that produced it, so reusing a checkpoint
    directory across two different runs with the same simulator name and batch numbering will pass
    both tests below. That predates spillover -- ``last_simulator_state`` was never scoped either --
    but spillover makes the consequence bigger. Tracked as
    ``gwmock/checkpoint-has-no-plan-identity``.

    Args:
        saved_simulator_name: ``last_simulator_name`` from the checkpoint.
        saved_batch_index: ``last_completed_batch_index`` from the checkpoint.
        simulator_name: The simulator about to run, or ``None`` to skip the check.
        batch_index: The batch about to run, or ``None`` to skip the check.

    Returns:
        ``True`` if the saved spillover belongs to this simulator and batch.
    """
    if simulator_name is not None and saved_simulator_name != simulator_name:
        return False
    # Spillover belongs to the batch immediately after the one that produced it. A plan whose batch
    # indices are not contiguous therefore rejects spillover it could have used -- which loses a
    # tail rather than misplacing one, the safe direction, and no such plan exists today.
    return not (batch_index is not None and saved_batch_index != batch_index - 1)


class CheckpointManager:
    """Manages checkpoint files for simulation recovery.

    A checkpoint is created after each successfully completed batch,
    allowing resumption from that point if the simulation is interrupted.

    Checkpoint file format:
    {
        "completed_batch_indices": [0, 1, 2, ...],
        "last_simulator_name": "signal",
        "last_completed_batch_index": 2,
        "last_simulator_state": {...},
        "last_simulator_spillover": ...,  # chunks continuing into the next segment, or null
        "config_sha256": "..."  # which configuration produced this run, or null if pre-1.5.0
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
        last_simulator_spillover: Any = None,
        config_sha256: str | None = None,
    ) -> None:
        """Save checkpoint after completing a batch.

        Args:
            completed_batch_indices: List of all completed batch indices so far
            last_simulator_name: Name of the simulator that completed the batch
            last_completed_batch_index: Index of the batch that just completed
            last_simulator_state: State dict of the simulator after completion
            last_simulator_spillover: Chunks that extend past the completed segment and belong to
                the next one -- the tail of any signal crossing the boundary.

                Carried **beside** the state rather than in it, deliberately. ``state`` is also
                serialized into every batch metadata record, and those are provenance documents
                meant to stay small and readable; spillover is raw samples, megabytes of them for a
                long inspiral. Putting it in ``state`` would bloat every metadata record and, since
                those are written with plain ``json``, fail outright on a ``TimeSeriesList``.

                Without this a resumed run starts with no spillover, so the tail is never placed and
                the segment after the resume point silently loses that content: a measured peak of
                8.6e-22 became 0.0, with the merger simply absent.

            config_sha256: Hash of the configuration that produced this run, so a resume can tell
                whether the checkpoint belongs to it. Without this a second config run from the same
                working directory resumes from the first's checkpoint and *skips its batches*:
                measured at 1 frame written where a clean run writes 3, exit code 0, no warning.

        Raises:
            OSError: If checkpoint cannot be written
        """
        checkpoint = {
            "completed_batch_indices": completed_batch_indices,
            "last_simulator_name": last_simulator_name,
            "last_completed_batch_index": last_completed_batch_index,
            "last_simulator_state": last_simulator_state,
            "last_simulator_spillover": last_simulator_spillover,
            "config_sha256": config_sha256,
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

    def get_last_simulator_state(self) -> dict[str, Any] | None:
        """Get the last completed batch state from checkpoint.

        Returns:
            State dict of the last simulator, or None if unavailable
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return None
        last_state = checkpoint.get("last_simulator_state")
        return last_state if isinstance(last_state, dict) else None

    def get_last_simulator_spillover(self, simulator_name: str | None = None, batch_index: int | None = None) -> Any:
        """Return the spillover chunks saved with the last completed batch, if any.

        ``None`` both when the checkpoint predates this field and when the last segment had no
        spillover, which are the same thing to a caller: there is nothing to carry in.

        Args:
            simulator_name: Restrict to spillover produced by this simulator. A plan can execute
                several, and the checkpoint holds one tail; without this the wrong simulator can
                receive it. ``None`` skips the check, for callers that have already established it.
            batch_index: The batch about to run. Spillover is only valid for the batch immediately
                following the one that produced it.

        Returns:
            The chunks that belong to the next segment, or ``None``.
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return None
        if not spillover_applies(
            checkpoint.get("last_simulator_name"),
            checkpoint.get("last_completed_batch_index"),
            simulator_name,
            batch_index,
        ):
            return None
        return checkpoint.get("last_simulator_spillover")

    def should_skip_batch(self, batch_index: int) -> bool:
        """Check if a batch has already been completed.

        Args:
            batch_index: Index of batch to check

        Returns:
            True if batch was already completed, False otherwise
        """
        return batch_index in self.get_completed_batch_indices()
