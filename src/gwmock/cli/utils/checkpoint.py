"""Checkpoint management for simulation recovery."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from gwmock.data.serialize.decoder import Decoder
from gwmock.data.serialize.encoder import Encoder

logger = logging.getLogger("gwmock")


def run_fingerprint(
    config_hashes: Iterable[str | None],
    output_directory: Path,
    metadata_directory: Path,
    referenced_inputs: Iterable[Path | str] = (),
) -> str:
    """Identify a run by everything that decides where its outputs go, not just its config file.

    The config file hash alone is not the run. Two invocations of the *same* file with different
    ``--output-dir`` produce identical hashes, so a checkpoint from the first is accepted by the
    second, which then skips the batches it records and writes nothing for them into the new
    directory -- measured at 2 frames where a clean run writes 3. That is the bug the fingerprint
    exists to stop, arriving through a different door.

    The output and metadata directories are resolved, so ``./out`` and an absolute path to the same
    place are one run rather than two, and a relative path is not mistaken for a different one when
    the working directory changes.

    Every batch's config hash goes in, not the first: a plan assembled from several metadata records
    can mix configurations, and taking one of them would let the rest through unchecked.

    **Two things it does not cover, both verified in review rather than assumed.**

    *Cosmetic edits refuse a valid resume.* The hash is over config bytes, so adding a comment or
    reindenting changes it and a legitimate resume is turned away. Annoying, and the safe direction;
    ``--ignore-checkpoint`` is the way out.

    **The population file is covered by content, not just by name.** A config names its catalogue by
    path, so replacing that file's bytes left every part of this identity unchanged: the checkpoint was
    accepted and the batches it recorded were skipped, giving one run some batches from the old
    catalogue and the rest from the new one. Measured before fixing -- a 3-batch run interrupted after
    the first batch and resumed over a rewritten CSV kept mass 30 in batch 0 while batches 1 and 2
    carried 81 and 82, exit code 0. Hashing it costs 18 ms for a 45 MB million-row catalogue, against a
    run measured in minutes.

    **Remote inputs are not covered at all, and the marker is not coverage.** A URL contributes a fixed
    marker, which adds nothing the config hash did not already carry: hashing the bytes would mean
    downloading the catalogue a second time purely to identify the run, and hashing the loader's cache
    would refuse every legitimate resume, because the first run computes this before any cache exists.
    So a re-fetch that returns different bytes between an interrupt and a resume -- ``refresh: true``, a
    cleared cache, a mutable URL -- mixes catalogues exactly as a local swap used to, undetected.
    :func:`warn_if_inputs_are_unverified` says so at the point it matters instead of leaving the
    docstring to carry it. Pinning the URL to an immutable revision, as the examples do, is the practical
    answer.

    *Only the population file is covered*, deliberately -- the input this was found through. Other paths
    a config can name (PSD files, waveform tables) remain invisible, and would need the same treatment
    if one of them ever bites.

    Args:
        config_hashes: Per-batch ``config_sha256`` values, in plan order. ``None`` entries are kept
            as a literal marker rather than dropped, so a plan with an unhashed batch does not
            fingerprint the same as one without that batch.
        output_directory: Where this run writes its data.
        metadata_directory: Where this run writes its metadata and index.
        referenced_inputs: Files whose *content* is part of this run's identity. Duplicates and order
            are normalised away: every batch carries the population config, so a three-batch plan
            presents the same path three times, and adding a batch must not change the input part of
            the identity. A path that cannot be read contributes a marker naming it, which still
            differs from the same path read successfully -- so a run that could not hash its input is
            never mistaken for one that did.

    Returns:
        A hex digest identifying the run.
    """
    parts = [hash_value if hash_value is not None else "<none>" for hash_value in config_hashes]
    parts.append(str(Path(output_directory).resolve()))
    parts.append(str(Path(metadata_directory).resolve()))
    # Sorted and de-duplicated, so the identity depends on which inputs a run reads and not on how many
    # batches happen to name them.
    for reference in sorted({str(reference) for reference in referenced_inputs}):
        parts.append(f"{reference}={_input_digest(reference)}")
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


# Schemes the population loader fetches rather than opens. Mirrors `gwmock_pop`'s own predicate
# (`loaders._fetch.is_population_url`) rather than testing for "://", which disagreed with it in both
# directions: a relative path under a directory named `data:` is local to the loader and was remote
# here, so its content went unhashed. Duplicated rather than imported because that helper is private;
# `test_run_identity_covers_the_population_file.py` compares the two whenever it can be imported, so
# drift is caught rather than assumed away.
_REMOTE_SCHEMES = frozenset({"http", "https", "s3", "zenodo"})


def _is_remote(reference: str) -> bool:
    """Whether the population loader would fetch *reference* instead of opening it."""
    return urlparse(reference).scheme.lower() in _REMOTE_SCHEMES


def _input_digest(reference: str) -> str:
    """Digest the content behind *reference*, or say why not.

    ``~`` is expanded, because the loader expands it before reading. Without that, a config naming
    ``~/population.csv`` failed to open here and recorded the same "could not hash" marker whatever the
    file held -- so the guard silently did nothing for exactly the paths people write by hand. Found in
    review, and the reason this function does not simply take a ``Path``.

    Line endings are normalised and trailing blank lines dropped before hashing. The run consumes the
    *parsed* catalogue, so a regenerated file that differs only in CRLF/LF or a trailing newline yields
    an identical population -- and refusing that resume costs a long run for no reason. Population files
    are machine-generated, so same-content regeneration is a normal workflow rather than a rare edit.
    This is not full parsing: a reordered or reformatted CSV still refuses, which is the safe direction.

    A remote reference is a marker rather than a fetch: identifying the run is not worth downloading the
    catalogue twice. **This means remote populations get no content coverage at all** -- the marker adds
    nothing the config hash did not already carry, and a re-fetch that returns different bytes between an
    interrupt and a resume is undetected. :func:`run_fingerprint` warns about that rather than implying
    the gap is closed.

    An unreadable reference is a marker too, not an exception. This runs before the plan executes, so a
    population staged later or a path typo has to reach its own error rather than a traceback from the
    identity check -- and the marker still differs from a successful digest, so "could not hash" is never
    confused with "hashed".
    """
    if _is_remote(reference):
        return "<remote>"
    path = Path(reference).expanduser()
    try:
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            # Chunked because a catalogue can be hundreds of megabytes; measured at ~2.5 GB/s, which is
            # 18 ms for a 45 MB million-row file. Normalisation is per chunk boundary-safe because it
            # only rewrites `\r\n` pairs, and a chunk is read at a 1 MiB boundary that can split one --
            # so the tail byte is carried into the next chunk.
            carry = b""
            while True:
                chunk = carry + handle.read(1 << 20)
                if not chunk:
                    break
                carry = chunk[-1:] if chunk.endswith(b"\r") else b""
                if carry:
                    chunk = chunk[:-1]
                digest.update(chunk.replace(b"\r\n", b"\n"))
            if carry:
                digest.update(carry.replace(b"\r", b"\n"))
        return digest.hexdigest()
    except OSError as error:
        logger.debug("Could not hash the run input %s: %s", path, error)
        return "<unhashed>"


def warn_if_inputs_are_unverified(referenced_inputs: Iterable[Path | str], resuming: bool) -> None:
    """Say, on a resume, which populations this run's identity could not actually verify.

    The guard's value is that a resume across two catalogues stops. For a remote or unreadable
    population it cannot: the digest is a marker, so the fingerprints match whatever the bytes were. That
    is a gap an operator can act on -- pin the URL, or stage the file locally -- and a gap they cannot see
    is indistinguishable from a guarantee.

    Only on a resume, and only when something is actually unverified: a first run has nothing to be
    inconsistent with, and warning on every clean run is how the message that matters gets ignored.

    Args:
        referenced_inputs: The population sources this run's identity was built from.
        resuming: Whether a checkpoint is being resumed from.
    """
    if not resuming:
        return
    # A marker rather than a digest is the definition of "not verified". Comparing two calls of
    # `_input_digest` would have been circular -- it is the same function, so it always agrees with
    # itself; the first version of this did exactly that and could never warn.
    unverified = sorted(
        {str(reference) for reference in referenced_inputs if _is_marker(_input_digest(str(reference)))}
    )
    if not unverified:
        return
    logger.warning(
        "This resume could not verify the population it is continuing from: %s. Their content is not part "
        "of the run's identity -- a remote catalogue is not fetched to identify a run, and one that cannot "
        "be read cannot be hashed -- so if those bytes changed since the interrupted run, this resume "
        "mixes two catalogues and nothing here will refuse it. Pin a remote URL to an immutable revision, "
        "or stage the catalogue locally, to get the check.",
        ", ".join(unverified),
    )


def _is_marker(digest: str) -> bool:
    """Whether :func:`_input_digest` gave up rather than hashing.

    Markers are bracketed and a sha256 hex digest never is, so this cannot mistake one for the other --
    which is also why the marker's exact spelling is not part of any contract.
    """
    return digest.startswith("<")


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
    is a certain cost against an uncertain one. **The residual is real:** every pre-fingerprint
    checkpoint stays exactly as exposed as before, and interrupted runs are precisely the population
    that resumes, so this is not a rare corner. It closes as those checkpoints age out.

    ``--ignore-checkpoint`` exists because refusing is otherwise a dead end for anything that cannot
    answer a prompt.

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
