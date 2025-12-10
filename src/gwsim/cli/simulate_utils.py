"""
Utilities for executing simulation plans via CLI.
"""

from __future__ import annotations

import atexit
import copy
import logging
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from gwsim.cli.utils.checkpoint import CheckpointManager
from gwsim.cli.utils.config import SimulatorConfig, resolve_class_path
from gwsim.cli.utils.hash import compute_file_hash
from gwsim.cli.utils.metadata import save_metadata_with_external_state
from gwsim.cli.utils.simulation_plan import (
    SimulationBatch,
    SimulationPlan,
    create_batch_metadata,
)
from gwsim.cli.utils.template import expand_template_variables
from gwsim.cli.utils.utils import handle_signal, import_attribute
from gwsim.simulator.base import Simulator

logger = logging.getLogger("gwsim")
logger.setLevel(logging.DEBUG)


def retry_with_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    state_restore_func: Any = None,
) -> Any:
    """Retry a function with exponential backoff and optional state restoration.

    Args:
        func: Callable to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        state_restore_func: Optional callable to restore state before each retry.
                           Called before each retry attempt (not before first attempt).

    Returns:
        Result of function call

    Raises:
        Exception: If all retries fail
    """
    delay = initial_delay
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:  # pylint: disable=broad-exception-caught
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.2fs...",
                    attempt + 1,
                    max_retries + 1,
                    str(e),
                    delay,
                    exc_info=e,
                )
                time.sleep(delay)
                delay *= backoff_factor

                # Restore state before retry if function provided
                if state_restore_func is not None:
                    try:
                        state_restore_func()
                        logger.debug("State restored before retry attempt %d", attempt + 2)
                    except Exception as restore_error:
                        logger.error("Failed to restore state before retry: %s", restore_error)
                        raise RuntimeError(f"Cannot retry: failed to restore state: {restore_error}") from restore_error
            else:
                logger.error("All %d attempts failed for batch: %s", max_retries + 1, str(e))

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected retry failure")


def update_metadata_index(
    metadata_directory: Path,
    output_files: list[Path],
    metadata_file_name: str,
    encoding: str = "utf-8",
) -> None:
    """Update the central metadata index file.

    The index maps data file names to their corresponding metadata files,
    enabling O(1) lookup to find metadata for a given data file.

    Args:
        metadata_directory: Directory where metadata files are stored
        output_files: List of output data file Paths
        metadata_file_name: Name of the metadata file (e.g., "signal-0.metadata.yaml")
        encoding: File encoding for reading/writing the index file
    """
    index_file = metadata_directory / "index.yaml"

    # Load existing index or create new
    if index_file.exists():
        try:
            with index_file.open(encoding=encoding) as f:
                index = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Failed to load metadata index: %s. Creating new index.", e)
            index = {}
    else:
        index = {}

    # Add entries for all output files
    for output_file in output_files:
        index[output_file.name] = metadata_file_name
        logger.debug("Index entry: %s -> %s", output_file.name, metadata_file_name)

    # Save updated index
    try:
        with index_file.open("w") as f:
            yaml.safe_dump(index, f, default_flow_style=False, sort_keys=True)
        logger.debug("Updated metadata index: %s", index_file)
    except (OSError, yaml.YAMLError) as e:
        logger.error("Failed to save metadata index: %s", e)
        raise


def instantiate_simulator(
    simulator_config: SimulatorConfig,
    simulator_name: str | None = None,
    global_simulator_arguments: dict[str, Any] | None = None,
) -> Simulator:
    """Instantiate a simulator from configuration.

    Creates a single simulator instance that will be reused across multiple batches.
    The simulator maintains state (RNG, counters, etc.) across iterations.

    Global simulator arguments are merged with simulator-specific arguments,
    with simulator-specific arguments taking precedence.

    Args:
        simulator_config: Configuration for this simulator
        simulator_name: Name of the simulator (used for class path resolution)
        global_simulator_arguments: Global fallback arguments for the simulator

    Returns:
        Instantiated Simulator

    Raises:
        ImportError: If simulator class cannot be imported
        TypeError: If simulator instantiation fails
    """
    class_spec = simulator_config.class_

    # Resolve short class names to full paths
    class_spec = resolve_class_path(class_spec, simulator_name)

    simulator_cls = import_attribute(class_spec)

    # Merge global and simulator-specific arguments
    # Simulator-specific arguments override global defaults
    if global_simulator_arguments:
        merged_arguments = {**global_simulator_arguments, **simulator_config.arguments}
    else:
        merged_arguments = simulator_config.arguments

    # Normalize keys: convert hyphens to underscores (YAML uses hyphens, Python uses underscores)
    normalized_arguments = {k.replace("-", "_"): v for k, v in merged_arguments.items()}

    simulator = simulator_cls(**normalized_arguments)

    logger.info("Instantiated simulator from class %s", class_spec)
    return simulator


def restore_batch_state(simulator: Simulator, batch: SimulationBatch) -> None:
    """Restore simulator state from batch metadata if available.

    This is used when reproducing a specific batch. It restores the RNG state,
    filter memory, and other stateful components that existed before this batch
    was generated.

    Args:
        simulator: Simulator instance
        batch: SimulationBatch potentially containing state snapshot

    Raises:
        ValueError: If state restoration fails
    """
    if batch.has_state_snapshot() and batch.pre_batch_state is not None:
        logger.debug(
            "[RESTORE] Batch %d: Restoring state from snapshot - state_keys=%s",
            batch.batch_index,
            list(batch.pre_batch_state.keys()),
        )
        try:
            logger.debug(
                "[RESTORE] Batch %d: Setting state dict - counter=%s",
                batch.batch_index,
                batch.pre_batch_state.get("counter"),
            )
            simulator.state = batch.pre_batch_state
            logger.debug(
                "[RESTORE] Batch %d: State restored successfully - new_counter=%s",
                batch.batch_index,
                simulator.counter,
            )
        except Exception as e:
            logger.error("Failed to restore batch state: %s", e)
            raise ValueError(f"Failed to restore state for batch {batch.batch_index}") from e
    else:
        logger.debug(
            "[RESTORE] Batch %d: No pre-batch state snapshot available (fresh generation)",
            batch.batch_index,
        )


def save_batch_metadata(
    simulator: Simulator,
    batch: SimulationBatch,
    metadata_directory: Path,
    output_files: list[Path],
    pre_batch_state: dict[str, Any] | None = None,
) -> None:
    """Save batch metadata including pre-batch simulator state and all output files.

    The metadata file uses batch-indexed naming ({simulator_name}-{batch_index}.metadata.yaml)
    to provide a single source of truth for all outputs from that batch. This handles
    cases where a single batch generates multiple output files (e.g., one per detector).

    An index file is also maintained to enable quick lookup of metadata for a given data file.

    Args:
        simulator: Simulator instance
        batch: SimulationBatch
        metadata_directory: Directory to save metadata
        output_files: List of Path objects for all output files generated by this batch
        pre_batch_state: State of simulator before batch generation (for reproducibility).
                        If None, uses current simulator state.
    """
    metadata_directory.mkdir(parents=True, exist_ok=True)

    # Use provided pre_batch_state or current simulator state
    state_to_save = pre_batch_state if pre_batch_state is not None else simulator.state

    metadata = create_batch_metadata(
        simulator_name=batch.simulator_name,
        batch_index=batch.batch_index,
        simulator_config=batch.simulator_config,
        globals_config=batch.globals_config,
        pre_batch_state=state_to_save,
        source=batch.source,
        author=batch.author,
        email=batch.email,
    )

    # Add output files to metadata for easy discovery
    # Store just the file names, not full paths
    metadata["output_files"] = [f.name for f in output_files]

    # Compute and add file hashes for integrity checking
    file_hashes = {}
    for output_file in output_files:
        try:
            file_hash = compute_file_hash(output_file)
            file_hashes[output_file.name] = file_hash
            logger.debug("Compute hash for %s: %s", output_file.name, file_hash)
        except OSError as e:
            logger.warning("Failed to compute hash for %s: %s", output_file.name, e)
            # Continue without failing - metadata is still useful

    metadata["file_hashes"] = file_hashes

    metadata_file_name = f"{batch.simulator_name}-{batch.batch_index}.metadata.yaml"
    metadata_file = metadata_directory / metadata_file_name
    logger.debug("Saving batch metadata to %s with %d output files", metadata_file, len(output_files))

    save_metadata_with_external_state(metadata=metadata, metadata_file=metadata_file)

    # Update the metadata index for quick lookup
    update_metadata_index(metadata_directory, output_files, metadata_file_name)


def process_batch(
    simulator: Simulator,
    batch_data: object,
    batch: SimulationBatch,
    output_directory: Path,
    overwrite: bool,
) -> list[Path]:
    """Process and save a single batch of generated data.

    A single batch may generate multiple output files (e.g., one per detector).
    This function handles both single and multiple output files.

    Args:
        simulator: Simulator instance
        batch_data: Generated batch data (may contain multiple outputs)
        batch: SimulationBatch metadata
        output_directory: Directory for output files
        overwrite: Whether to overwrite existing files

    Returns:
        List of Path objects for all generated output files
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    logger.debug(
        "[PROCESS] Batch %s: Saving data - counter=%s, file_template=%s",
        batch.batch_index,
        simulator.counter,
        batch.simulator_config.output.file_name,
    )

    # Build output configuration
    output_config = batch.simulator_config.output
    file_name_template = output_config.file_name
    output_args = output_config.arguments.copy() if output_config.arguments else {}

    # Save data with output directory
    logger.debug(
        "Saving batch data for %s batch %d",
        batch.simulator_name,
        batch.batch_index,
    )

    # Resolve the output file names (may be multiple if template contains arrays)
    output_files = expand_template_variables(value=file_name_template, simulator_instance=simulator)

    # Normalize to list of Paths
    if isinstance(output_files, str):
        output_files_list = [output_directory / Path(output_files)]
    else:
        # If it's an array (multiple detectors), flatten it
        output_files_list = [output_directory / Path(str(f)) for f in np.array(output_files).flatten()]

    logger.debug(
        "[PROCESS] Batch %s: Resolved filenames - %s", batch.batch_index, [str(f.name) for f in output_files_list]
    )

    simulator.save_data(
        data=batch_data,
        file_name=file_name_template,
        output_directory=output_directory,
        overwrite=overwrite,
        **output_args,
    )

    logger.debug("[PROCESS] Batch %s: Data saved - counter=%s", batch.batch_index, simulator.counter)

    return output_files_list


def setup_signal_handlers(checkpoint_dirs: list[Path]) -> None:
    """Set up signal handlers for graceful shutdown.

    Args:
        checkpoint_dirs: List of checkpoint directories to clean up
    """

    def cleanup_checkpoints():
        """Clean up temporary checkpoint files."""
        for checkpoint_dir in checkpoint_dirs:
            for backup_file in checkpoint_dir.glob("*.bak"):
                try:
                    backup_file.unlink()
                    logger.debug("Cleaned up backup file: %s", backup_file)
                except OSError as e:
                    logger.warning("Failed to clean up backup file %s: %s", backup_file, e)

    atexit.register(cleanup_checkpoints)
    signal.signal(signal.SIGINT, handle_signal(cleanup_checkpoints))
    signal.signal(signal.SIGTERM, handle_signal(cleanup_checkpoints))


def validate_plan(plan: SimulationPlan) -> None:
    """Validate simulation plan before execution.

    Args:
        plan: SimulationPlan to validate

    Raises:
        ValueError: If plan validation fails
    """
    logger.info("Validating simulation plan with %d batches", plan.total_batches)

    if plan.total_batches == 0:
        raise ValueError("Simulation plan contains no batches")

    # Validate each batch
    for batch in plan.batches:
        if not batch.simulator_name:
            raise ValueError("Batch has empty simulator name")
        if batch.batch_index < 0:
            raise ValueError(f"Batch {batch.batch_index} has invalid index")

        # Validate output configuration
        output_config = batch.simulator_config.output
        if not output_config.file_name:
            raise ValueError(f"Batch {batch.simulator_name}-{batch.batch_index} missing file_name")

    logger.info("Simulation plan validation completed successfully")


def execute_plan(  # pylint: disable=too-many-locals
    plan: SimulationPlan,
    output_directory: Path,
    metadata_directory: Path,
    overwrite: bool,
    max_retries: int = 3,
) -> None:
    """Execute a complete simulation plan.

    The key insight: Simulators are stateful objects. Each simulator is instantiated
    once and then generates multiple batches by calling next() repeatedly. State
    (RNG, counters, filters) accumulates across batches.

    Checkpoint behavior:
    1. After each successfully completed batch, save checkpoint with updated state
    2. Checkpoint contains: completed batch indices, simulator state
    3. On next run, already-completed batches are skipped (resumption)
    4. On successful completion of all batches, checkpoint is cleaned up

    Workflow:
    1. Group batches by simulator name
    2. Load checkpoint to find already-completed batches
    3. For each simulator:
       a. Create ONE simulator instance
       b. For each batch of that simulator:
          - Skip if already completed (from checkpoint)
          - Restore state if reproducing from metadata
          - Call next(simulator) to generate batch (increments state)
          - Save batch output and metadata
          - Save checkpoint with updated state (for resumption)

    Args:
        plan: SimulationPlan to execute
        output_directory: Directory for output files
        metadata_directory: Directory for metadata files
        overwrite: Whether to overwrite existing files
        max_retries: Maximum retries per batch
    """
    logger.info("Executing simulation plan: %d batches", plan.total_batches)

    validate_plan(plan)
    setup_signal_handlers([plan.checkpoint_directory] if plan.checkpoint_directory else [])

    # Initialize checkpoint manager for resumption support
    checkpoint_manager = CheckpointManager(plan.checkpoint_directory)
    completed_batch_indices = checkpoint_manager.get_completed_batch_indices()

    if completed_batch_indices:
        logger.info("Loaded checkpoint: %d batches already completed", len(completed_batch_indices))
    else:
        logger.debug("No checkpoint found or no batches completed yet")

    # Group batches by simulator name to execute sequentially per simulator
    simulator_batches: dict[str, list[SimulationBatch]] = {}
    for batch in plan.batches:
        if batch.simulator_name not in simulator_batches:
            simulator_batches[batch.simulator_name] = []
        simulator_batches[batch.simulator_name].append(batch)

    logger.info("Executing %d simulators", len(simulator_batches))

    with tqdm(total=plan.total_batches, desc="Executing simulation plan") as p_bar:
        for simulator_name, batches in simulator_batches.items():
            logger.info("Starting simulator: %s with %d batches", simulator_name, len(batches))

            # Create ONE simulator instance for all batches of this simulator
            # Extract global simulator arguments from the first batch's global config
            global_sim_args = batches[0].globals_config.simulator_arguments if batches else {}
            simulator = instantiate_simulator(batches[0].simulator_config, simulator_name, global_sim_args)

            # Process batches sequentially, maintaining state across them
            for batch_idx, batch in enumerate(batches):
                # Skip batches that were already completed (for resumption after interrupt)
                if checkpoint_manager.should_skip_batch(batch.batch_index):
                    logger.info(
                        "Skipping batch %d (already completed from checkpoint)",
                        batch.batch_index,
                    )
                    continue

                try:
                    logger.debug(
                        "Executing batch %d/%d for simulator %s",
                        batch_idx + 1,
                        len(batches),
                        simulator_name,
                    )

                    # Capture pre-batch state first for potential retries
                    logger.debug(
                        "[EXECUTE] Batch %s: Before restore - counter=%s, has_state_snapshot=%s",
                        batch.batch_index,
                        simulator.counter,
                        batch.has_state_snapshot(),
                    )
                    restore_batch_state(simulator, batch)
                    logger.debug("[EXECUTE] Batch %s: After restore - counter=%s", batch.batch_index, simulator.counter)
                    pre_batch_state = copy.deepcopy(simulator.state)
                    logger.debug(
                        "[EXECUTE] Batch %s: Captured pre_batch_state - keys=%s",
                        batch.batch_index,
                        list(pre_batch_state.keys()),
                    )

                    def execute_batch(
                        _simulator=simulator,
                        _batch=batch,
                        _output_directory=output_directory,
                        _pre_batch_state=pre_batch_state,
                    ):
                        """Execute a single batch with state management."""
                        # Generate data by calling next() - this advances simulator state
                        logger.debug("[BATCH] %s: Before next() - counter=%s", _batch.batch_index, _simulator.counter)
                        batch_data = _simulator.simulate()
                        logger.debug("[BATCH] %s: After next() - counter=%s", _batch.batch_index, _simulator.counter)

                        # Save the generated data and get all output file paths
                        output_files = process_batch(
                            simulator=_simulator,
                            batch_data=batch_data,
                            batch=_batch,
                            output_directory=_output_directory,
                            overwrite=overwrite,
                        )

                        # Only save metadata if data save succeeded
                        # This ensures metadata only exists for successfully saved data
                        save_batch_metadata(
                            _simulator,
                            _batch,
                            metadata_directory,
                            output_files,
                            pre_batch_state=_pre_batch_state,
                        )
                        # Update the state after successful save
                        _simulator.update_state()

                    def restore_state_for_retry(_simulator=simulator, _pre_batch_state=pre_batch_state):
                        """Restore simulator state to pre-batch state before retry."""
                        _simulator.state = copy.deepcopy(_pre_batch_state)

                    # Execute batch with retry mechanism that restores state on failure
                    retry_with_backoff(
                        execute_batch,
                        max_retries=max_retries,
                        state_restore_func=restore_state_for_retry,
                    )

                    # After successful completion, save checkpoint with updated state
                    # At this point, state has been incremented by next() -> update_state()
                    # Save checkpoint to enable resumption if interrupted before next batch
                    completed_batch_indices.add(batch.batch_index)
                    checkpoint_manager.save_checkpoint(
                        completed_batch_indices=sorted(completed_batch_indices),
                        last_simulator_name=simulator_name,
                        last_completed_batch_index=batch.batch_index,
                        last_simulator_state=copy.deepcopy(simulator.state),
                    )
                    logger.debug(
                        "Checkpoint saved after batch %d - state counter=%s",
                        batch.batch_index,
                        simulator.counter,
                    )
                    p_bar.update(1)

                except Exception as e:
                    logger.error(
                        "Failed to execute batch %d for simulator %s after %d retries: %s",
                        batch.batch_index,
                        simulator_name,
                        max_retries,
                        e,
                    )
                    raise

    # All batches completed successfully - clean up checkpoint files
    checkpoint_manager.cleanup()
    logger.info("All batches completed successfully. Checkpoint files cleaned up.")
