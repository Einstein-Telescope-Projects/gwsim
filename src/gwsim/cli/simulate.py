"""
A sub-command to handle data generation using simulation plans.
"""

from __future__ import annotations

import atexit
import copy
import logging
import signal
import time
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from tqdm import tqdm

from gwsim.cli.utils.config import SimulatorConfig, load_config, resolve_class_path
from gwsim.cli.utils.simulation_plan import (
    SimulationBatch,
    SimulationPlan,
    create_batch_metadata,
    create_plan_from_config,
)
from gwsim.cli.utils.utils import handle_signal, import_attribute
from gwsim.simulator.base import Simulator
from gwsim.utils.io import get_file_name_from_template

logger = logging.getLogger("gwsim")


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
            "Restoring simulator state for batch %d from pre-batch snapshot",
            batch.batch_index,
        )
        try:
            simulator.state = batch.pre_batch_state
        except Exception as e:
            logger.error("Failed to restore batch state: %s", e)
            raise ValueError(f"Failed to restore state for batch {batch.batch_index}") from e
    else:
        logger.debug(
            "No pre-batch state snapshot available for batch %d, using fresh state",
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
    )

    # Add output files to metadata for easy discovery
    # Store just the file names, not full paths
    metadata["output_files"] = [f.name for f in output_files]

    metadata_file_name = f"{batch.simulator_name}-{batch.batch_index}.metadata.yaml"
    metadata_file = metadata_directory / metadata_file_name
    logger.debug("Saving batch metadata to %s with %d output files", metadata_file, len(output_files))

    with metadata_file.open("w") as f:
        yaml.safe_dump(metadata, f)

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
    output_files = get_file_name_from_template(
        template=file_name_template,
        instance=simulator,
        output_directory=output_directory,
    )

    # Normalize to list of Paths
    if isinstance(output_files, Path):
        output_files_list = [output_files]
    else:
        # If it's an array (multiple detectors), flatten it
        output_files_list = [Path(str(f)) for f in output_files.flatten()]

    simulator.save_data(
        data=batch_data,
        file_name=file_name_template,
        output_directory=output_directory,
        overwrite=overwrite,
        **output_args,
    )

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

    Workflow:
    1. Group batches by simulator name
    2. For each simulator:
       a. Create ONE simulator instance
       b. For each batch of that simulator:
          - Restore state if reproducing from metadata
          - Call next(simulator) to generate batch
          - Save batch output and metadata
          - State is captured after generation for reproducibility

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
                try:
                    logger.debug(
                        "Executing batch %d/%d for simulator %s",
                        batch_idx + 1,
                        len(batches),
                        simulator_name,
                    )

                    # Capture pre-batch state first for potential retries
                    restore_batch_state(simulator, batch)
                    pre_batch_state = copy.deepcopy(simulator.state)

                    def execute_batch(
                        _simulator=simulator,
                        _batch=batch,
                        _output_directory=output_directory,
                        _pre_batch_state=pre_batch_state,
                    ):
                        """Execute a single batch with state management."""
                        # Generate data by calling next() - this advances simulator state
                        batch_data = next(_simulator)

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

                    def restore_state_for_retry(_simulator=simulator, _pre_batch_state=pre_batch_state):
                        """Restore simulator state to pre-batch state before retry."""
                        _simulator.state = copy.deepcopy(_pre_batch_state)

                    # Execute batch with retry mechanism that restores state on failure
                    retry_with_backoff(
                        execute_batch,
                        max_retries=max_retries,
                        state_restore_func=restore_state_for_retry,
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


def simulate_command(
    config_file_name: Annotated[str, typer.Argument(help="Configuration file path")],
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing files")] = False,
    metadata: Annotated[bool, typer.Option("--metadata", help="Generate metadata files")] = True,
) -> None:
    """Generate gravitational wave simulation data using specified simulators.

    This command creates a simulation plan from the provided YAML configuration,
    validates all parameters, and executes the simulation batches with state
    tracking for reproducibility.

    Args:
        config_file_name: Path to YAML configuration file
        overwrite: Whether to overwrite existing files
        metadata: Whether to save metadata files

    Returns:
        None
    """
    logger.info("Starting simulation from config: %s", config_file_name)

    try:
        # Load configuration
        config = load_config(file_name=Path(config_file_name))
        logger.debug("Configuration loaded successfully from %s", config_file_name)

        # Create simulation plan from configuration
        checkpoint_dir = Path(".gwsim_checkpoints")
        plan = create_plan_from_config(config, checkpoint_dir)
        logger.info("Created simulation plan with %d batches", len(plan.batches))

        # Extract output directories from globals
        # Resolve relative paths relative to working_directory
        working_dir = Path(config.globals.working_directory or ".")  # pylint: disable=no-member
        output_dir_config = config.globals.output_directory or "output"  # pylint: disable=no-member
        output_dir = (
            Path(output_dir_config) if Path(output_dir_config).is_absolute() else working_dir / output_dir_config
        )

        metadata_dir_config = config.globals.metadata_directory or "metadata"  # pylint: disable=no-member
        metadata_dir = (
            (
                Path(metadata_dir_config)
                if Path(metadata_dir_config).is_absolute()
                else working_dir / metadata_dir_config
            )
            if metadata
            else None
        )

        logger.debug("Output directory: %s", output_dir)
        if metadata_dir:
            logger.debug("Metadata directory: %s", metadata_dir)

        # Set up signal handlers for graceful shutdown
        checkpoint_dirs = [plan.checkpoint_directory] if plan.checkpoint_directory else []
        setup_signal_handlers(checkpoint_dirs)

        # Validate plan before execution
        validate_plan(plan)
        logger.info("Simulation plan validation passed")

        # Execute the plan
        execute_plan(
            plan=plan,
            output_directory=output_dir,
            metadata_directory=metadata_dir or Path("metadata"),
            overwrite=overwrite,
            max_retries=3,
        )

        logger.info("Simulation completed successfully. Output written to %s", output_dir)

    except Exception as e:
        logger.error("Simulation failed: %s", str(e), exc_info=True)
        raise typer.Exit(code=1) from e
