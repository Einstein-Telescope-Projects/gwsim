"""
A sub-command to handle data generation.
"""

from __future__ import annotations

import atexit
import logging
import signal
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import typer
from tqdm import tqdm

from ..simulator.base import Simulator
from .config import get_config_value, load_config, process_config, resolve_class_path
from .utils.retry import RetryManager
from .utils.template import TemplateValidator
from .utils.utils import get_file_name_from_template, handle_signal, import_attribute, save_file_safely

logger = logging.getLogger("gwsim")


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""

    file_name_template: str
    output_directory: Path
    metadata_directory: Path
    output_arguments: dict[str, Any]
    overwrite: bool
    metadata: bool
    checkpoint_file: Path
    checkpoint_file_backup: Path


def get_simulator(simulator_name: str, simulator_config: dict, resolved_class_path: str) -> Simulator:
    """Get the simulator from a dictionary of configuration.

    Args:
        simulator_name (str): Name of the simulator for logging.
        simulator_config (dict): Simulator-specific configuration.
        resolved_class_path (str): Fully resolved class path.

    Raises:
        KeyError: If 'class' is not in simulator_config.
        KeyError: If 'arguments' is not in simulator_config.

    Returns:
        Simulator: An instance of a simulator.
    """
    if "class" not in simulator_config:
        raise KeyError(
            f"Failed to initialize simulator '{simulator_name}'. "
            "'class' is not found in the simulator configuration."
        )

    if "arguments" not in simulator_config:
        raise KeyError(
            f"Failed to initialize simulator '{simulator_name}'. "
            "'arguments' is not found in the simulator configuration."
        )

    simulator_cls = import_attribute(resolved_class_path)
    simulator = simulator_cls(**simulator_config["arguments"])

    # Print the information.
    logger.info("Simulator '%s' class: %s", simulator_name, resolved_class_path)
    logger.info("Simulator '%s' arguments: %s", simulator_name, simulator_config["arguments"])

    return simulator


def clean_up_generate(checkpoint_file: Path, checkpoint_file_backup: Path) -> None:
    """Clean-up function to be called when the signal is received.

    Args:
        checkpoint_file (Path): Path to the checkpoint file.
        checkpoint_file_backup (Path): Path to the backup checkpoint file.

    Returns:
        None
    """

    # Check whether a backup checkpoint file exists.
    if checkpoint_file_backup.is_file():
        logger.warning("Interrupted while creating a checkpoint file. Restoring the checkpoint file from a backup.")

        try:
            checkpoint_file_backup.rename(checkpoint_file)
            logger.info("Checkpoint file restored from backup.")
        except (OSError, FileNotFoundError, PermissionError) as e:
            logger.error("Failed to restore checkpoint from backup: %s", e)
            logger.warning("Continuing without checkpoint restoration.")
    else:
        logger.debug("No backup checkpoint file found. Nothing to clean up.")


def setup_signal_handlers(checkpoint_file: Path, checkpoint_file_backup: Path) -> None:
    """Set up signal handlers for graceful shutdown.

    Args:
        checkpoint_file (Path): Path to checkpoint file.
        checkpoint_file_backup (Path): Path to backup checkpoint file.
    """
    clean_up_fn = partial(clean_up_generate, checkpoint_file, checkpoint_file_backup)

    # Register clean-up for normal exit
    atexit.register(clean_up_fn)

    # Register cleanup for Ctrl+C and termination
    signal.signal(signal.SIGINT, handle_signal(clean_up_fn))
    signal.signal(signal.SIGTERM, handle_signal(clean_up_fn))


def process_batch_with_rollback(
    simulator: Simulator, batch: object, config: BatchProcessingConfig, detector: str | None = None
) -> None:
    """Process a single batch of generated data with rollback capability.

    Args:
        simulator: The data simulator.
        batch: Generated data batch.
        config: Configuration object containing all processing parameters.
        detector: Optional detector name for multi-detector scenarios.
    """
    # Capture state before processing for rollback capability
    pre_processing_state = simulator.state.copy()

    try:
        process_batch(simulator, batch, config, detector)
    except (
        OSError,
        PermissionError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        ValueError,
        AttributeError,
    ) as e:
        # Rollback on any failure
        simulator.state = pre_processing_state
        logger.warning("Batch processing failed, rolled back state: %s", e)
        raise


def process_batch(
    simulator: Simulator, batch: object, config: BatchProcessingConfig, detector: str | None = None
) -> None:
    """Process a single batch of generated data.

    Args:
        simulator: The data simulator.
        batch: Generated data batch.
        config: Configuration object containing all processing parameters.
        detector: Optional detector name for multi-detector scenarios.
    """
    # Get the file name from template, handling detector placeholders
    file_name_template = config.file_name_template
    if detector and "{detector}" in file_name_template:
        file_name_template = file_name_template.replace("{detector}", detector)

    file_name = Path(get_file_name_from_template(file_name_template, simulator))
    batch_file_name = config.output_directory / file_name

    # Check whether the file exists
    if batch_file_name.is_file():
        if not config.overwrite:
            raise FileExistsError("Exiting. Use --overwrite to allow overwriting.")
        logger.warning("%s already exists. Overwriting the existing file.", batch_file_name)

    # Prepare output arguments, handling detector placeholders in channel names
    output_arguments = config.output_arguments.copy()
    if detector and "channel" in output_arguments:
        channel = output_arguments["channel"]
        if "{detector}" in channel:
            output_arguments["channel"] = channel.replace("{detector}", detector)

    # Handle list of channels with detector placeholders
    if detector and "channels" in output_arguments:
        channels = output_arguments["channels"]
        if isinstance(channels, list):
            output_arguments["channels"] = [
                ch.replace("{detector}", detector) if "{detector}" in ch else ch for ch in channels
            ]

    # Write the batch of data to file.
    simulator.save_batch(batch, batch_file_name, overwrite=config.overwrite, **output_arguments)

    # Write the metadata to file.
    if config.metadata:
        metadata_file_name = config.metadata_directory / file_name.with_suffix(".json")
        simulator.save_metadata(file_name=metadata_file_name, overwrite=config.overwrite)

    # Update the state if the data is saved successfully (backward compatibility)
    if hasattr(simulator, "update_state"):
        simulator.update_state()
    # Note: New StateAttribute-based simulators advance state automatically

    # Create checkpoint file.
    save_file_safely(
        file_name=config.checkpoint_file,
        backup_file_name=config.checkpoint_file_backup,
        save_function=simulator.save_state,
        overwrite=True,
    )


@dataclass
class GenerationSetup:
    """Configuration setup for generation process."""

    working_directory: Path
    output_directory: Path
    metadata_directory: Path
    checkpoint_file: Path
    checkpoint_file_backup: Path


def validate_configuration_phase(processed_config: dict) -> None:
    """Validate all templates and configurations before generation starts.

    Args:
        processed_config: Processed configuration dictionary

    Raises:
        ValueError: If any validation fails
    """
    logger.info("Validating configuration...")

    for simulator_name, simulator_config in processed_config["simulators"].items():
        # Validate template syntax
        output_config = simulator_config.get("output", {})
        file_name_template = output_config.get(
            "file_name", f"{simulator_name}-{{{{ start_time }}}}-{{{{ duration }}}}.gwf"
        )

        is_valid, errors = TemplateValidator.validate_template(file_name_template, simulator_name)
        if not is_valid:
            raise ValueError(f"Invalid template for simulator '{simulator_name}': {errors}")

        # Validate required configuration keys
        if "class" not in simulator_config:
            raise ValueError(f"Missing 'class' in configuration for simulator '{simulator_name}'")

        if "arguments" not in simulator_config:
            raise ValueError(f"Missing 'arguments' in configuration for simulator '{simulator_name}'")

    logger.info("Configuration validation completed successfully")


def setup_generation_directories(globals_config: dict, metadata: bool) -> GenerationSetup:
    """Setup directories and paths for generation.

    Args:
        globals_config: Global configuration dictionary
        metadata: Whether metadata generation is enabled

    Returns:
        GenerationSetup with configured paths
    """
    working_directory = Path(get_config_value(globals_config, "working-directory", "."))

    # Setup base directories
    checkpoint_file = working_directory / "checkpoint.json"
    checkpoint_file_backup = working_directory / "checkpoint.json.bak"

    output_directory = working_directory / get_config_value(globals_config, "output-directory", "output")
    output_directory.mkdir(exist_ok=True)

    metadata_directory = working_directory / get_config_value(globals_config, "metadata-directory", "metadata")
    if metadata:
        metadata_directory.mkdir(exist_ok=True)

    return GenerationSetup(
        working_directory=working_directory,
        output_directory=output_directory,
        metadata_directory=metadata_directory,
        checkpoint_file=checkpoint_file,
        checkpoint_file_backup=checkpoint_file_backup,
    )


def create_simulator_instance(simulator_name: str, simulator_config: dict) -> Simulator:
    """Create and configure a simulator instance.

    Args:
        simulator_name: Name of the simulator
        simulator_config: Simulator configuration

    Returns:
        Configured simulator instance
    """
    class_spec = simulator_config["class"]
    resolved_class_path = resolve_class_path(class_spec, simulator_name)
    return get_simulator(simulator_name, simulator_config, resolved_class_path)


def setup_batch_config(
    simulator_name: str,
    simulator_config: dict,
    setup: GenerationSetup,
    overwrite: bool,
    metadata: bool,
) -> BatchProcessingConfig:
    """Setup batch processing configuration for a simulator.

    Args:
        simulator_name: Name of the simulator
        simulator_config: Simulator configuration
        setup: Generation setup paths
        overwrite: Whether to overwrite existing files
        metadata: Whether to generate metadata

    Returns:
        BatchProcessingConfig instance
    """
    output_config = simulator_config.get("output", {})
    file_name_template = output_config.get("file_name", f"{simulator_name}-{{{{ start_time }}}}-{{{{ duration }}}}.gwf")
    output_arguments = output_config.get("arguments", {})

    simulator_checkpoint = setup.checkpoint_file.parent / f"checkpoint_{simulator_name}.json"

    return BatchProcessingConfig(
        file_name_template=file_name_template,
        output_directory=setup.output_directory,
        metadata_directory=setup.metadata_directory,
        output_arguments=output_arguments,
        overwrite=overwrite,
        metadata=metadata,
        checkpoint_file=simulator_checkpoint,
        checkpoint_file_backup=simulator_checkpoint.with_suffix(".json.bak"),
    )


def execute_simulator_with_retry(
    simulator: Simulator,
    simulator_name: str,
    batch_config: BatchProcessingConfig,
    detectors: list[str],
    max_retries: int = 3,
) -> None:
    """Execute simulator with retry capability.

    Args:
        simulator: The simulator instance
        simulator_name: Name of the simulator
        batch_config: Batch processing configuration
        detectors: List of detectors
        max_retries: Maximum number of retry attempts
    """
    retry_manager = RetryManager(max_retries=max_retries)

    def single_execution():
        return execute_simulator_with_rollback(simulator, simulator_name, batch_config, detectors)

    retry_manager.retry_with_backoff(single_execution)


def execute_simulator_with_rollback(
    simulator: Simulator,
    simulator_name: str,
    batch_config: BatchProcessingConfig,
    detectors: list[str],
) -> None:
    """Execute simulator with rollback capability for batch failures.

    Args:
        simulator: The simulator instance
        simulator_name: Name of the simulator
        batch_config: Batch processing configuration
        detectors: List of detectors
    """
    uses_detector_placeholder = "{detector}" in batch_config.file_name_template or any(
        "{detector}" in str(v) for v in batch_config.output_arguments.values()
    )

    if uses_detector_placeholder and detectors:
        # Multi-detector simulator: process each detector separately
        logger.info("Multi-detector mode: generating %s data for %d detectors", simulator_name, len(detectors))
        for detector in detectors:
            logger.info("Generating %s data for detector: %s", simulator_name, detector)

            # Check if simulator supports multi-detector operation
            if hasattr(simulator, "set_detector"):
                simulator.set_detector(detector)

            # Process batches for this detector with rollback
            for batch in tqdm(simulator, desc=f"Generating {simulator_name} data for {detector}"):
                process_batch_with_rollback(simulator=simulator, batch=batch, config=batch_config, detector=detector)
    else:
        # Single simulator, population simulator, or network-wide simulator
        logger.info("Single-mode: generating %s data (detector-agnostic)", simulator_name)
        for batch in tqdm(simulator, desc=f"Generating {simulator_name} data"):
            process_batch_with_rollback(simulator=simulator, batch=batch, config=batch_config)


def execute_simulator(
    simulator: Simulator,
    simulator_name: str,
    batch_config: BatchProcessingConfig,
    detectors: list[str],
) -> None:
    """Execute simulator with appropriate detector handling.

    Args:
        simulator: The simulator instance
        simulator_name: Name of the simulator
        batch_config: Batch processing configuration
        detectors: List of detectors
    """
    uses_detector_placeholder = "{detector}" in batch_config.file_name_template or any(
        "{detector}" in str(v) for v in batch_config.output_arguments.values()
    )

    if uses_detector_placeholder and detectors:
        # Multi-detector simulator: process each detector separately
        logger.info("Multi-detector mode: generating %s data for %d detectors", simulator_name, len(detectors))
        for detector in detectors:
            logger.info("Generating %s data for detector: %s", simulator_name, detector)

            # Check if simulator supports multi-detector operation
            if hasattr(simulator, "set_detector"):
                simulator.set_detector(detector)

            # Process batches for this detector
            for batch in tqdm(simulator, desc=f"Generating {simulator_name} data for {detector}"):
                process_batch(simulator=simulator, batch=batch, config=batch_config, detector=detector)
    else:
        # Single simulator, population simulator, or network-wide simulator
        logger.info("Single-mode: generating %s data (detector-agnostic)", simulator_name)
        for batch in tqdm(simulator, desc=f"Generating {simulator_name} data"):
            process_batch(simulator=simulator, batch=batch, config=batch_config)


def process_single_simulator(
    simulator_name: str,
    simulator_config: dict,
    globals_config: dict,
    setup: GenerationSetup,
    overwrite: bool,
    metadata: bool,
) -> None:
    """Process a single simulator.

    Args:
        simulator_name: Name of the simulator
        simulator_config: Simulator configuration
        globals_config: Global configuration
        setup: Generation setup paths
        overwrite: Whether to overwrite existing files
        metadata: Whether to generate metadata
    """
    logger.info("Processing simulator: %s", simulator_name)

    # Create simulator instance
    simulator = create_simulator_instance(simulator_name, simulator_config)

    # Setup batch processing configuration
    batch_config = setup_batch_config(simulator_name, simulator_config, setup, overwrite, metadata)

    # Load checkpoint if exists
    if batch_config.checkpoint_file.is_file():
        simulator.load_state(file_name=batch_config.checkpoint_file)

    # Execute simulator with retry and rollback capabilities
    detectors = globals_config.get("detectors", [])
    max_retries = globals_config.get("max_retries", 3)
    execute_simulator_with_retry(simulator, simulator_name, batch_config, detectors, max_retries)


def simulate_command(
    config_file_name: Annotated[str, typer.Argument(help="Configuration file path")],
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing files")] = False,
    metadata: Annotated[bool, typer.Option("--metadata", help="Generate metadata files")] = False,
) -> None:
    """Generate gravitational wave simulation data using specified simulators.

    This command processes simulators sequentially based on the configuration file.
    Each simulator can operate in either single-mode (detector-agnostic) or
    multi-detector mode depending on the file name template and output configuration.

    Args:
        config_file_name (str): Path to the configuration file.
        overwrite (bool): If True, overwrite the existing file, otherwise raise an error if output
            already exists.
        metadata (bool): If True, write the metadata to file.

    Raises:
        FileExistsError: If output file exists and overwrite is False, raise an error.

    Returns:
        None
    """
    # Load and process configuration
    raw_config = load_config(file_name=Path(config_file_name))
    processed_config = process_config(raw_config)
    globals_config = processed_config["globals"]

    # Validation phase - fail fast on configuration errors
    validate_configuration_phase(processed_config)

    # Setup directories and paths
    setup = setup_generation_directories(globals_config, metadata)

    # Set up signal handlers
    setup_signal_handlers(setup.checkpoint_file, setup.checkpoint_file_backup)

    # Process each simulator sequentially
    for simulator_name, simulator_config in processed_config["simulators"].items():
        process_single_simulator(
            simulator_name=simulator_name,
            simulator_config=simulator_config,
            globals_config=globals_config,
            setup=setup,
            overwrite=overwrite,
            metadata=metadata,
        )
