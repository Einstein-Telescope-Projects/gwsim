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
from typing import Any

import typer
from tqdm import tqdm

from ..generator.base import Generator
from .config import get_config_value, load_config
from .utils import get_file_name_from_template, handle_signal, import_attribute, save_file_safely

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


def get_generator(config: dict) -> Generator:
    """Get the generator from a dictionary of configuration.

    Args:
        config (dict): A dictionary of configuration.

    Raises:
        KeyError: If 'generator' is not in config.
        KeyError: If 'class' is not in config['generator'].
        KeyError: If 'arguments' is not in config['generator'].

    Returns:
        Generator: An instance of a generator.
    """
    if "generator" not in config:
        raise KeyError("Failed to initialize a generator. 'generator' is not found in the configuration file.")

    if "class" not in config["generator"]:
        raise KeyError(
            "Failed to initialize a generator."
            "'class' is not found in the 'generator' section in the configuration file."
        )

    if "arguments" not in config["generator"]:
        raise KeyError(
            "Failed to initialize a generator."
            "'arguments' is not found in the 'generator' section in the configuration file."
        )

    generator_cls = import_attribute(config["generator"]["class"])
    generator = generator_cls(**config["generator"]["arguments"])

    # Print the information.
    logger.info("Generator class: %s", config["generator"]["class"])
    logger.info("Generator arguments: %s", config["generator"]["arguments"])

    return generator


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


def setup_directories(config: dict, metadata: bool) -> tuple[Path, Path, Path, Path, Path]:
    """Set up working directories and checkpoint files.

    Args:
        config (dict): Configuration dictionary.
        metadata (bool): Whether to create metadata directory.

    Returns:
        tuple[Path, Path, Path, Path, Path]: working_directory, checkpoint_file,
            checkpoint_file_backup, output_directory, metadata_directory
    """
    working_directory = Path(get_config_value(config=config, key="working-directory", default_value="."))

    # Checkpoint files
    checkpoint_file = working_directory / "checkpoint.json"
    checkpoint_file_backup = working_directory / "checkpoint.json.bak"

    # Output directory
    output_directory = working_directory / "output/"
    output_directory.mkdir(exist_ok=True)

    # Metadata directory
    metadata_directory = working_directory / "metadata/"
    if metadata:
        metadata_directory.mkdir(exist_ok=True)

    return working_directory, checkpoint_file, checkpoint_file_backup, output_directory, metadata_directory


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


def process_batch(generator: Generator, batch: object, config: BatchProcessingConfig) -> None:
    """Process a single batch of generated data.

    Args:
        generator: The data generator.
        batch: Generated data batch.
        config: Configuration object containing all processing parameters.
    """
    # Get the file name from template
    file_name = Path(get_file_name_from_template(config.file_name_template, generator))
    batch_file_name = config.output_directory / file_name

    # Check whether the file exists
    if batch_file_name.is_file():
        if not config.overwrite:
            raise FileExistsError("Exiting. Use --overwrite to allow overwriting.")
        logger.warning("%s already exists. Overwriting the existing file.", batch_file_name)

    # Write the batch of data to file.
    generator.save_batch(batch, batch_file_name, overwrite=config.overwrite, **config.output_arguments)

    # Write the metadata to file.
    if config.metadata:
        metadata_file_name = config.metadata_directory / file_name.with_suffix(".json")
        generator.save_metadata(file_name=metadata_file_name, overwrite=config.overwrite)

    # Update the state if the data is saved successfully.
    generator.update_state()

    # Create checkpoint file.
    save_file_safely(
        file_name=config.checkpoint_file,
        backup_file_name=config.checkpoint_file_backup,
        save_function=generator.save_state,
        overwrite=True,
    )


def generate_command(
    config_file_name: str = typer.Argument(..., help="Configuration file path"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing files"),
    metadata: bool = typer.Option(False, "--metadata", help="Generate metadata files"),
) -> None:
    """Generate mock data based on the configuration file.

    Args:
        config_file_name (str): Name of the configuration file.
        overwrite (bool): If True, overwrite the existing file, otherwise raise an error if output
            already exists.
        metadata (bool): If True, write the metadata to file.

    Raises:
        FileExistsError: If output file exists and overwrite is False, raise an error.

    Returns:
        None
    """
    config = load_config(file_name=Path(config_file_name))

    # Set up directories and files
    _, checkpoint_file, checkpoint_file_backup, output_directory, metadata_directory = setup_directories(
        config, metadata
    )

    # Set up signal handlers
    setup_signal_handlers(checkpoint_file, checkpoint_file_backup)

    # Get the generator
    generator = get_generator(config)

    # Get configuration values
    file_name_template = get_config_value(config=config["output"], key="file_name")
    output_arguments = get_config_value(config=config["output"], key="arguments", default_value={})

    # Load the checkpoint file if it exists
    if checkpoint_file.is_file():
        generator.load_state(file_name=checkpoint_file)

    # Create batch processing configuration
    batch_config = BatchProcessingConfig(
        file_name_template=file_name_template,
        output_directory=output_directory,
        metadata_directory=metadata_directory,
        output_arguments=output_arguments,
        overwrite=overwrite,
        metadata=metadata,
        checkpoint_file=checkpoint_file,
        checkpoint_file_backup=checkpoint_file_backup,
    )

    # Generate data
    for batch in tqdm(generator, desc="Generating data"):
        process_batch(generator=generator, batch=batch, config=batch_config)
