"""
Main command line tool to generate mock data.
"""

from __future__ import annotations

import atexit
import enum
import logging
import signal
from functools import partial
from pathlib import Path

import click
from tqdm import tqdm

from ..generator.base import Generator
from .config import get_config_value, load_config, save_config
from .default_config import DEFAULT_CONFIG
from .utils import (get_file_name_from_template, handle_signal,
                    import_attribute, save_file_safely)

logger = logging.getLogger("gwsim")


class LoggingLevel(enum.Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@click.group("main")
@click.option(
    "--logging-level",
    type=click.Choice(LoggingLevel, case_sensitive=False),
    default=LoggingLevel.INFO,
    help="Logging level.",
)
def main(logging_level: LoggingLevel):
    logger.setLevel(logging_level.value)


@main.command("default-config", help="Write a default configuration file to disk.")
@click.option("--output", type=str, help="File name of the output", default="config.yaml")
@click.option("--overwrite", is_flag=True, help="If flagged, the existing file will be overwritten.")
def default_config(output: str, overwrite: bool) -> None:
    """Write the default configuration file to disk.

    Args:
        output (str): Name of the output file.
        overwrite (bool): If True, overwrite the existing file, otherwise raise an error if output already exists.

    Raises:
        FileExistsError: If file_name exists and overwrite is False, raise an error.
    """
    save_config(file_name=output, config=DEFAULT_CONFIG, overwrite=overwrite)


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
            "Failed to initialize a generator. 'class' is not found in the 'generator' section in the configuration file."
        )

    if "arguments" not in config["generator"]:
        raise KeyError(
            "Failed to initialize a generator. 'arguments' is not found in the 'generator' section in the configuration file."
        )

    generator_cls = import_attribute(config["generator"]["class"])
    generator = generator_cls(**config["generator"]["arguments"])

    # Print the information.
    logger.info("Generator class: %s", config["generator"]["class"])
    logger.info("Generator arguments: %s", config["generator"]["arguments"])

    return generator


def clean_up_generate(checkpoint_file: Path, checkpoint_file_backup: Path):

    # Check whether a backup checkpoint file exists.
    if checkpoint_file_backup.is_file():
        logger.warning("Interrupted while creating a checkpoint file. Restoring the checkpoint file from a backup.")

        try:
            checkpoint_file_backup.rename(checkpoint_file)
            logger.info("Checkpoint file restored from backup.")
        except Exception as e:
            logger.error("Failed to restore checkpoint from backup: %s", e)
    else:
        logger.debug("No backup checkpoint file found. Nothing to clean up.")


@main.command("generate", help="Generate mock data.")
@click.argument("config_file_name", type=str)
@click.option("--overwrite", is_flag=True, help="If flagged, overwrite the existing file.")
@click.option("--metadata", is_flag=True, help="If flagged, write the metadata to file.")
def generate(config_file_name: str, overwrite: bool, metadata: bool) -> None:
    config = load_config(file_name=config_file_name)

    # Working directory.
    working_directory = Path(get_config_value(config=config, key="working-directory", default_value="."))

    # Checkpoint file
    checkpoint_file = working_directory / "checkpoint.json"

    # Backup checkpoint file
    checkpoint_file_backup = working_directory / "checkpoint.json.bak"

    # Output directory
    output_directory = working_directory / "output/"
    output_directory.mkdir(exist_ok=True)

    # Metadata directory
    metadata_directory = working_directory / "metadata/"
    if metadata:
        metadata_directory.mkdir(exist_ok=True)

    # Set up the signal handler as soon as the working directory is defined.
    clean_up_fn = partial(clean_up_generate, checkpoint_file, checkpoint_file_backup)

    # Register clean-up for normal exit
    atexit.register(clean_up_fn)

    # Register cleanup for Ctrl+C and termination
    signal.signal(signal.SIGINT, handle_signal(clean_up_fn))
    signal.signal(signal.SIGTERM, handle_signal(clean_up_fn))

    # Get the generator.
    generator = get_generator(config)

    # Get the file name template.
    file_name_template = get_config_value(config=config["output"], key="file_name")

    # Get the extra arguments for saving data.
    output_arguments = get_config_value(config=config["output"], key="arguments", default_value={})

    # Load the checkpoint file if it exists.
    if checkpoint_file.is_file():
        generator.load_state(file_name=checkpoint_file)

    for batch in tqdm(generator, initial=generator.batch_counter, total=len(generator), desc="Generating data"):
        # Get the file name from template
        file_name = Path(get_file_name_from_template(file_name_template, generator))

        # Get the batch file name.
        batch_file_name = output_directory / file_name

        # Save the batch of data to file.

        # Check whether the file exists
        if batch_file_name.is_file():
            if not overwrite:
                raise FileExistsError("Exiting. Use --overwrite to allow overwriting.")
            logger.warning("%s already exists. Overwriting the existing file.", batch_file_name)

        # Write the batch of data to file.
        generator.save_batch(batch, batch_file_name, overwrite=overwrite, **output_arguments)

        # Write the metadata to file.
        if metadata:
            # Get the metadata file name.
            metadata_file_name = metadata_directory / file_name.with_suffix(".json")

            generator.save_metadata(file_name=metadata_file_name, overwrite=overwrite)

        # Update the state if the data is saved successfully.
        generator.update_state()

        # Create checkpoint file.
        save_file_safely(
            file_name=checkpoint_file,
            backup_file_name=checkpoint_file_backup,
            save_function=generator.save_state,
            overwrite=True,
        )
