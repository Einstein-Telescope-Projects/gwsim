"""
Main command line tool to generate mock data.
"""

from __future__ import annotations

import enum
import logging

import click

from .default_config import default_config
from .generate import generate

logger = logging.getLogger("gwsim")


class LoggingLevel(enum.Enum):
    """Logging levels for the CLI."""

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
    """Main command line tool."""
    logger.setLevel(logging_level.value)


main.add_command(default_config)
main.add_command(generate)
