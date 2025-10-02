"""
Main command line tool to generate mock data.
"""

from __future__ import annotations

import enum
import logging

import typer
from rich.console import Console
from rich.logging import RichHandler

from .default_config import default_config_command
from .generate import generate_command

logger = logging.getLogger("gwsim")
console = Console()


class LoggingLevel(str, enum.Enum):
    """Logging levels for the CLI."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Create the main Typer app
app = typer.Typer(
    name="gwsim",
    help="Gravitational Wave Simulation Data Generator",
    rich_markup_mode="rich",
)


def setup_logging(level: LoggingLevel = LoggingLevel.INFO) -> None:
    """Set up logging with Rich handler."""
    logging.basicConfig(
        level=level.value,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.callback()
def main(
    logging_level: LoggingLevel = LoggingLevel.INFO,
) -> None:
    """Gravitational Wave Simulation Data Generator.

    Generate synthetic gravitational wave detector data including noise, signals, and glitches.
    """
    setup_logging(logging_level)
    logger.info("Starting gwsim with logging level: %s", logging_level.value)


# Import and register commands after app is created
def register_commands() -> None:
    """Register all CLI commands."""

    app.command("generate")(generate_command)
    app.command("default-config")(default_config_command)


register_commands()
