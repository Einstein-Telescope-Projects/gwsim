"""A package to simulate a population of gravitational waves."""

from __future__ import annotations

from . import utils
from .utils.log import setup_logger
from .version import __version__

setup_logger()

__all__ = ["__version__", "utils"]
