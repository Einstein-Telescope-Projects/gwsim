"""Utility functions for logging."""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, distribution
from importlib.metadata import version as get_package_version
from pathlib import Path

from gwsim.version import __version__

logger = logging.getLogger("gwsim")


def get_version_information() -> str:
    """Get the version information.

    Returns:
        str: Version information.
    """
    return __version__


def _get_dependencies_from_distribution() -> list[str]:
    """Extract main dependencies from the installed gwsim distribution.

    This uses importlib.metadata to query the installed package metadata,
    which works in both development and installed environments.

    Returns:
        List of package names from the distribution's requires metadata.
        Returns empty list if gwsim distribution cannot be found.
    """
    try:
        dist = distribution("gwsim")
        if dist.requires is None:
            return []

        # Parse dependency specifiers from the requires list
        # Format: "package_name (>=version); extra == 'condition'" or "package_name>=version"
        package_names = []
        for req in dist.requires:
            # Remove version specifiers and extras
            # Split on common delimiters: >, <, =, !, [, ;, space
            package_name = (
                req.split(">")[0].split("<")[0].split("=")[0].split("!")[0].split("[")[0].split(";")[0].strip()
            )
            if package_name and package_name not in package_names:  # Avoid duplicates
                package_names.append(package_name)

        return package_names
    except PackageNotFoundError:
        logger.debug("gwsim distribution not found - unable to extract dependencies")
        return []
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("Failed to extract dependencies from distribution: %s", e)
        return []


def get_dependency_versions() -> dict[str, str | None]:
    """Get versions of gwsim and its main dependencies.

    This retrieves version information for gwsim and all dependencies listed
    in the installed distribution's metadata (from pyproject.toml at build time),
    useful for metadata and reproducibility tracking.

    Works in both development and installed environments since it queries
    the installed distribution metadata rather than the source pyproject.toml.

    Returns:
        Dictionary with package names as keys and version strings as values.
        If a package version cannot be determined, the value is None.
        Always includes 'gwsim' as the first key.
    """
    versions: dict[str, str | None] = {}

    # Always include gwsim first
    try:
        versions["gwsim"] = get_package_version("gwsim")
    except PackageNotFoundError:
        versions["gwsim"] = None

    # Get dependencies from the installed distribution metadata
    dependencies = _get_dependencies_from_distribution()

    for package in dependencies:
        try:
            versions[package] = get_package_version(package)
        except PackageNotFoundError:
            # Package not installed or version cannot be determined
            versions[package] = None

    return versions


def setup_logger(
    outdir: str = ".", label: str | None = None, log_level: str | int = "INFO", print_version: bool = False
):
    """Setup logging output: call at the start of the script to use

    Args:
        outdir (str, optional): If supplied, write the logging output to outdir/label.log. Defaults to '.'.
        label (str, optional): If supplied, write the logging output to outdir/label.log. Defaults to None.
        log_level (str, optional): ['debug', 'info', 'warning']
            Either a string from the list above, or an integer as specified
            in https://docs.python.org/2/library/logging.html#logging-levels
            Defaults to 'INFO'.
        print_version (bool): If true, print version information. Defaults to False.
    """

    if isinstance(log_level, str):
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError as err:
            raise ValueError(f"log_level {log_level} not understood") from err
    else:
        level = int(log_level)

    logger.propagate = False
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)-8s: %(message)s", datefmt="%H:%M")
        )
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers) and label:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        log_file = f"{outdir}/{label}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s", datefmt="%H:%M"))

        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    if print_version:
        version = get_version_information()
        logger.info("Running gwsim version: %s", version)
