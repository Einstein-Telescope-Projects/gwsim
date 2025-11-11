"""A utility module for loading gravitational wave interferometer configurations."""

from __future__ import annotations

import configparser
from pathlib import Path

from pycbc.detector import add_detector_on_earth


def load_interferometer_config(config_file: str | Path) -> str:  # pylint: disable=too-many-locals
    """
    Load a .interferometer config file and add its detector using pycbc.detector.add_detector_on_earth.

    Args:
        config_file (str | Path): The path to the config file.

    Returns:
        str: Added detector name (e.g., "E1").
    """
    # Load the .interferometer config file
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found.")

    config = configparser.ConfigParser()
    config.read(config_file)

    sections = config.sections()
    if len(sections) != 1:
        raise ValueError(f"Expected only one detector for config file, found {len(sections)}.")

    section = sections[0]
    params = config[section]
    det_suffix = section

    try:
        # Parse parameters (assume radians for angles/lat/lon, meters for lengths/heights)
        latitude = float(params["LATITUDE"].split(";")[0].strip())
        longitude = float(params["LONGITUDE"].split(";")[0].strip())
        height = float(params.get("ELEVATION", "0").split(";")[0].strip())
        xangle = float(params["X_AZIMUTH"].split(";")[0].strip())
        yangle = float(params["Y_AZIMUTH"].split(";")[0].strip())
        xaltitude = float(params.get("X_ALTITUDE", "0").split(";")[0].strip())
        yaltitude = float(params.get("Y_ALTITUDE", "0").split(";")[0].strip())
        xlength = float(params.get("X_LENGTH", "10000").split(";")[0].strip())
        ylength = float(params.get("Y_LENGTH", "10000").split(";")[0].strip())
    except (KeyError, ValueError, IndexError) as e:
        raise ValueError(f"Error parsing config parameter in {config_file}: {e}") from e

    # Add detector configuration
    add_detector_on_earth(
        name=det_suffix,
        latitude=latitude,
        longitude=longitude,
        height=height,
        xangle=xangle,
        yangle=yangle,
        xaltitude=xaltitude,
        yaltitude=yaltitude,
        xlength=xlength,
        ylength=ylength,
    )

    return det_suffix
