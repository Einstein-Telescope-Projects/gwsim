"""A utility module for loading gravitational wave interferometer configurations."""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import numpy as np
from pycbc.detector import add_detector_on_earth

# The default base path for detector configuration files
DEFAULT_DETECTOR_BASE_PATH = Path(__file__).parent / "detectors"


def _bilby_to_pycbc_detector_parameters(bilby_params: dict) -> dict:
    """
    Convert Bilby detector parameters to PyCBC-compatible parameters.

    This function handles the conversion of units and conventions between Bilby and PyCBC,
    including latitude/longitude to radians, length from km to meters, and azimuth adjustments
    due to different reference conventions (Bilby: from East counterclockwise; PyCBC/LAL: from North clockwise).

    Args:
        bilby_params (dict): Dictionary of Bilby parameters (e.g., 'latitude', 'xarm_azimuth', etc.).

    Returns:
        dict: Dictionary of converted PyCBC parameters.
    """
    pycbc_params = {
        "name": bilby_params["name"],
        "latitude": np.deg2rad(bilby_params["latitude"]),
        "longitude": np.deg2rad(bilby_params["longitude"]),
        "height": bilby_params["elevation"],
        "xangle": (np.pi / 2 - np.deg2rad(bilby_params["xarm_azimuth"])) % (2 * np.pi),
        "yangle": (np.pi / 2 - np.deg2rad(bilby_params["yarm_azimuth"])) % (2 * np.pi),
        "xaltitude": bilby_params["xarm_tilt"],
        "yaltitude": bilby_params["yarm_tilt"],
        "xlength": bilby_params["length"] * 1000,
        "ylength": bilby_params["length"] * 1000,
    }

    return pycbc_params


def load_interferometer_config(config_file: str | Path, encoding: str = "utf-8") -> str:
    """
    Load a .interferometer config file and add its detector using pycbc.detector.add_detector_on_earth.

    Args:
        config_file: The path to the config file.
        encoding: The file encoding to use when reading the config file. Default is 'utf-8'.

    Returns:
        str: Added detector name (e.g., "E1").
    """
    # Load the .interferometer config file
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found.")

    bilby_params = {}
    with config_file.open(encoding=encoding) as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == "#" or line[0] == "\n":
                continue
            split_line = line.split("=")
            key = split_line[0].strip()
            if key == "power_spectral_density":
                continue
            value = literal_eval("=".join(split_line[1:]))
            bilby_params[key] = value

    params = _bilby_to_pycbc_detector_parameters(bilby_params)
    det_name = params["name"]

    add_detector_on_earth(
        name=det_name,
        latitude=params["latitude"],
        longitude=params["longitude"],
        height=params["height"],
        xangle=params["xangle"],
        yangle=params["yangle"],
        xaltitude=params["xaltitude"],
        yaltitude=params["yaltitude"],
        xlength=params["xlength"],
        ylength=params["ylength"],
    )

    return det_name
