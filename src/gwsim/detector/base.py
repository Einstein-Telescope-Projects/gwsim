"""A module to handle gravitational wave detector configurations,"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pycbc.detector

from gwsim.detector.utils import DEFAULT_DETECTOR_BASE_PATH, load_interferometer_config

# Store the original for reference
_original_get_available_detectors = get_available_detectors


# Define the path to the available .interferometer config files
detectors_dir = str(Path(__file__).parent / "detectors")
det_dir_path = Path(detectors_dir)
if not det_dir_path.exists() or not det_dir_path.is_dir():
    print(
        f"\n *** Warning: Detector config directory {det_dir_path.absolute()} does not exist or is not a directory. ***"
    )


def _bilby_to_pycbc_parameters(bilby_params: dict) -> dict:
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
    pycbc_params = dict()

    pycbc_params["name"] = bilby_params["name"]
    pycbc_params["latitude"] = np.deg2rad(bilby_params["latitude"])
    pycbc_params["longitude"] = np.deg2rad(bilby_params["longitude"])
    pycbc_params["height"] = bilby_params["elevation"]
    pycbc_params["xangle"] = (np.pi / 2 - np.deg2rad(bilby_params["xarm_azimuth"])) % (2 * np.pi)
    pycbc_params["yangle"] = (np.pi / 2 - np.deg2rad(bilby_params["yarm_azimuth"])) % (2 * np.pi)
    pycbc_params["xaltitude"] = bilby_params["xarm_tilt"]
    pycbc_params["yaltitude"] = bilby_params["yarm_tilt"]
    pycbc_params["xlength"] = bilby_params["length"] * 1000
    pycbc_params["ylength"] = bilby_params["length"] * 1000

    return pycbc_params


def load_interferometer_config(config_name: str, config_dir: str = detectors_dir) -> str:
    """
    Load a .interferometer config file and add its detector using pycbc.detector.add_detector_on_earth.

    Args:
        config_name (str): The base name of the config file (e.g., "E1_Triangle_Sardinia").
        config_dir (str, optional): Directory where .interferometer files are stored. Default is detectors_dir.

    Returns:
        str: Added detector name (e.g., "E1").
    """
    # Load the .interferometer config file
    path = Path(config_dir)
    file_path = path / f"{config_name}.interferometer"
    if not file_path.exists():
        raise FileNotFoundError(f"Config file {file_path} not found.")

    # Extract the parameters
    bilby_params = dict()
    with open(file_path) as parameter_file:
        lines = parameter_file.readlines()
        for line in lines:
            if line[0] == "#" or line[0] == "\n":
                continue
            split_line = line.split("=")
            key = split_line[0].strip()
            value = eval("=".join(split_line[1:]))
            bilby_params[key] = value

    params = _bilby_to_pycbc_parameters(bilby_params)
    det_name = params["name"]

    # Add detector configuration
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


def extended_get_available_detectors(config_dir: str = detectors_dir) -> list[str]:
    """
    Extended version of pycbc.detector.get_available_detectors that includes both built-in detectors
    and available .interferometer config names.

    Args:
        config_dir (str, optional): Directory where .interferometer files are stored (default: detectors_dir).

    Returns:
        list[str]: Sorted list of available detector names, including configs.
    """
    built_in_dets = _original_get_available_detectors()
    path = Path(config_dir)
    config_files = [f.stem for f in path.glob("*.interferometer")]

    return sorted(set(built_in_dets + config_files))
logger = logging.getLogger("gwsim")


class Detector:
    """A wrapper class around pycbc.detector.Detector that
    handles custom detector configurations from .interferometer files
    """

    def __init__(self, name: str | None = None, configuration_file: str | Path | None = None):
        """
        Initialize Detector class.
        If `detector_name` is a built-in PyCBC detector, use it directly.
        Otherwise, load from the corresponding .interferometer config file.

        Args:
            detector_name (str): The detector name or config name (e.g., 'V1' or 'E1_Triangle_Sardinia').
            config_dir (str, optional): Directory where .interferometer files are stored (default: detectors_dir).
        """
        if name is not None and configuration_file is None:
            try:
                self._detector = pycbc.detector.Detector(str(name))
                self.name = str(name)
            except ValueError as e:
                logger.warning("Detector name '%s' not found in PyCBC: %s", name, e)
                logger.warning("Setting up detector with no configuration.")
                self._detector = None
                self.name = str(name)
        elif configuration_file is not None:
            configuration_file = Path(configuration_file)

            if configuration_file.is_file():

                logger.debug("Loading detector from configuration file: %s", configuration_file)

                prefix = load_interferometer_config(config_file=configuration_file)

            elif (DEFAULT_DETECTOR_BASE_PATH / configuration_file).is_file():

                logger.debug("Loading detector from default path: %s", configuration_file)

                prefix = load_interferometer_config(config_file=DEFAULT_DETECTOR_BASE_PATH / configuration_file)
            else:
                raise FileNotFoundError(f"Configuration file '{configuration_file}' not found.")
            self._detector = pycbc.detector.Detector(prefix)
            self.name = prefix
            if name is not None:
                logger.warning("Using the detector prefix as the name. Ignoring provided name: %s", name)
        else:
            raise ValueError("Either name or configuration_file must be provided.")

    def is_configured(self) -> bool:
        """
        Check if the detector is properly configured.
        """
        return isinstance(self._detector, pycbc.detector.Detector)

    def antenna_pattern(
        self, right_ascension, declination, polarization, t_gps, frequency=0, polarization_type="tensor"
    ):
        """
        Return the antenna pattern for the detector.
        """
        if not self.is_configured():
            raise ValueError(f"Detector '{self.name}' is not configured.")
        detector = cast(pycbc.detector.Detector, self._detector)
        return detector.antenna_pattern(right_ascension, declination, polarization, t_gps, frequency, polarization_type)

    def time_delay_from_earth_center(self, right_ascension, declination, t_gps):
        """
        Return the time delay from the Earth center for the detector.
        """
        if not self.is_configured():
            raise ValueError(f"Detector '{self.name}' is not configured.")
        detector = cast(pycbc.detector.Detector, self._detector)
        return detector.time_delay_from_earth_center(right_ascension, declination, t_gps)

    def __getattr__(self, attr):
        """
        Delegate attributes to the underlying _detector.
        """
        return getattr(self._detector, attr)

    def __str__(self):
        """
        Return a string representation of the detector name, stripped to the base part.
        """
        return self.name

    @staticmethod
    def get_detector(name: str | Path) -> Detector | str:
        """A helper function to get a Detector instance or return the name string.

        Args:
            name: Name of the detector (e.g., 'H1', 'L1') or configuration.

        Returns:
            Detector instance if loading is successful, otherwise returns the name string.
        """
        # First check if name corresponds to a configuration file
        if Path(name).is_file() or (DEFAULT_DETECTOR_BASE_PATH / name).is_file():
            return Detector(configuration_file=name)
        return Detector(name=str(name))
