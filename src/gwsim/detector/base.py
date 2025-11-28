"""A module to handle gravitational wave detector configurations,"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from pycbc.detector import Detector as PyCBCDetector

from gwsim.detector.utils import DEFAULT_DETECTOR_BASE_PATH, load_interferometer_config

logger = logging.getLogger("gwsim")


def _load_all_detector_configuration_files(config_dir: Path = DEFAULT_DETECTOR_BASE_PATH) -> None:
    # Glob all files in the config_dir with .interferometer extension
    for config_file in config_dir.glob("*.interferometer"):
        try:
            load_interferometer_config(config_file)
            logger.debug("Loaded detector configuration from %s", config_file)
        except (OSError, ValueError) as e:
            logger.warning("Failed to load detector configuration from %s: %s", config_file, e)


_load_all_detector_configuration_files(config_dir=DEFAULT_DETECTOR_BASE_PATH)


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
        self._metadata = {
            "arguments": {
                "name": name,
                "configuration_file": configuration_file,
            }
        }
        if name is not None and configuration_file is None:
            try:
                self._detector = PyCBCDetector(str(name))
                self.name = str(name)
            except ValueError as e:
                logger.warning("Detector name '%s' not found in PyCBC: %s", name, e)
                logger.warning("Setting up detector with no configuration.")
                self._detector = None
                self.name = str(name)
        elif name is None and configuration_file is not None:
            configuration_file = Path(configuration_file)

            if configuration_file.is_file():

                logger.debug("Loading detector from configuration file: %s", configuration_file)

                prefix = load_interferometer_config(config_file=configuration_file)

            elif (DEFAULT_DETECTOR_BASE_PATH / configuration_file).is_file():

                logger.debug("Loading detector from default path: %s", configuration_file)

                prefix = load_interferometer_config(config_file=DEFAULT_DETECTOR_BASE_PATH / configuration_file)
            else:
                raise FileNotFoundError(f"Configuration file '{configuration_file}' not found.")
            self._detector = PyCBCDetector(prefix)
            self.name = prefix
        elif name is not None and configuration_file is not None:
            raise ValueError("Specify either 'name' or 'configuration_file', not both.")
        else:
            raise ValueError("Either 'name' or 'configuration_file' must be provided.")

        self.configuration_file = configuration_file

    def is_configured(self) -> bool:
        """
        Check if the detector is properly configured.
        """
        return self._detector is not None

    def antenna_pattern(
        self, right_ascension, declination, polarization, t_gps, frequency=0, polarization_type="tensor"
    ):
        """
        Return the antenna pattern for the detector.
        """
        if not self.is_configured():
            raise ValueError(f"Detector '{self.name}' is not configured.")
        detector = cast(PyCBCDetector, self._detector)
        return detector.antenna_pattern(right_ascension, declination, polarization, t_gps, frequency, polarization_type)

    def time_delay_from_earth_center(self, right_ascension, declination, t_gps):
        """
        Return the time delay from the Earth center for the detector.
        """
        if not self.is_configured():
            raise ValueError(f"Detector '{self.name}' is not configured.")
        detector = cast(PyCBCDetector, self._detector)
        return detector.time_delay_from_earth_center(right_ascension, declination, t_gps)

    def __getattr__(self, attr):
        """
        Delegate attributes to the underlying _detector.
        """
        return getattr(self._detector, attr)

    def __str__(self) -> str:
        """
        Return a string representation of the detector name, stripped to the base part.

        Returns:
            str: The detector name.
        """
        return self.name

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Detector instance.

        Returns:
            str: A string representation of the Detector instance.
        """
        return f"Detector(name={self.name}, configured={self.is_configured()})"

    @staticmethod
    def get_detector(name: str | Path) -> Detector:
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

    @property
    def metadata(self) -> dict:
        """Get a dictionary of metadata.

        Returns:
            dict: A dictionary of metadata.
        """
        return self._metadata
