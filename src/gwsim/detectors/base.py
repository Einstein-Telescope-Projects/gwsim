from __future__ import annotations
import configparser
from pathlib import Path

import pycbc.detector
from pycbc.detector import get_available_detectors, add_detector_on_earth

# Store the original for reference
_original_get_available_detectors = get_available_detectors

# Define the path to the available .interferometer config files
detectors_dir = str(Path(__file__).parent / "detectors")
det_dir_path = Path(detectors_dir)
if not det_dir_path.exists() or not det_dir_path.is_dir():
    print(
        f"\n *** Warning: Detector config directory {det_dir_path.absolute()} does not exist or is not a directory. ***")


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

    config = configparser.ConfigParser()
    config.read(file_path)

    sections = config.sections()
    if len(sections) != 1:
        raise ValueError(f"Expected only one detector for config file, found {len(sections)}.")

    section = sections[0]
    params = config[section]
    det_suffix = section

    try:
        # Parse parameters (assume radians for angles/lat/lon, meters for lengths/heights)
        latitude = float(params['LATITUDE'].split(';')[0].strip())
        longitude = float(params['LONGITUDE'].split(';')[0].strip())
        height = float(params.get('ELEVATION', '0').split(';')[0].strip())
        xangle = float(params['X_AZIMUTH'].split(';')[0].strip())
        yangle = float(params['Y_AZIMUTH'].split(';')[0].strip())
        xaltitude = float(params.get('X_ALTITUDE', '0').split(';')[0].strip())
        yaltitude = float(params.get('Y_ALTITUDE', '0').split(';')[0].strip())
        xlength = float(params.get('X_LENGTH', '10000').split(';')[0].strip())
        ylength = float(params.get('Y_LENGTH', '10000').split(';')[0].strip())
    except (KeyError, ValueError, IndexError) as e:
        raise ValueError(f"Error parsing config parameter in {file_path}: {e}")

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
        ylength=ylength
    )

    return det_suffix


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
    config_files = [f.stem for f in path.glob('*.interferometer')]

    return sorted(set(built_in_dets + config_files))


# Monkey-patch get_available_detectors to include config/group names
get_available_detectors = extended_get_available_detectors


class Detector:
    """A wrapper class around pycbc.detector.Detector that handles custom detector configurations from .interferometer files"""

    def __init__(
        self,
        detector_name: str,
        config_dir: str = detectors_dir
    ):
        """
        Initialize Detector class.
        If `detector_name` is a built-in PyCBC detector, use it directly.
        Otherwise, load from the corresponding .interferometer config file.

        Args:
            detector_name (str): The detector name or config name (e.g., 'V1' or 'E1_Triangle_Sardinia').
            config_dir (str, optional): Directory where .interferometer files are stored (default: detectors_dir).
        """

        if detector_name in _original_get_available_detectors():
            det_suffix = detector_name
        else:
            # Load the config and add detector configuration
            det_suffix = load_interferometer_config(
                config_name=detector_name, config_dir=config_dir)
            if not det_suffix:
                raise ValueError(f"No detector loaded from config '{detector_name}'.")

        self._detector = pycbc.detector.Detector(det_suffix)

    def antenna_pattern(self, right_ascension, declination, polarization, t_gps, frequency=0, polarization_type='tensor'):
        """
        Return the antenna pattern for the detector.
        """
        return self._detector.antenna_pattern(right_ascension, declination, polarization, t_gps, frequency, polarization_type)

    def time_delay_from_earth_center(self, right_ascension, declination, t_gps):
        """
        Return the time delay from the Earth center for the detector.
        """
        return self._detector.time_delay_from_earth_center(right_ascension, declination, t_gps)

    def __getattr__(self, attr):
        """
        Delegate attributes to the underlying _detector.
        """
        return getattr(self._detector, attr)

    def __str__(self):
        """
        Return a string representation of the detector name, stripped to the base part.
        """
        return self.name.split('_')[0].strip()
