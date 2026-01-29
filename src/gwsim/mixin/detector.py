# pylint: disable=too-many-nested-blocks, too-many-branches
# ruff: noqa: PLR0912
"""Detector mixin for simulators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from scipy.interpolate import interp1d

from gwsim.calibration.calibration import CalibrationModel
from gwsim.data.time_series.time_series import TimeSeries
from gwsim.detector.base import Detector


class DetectorMixin:  # pylint: disable=too-few-public-methods
    """Mixin class to add detector information to simulators."""

    def __init__(self, detectors: list[str] | None = None, **kwargs):  # pylint: disable=unused-argument
        """Initialize the DetectorMixin.

        Args:
            detectors (list[str] | None): List of detector names. If None, use all available detectors.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self._metadata = {"detector": {"arguments": {"detectors": detectors}}}
        self.detectors = detectors

    @property
    def detectors(self) -> list[Detector]:
        """Get the list of detectors.

        Returns:
            List of detector names or Detector instances, or None if not set.
        """
        return self._detectors

    @detectors.setter
    def detectors(self, value: list[str] | None) -> None:
        """Set the list of detectors.

        Args:
            value (list[str | Path | Detector] | None):
                List of detector names, config file paths, or Detector instances.
                If None, no detectors are set.
        """
        if value is None:
            self._detectors = []
        elif isinstance(value, list):
            self._detectors = []
            for det in value:

                # NEW: detector specified as a dict (with optional calibration)
                if isinstance(det, dict):
                    if "name" not in det:
                        raise ValueError("Detector dict must contain 'name' key.")
                    name = det["name"]
                    detector = Detector(name=name)

                    if "calibration" in det:
                        cal_cfg = det["calibration"]
                        for key in ("file", "minimum_frequency", "maximum_frequency", "n_points"):
                            if key not in cal_cfg:
                                raise ValueError(f"Calibration config for '{name}' missing required key: '{key}'")
                        cal_file = Path(cal_cfg["file"])

                        try:
                            with open(cal_file, encoding="utf-8") as f:
                                recalib_params = json.load(f)
                        except FileNotFoundError as e:
                            raise FileNotFoundError(
                                f"Calibration file not found for detector '{name}': {cal_file}"
                            ) from e
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Invalid JSON in calibration file for detector '{name}': {cal_file}"
                            ) from e

                        detector.calibration = CalibrationModel(
                            recalibration_parameters=recalib_params,
                            detector_name=name,
                            minimum_frequency=cal_cfg["minimum_frequency"],
                            maximum_frequency=cal_cfg["maximum_frequency"],
                            n_points=cal_cfg["n_points"],
                        )
                    else:
                        detector.calibration = None

                    self._detectors.append(detector)

                else:
                    detector = Detector(name=str(det))
                    detector.calibration = None
                    self._detectors.append(detector)

        else:
            raise ValueError("detectors must be a list.")

    def detectors_are_configured(self) -> bool:
        """Check if all detectors are configured.

        Returns:
            True if all detectors are configured, False otherwise.
        """
        return all(det.is_configured() for det in self.detectors)

    def apply_calibration_fd(
        self,
        detector: Detector,
        polarizations_fd: dict[str, FrequencySeries],
    ) -> dict[str, FrequencySeries]:
        """Apply detector calibration in the frequency domain.

        This method applies the detector calibration transfer function
        to frequency-domain plus and cross polarizations.

        Calibration is applied before any inverse Fourier transform
        to avoid redundant FFTs.

        Args:
            detector:
                Detector instance whose calibration will be applied.
            polarizations_fd:
                Dictionary with 'plus' and 'cross' FrequencySeries.

        Returns:
            Dictionary with calibrated FrequencySeries.
        """
        calibration = getattr(detector, "calibration", None)
        if calibration is None:
            return polarizations_fd
        hp = polarizations_fd["plus"]
        hc = polarizations_fd["cross"]

        freqs = hp.frequencies.to_value()
        # Filter out DC component (0 Hz) to avoid log10(0) = -inf
        valid_mask = freqs > 0
        freqs_valid = freqs[valid_mask]
        cal_valid = calibration.transfer_function(freqs_valid)
        # Reconstruct full calibration array with 1.0 for DC (no calibration applied)
        cal = np.ones_like(freqs)
        cal[valid_mask] = cal_valid
        hp_cal = hp * cal
        hc_cal = hc * cal

        return {"plus": hp_cal, "cross": hc_cal}

    def project_polarizations(  # pylint: disable=too-many-locals,unused-argument
        self,
        polarizations: dict[str, GWpyTimeSeries],
        right_ascension: float,
        declination: float,
        polarization_angle: float,
        earth_rotation: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Project waveform polarizations onto detectors using antenna patterns.

        This method projects the plus and cross polarizations of a gravitational wave
        onto each detector in the network, accounting for antenna response and
        time delays.

        Args:
            polarizations: Dictionary with 'plus' and 'cross' keys containing
                TimeSeries objects of the waveform polarizations.
            right_ascension: RA of source in radians.
            declination: Declination of source in radians.
            polarization_angle: Polarization angle in radians.
            earth_rotation: If True, account for Earth's rotation by computing
                antenna patterns at multiple times and interpolating.
                Defaults to True.

        Returns:
            Dictionary mapping detector names (str) to projected TimeSeries objects.
            Keys are detector names, values are projected strain TimeSeries.

        Raises:
            ValueError: If detectors are not configured.
            ValueError: If polarizations dict doesn't contain 'plus' and 'cross' keys,
                or if detector is not initialized.
            TypeError: If polarizations values are not TimeSeries objects.
        """
        # Validate the detector list
        if not self.detectors_are_configured():
            raise ValueError("Detectors are not configured in the simulator.")

        # Validate inputs
        if not isinstance(polarizations, dict):
            raise TypeError("polarizations must be a dictionary")
        if "plus" not in polarizations or "cross" not in polarizations:
            raise ValueError("polarizations dict must contain 'plus' and 'cross' keys")
        if not isinstance(polarizations["plus"], GWpyTimeSeries):
            raise TypeError("polarizations['plus'] must be a GWpyTimeSeries")
        if not isinstance(polarizations["cross"], GWpyTimeSeries):
            raise TypeError("polarizations['cross'] must be a GWpyTimeSeries")

        hp = polarizations["plus"]
        hc = polarizations["cross"]

        # Convert TimeSeries data to numpy arrays for computation
        # Interpolate the hp and hc data to ensure smooth evaluation
        time_array = cast(np.ndarray, hp.times.to_value())
        reference_time = 0.5 * (time_array[0] + time_array[-1])

        # Compute the time_array minus the reference_time to avoid the systematic large time values
        time_array_wrt_reference = time_array - reference_time

        hp_func = interp1d(time_array_wrt_reference, hp.to_value(), kind="cubic", bounds_error=False, fill_value=0.0)
        hc_func = interp1d(time_array_wrt_reference, hc.to_value(), kind="cubic", bounds_error=False, fill_value=0.0)

        if earth_rotation:
            # Calculate the time delays first
            time_delays = [
                det.time_delay_from_earth_center(
                    right_ascension=right_ascension, declination=declination, t_gps=time_array
                )
                for det in self.detectors
            ]

        else:
            # Calculate the antenna patterns at the reference time
            reference_time = 0.5 * (time_array[0] + time_array[-1])
            antenna_patterns = [
                det.antenna_pattern(
                    right_ascension=right_ascension,
                    declination=declination,
                    polarization=polarization_angle,
                    t_gps=reference_time,
                    polarization_type="tensor",
                )
                for det in self.detectors
            ]

            # Calculate the time delays at the reference time
            time_delays = [
                det.time_delay_from_earth_center(
                    right_ascension=right_ascension, declination=declination, t_gps=reference_time
                )
                for det in self.detectors
            ]

        # Evaluate the detector responses
        detector_responses = np.zeros((len(self.detectors), len(time_array)))
        for i, det in enumerate(self.detectors):
            time_delay = time_delays[i]

            # Shift the waveform data according to time delays
            shifted_times = time_array_wrt_reference + time_delay

            if earth_rotation:
                # Evaluate antenna patterns exactly at the delayed times
                fp_vals, fc_vals = det.antenna_pattern(
                    right_ascension=right_ascension,
                    declination=declination,
                    polarization=polarization_angle,
                    t_gps=time_array + time_delay,
                    polarization_type="tensor",
                )
            else:
                # Use constant antenna patterns (from earlier calculation)
                fp_vals, fc_vals = antenna_patterns[i]

            hp_shifted = hp_func(shifted_times)
            hc_shifted = hc_func(shifted_times)

            detector_responses[i, :] = fp_vals * hp_shifted + fc_vals * hc_shifted

        # Create TimeSeries for projected strain
        start_time = cast(float, time_array[0])
        projected_ts = TimeSeries(data=detector_responses, start_time=start_time, sampling_frequency=hp.sample_rate)
        return projected_ts

    @property
    def metadata(self) -> dict:
        """Get metadata including detector information.

        Returns:
            Dictionary containing the list of detectors.
        """
        return self._metadata
