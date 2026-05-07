"""Detector mixin for simulators."""

from __future__ import annotations

import lal
import numpy as np
from gwmock_signal import CustomDetector
from gwmock_signal.projection import project_polarizations_to_network
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock.data.time_series.time_series import TimeSeries
from gwmock.signal.adapter import SignalAdapter


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
    def detectors(self) -> list[str | CustomDetector]:
        """Get the list of detectors.

        Returns:
            List of resolved public detector specs, or an empty list if not set.
        """
        return self._detectors

    @detectors.setter
    def detectors(self, value: list[str] | None) -> None:
        """Set the list of detectors.

        Args:
            value (list[str] | None):
                List of detector names, preset aliases, or config file paths.
                If None, no detectors are set.
        """
        if value is None:
            self._detectors = []
        elif isinstance(value, list):
            self._detectors = list(SignalAdapter.resolve_detector_network(value).detector_names)
        else:
            raise ValueError("detectors must be a list.")

    def detectors_are_configured(self) -> bool:
        """Check if all detectors are configured.

        Returns:
            True if all detectors are configured, False otherwise.
        """
        return all(
            isinstance(detector, CustomDetector) or detector in lal.cached_detector_by_prefix
            for detector in self.detectors
        )

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

        projected_by_detector = project_polarizations_to_network(
            polarizations,
            self.detectors,
            right_ascension=right_ascension,
            declination=declination,
            polarization_angle=polarization_angle,
            earth_rotation=earth_rotation,
        )
        detector_names = [detector if isinstance(detector, str) else detector.name for detector in self.detectors]
        detector_responses = np.vstack([projected_by_detector[detector_name].value for detector_name in detector_names])
        reference_series = projected_by_detector[detector_names[0]]
        return TimeSeries(
            data=detector_responses,
            start_time=reference_series.t0,
            sampling_frequency=reference_series.sample_rate,
        )

    @property
    def metadata(self) -> dict:
        """Get metadata including detector information.

        Returns:
            Dictionary containing the list of detectors.
        """
        return self._metadata
