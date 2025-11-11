"""Detector mixin for simulators."""

from __future__ import annotations

from pathlib import Path

from gwsim.detector.base import Detector


class DetectorMixin:  # pylint: disable=too-few-public-methods
    """Mixin class to add detector information to simulators."""

    def __init__(self, detectors: list[str | Detector] | None = None, **kwargs):  # pylint: disable=unused-argument
        """Initialize the DetectorMixin.

        Args:
            detectors (list[str] | None): List of detector names. If None, use all available detectors.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.detectors = detectors

    @property
    def detectors(self) -> list[str | Detector] | None:
        """Get the list of detectors.

        Returns:
            List of detector names or Detector instances, or None if not set.
        """
        return self._detectors

    @detectors.setter
    def detectors(self, value: list[str | Path | Detector] | None) -> None:
        """Set the list of detectors.

        Args:
            value (list[str | Path | Detector] | None):
                List of detector names, config file paths, or Detector instances.
                If None, no detectors are set.
        """
        if value is None:
            self._detectors = None
        if isinstance(value, list):
            if all(isinstance(det, Detector) for det in value):
                self._detectors = value
                return
            if all(isinstance(det, str) for det in value) or all(isinstance(det, Path) for det in value):
                self._detectors = [Detector.get_detector(det) for det in value]
                return
            raise ValueError("All elements in detectors list must be of the same type: str, Path, or Detector.")
        raise ValueError("Detectors must be a list of str, Path, or Detector instances, or None.")

    @property
    def metadata(self) -> dict:
        """Get metadata including detector information.

        Returns:
            Dictionary containing the list of detectors.
        """
        metadata = {
            "detectors": [str(det) for det in self.detectors] if self.detectors is not None else [],
        }
        return metadata
