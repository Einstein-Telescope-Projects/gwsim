"""Detector module for GWSim."""

from __future__ import annotations

from pathlib import Path

from .base import Detector, get_available_detectors, load_interferometer_config

# The default base path for detector configuration files
DEFAULT_DETECTOR_BASE_PATH = Path(__file__).parent / "detectors"

__all__ = ['DEFAULT_DETECTOR_BASE_PATH', 'Detector', 'get_available_detectors', 'load_interferometer_config']
