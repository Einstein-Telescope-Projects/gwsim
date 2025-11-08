"""Detector module for GWSim."""

from __future__ import annotations

from .base import Detector, get_available_detectors, load_interferometer_config

__all__ = ['Detector', 'get_available_detectors', 'load_interferometer_config']
