# ruff: noqa PLC0415

"""Dynamic discovery of available options from gwmock sub-packages."""

from __future__ import annotations

import logging
from importlib.resources import files

logger = logging.getLogger("gwmock")


def discover_geometries() -> list[str]:
    """Return available network geometry presets from gwmock-signal."""
    try:
        from gwmock_signal.network import Network

        return Network.list_names()
    except Exception:
        logger.warning("Could not discover geometries from gwmock-signal")
        return []


def discover_psds() -> list[str]:
    """Return available PSD preset names from gwmock-noise."""
    try:
        psd_dir = files("gwmock_noise.data.psd")
        return sorted(r.name.removesuffix(".txt") for r in psd_dir.iterdir() if r.is_file() and r.name.endswith(".txt"))
    except Exception:
        logger.warning("Could not discover PSDs from gwmock-noise")
        return []


def discover_source_types() -> list[str]:
    """Return registered source types from gwmock-signal."""
    try:
        from gwmock_signal.registry import list_registered_source_types

        return list(list_registered_source_types())
    except Exception:
        logger.warning("Could not discover source types from gwmock-signal")
        return []


def discover_waveform_models() -> list[str]:
    """Return available waveform models from the default backend in gwmock-signal."""
    try:
        from gwmock_signal.waveform.factory import WaveformFactory

        return WaveformFactory().list_models()
    except Exception:
        logger.warning("Could not discover waveform models from gwmock-signal")
        return []


def discover_glitch_models() -> list[str]:
    """Return supported glitch model kinds from gwmock-noise."""
    try:
        from gwmock_noise.glitches.models import supported_glitch_kinds

        return sorted(supported_glitch_kinds())
    except Exception:
        logger.warning("Could not discover glitch models from gwmock-noise")
        return []


def discover_population_presets() -> list[str]:
    """Return available population presets from gwmock-pop."""
    try:
        from gwmock_pop.configs import list_presets

        return list_presets()
    except Exception:
        logger.warning("Could not discover population presets from gwmock-pop")
        return []
