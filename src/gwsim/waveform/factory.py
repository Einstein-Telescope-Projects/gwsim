"""Waveform Factory for generating gravitational waveforms using various models."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

from gwpy.timeseries import TimeSeries
from pycbc.waveform import td_approximants

from gwsim.waveform.pycbc_wrapper import pycbc_waveform_wrapper

logger = logging.getLogger("gwsim")


class WaveformFactory:
    """Factory class for generating gravitational waveforms using various models."""

    def __init__(self):
        """Initialize the WaveformFactory with built-in models."""
        self._models: dict[str, Callable] = dict.fromkeys(td_approximants(), pycbc_waveform_wrapper)

    def register_model(self, name: str, factory_func: Callable | str) -> None:
        """Register a new waveform model.

        Args:
            name: Name of the waveform model.
            factory_func: A callable that generates the waveform or a string path to import it.
        """
        if isinstance(factory_func, str):
            # Import from path
            module_path, func_name = factory_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            factory_func: Callable = getattr(module, func_name)

        self._models[name] = factory_func
        logger.info("Registered waveform model: %s", name)

    def get_model(self, name: str) -> Callable:
        """Get a waveform model by name.

        Args:
            name: Name of the waveform model.

        Returns:
            Callable: The waveform generation function.
        """
        if name in self._models:
            return self._models[name]
        raise ValueError(f"Waveform model '{name}' not found. " f"Available: {list(self._models.keys())}.")

    def list_models(self) -> list[str]:
        """List all registered waveform models.

        Returns:
            list[str]: List of waveform model names.
        """
        return list(self._models.keys())

    def generate(
        self,
        waveform_model: str,
        parameters: dict[str, Any],
        **extra_params,
    ) -> dict[str, TimeSeries]:
        """Generate a waveform using the specified model and parameters.

        Args:
            waveform_model: Name of the waveform model to use.
            parameters: Parameters for the waveform generation.
            extra_params: Additional parameters to pass to the waveform function.

        Returns:
            dict[str, TimeSeries]: Generated waveform data.
        """
        waveform_func = self.get_model(waveform_model)

        # Merge parameters
        all_params = {"waveform_model": waveform_model, **parameters, **extra_params}

        return waveform_func(**all_params)
