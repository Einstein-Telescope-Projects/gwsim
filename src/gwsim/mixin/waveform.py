"""Mixin for waveform generation in signal simulators."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from gwsim.waveform.factory import WaveformFactory

logger = logging.getLogger("gwsim")


class WaveformMixin:  # pylint: disable=too-few-public-methods
    """Mixin class for waveform generation in signal simulators."""

    def __init__(
        self,
        waveform_model: str | Callable = "IMRPhenomXPHM",
        waveform_arguments: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the WaveformMixin.

        Args:
            waveform_model: Name (from registry) or callable for waveform generation.
            waveform_arguments: Fixed parameters to pass to waveform model.
            parameter_mapping: Dict mapping population column names to waveform parameter names.
            minimum_frequency: Minimum GW frequency for waveform generation.
            **kwargs: Additional arguments for other mixins.
        """
        super().__init__(**kwargs)
        self.waveform_factory = WaveformFactory()
        if waveform_model not in self.waveform_factory.list_models():
            # Register the model if not already registered
            self.waveform_factory.register_model(name=str(waveform_model), factory_func=waveform_model)
        self.waveform_model = str(waveform_model)
        self.waveform_arguments = waveform_arguments or {}

    @property
    def metadata(self) -> dict:
        """Include waveform metadata."""
        metadata = {
            "waveform_model": self.waveform_model,
            "waveform_arguments": self.waveform_arguments,
        }
        return metadata
