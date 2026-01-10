"""Compact Binary Coalescence (CBC) signal simulation module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from gwsim.mixin.cbc_population_reader import CBCPopulationReaderMixin
from gwsim.signal.base import SignalSimulator


class CBCSignalSimulator(CBCPopulationReaderMixin, SignalSimulator):  # pylint: disable=too-many-ancestors
    """CBC Signal Simulator class."""

    def __init__(  # noqa: PLR0913
        self,
        population_file: str | Path,
        population_parameter_name_mapper: dict[str, str] | None = None,
        population_cache_dir: str | Path | None = None,
        population_download_timeout: int = 300,
        waveform_model: str | Callable = "IMRPhenomXPHM",
        waveform_arguments: dict[str, Any] | None = None,
        start_time: int = 0,
        duration: float = 1024,
        sampling_frequency: float = 4096,
        max_samples: int | None = None,
        dtype: type = np.float64,
        detectors: list[str] | None = None,
        minimum_frequency: float = 5,
        **kwargs,
    ) -> None:
        """Initialize the CBC signal simulator.

        Args:
            population_file: Path to the population file.
            population_parameter_name_mapper: Dict mapping population column names to simulator parameter names.
            population_cache_dir: Directory to cache downloaded population files.
            population_download_timeout: Timeout in seconds for downloading population files. Default is 300.
            waveform_model: Name (from registry) or callable for waveform generation.
            waveform_arguments: Fixed parameters to pass to waveform model.
            start_time: Start time of the first signal segment in GPS seconds. Default is 0.
            duration: Duration of each signal segment in seconds. Default is 1024.
            sampling_frequency: Sampling frequency of the signals in Hz. Default is 4096.
            max_samples: Maximum number of samples to generate. None means infinite.
            dtype: Data type for the time series data. Default is np.float64.
            detectors: List of detector names. Default is None.
            minimum_frequency: Minimum GW frequency for waveform generation. Default is 5 Hz.
            **kwargs: Additional arguments absorbed by subclasses and mixins.
        """
        super().__init__(
            population_file=population_file,
            population_parameter_name_mapper=population_parameter_name_mapper,
            population_cache_dir=population_cache_dir,
            population_download_timeout=population_download_timeout,
            waveform_model=waveform_model,
            waveform_arguments=waveform_arguments,
            start_time=start_time,
            duration=duration,
            sampling_frequency=sampling_frequency,
            max_samples=max_samples,
            dtype=dtype,
            detectors=detectors,
            minimum_frequency=minimum_frequency,
            **kwargs,
        )
