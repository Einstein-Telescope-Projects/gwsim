"""Base class for signal simulators."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from gwsim.data.time_series.time_series_list import TimeSeriesList
from gwsim.mixin.detector import DetectorMixin
from gwsim.mixin.population_reader import PopulationReaderMixin
from gwsim.mixin.time_series import TimeSeriesMixin
from gwsim.mixin.waveform import WaveformMixin
from gwsim.simulator.base import Simulator

logger = logging.getLogger("gwsim")


class SignalSimulator(PopulationReaderMixin, WaveformMixin, TimeSeriesMixin, DetectorMixin, Simulator):
    """Base class for signal simulators."""

    def __init__(  # noqa: PLR0913
        self,
        population_file: str | Path,
        population_parameter_name_mapper: dict[str, str] | None = None,
        population_sort_by: str | None = None,
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
        """Initialize the base signal simulator.

        Args:
            population_file: Path to the population file.
            population_parameter_name_mapper: Dict mapping population column names to simulator parameter names.
            population_sort_by: Column name to sort the population by.
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
        waveform_arguments = waveform_arguments or {}
        required_waveform_arguments = {
            "minimum_frequency": minimum_frequency,
            "sampling_frequency": sampling_frequency,
        }
        for key, value in required_waveform_arguments.items():
            if key not in waveform_arguments:
                logger.info("%s not specified in waveform_arguments; setting to %s", key, value)
                waveform_arguments[key] = value

        super().__init__(
            population_file=population_file,
            population_parameter_name_mapper=population_parameter_name_mapper,
            population_sort_by=population_sort_by,
            population_cache_dir=population_cache_dir,
            population_download_timeout=population_download_timeout,
            waveform_model=waveform_model,
            waveform_arguments=waveform_arguments,
            detectors=detectors,
            start_time=start_time,
            duration=duration,
            sampling_frequency=sampling_frequency,
            max_samples=max_samples,
            dtype=dtype,
            **kwargs,
        )

    def _simulate(self, *args, **kwargs) -> TimeSeriesList:
        """Simulate signals for the current segment.

        Returns:
            TimeSeriesList: List of simulated signals.
        """
        output = []

        while True:
            # Get the next injection parameters
            parameters = self.get_next_injection_parameters()

            # If the parameters are None, break the loop
            if parameters is None:
                break

            # Get the polarizations
            polarizations = self.waveform_factory.generate(
                waveform_model=self.waveform_model, parameters=parameters, **self.waveform_arguments
            )

            # Project onto detectors
            strain = self.project_polarizations(
                polarizations=polarizations,
                right_ascension=parameters["right_ascension"],
                declination=parameters["declination"],
                polarization_angle=parameters["polarization_angle"],
                **self.waveform_arguments,
            )

            # Register the parameters
            strain.metadata.update({"injection_parameters": parameters})

            output.append(strain)

            # Check whether the start time of the strain is at or after the end time of the current segment
            if strain.start_time >= self.end_time:
                break
        return TimeSeriesList(output)

    @property
    def metadata(self) -> dict:
        """Get the metadata of the simulator.

        Returns:
            Metadata dictionary.
        """
        meta = super().metadata
        return meta

    def update_state(self) -> None:
        """Update internal state after each sample generation.

        This method can be overridden by subclasses to update any internal state
        after generating a sample. The default implementation does nothing.
        """
        self.counter = cast(int, self.counter) + 1
        self.start_time += self.duration
