"""Base class for glitch simulators."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

from gwsim.mixin.detector import DetectorMixin
from gwsim.mixin.population_reader import PopulationReaderMixin
from gwsim.mixin.randomness import RandomnessMixin
from gwsim.mixin.time_series import TimeSeriesMixin
from gwsim.simulator.base import Simulator


class GlitchSimulator(PopulationReaderMixin, TimeSeriesMixin, RandomnessMixin, DetectorMixin, Simulator):
    """Base class for glitch simulators."""

    def __init__(
        self,
        population_file: str | Path,
        population_file_type: str,
        start_time: int = 0,
        duration: float = 1024,
        sampling_frequency: float = 4096,
        max_samples: int | None = None,
        dtype: type = np.float64,
        seed: int | None = None,
        detectors: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the base glitch simulator.

        Args:
            population_file: Path to the population file.
            population_file_type: Type of the population file.
            start_time: Start time of the first glitch segment in GPS seconds. Default is 0.
            duration: Duration of each glitch segment in seconds. Default is 1024.
            sampling_frequency: Sampling frequency of the glitches in Hz. Default is 4096.
            max_samples: Maximum number of samples to generate. None means infinite.
            dtype: Data type for the time series data. Default is np.float64.
            seed: Seed for the random number generator. If None, the RNG is not initialized.
            detectors: List of detector names. Default is None.
            **kwargs: Additional arguments absorbed by subclasses and mixins.
        """
        super().__init__(
            population_file=population_file,
            population_file_type=population_file_type,
            start_time=start_time,
            duration=duration,
            sampling_frequency=sampling_frequency,
            max_samples=max_samples,
            dtype=dtype,
            seed=seed,
            detectors=detectors,
            **kwargs,
        )

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
