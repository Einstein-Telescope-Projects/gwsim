"""Stationary Gaussian noise simulator using Bilby."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pycbc.psd
from pycbc.noise import noise_from_psd
from pycbc.types.frequencyseries import FrequencySeries

from gwsim.noise.stationary_gaussian import StationaryGaussianNoiseSimulator

logger = logging.getLogger("gwsim")


class PyCBCStationaryGaussianNoiseSimulator(
    StationaryGaussianNoiseSimulator
):  # pylint: disable=too-many-ancestors, duplicate-code
    """Stationary Gaussian noise simulator using Bilby."""

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        frequency_array: np.ndarray[Any, np.dtype[Any]] | None = None,
        psd_array: np.ndarray[Any, np.dtype[Any]] | None = None,
        psd_file: str | None = None,
        label: str | None = None,
        low_frequency_cutoff: float = 5.0,
        sampling_frequency: float = 4096,
        duration: float = 4,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
        detectors: list[str] | None = None,
        **kwargs,
    ):
        """Initialize Bilby stationary Gaussian noise simulator.

        Args:
            frequency_array (np.ndarray): Frequency array for the PSD.
            psd_array (np.ndarray): PSD values corresponding to the frequency array.
            psd_file (str): Path to a file containing the PSD.
            label (str): Label for a predefined PyCBC PSD.
            low_frequency_cutoff (float): Low frequency cutoff for the PSD. Default is 5.0 Hz.
            sampling_frequency (float): Sampling frequency in Hz. Default is 4096.
            duration (float): Duration of each segment in seconds. Default is 4.
            start_time (float): Start time in GPS seconds. Default is 0.
            max_samples (int | None): Maximum number of samples. None means infinite.
            seed (int | None): Random seed. If None, RNG is not initialized.
            detectors (list[str] | None): List of detector names. Default is None.
            **kwargs: Additional arguments.
        """
        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
            detectors=detectors,
            **kwargs,
        )
        self.frequency_array = frequency_array
        self.psd_array = psd_array
        self.psd_file = psd_file
        self.label = label
        self.low_frequency_cutoff = low_frequency_cutoff
        self._setup_psd()

    def _setup_psd(self) -> None:
        if self.frequency_array is not None and self.psd_array is not None:
            logger.info("Setting PSD values below low_frequency_cutoff = %s to zero.", self.low_frequency_cutoff)
            self.psd_array[self.frequency_array < self.low_frequency_cutoff] = 0.0
            self.psd = FrequencySeries(
                initial_array=self.psd_array, delta_f=self.frequency_array[1] - self.frequency_array[0]
            )
        elif self.psd_file is not None:
            data = np.loadtxt(self.psd_file)
            frequency_array = data[:, 0]
            psd_values = data[:, 1]
            logger.info("Setting PSD values below low_frequency_cutoff = %s to zero.", self.low_frequency_cutoff)
            psd_values[frequency_array < self.low_frequency_cutoff] = 0.0
            self.psd = FrequencySeries(initial_array=psd_values, delta_f=data[1, 0] - data[0, 0])
        elif self.label is not None:
            self.psd = pycbc.psd.from_string(
                psd_name=self.label,
                length=int(self.duration * self.sampling_frequency // 2 + 1),
                delta_f=1.0 / self.duration,
                low_freq_cutoff=self.low_frequency_cutoff,
            )
        else:
            raise ValueError("Either frequency_array and psd_array or psd_file must be provided.")

    def simulate(self, *args, **kwargs) -> np.ndarray:
        """Simulate a noise segment.

        Returns:
            np.ndarray: Simulated noise segment as a numpy array.
        """
        if self.rng is None:
            raise RuntimeError("Random number generator not initialized. Set seed in constructor.")
        return noise_from_psd(
            length=int(self.duration * self.sampling_frequency),
            delta_t=1.0 / self.sampling_frequency,
            psd=self.psd,
            seed=int(self.rng.integers(0, 2**31 - 1)),
        ).numpy()
