"""Stationary Gaussian noise simulator using Bilby."""

from __future__ import annotations

from typing import Any

import numpy as np
from bilby.core.utils.random import Generator as BilbyGenerator
from bilby.core.utils.series import infft
from bilby.gw.detector.psd import PowerSpectralDensity
from numpy.random import Generator

from gwsim.noise.stationary_gaussian import StationaryGaussianNoiseSimulator


class BilbyStationaryGaussianNoiseSimulator(StationaryGaussianNoiseSimulator):
    """Stationary Gaussian noise simulator using Bilby."""

    def __init__(
        self,
        frequency_array: np.ndarray[Any, np.dtype[Any]] | None = None,
        psd_array: np.ndarray[Any, np.dtype[Any]] | None = None,
        psd_file: str | None = None,
        sampling_frequency: float = 4096,
        duration: float = 4,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """Initialize Bilby stationary Gaussian noise simulator.

        Args:
            frequency_array: Frequency array for the PSD.
            psd_array: PSD values corresponding to the frequency array.
            psd_file: Path to a file containing the PSD.
            psd: Path to PSD file or numpy array with PSD values, or label of PSD
            sampling_frequency: Sampling frequency in Hz. Default is 4096.
            duration: Duration of each segment in seconds. Default is 4.
            start_time: Start time in GPS seconds. Default is 0.
            max_samples: Maximum number of samples. None means infinite.
            seed: Random seed. If None, RNG is not initialized.
            **kwargs: Additional arguments.
        """
        self.frequency_array = frequency_array
        self.psd_array = psd_array
        self.psd_file = psd_file
        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
            **kwargs,
        )

    def _setup_psd(self) -> None:
        if self.frequency_array is not None and self.psd_array is not None:
            self.psd = PowerSpectralDensity(frequency_array=self.frequency_array, psd_array=self.psd_array)
        elif self.psd_file is not None:
            self.psd = PowerSpectralDensity.from_power_spectral_density_file(self.psd_file)
        else:
            raise ValueError("Either frequency_array and psd_array or psd_file must be provided.")

    @property
    def rng(self) -> Generator | None:
        """Get the random number generator.

        Returns:
            Random number generator instance or None if no seed was set.
        """
        return self._rng

    @rng.setter
    def rng(self, value: Generator | None) -> None:
        """Set the random number generator.

        Args:
            value: Random number generator instance.
        """
        self._rng = value
        # Override the bilby RNG
        if value is not None:
            BilbyGenerator.rng = value

    def simulate(self, *args, **kwargs) -> np.ndarray:
        """Simulate a noise segment.

        Returns:
            np.ndarray: Simulated noise segment as a numpy array.
        """
        if self.rng is None:
            raise RuntimeError("Random number generator not initialized. Set seed in constructor.")
        # Placeholder implementation; replace with actual Bilby PSD-based noise generation
        frequency_domain_strain, _frequencies = self.psd.get_noise_realisation(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        return infft(frequency_domain_strain=frequency_domain_strain, sampling_frequency=self.sampling_frequency)
