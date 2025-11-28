"""Colored noise simulator implementation."""

from __future__ import annotations

import numpy as np
import pycbc.psd as psd_module

from gwsim.noise.base import NoiseSimulator
from gwsim.simulator.mixin.gwf import GWFOutputMixin


class ColoredNoiseSimulator(GWFOutputMixin, NoiseSimulator):
    """Colored noise simulator using scipy signal processing.

    Generates noise with specified power spectral density (PSD) characteristics.
    """

    def __init__(
        self,
        psd_type: str = "aLIGO",
        low_frequency_cutoff: float = 20.0,
        sampling_frequency: float = 4096,
        duration: float = 4,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """Initialize colored noise simulator.

        Args:
            psd_type: Type of PSD to use ('aLIGO', '1/f', 'custom'). Default is 'aLIGO'.
            low_frequency_cutoff: Low frequency cutoff in Hz. Default is 20.0.
            sampling_frequency: Sampling frequency in Hz. Default is 4096.
            duration: Duration of each segment in seconds. Default is 4.
            start_time: Start time in GPS seconds. Default is 0.
            max_samples: Maximum number of samples. None means infinite.
            seed: Random seed. If None, RNG is not initialized.
            **kwargs: Additional arguments.
        """
        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
            **kwargs,
        )
        self.psd_type = psd_type
        self.low_frequency_cutoff = low_frequency_cutoff
        self._generate_filter()

    def _generate_filter(self) -> None:
        """Generate the filter for coloring the noise."""
        # Generate frequency array
        n_samples = int(self.duration * self.sampling_frequency)
        freqs = np.fft.fftfreq(n_samples, 1 / self.sampling_frequency)
        freqs = freqs[: n_samples // 2 + 1]  # Only positive frequencies

        # Generate PSD based on type
        if self.psd_type == "aLIGO":
            # Simplified aLIGO-like PSD (1/f^7 at low freq, flat at high freq)
            psd = np.ones_like(freqs)
            low_mask = freqs < 60
            psd[low_mask] = (60 / np.maximum(freqs[low_mask], self.low_frequency_cutoff)) ** 7
        elif self.psd_type == "1/f":
            # 1/f noise (pink noise)
            psd = 1 / np.maximum(freqs, self.low_frequency_cutoff)
        else:
            # Default to flat (white noise)
            psd = np.ones_like(freqs)

        # Store the square root of PSD for filtering
        self._filter_response = np.sqrt(psd)

    def next(self) -> np.ndarray:
        """Generate next colored noise segment.

        Returns:
            np.ndarray: Colored noise time series.
        """
        if self.rng is None:
            raise RuntimeError("Random number generator not initialized. Set seed in constructor.")

        n_samples = int(self.duration * self.sampling_frequency)

        # Generate white noise
        white_noise = self.rng.normal(0, 1, n_samples)

        # Transform to frequency domain
        noise_fft = np.fft.rfft(white_noise)

        # Apply coloring filter
        colored_fft = noise_fft * self._filter_response

        # Transform back to time domain
        colored_noise = np.fft.irfft(colored_fft, n=n_samples)

        return colored_noise


class StationaryGaussianNoiseSimulator(GWFOutputMixin, NoiseSimulator):
    """Stationary Gaussian noise simulator.

    Generates noise from a specified power spectral density using scipy.
    More sophisticated than ColoredNoiseSimulator for realistic detector noise.
    """

    def __init__(
        self,
        psd_data: str | np.ndarray | None = None,
        scale: float = 1e-23,
        sampling_frequency: float = 4096,
        duration: float = 4,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """Initialize stationary Gaussian noise simulator.

        Args:
            psd_data: Path to PSD file or numpy array with PSD values. If None, uses simple model.
            scale: Overall amplitude scale factor. Default is 1e-23.
            sampling_frequency: Sampling frequency in Hz. Default is 4096.
            duration: Duration of each segment in seconds. Default is 4.
            start_time: Start time in GPS seconds. Default is 0.
            max_samples: Maximum number of samples. None means infinite.
            seed: Random seed. If None, RNG is not initialized.
            **kwargs: Additional arguments.
        """
        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
            **kwargs,
        )
        self.scale = scale
        self._setup_psd(psd_data)

    def _setup_psd(self, psd_data: str | np.ndarray | None) -> None:
        """Setup the power spectral density."""
        n_samples = int(self.duration * self.sampling_frequency)
        freqs = np.fft.fftfreq(n_samples, 1 / self.sampling_frequency)
        freqs = freqs[: n_samples // 2 + 1]  # Only positive frequencies

        if isinstance(psd_data, str):
            # Load PSD from file (assuming two-column format: frequency, PSD)
            try:
                data = np.loadtxt(psd_data)
                psd_freqs, psd_values = data[:, 0], data[:, 1]
                # Interpolate to our frequency grid
                self._psd = np.interp(freqs, psd_freqs, psd_values)
            except (FileNotFoundError, IndexError, ValueError) as e:
                raise ValueError(f"Could not load PSD from file {psd_data}: {e}") from e
        elif isinstance(psd_data, np.ndarray):
            # Use provided PSD array
            if len(psd_data) != len(freqs):
                raise ValueError(f"PSD array length {len(psd_data)} doesn't match frequency grid {len(freqs)}")
            self._psd = psd_data.copy()
        else:
            # Generate simple analytical PSD
            self._psd = self._analytical_psd(freqs)

        # Apply scale factor
        self._psd *= self.scale**2

    def _analytical_psd(self, freqs: np.ndarray) -> np.ndarray:
        """Generate simple analytical PSD model.

        Args:
            freqs: Frequency array.

        Returns:
            np.ndarray: PSD values.
        """
        # Simple model: 1/f^7 below 60 Hz, flat above
        psd = np.ones_like(freqs)
        low_mask = freqs < 60
        psd[low_mask] = (60 / np.maximum(freqs[low_mask], 20)) ** 7
        return psd

    def next(self) -> np.ndarray:
        """Generate next noise segment from PSD.

        Returns:
            np.ndarray: Noise time series with specified PSD.
        """
        if self.rng is None:
            raise RuntimeError("Random number generator not initialized. Set seed in constructor.")

        n_samples = int(self.duration * self.sampling_frequency)

        # Generate complex white noise in frequency domain
        real_part = self.rng.normal(0, 1, len(self._psd))
        imag_part = self.rng.normal(0, 1, len(self._psd))
        white_fft = real_part + 1j * imag_part

        # Apply PSD coloring (multiply by sqrt of PSD)
        colored_fft = white_fft * np.sqrt(self._psd / 2)  # Factor of 2 for two-sided to one-sided

        # Ensure proper symmetry for real output
        colored_fft[0] = colored_fft[0].real  # DC component must be real
        if len(colored_fft) > 1 and n_samples % 2 == 0:
            colored_fft[-1] = colored_fft[-1].real  # Nyquist component must be real

        # Transform to time domain
        noise = np.fft.irfft(colored_fft, n=n_samples)

        return noise


# PyCBC-based colored noise simulator
try:
    from pycbc.noise.gaussian import noise_from_psd
    from pycbc.psd import aLIGOZeroDetHighPower

    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False


class PyCBCColoredNoiseSimulator(GWFOutputMixin, NoiseSimulator):
    """Colored noise simulator using PyCBC.

    Generates colored noise using PyCBC's noise generation functions.
    Provides access to PyCBC's built-in PSD models and noise generation.
    """

    def __init__(
        self,
        psd_name: str = "aLIGOZeroDetHighPower",
        low_frequency_cutoff: float = 20.0,
        sampling_frequency: float = 4096,
        duration: float = 4,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """Initialize PyCBC colored noise simulator.

        Args:
            psd_name: Name of PyCBC PSD to use. Default is 'aLIGOZeroDetHighPower'.
            low_frequency_cutoff: Low frequency cutoff in Hz. Default is 20.0.
            sampling_frequency: Sampling frequency in Hz. Default is 4096.
            duration: Duration of each segment in seconds. Default is 4.
            start_time: Start time in GPS seconds. Default is 0.
            max_samples: Maximum number of samples. None means infinite.
            seed: Random seed. If None, RNG is not initialized.
            **kwargs: Additional arguments.
        """
        if not PYCBC_AVAILABLE:
            raise ImportError("PyCBC is required for PyCBCColoredNoiseSimulator")

        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
            **kwargs,
        )
        self.psd_name = psd_name
        self.low_frequency_cutoff = low_frequency_cutoff
        self._setup_psd()

    def _setup_psd(self) -> None:
        """Setup the PyCBC PSD."""
        # Get the PSD function from PyCBC
        if self.psd_name == "aLIGOZeroDetHighPower":
            psd_func = aLIGOZeroDetHighPower
        else:
            if hasattr(psd_module, self.psd_name):
                psd_func = getattr(psd_module, self.psd_name)
            else:
                raise ValueError(f"Unknown PyCBC PSD: {self.psd_name}")

        # Create frequency series
        n_samples = int(self.duration * self.sampling_frequency)
        df = self.sampling_frequency / n_samples

        # Generate PSD
        self._psd = psd_func(
            length=int(self.sampling_frequency / df / 2) + 1, delta_f=df, low_freq_cutoff=self.low_frequency_cutoff
        )

    def next(self) -> np.ndarray:
        """Generate next colored noise segment using PyCBC.

        Returns:
            np.ndarray: Colored noise time series.
        """
        # Generate noise using PyCBC
        noise_ts = noise_from_psd(
            length=int(self.duration * self.sampling_frequency),
            delta_t=1.0 / self.sampling_frequency,
            psd=self._psd,
            seed=self.seed if self.seed is not None else None,
        )

        # Convert to numpy array
        return np.array(noise_ts.data)


# Bilby-based colored noise simulator
try:
    from bilby.core.utils import create_frequency_series
    from bilby.gw.detector import PowerSpectralDensity

    BILBY_AVAILABLE = True
except ImportError:
    BILBY_AVAILABLE = False


class BilbyColoredNoiseSimulator(GWFOutputMixin, NoiseSimulator):
    """Colored noise simulator using Bilby.

    Generates colored noise using Bilby's detector and PSD utilities.
    Provides access to Bilby's PSD models and noise generation capabilities.
    """

    def __init__(
        self,
        psd_name: str = "aLIGO",
        low_frequency_cutoff: float = 20.0,
        sampling_frequency: float = 4096,
        duration: float = 4,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """Initialize Bilby colored noise simulator.

        Args:
            psd_name: Name of Bilby PSD to use. Default is 'aLIGO'.
            low_frequency_cutoff: Low frequency cutoff in Hz. Default is 20.0.
            sampling_frequency: Sampling frequency in Hz. Default is 4096.
            duration: Duration of each segment in seconds. Default is 4.
            start_time: Start time in GPS seconds. Default is 0.
            max_samples: Maximum number of samples. None means infinite.
            seed: Random seed. If None, RNG is not initialized.
            **kwargs: Additional arguments.
        """
        if not BILBY_AVAILABLE:
            raise ImportError("Bilby is required for BilbyColoredNoiseSimulator")

        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
            **kwargs,
        )
        self.psd_name = psd_name
        self.low_frequency_cutoff = low_frequency_cutoff
        self._setup_psd()

    def _setup_psd(self) -> None:
        """Setup the Bilby PSD."""
        # Create frequency series
        frequency_array = create_frequency_series(sampling_frequency=self.sampling_frequency, duration=self.duration)

        # Create PSD object
        self._psd = PowerSpectralDensity(psd_name=self.psd_name, frequency_array=frequency_array)

    def next(self) -> np.ndarray:
        """Generate next colored noise segment using Bilby.

        Returns:
            np.ndarray: Colored noise time series.
        """
        if self.rng is None:
            raise RuntimeError("Random number generator not initialized. Set seed in constructor.")

        # Generate white noise in time domain
        n_samples = int(self.duration * self.sampling_frequency)
        white_noise = self.rng.normal(0, 1, n_samples)

        # Transform to frequency domain
        noise_fft = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(n_samples, 1 / self.sampling_frequency)

        # Get PSD values at these frequencies
        psd_values = self._psd.power_spectral_density_interpolated(freqs)

        # Apply coloring
        colored_fft = noise_fft * np.sqrt(psd_values)

        # Transform back to time domain
        colored_noise = np.fft.irfft(colored_fft, n=n_samples)

        return colored_noise
