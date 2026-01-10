"""Glitch simulator based on `gengli`."""

# pylint: disable=duplicate-code

from __future__ import annotations

import logging
from pathlib import Path

import gengli  # pylint: disable=E0401
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey

from gwsim.data.time_series.time_series import TimeSeries
from gwsim.data.time_series.time_series_list import TimeSeriesList
from gwsim.glitch.base import GlitchSimulator

logger = logging.getLogger("gwsim")

# The default base path for PSD files
DEFAULT_PSD_PATH = Path(__file__).parent.parent / "detector/noise_curves"


class GengliGlitchSimulator(GlitchSimulator):
    """Glitch simulator based on `gengli`.

    This class simulates gravitational wave detector glitches using the gengli package for glitch generation.
    Glitches are generated as white time series and colored using the provided PSD

    Only 'Blip' glitches generation is supported for now!

    The population file provides two injection parameters: SNR, and GPS time.
    """

    def __init__(  # noqa: PLR0913
        self,
        psd_file: str | Path,
        population_file: str | Path,
        population_file_type: str | None = None,
        start_time: int = 0,
        duration: float = 1024,
        sampling_frequency: float = 4096,
        max_samples: int | None = None,
        dtype: type = np.float64,
        seed: int | None = None,
        detectors: str | None = None,
        low_frequency_cutoff: float = 5.0,
        high_frequency_cutoff: float | None = None,
        gengli_detector: str = "L1",
        **kwargs,
    ):
        """Initialize the gengli glitch simulator.

        Args:
            psd_file: Path to file containing Power Spectral Density array with shape (N, 2),
                where the first column is frequency (Hz) and the second is PSD values.
            population_file: Path to the population file.
            population_file_type: Type of the population file.
            start_time: Start time of the first glitch segment in GPS seconds. Default is 0.
            duration: Duration of each glitch segment in seconds. Default is 1024.
            sampling_frequency: Sampling frequency of the glitches in Hz. Default is 4096.
            max_samples: Maximum number of samples to generate. None means infinite.
            dtype: Data type for the time series data. Default is np.float64.
            seed: Seed for the random number generator. If None, the RNG is not initialized.
            detectors: List of detector names. Default is None.
            low_frequency_cutoff: Lower frequency cutoff in Hz. Default is 2.0.
            high_frequency_cutoff: Upper frequency cutoff in Hz. Default is Nyquist frequency.
            gengli_detector: Detector name for gengli glitch generator. Default is "L1".
            **kwargs: Additional arguments absorbed by subclasses and mixins.
        """
        if len(detectors) > 1:
            raise ValueError(f"Multiple detectors were provided ({detectors}). Only one expected.")

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

        self.psd = psd_file
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = (
            high_frequency_cutoff
            if (high_frequency_cutoff is not None and high_frequency_cutoff <= sampling_frequency / 2)
            else sampling_frequency // 2
        )
        self.gengli_detector = gengli_detector

        # Load psd file
        self._psd_array = self._load_spectral_data(self.psd)

    def _load_spectral_data(self, file_path: str | Path) -> np.ndarray:  # pylint: disable=duplicate-code
        """Load spectral data from file.

        Args:
            file_path: Path to file containing spectral data.

        Returns:
            Loaded array.

        Raises:
            ValueError: If file format is not supported.
            TypeError: If file_path is not a string or Path.
        """
        if not isinstance(file_path, (str, Path)):
            raise TypeError("file_path must be a string or Path.")

        path = Path(file_path)
        if not path.exists():
            psd_dir = DEFAULT_PSD_PATH
            path = next(iter(psd_dir.rglob(path.name)))

        if path.suffix == ".npy":
            return np.load(path)
        if path.suffix == ".txt":
            return np.loadtxt(path)
        if path.suffix == ".csv":
            return np.loadtxt(path, delimiter=",")
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .npy, .txt, or .csv.")

    def _initialize_psd(self, psd_data: np.ndarray, duration: float) -> np.ndarray:  # pylint: disable=duplicate-code
        """Initialize PSD interpolation for the frequency range.

        Args:
            psd_data: The PSD data array with shape (N, 2), where columns are frequency and PSD values.
            duration: Duration of the time series in seconds.

        Returns:
            np.ndarray: Interpolated and windowed PSD values for the relevant frequencies.
        """
        if psd_data.shape[1] != 2:  # noqa: PLR2004
            raise ValueError("PSD file must have shape (N, 2).")

        # Frequency properties
        df = 1.0 / duration
        n_samples = int(duration * self.sampling_frequency.value)
        k_min = np.ceil(self.low_frequency_cutoff / df).astype(int)
        k_max = np.floor(self.high_frequency_cutoff / df).astype(int) + 1
        frequency = np.arange(0.0, n_samples / 2.0 + 1) * df

        # Interpolate the PSD to the relevant frequencies
        freqs = frequency[k_min:k_max]
        psd_interp = interp1d(psd_data[:, 0], psd_data[:, 1], bounds_error=False, fill_value="extrapolate")(freqs)

        # Add a roll-off at the edges using a Tukey window
        window = tukey(len(freqs), alpha=1e-3)

        return psd_interp * window

    def _color_glitch(self, glitch: np.ndarray, psd: np.ndarray, duration: float) -> np.ndarray:
        """Color the glitch with the PSD.

        Args:
            glitch: White glitch time series data as a numpy array.
            psd: PSD values for the frequency bins.
            duration: Duration of the glitch in seconds.

        Returns:
            np.ndarray: Colored glitch array.
        """
        # Define frequency properties
        df = 1.0 / duration
        n_samples = int(duration * self.sampling_frequency.value)
        k_min = np.ceil(self.low_frequency_cutoff / df).astype(int)
        k_max = np.floor(self.high_frequency_cutoff / df).astype(int) + 1

        white_glitch_fd = np.fft.rfft(glitch) / self.sampling_frequency.value

        # Color glitch in frequency domain
        colored_glitch_fd = np.zeros_like(white_glitch_fd, dtype=np.complex128)
        colored_glitch_fd[k_min:k_max] = white_glitch_fd[k_min:k_max] * np.sqrt(psd)

        colored_glitch = np.fft.irfft(colored_glitch_fd, n=n_samples) * self.sampling_frequency.value

        return colored_glitch

    def _simulate(self, *args, **kwargs) -> TimeSeriesList:
        """Simulate glitches for the current segment.

        Returns:
            TimeSeriesList: List of simulated glitches.
        """
        if self.rng is None:
            raise RuntimeError("Random number generator not initialized. Set seed in constructor.")

        # Initialize gengli glitch generator
        gengli_generator = gengli.glitch_generator(self.gengli_detector)

        output = []

        while True:
            # Get the next injection parameters
            parameters = self.get_next_injection_parameters()

            # If the parameters are None, break the loop
            if parameters is None:
                break

            # Get the raw glitch
            glitch_instance = gengli_generator.get_glitch(
                seed=self.rng.integers(0, 2**32 - 1),
                snr=parameters["snr"],
                srate=self.sampling_frequency.value,
                glitch_type="Blip",
            )

            # Define glitch time properties
            glitch_duration = len(glitch_instance) / self.sampling_frequency.value
            glitch_start_time = parameters["gps_time"] - glitch_duration / 2  # Center the glitch at gps_time

            # Interpolate the PSD
            psd_instance = self._initialize_psd(self._psd_array, glitch_duration)

            # Create container for SINGLE detector
            glitch_data = np.zeros((1, int(glitch_duration * self.sampling_frequency.value)))

            # Color the glitch
            glitch_data[0, :] = self._color_glitch(glitch_instance, psd_instance, glitch_duration)

            strain = TimeSeries(
                data=glitch_data, start_time=glitch_start_time, sampling_frequency=self.sampling_frequency.value
            )

            # Register the parameters
            strain.metadata.update({"injection_parameters": parameters})

            output.append(strain)

            # Check whether the start time of the strain is at or after the end time of the current segment
            if strain.start_time >= self.end_time:
                break
        return TimeSeriesList(output)
