"""Colored noise simulator for gravitational wave detectors."""

# pylint: disable=duplicate-code

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey

from gwsim.data.time_series.time_series import TimeSeries
from gwsim.data.time_series.time_series_list import TimeSeriesList
from gwsim.noise.base import NoiseSimulator
from gwsim.simulator.state import StateAttribute

logger = logging.getLogger("gwsim")

# The default base path for PSD files
DEFAULT_PSD_PATH = Path(__file__).parent.parent / "detector/noise_curves"


class ColoredNoiseSimulator(NoiseSimulator):  # pylint: disable=too-many-instance-attributes
    """Colored noise simulator for gravitational wave detectors.

    This class generates noise time series with a specified power spectral density (PSD).
    It uses an overlap-add method with windowing to produce smooth, continuous time series
    across segment boundaries.

    The simulator maintains state between batches to ensure continuity of the noise
    time series across multiple calls to simulate().
    """

    # State attribute to track the previous strain buffer for continuity
    previous_strain = StateAttribute(default=None)

    def __init__(
        self,
        psd_file: str | Path,
        detectors: list[str],
        sampling_frequency: float = 4096,
        duration: float = 4,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        **kwargs,
    ):
        """Initialize the colored noise simulator.

        Args:
            psd_file: Path to file containing Power Spectral Density array with shape (N, 2),
                where the first column is frequency (Hz) and the second is PSD values.
            detectors: List of detector names (e.g., ['H1', 'L1']).
            sampling_frequency: Sampling frequency in Hz. Default is 4096.
            duration: Duration of each noise segment in seconds. Default is 4.
            start_time: GPS start time for the time series. Default is 0.
            max_samples: Maximum number of samples to generate. None means infinite.
            seed: Seed for random number generation. If None, RNG is not initialized.
            low_frequency_cutoff: Lower frequency cutoff in Hz. Default is 2.0.
            high_frequency_cutoff: Upper frequency cutoff in Hz. Default is Nyquist frequency.
            **kwargs: Additional arguments passed to parent classes.

        Raises:
            ValueError: If detectors list is empty.
            ValueError: If duration is too short for proper noise generation.
        """
        if not detectors or len(detectors) == 0:
            raise ValueError("detectors must contain at least one detector.")

        super().__init__(  # pylint: disable=duplicate-code
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
            detectors=detectors,
            **kwargs,
        )

        self.psd_file = psd_file
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = (
            high_frequency_cutoff
            if (high_frequency_cutoff is not None and high_frequency_cutoff <= sampling_frequency / 2)
            else sampling_frequency // 2
        )

        # Initialize noise generation properties
        self._n_det = len(detectors)
        self._initialize_window_properties()
        self._initialize_frequency_properties()
        self._initialize_psd()

        # Initialize the previous strain buffer (will be populated on first simulate call)
        self.previous_strain = np.zeros((self._n_det, self._n_chunk))
        self._temp_strain_buffer: np.ndarray | None = None

    def _initialize_window_properties(self) -> None:
        """Initialize window properties for connecting noise realizations.

        Raises:
            ValueError: If the duration is too short for proper noise generation.
        """
        self._t_window = 2048
        self._f_window = 1.0 / self._t_window
        self._t_overlap = self._t_window / 2.0
        self._n_overlap = int(self._t_overlap * self.sampling_frequency.value)

        # Create overlap windows for smooth transitions
        t_overlap_array = np.linspace(0, self._t_overlap, self._n_overlap)
        self._w0 = 0.5 + np.cos(2 * np.pi * self._f_window * t_overlap_array) / 2
        self._w1 = 0.5 + np.sin(2 * np.pi * self._f_window * t_overlap_array - np.pi / 2) / 2

        # Safety check to ensure proper noise generation
        if self.duration.value < self._t_window / 2:
            raise ValueError(
                f"Duration ({self.duration.value:.1f} seconds) must be at least "
                f"{self._t_window / 2:.1f} seconds to ensure noise continuity."
            )

    def _initialize_frequency_properties(self) -> None:
        """Initialize frequency and time properties for noise generation."""
        self._t_chunk = self._t_window
        self._df_chunk = 1.0 / self._t_chunk
        self._n_chunk = int(self._t_chunk * self.sampling_frequency.value)
        self._k_min_chunk = int(self.low_frequency_cutoff / self._df_chunk)
        self._k_max_chunk = int(self.high_frequency_cutoff / self._df_chunk) + 1
        self._frequency_chunk = np.arange(0.0, self._n_chunk / 2.0 + 1) * self._df_chunk
        self._n_freq_chunk = len(self._frequency_chunk[self._k_min_chunk : self._k_max_chunk])
        self._dt = 1.0 / self.sampling_frequency.value

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

    def _initialize_psd(self) -> None:  # pylint: disable=duplicate-code
        """Initialize PSD interpolation for the frequency range.

        Raises:
            ValueError: If PSD array doesn't have shape (N, 2).
        """
        psd_data = self._load_spectral_data(self.psd_file)

        expected_shape = (None, 2)
        if psd_data.shape[1] != expected_shape[1]:
            raise ValueError("PSD file must have shape (N, 2).")

        # Interpolate the PSD to the relevant frequencies
        freqs = self._frequency_chunk[self._k_min_chunk : self._k_max_chunk]
        psd_interp = interp1d(psd_data[:, 0], psd_data[:, 1], bounds_error=False, fill_value="extrapolate")(freqs)

        # Add a roll-off at the edges using a Tukey window
        window = tukey(self._n_freq_chunk, alpha=1e-3)
        self._psd = psd_interp * window

    def _generate_single_realization(self) -> np.ndarray:
        """Generate a single noise realization in the time domain.

        Returns:
            Time series array with shape (n_detectors, n_samples).
        """
        if self.rng is None:
            raise RuntimeError("Random number generator not initialized. Set seed in constructor.")

        freq_series = np.zeros((self._n_det, self._frequency_chunk.size), dtype=np.complex128)

        # Generate white noise and color it with the PSD
        white_strain = (
            self.rng.standard_normal((self._n_det, self._n_freq_chunk))
            + 1j * self.rng.standard_normal((self._n_det, self._n_freq_chunk))
        ) / np.sqrt(2)
        colored_strain = white_strain * np.sqrt(self._psd * 0.5 / self._df_chunk)
        freq_series[:, self._k_min_chunk : self._k_max_chunk] += colored_strain

        # Transform to time domain
        time_series = np.fft.irfft(freq_series, n=self._n_chunk, axis=1) * self._df_chunk * self._n_chunk

        return time_series

    def _simulate(self, *args, **kwargs) -> TimeSeriesList:
        """Simulate colored noise for all detectors.

        Returns:
            TimeSeriesList containing a single TimeSeries with shape (n_detectors, n_samples).
        """
        n_frame = int(self.duration.value * self.sampling_frequency.value)

        # Load previous strain, or generate new if all zeros
        if self.previous_strain.shape[-1] < self._n_overlap:
            raise ValueError(
                f"previous_strain has only {self.previous_strain.shape[-1]} samples per detector, "
                f"but expected at least {self._n_overlap}."
            )

        strain_buffer = self.previous_strain[:, -self._n_chunk :]
        if np.all(strain_buffer == 0):
            strain_buffer = self._generate_single_realization()

        # Apply the final part of the window
        strain_buffer[:, -self._n_overlap :] *= self._w0

        # Extend the strain buffer until it has more valid data than a single frame
        while strain_buffer.shape[-1] - self._n_chunk - self._n_overlap < n_frame:
            new_strain = self._generate_single_realization()
            new_strain[:, : self._n_overlap] *= self._w1
            new_strain[:, -self._n_overlap :] *= self._w0
            strain_buffer[:, -self._n_overlap :] += new_strain[:, : self._n_overlap]
            strain_buffer[:, -self._n_overlap :] *= 1 / np.sqrt(self._w0**2 + self._w1**2)
            strain_buffer = np.concatenate((strain_buffer, new_strain[:, self._n_overlap :]), axis=1)

        # Extract the frame data
        output_strain = strain_buffer[:, self._n_chunk : (self._n_chunk + n_frame)]

        # Store the output strain temporarily for state update
        self._temp_strain_buffer = output_strain

        return TimeSeriesList(
            [TimeSeries(data=output_strain, start_time=self.start_time, sampling_frequency=self.sampling_frequency)]
        )

    def update_state(self) -> None:
        """Update internal state after each sample generation.

        Updates the previous_strain buffer to ensure continuity across batches.
        """
        # Call parent's update_state first (increments counter, advances start_time, saves rng_state)
        super().update_state()

        # Update the previous strain buffer for continuity
        if self._temp_strain_buffer is not None:
            self.previous_strain = self._temp_strain_buffer
            self._temp_strain_buffer = None

    @property
    def metadata(self) -> dict:
        """Get metadata including colored noise configuration.

        Returns:
            Dictionary containing metadata.
        """
        meta = super().metadata
        meta["colored_noise"] = {
            "arguments": {
                "psd_file": str(self.psd_file),
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self.high_frequency_cutoff,
            }
        }
        return meta
