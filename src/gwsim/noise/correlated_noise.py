# pylint: disable=duplicate-code
"""Correlated noise simulator for multiple gravitational wave detectors."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cholesky
from scipy.sparse import block_diag, coo_matrix

from gwsim.data.time_series.time_series import TimeSeries
from gwsim.data.time_series.time_series_list import TimeSeriesList
from gwsim.noise.base import NoiseSimulator

logger = logging.getLogger("gwsim")


class CorrelatedNoiseSimulator(NoiseSimulator):  # pylint: disable=too-many-instance-attributes
    """Correlated noise simulator for multiple gravitational wave detectors

    This class generates noise time series with specified power spectral density (PSD)
    and cross-spectral density (CSD) for multiple detectors. The correlations between
    detectors are modeled using Cholesky decomposition of the spectral matrix.

    The noise generation uses an overlap-add method with windowing to produce
    smooth, continuous time series across segment boundaries.
    """

    def __init__(
        self,
        psd_file: str | Path,
        csd_file: str | Path,
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
        """Initialize the correlated noise simulator.

        Args:
            psd_file: Path to file containing Power Spectral Density array with shape (N, 2),
                where the first column is frequency (Hz) and the second is PSD values.
            csd_file: Path to file containing Cross Spectral Density array with shape (N, 2),
                where the first column is frequency (Hz) and the second is complex CSD values.
            detectors: List of detector names (e.g., ['E1', 'E2', 'E3']).
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
        self.csd_file = csd_file
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = (
            high_frequency_cutoff
            if (high_frequency_cutoff is not None and high_frequency_cutoff <= sampling_frequency / 2)
            else sampling_frequency / 2
        )

        # Initialize noise generation properties
        self._n_det = len(detectors)
        self._initialize_window_properties()
        self._initialize_frequency_properties()
        self._initialize_psd_csd()
        self._spectral_matrix = self._compute_spectral_matrix_cholesky()

    def _initialize_window_properties(self) -> None:
        """Initialize window properties for overlap-add noise generation."""
        self._f_window = self.low_frequency_cutoff / 100
        self._t_window = 1 / self._f_window
        self._t_overlap = self._t_window / 2.0
        self._n_overlap = int(self._t_overlap * self.sampling_frequency.value)

        # Create overlap windows for smooth transitions
        t_overlap_array = np.linspace(0, self._t_overlap, self._n_overlap)
        self._w0 = 0.5 + np.cos(2 * np.pi * self._f_window * t_overlap_array) / 2
        self._w1 = 0.5 + np.sin(2 * np.pi * self._f_window * t_overlap_array - np.pi / 2) / 2

    def _initialize_frequency_properties(self) -> None:
        """Initialize frequency and time properties for noise generation."""
        self._t_segment = self._t_window * 3
        self._df = 1.0 / self._t_segment
        self._dt = 1.0 / self.sampling_frequency.value
        self._n_samples = int(self._t_segment * self.sampling_frequency.value)
        self._k_min = int(self.low_frequency_cutoff / self._df)
        self._k_max = int(self.high_frequency_cutoff / self._df) + 1
        self._frequency = np.arange(0.0, self._n_samples / 2.0 + 1) * self._df
        self._n_freq = len(self._frequency[self._k_min : self._k_max])

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
        if isinstance(file_path, (str, Path)):
            path = Path(file_path)
            if path.suffix == ".npy":
                return np.load(path)
            if path.suffix == ".txt":
                return np.loadtxt(path)
            if path.suffix == ".csv":
                return np.loadtxt(path, delimiter=",")
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .npy, .txt, or .csv.")
        raise TypeError("file_path must be a string or Path.")

    def _initialize_psd_csd(self) -> None:  # pylint: disable=duplicate-code
        """Initialize PSD and CSD interpolations for the frequency range.

        Raises:
            ValueError: If PSD or CSD arrays don't have shape (N, 2).
        """
        psd_data = self._load_spectral_data(self.psd_file)
        csd_data = self._load_spectral_data(self.csd_file)

        expected_shape = (None, 2)
        if psd_data.shape[1] != expected_shape[1] or csd_data.shape[1] != expected_shape[1]:
            raise ValueError("PSD and CSD files must have shape (N, 2).")

        # Interpolate to the relevant frequencies
        freqs = self._frequency[self._k_min : self._k_max]
        self._psd = interp1d(psd_data[:, 0], psd_data[:, 1], bounds_error=False, fill_value="extrapolate")(freqs)

        csd_real = interp1d(csd_data[:, 0], csd_data[:, 1].real, bounds_error=False, fill_value="extrapolate")(freqs)
        csd_imag = interp1d(csd_data[:, 0], csd_data[:, 1].imag, bounds_error=False, fill_value="extrapolate")(freqs)
        csd_complex = csd_real + 1j * csd_imag
        self._csd_magnitude = np.abs(csd_complex)
        self._csd_phase = np.angle(csd_complex)

    def _compute_spectral_matrix_cholesky(self) -> coo_matrix:
        """Compute the Cholesky decomposition of the spectral matrix.

        Returns:
            Sparse COO matrix containing the block-diagonal Cholesky decomposition.
        """
        # Compute diagonal elements
        d0 = self._psd * 0.25 / self._df
        d1 = self._csd_magnitude * 0.25 / self._df * np.exp(-1j * self._csd_phase)

        # Build Cholesky decomposition of the spectral matrix in block-diagonal form
        spectral_matrix = np.empty((self._n_freq, self._n_det, self._n_det), dtype=np.complex128)
        for n in range(self._n_freq):
            submatrix = np.array(
                [
                    [d0[n] if row == col else d1[n] if row < col else np.conj(d1[n]) for row in range(self._n_det)]
                    for col in range(self._n_det)
                ]
            )
            spectral_matrix[n, :, :] = cholesky(submatrix)

        return block_diag(spectral_matrix, format="coo")

    def _generate_single_realization(self) -> np.ndarray:
        """Generate a single noise realization in the time domain.

        Returns:
            Time series array with shape (n_detectors, n_samples).
        """
        if self.rng is None:
            raise RuntimeError("Random number generator not initialized. Set seed in constructor.")

        freq_series = np.zeros((self._n_det, self._frequency.size), dtype=np.complex128)

        # Generate white noise and color it with the spectral matrix
        white_strain = self.rng.standard_normal(self._n_freq * self._n_det) + 1j * self.rng.standard_normal(
            self._n_freq * self._n_det
        )
        colored_strain = self._spectral_matrix.dot(white_strain)

        # Split the frequency strain for each detector
        freq_series[:, self._k_min : self._k_max] += np.transpose(
            np.reshape(colored_strain, (self._n_freq, self._n_det))
        )

        # Transform each frequency strain into the time domain
        time_series = np.fft.irfft(freq_series, n=self._n_samples, axis=1) * self._df * self._n_samples

        return time_series

    def _simulate(self, *args, **kwargs) -> TimeSeriesList:
        """Simulate correlated noise for all detectors.

        Returns:
            TimeSeriesList containing a single TimeSeries with shape (n_detectors, n_samples).
        """
        n_frame = int(self.duration.value * self.sampling_frequency.value)

        # Generate the initial single noise realization and apply the final part of the window
        strain_buffer = self._generate_single_realization()
        strain_buffer[:, -self._n_overlap :] *= self._w0

        # Extend the strain buffer until it has more valid data than a single frame
        while strain_buffer.shape[-1] - self._n_overlap < n_frame:
            new_strain = self._generate_single_realization()
            new_strain[:, : self._n_overlap] *= self._w1
            new_strain[:, -self._n_overlap :] *= self._w0
            strain_buffer[:, -self._n_overlap :] += new_strain[:, : self._n_overlap]
            strain_buffer = np.concatenate((strain_buffer, new_strain[:, self._n_overlap :]), axis=1)

        # Extract the frame and create TimeSeries
        data = strain_buffer[:, :n_frame]

        return TimeSeriesList(
            [TimeSeries(data=data, start_time=self.start_time, sampling_frequency=self.sampling_frequency)]
        )

    @property
    def metadata(self) -> dict:
        """Get metadata including correlated noise configuration.

        Returns:
            Dictionary containing metadata.
        """
        meta = super().metadata
        meta["correlated_noise"] = {
            "arguments": {
                "psd_file": str(self.psd_file),
                "csd_file": str(self.csd_file),
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self.high_frequency_cutoff,
            }
        }
        return meta
