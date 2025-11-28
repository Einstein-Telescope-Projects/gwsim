from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cholesky
from scipy.sparse import block_diag, coo_matrix

from ..utils.random import get_state
from .base import BaseNoise


class CorrelatedNoise(BaseNoise):
    """
    Generate correlated noise time series for multiple gravitational wave detectors.

    This class generates noise time series with specified power spectral density (PSD)
    and cross-spectral density (CSD) for multiple detectors
    """

    def __init__(
        self,
        detector_names: list[str],
        psd: str,
        csd: str,
        sampling_frequency: float,
        duration: float,
        flow: float | None = 2,
        fhigh: float | None = None,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
    ):
        """
        Initialiser for the CorrelatedNoise class.

        This class generates noise time series with specified power spectral density (PSD)
        and cross-spectral density (CSD) for multiple detectors

        Args:
            detector_names (List[str]): Names of the detectors.
            psd (str): Path to the file containing the Power Spectral Density array, with shape (N, 2), where the first column is frequency (Hz) and the second is PSD values.
            csd (str): Path to the file containing Cross Power Spectral Density array, with shape (N, 2), where the first column is frequency (Hz) and the second is complex CSD values.
            sampling_frequency (float): Sampling frequency in Hz.
            duration (float): Length of the noise time series in seconds.
            flow (float, optional): Lower frequency cut-off in Hz. Defaults to 2.0.
            fhigh (float, optional): Upper frequency cut-off in Hz. Defaults to Nyquist frequency.
            start_time (float, optional): GPS start time for the time series. Defaults to 0.
            max_samples (int, optional): Maximum number of samples to generate. Defaults to None.
            seed (int, optional): Seed for pseudo-random number generation. Defaults to None.
        """

        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
        )
        self.detector_names = detector_names
        self.N_det = len(detector_names)
        if self.N_det == 0:
            raise ValueError("detector_names must contain at least one detector.")
        self.flow = flow
        self.fhigh = fhigh if (fhigh is not None and fhigh <= sampling_frequency / 2) else sampling_frequency / 2

        # Initialize
        self._initialize_window_properties()
        self._initialize_frequency_properties()
        self._initialize_psd_csd(psd, csd)
        self.spectral_matrix = self.spectral_matrix_cholesky_decomposition()

    def _initialize_window_properties(self) -> None:
        """
        Initialize window properties for noise generation
        """
        self.f_window = self.flow / 100
        self.T_window = 1 / self.f_window
        self.T_overlap = self.T_window / 2.0
        self.N_overlap = int(self.T_overlap * self.sampling_frequency)
        self.w0 = 0.5 + np.cos(2 * np.pi * self.f_window * np.linspace(0, self.T_overlap, self.N_overlap)) / 2
        self.w1 = (
            0.5 + np.sin(2 * np.pi * self.f_window * np.linspace(0, self.T_overlap, self.N_overlap) - np.pi / 2) / 2
        )

    def _initialize_frequency_properties(self) -> None:
        """
        Initialize frequency and time properties for noise generation
        """
        self.T = self.T_window * 3
        self.df = 1.0 / self.T
        self.dt = 1.0 / self.sampling_frequency
        self.N = int(self.T * self.sampling_frequency)
        self.kmin = int(self.flow / self.df)
        self.kmax = int(self.fhigh / self.df) + 1
        self.frequency = np.arange(0.0, self.N / 2.0 + 1) * self.df
        self.N_freq = len(self.frequency[self.kmin : self.kmax])

    def _load_array(self, arr_path: str) -> np.ndarray:
        """
        Load an array from a file path

        Args:
            arr_path (str): Path to the file containing the input array

        Returns:
            np.ndarray: The loaded array
        """
        if isinstance(arr_path, str):
            path = Path(arr_path)
            if path.suffix == ".npy":
                return np.load(path)
            elif path.suffix == ".txt":
                return np.loadtxt(path)
            elif path.suffix == ".csv":
                return np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
            else:
                raise ValueError(f"Unsupported file format for {path}")
        else:
            raise TypeError("psd and csd must be a string with path to a file")

    def _initialize_psd_csd(self, psd: str, csd: str) -> None:
        """
        Initialize PSD and CSD interpolations for frequency range

        Args:
            psd (str): Path to the file containing the Power Spectral Density array, with shape (N, 2), where the first column is frequency (Hz) and the second is PSD values.
            csd (str): Path to the file containing Cross Power Spectral Density array, with shape (N, 2), where the first column is frequency (Hz) and the second is complex CSD values.

        Raises:
            ValueError: If the shape of the psd or csd is different form (N, 2), raise ValueError
        """

        # Load psd/csd
        psd = self._load_array(psd)
        csd = self._load_array(csd)

        if psd.shape[1] != 2 or csd.shape[1] != 2:
            raise ValueError("PSD and CSD must have shape (N, 2)")

        # Interpolate the PSD and CSD to the relevant frequencies
        freqs = self.frequency[self.kmin : self.kmax]
        self.psd = interp1d(psd[:, 0], psd[:, 1], bounds_error=False, fill_value="extrapolate")(freqs)
        csd_real = interp1d(csd[:, 0], csd[:, 1].real, bounds_error=False, fill_value="extrapolate")(freqs)
        csd_imag = interp1d(csd[:, 0], csd[:, 1].imag, bounds_error=False, fill_value="extrapolate")(freqs)
        self.csd_magnitude = np.abs(csd_real + 1j * csd_imag)
        self.csd_phase = np.angle(csd_real + 1j * csd_imag)

    def spectral_matrix_cholesky_decomposition(self) -> coo_matrix:
        """
        Compute the Cholesky decomposition of the spectral matrix.

        Returns:
            scipy sparse coo_matrix: Cholesky decomposition of the spectral matrix
        """
        # Take the principal diagonals of the spectral matrix
        d0 = self.psd * 0.25 / self.df
        d1 = self.csd_magnitude * 0.25 / self.df * np.exp(-1j * self.csd_phase)

        # Build Cholesky decomposition of the spectral matrix in block diagonal form
        spectral_matrix = np.empty((self.N_freq, self.N_det, self.N_det), dtype=np.complex128)
        for n in range(self.N_freq):
            submatrix = np.array(
                [
                    [d0[n] if row == col else d1[n] if row < col else np.conj(d1[n]) for row in range(self.N_det)]
                    for col in range(self.N_det)
                ]
            )
            spectral_matrix[n, :, :] = cholesky(submatrix)

        return block_diag(spectral_matrix, format="coo")

    def single_noise_realization(self, spectral_matrix: coo_matrix) -> np.ndarray:
        """
        Generate a single noise realization in the frequency domain for each detector, and then transform it into the time domain.

        Args:
            spectral_matrix (scipy sparse csc_matrix): Cholesky decomposition of the spectral matrix

        Returns:
            np.ndarray: time_series
        """
        freq_series = np.zeros((self.N_det, self.frequency.size), dtype=np.complex128)

        # generate white noise and color it with the spectral matrix
        white_strain = self.rng.standard_normal(self.N_freq * self.N_det) + 1j * self.rng.standard_normal(
            self.N_freq * self.N_det
        )
        colored_strain = spectral_matrix.dot(white_strain)

        # Split the frequency strain for each detector
        freq_series[:, self.kmin : self.kmax] += np.transpose(np.reshape(colored_strain, (self.N_freq, self.N_det)))

        # transform each frequency strain into the time domain
        time_series = np.fft.irfft(freq_series, n=self.N, axis=1) * self.df * self.N

        return time_series

    def next(self) -> np.ndarray:
        """
        Generate a noise realization in the time domain for each detector.

        Returns:
            np.ndarray: time series for each detector
        """
        N_frame = int(self.duration * self.sampling_frequency)

        # Generate the initial single noise realization and apply the final part of the window
        strain_buffer = self.single_noise_realization(self.spectral_matrix)
        strain_buffer[:, -self.N_overlap :] *= self.w0

        # Extend the strain buffer until it has more valid data than a single frame
        while strain_buffer.shape[-1] - self.N_overlap < N_frame:
            new_strain = self.single_noise_realization(self.spectral_matrix)
            new_strain[:, : self.N_overlap] *= self.w1
            new_strain[:, -self.N_overlap :] *= self.w0
            strain_buffer[:, -self.N_overlap :] += new_strain[:, : self.N_overlap]
            strain_buffer = np.concatenate((strain_buffer, new_strain[:, self.N_overlap :]), axis=1)

        return strain_buffer[:, :N_frame]

    def update_state(self) -> None:
        """Update the internal state for the next batch."""
        self.sample_counter += 1
        self.start_time += self.duration
        if self.rng is not None:
            self.rng_state = get_state()

    def save_batch(self, batch: np.ndarray, file_name: str | Path, overwrite: bool = False, **kwargs) -> None:
        """
        Args:
            batch (np.ndarray): One batch of data with shape (D, 2), where D is the number of detectors.
            file_name (str | Path): File name.
            overwrite (bool, optional): If True, overwrite existing file. Defaults to False.
            **kwargs: Optional keyword arguments, e.g., 'dataset_name' for HDF5 or 'channel' for GWF.

        Raises:
            ValueError: If the file suffix is not supported (supported: '.h5', '.hdf5', '.gwf').
            ValueError: If file_name does not start with 'E-'.
        """
        file_name = Path(file_name)

        if batch.shape[0] != self.N_det:
            raise ValueError(f"Batch first dimension ({batch.shape[0]}) must match number of detectors ({self.N_det}).")

        for i, det_name in enumerate(self.detector_names):
            # Adjust filename per detector
            det_file_name = self._adjust_filename(file_name=file_name, insert=det_name)

            if file_name.suffix in [".h5", ".hdf5"]:
                # Prepare dataset name
                dataset_name = kwargs.get("dataset_name", "strain")
                det_dataset_name = f"{det_name}:{dataset_name}"
                self._save_batch_hdf5(
                    batch=batch[i, :], file_name=det_file_name, overwrite=overwrite, dataset_name=det_dataset_name
                )
            elif file_name.suffix == ".gwf":
                # Prepare channel
                channel = kwargs.get("channel", "strain")
                det_channel = f"{det_name}:{channel}"
                self._save_batch_gwf(
                    batch=batch[i, :], file_name=det_file_name, overwrite=overwrite, channel=det_channel
                )
            else:
                raise ValueError(
                    f"Suffix of file_name = {file_name} is not supported. Use ['.h5', '.hdf5'] for HDF5 files,"
                    "and '.gwf' for frame files."
                )

    def _adjust_filename(self, file_name: Path, insert: str) -> Path:
        """If the file name starts with 'E-', insert `insert` right after it."""
        stem = file_name.stem
        if stem.startswith("E-"):
            stem = f"E-{insert}_{stem[2:]}"
        else:
            raise ValueError(f"Invalid filename '{file_name.name}'. Must start with 'E-'.")
        return file_name.with_stem(stem)
