from __future__ import annotations

from typing import Any
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import block_diag, coo_matrix
from scipy.linalg import cholesky
from gwpy.timeseries import TimeSeries
from gwpy.detector import Channel
from gwpy import time
from pathlib import Path

from .noise.base import BaseNoise


class CorrelatedNoise(BaseNoise):
    """
    Generate correlated noise time series for multiple gravitational wave detectors.

    This class generates noise time series with specified power spectral density (PSD)
    and cross-spectral density (CSD) for multiple detectors, using a Cholesky decomposition
    of the spectral matrix to ensure proper correlation.
    """

    def __init__(
        self,
        detector_names: list[str],
        psd: np.ndarray,
        csd: np.ndarray,
        sampling_frequency: float,
        duration: float,
        flow: float | None = 2,
        fhigh: float | None = None,
        gps_epoch: float | None = None,
        batch_size: int = 1,
        max_samples: int | None = None,
        seed: int | None = None,
    ):
    """
    Initialiser for the CorrelatedNoise class.

    This class generates noise time series with specified power spectral density (PSD)
    and cross-spectral density (CSD) for multiple detectors, using a Cholesky decomposition
    of the spectral matrix to ensure proper correlation.

    Args:
        detector_names (List[str]): Names of the detectors.
        psd (np.ndarray): Power Spectral Density with shape (N, 2), where the first column is frequency (Hz) and the second is PSD values.
        csd (np.ndarray): Cross Power Spectral Density with shape (N, 2), where the first column is frequency (Hz) and the second is complex CSD values.
        sampling_frequency (float): Sampling frequency in Hz.
        duration (float): Length of the noise time series in seconds.
        flow (float, optional): Lower frequency cut-off in Hz. Defaults to 2.0.
        fhigh (float, optional): Upper frequency cut-off in Hz. Defaults to Nyquist frequency.
        gps_epoch (float, optional): GPS start time for the time series. Defaults to 01/01/2025 GPS time.
        batch_size (int, optional): Number of samples per batch. Defaults to 1.
        max_samples (int, optional): Maximum number of samples to generate. Defaults to None.
        seed (int, optional): Seed for pseudo-random number generation. Defaults to None.
    """
    super().__init__(
        sampling_frequency=sampling_frequency,
        duration=duration,
        batch_size=batch_size,
        max_samples=max_samples,
        seed=seed
    )
    self.detector_names = detector_names
    self.gps_epoch = np.floor(gps_epoch) if gps_epoch else np.floor(
        time.to_gps(datetime.datetime(year=2025, month=1, day=1)))
    self.flow = flow
    self.fhigh = fhigh if (fhigh is not None and fhigh <=
                           sampling_frequency / 2) else sampling_frequency / 2

    self._initialize_window_properties()
    self._initialize_frequency_properties()
    self._initialize_psd_csd(psd, csd)
    self.spectral_matrix = self.spectral_matrix_cholesky_decomposition()

    def _initialize_window_properties(self) -> None:
        """
        Initialize window properties for noise generation
        """
        self.f_window = self.flow/100
        self.T_window = 1/self.f_window
        self.T_overlap = self.T_window/2.
        self.N_overlap = int(self.T_overlap*self.sampling_frequency)
        self.w0 = 0.5 + np.cos(2*np.pi*self.f_window*np.linspace(0,
                               self.T_overlap, self.N_overlap))/2
        self.w1 = 0.5 + np.sin(2*np.pi*self.f_window*np.linspace(0,
                               self.T_overlap, self.N_overlap) - np.pi/2)/2

    def _initialize_frequency_properties(self) -> None:
        """
        Initialize frequency and time properties for noise generation
        """
        self.T = self.T_window*3
        self.df = 1./self.T
        self.dt = 1./self.sampling_frequency
        self.N = int(self.T*self.sampling_frequency)
        self.kmin = int(self.flow/self.df)
        self.kmax = int(self.fhigh/self.df)+1
        self.frequency = np.arange(0., self.N/2.+1)*self.df
        self.N_freq = len(self.frequency[self.kmin:self.kmax])

    def _initialize_psd_csd(self, psd: np.ndarray, csd: np.ndarray) -> None:
        """
        Initialize PSD and CSD interpolations for frequency range
        """
        self.psd = interp1d(psd[:, 0], psd[:, 1], bounds_error=False,
                            fill_value='extrapolate')(self.frequency[self.kmin:self.kmax])
        csd_real = interp1d(csd[:, 0], csd[:, 1].real, bounds_error=False,
                            fill_value='extrapolate')(self.frequency[self.kmin:self.kmax])
        csd_imag = interp1d(csd[:, 0], csd[:, 1].imag, bounds_error=False,
                            fill_value='extrapolate')(self.frequency[self.kmin:self.kmax])
        self.csd_magnitude = np.abs(csd_real + 1j*csd_imag)
        self.csd_phase = np.angle(csd_real + 1j*csd_imag)

    def spectral_matrix_cholesky_decomposition(self) -> coo_matrix:
        """
        Auxiliary function to compute the Cholesky decomposition of the spectral matrix.

        Returns:
            scipy sparse coo_matrix: Cholesky decomposition of the spectral matrix
        """
        # take the principal diagonals of the spectral matrix
        d0 = self.psd*0.25/self.df
        d1 = self.csd_magnitude*0.25/self.df * np.exp(-1j*self.csd_phase)

        # Build Cholesky decomposition of the spectral matrix in block diagonal form
        spectral_matrix = np.empty((self.N_freq, self.N_det, self.N_det), dtype=np.complex128)
        for n in range(self.N_freq):
            submatrix = np.array([[d0[n] if row == col else d1[n] if row < col else np.conj(
                d1[n]) for row in range(self.N_det)] for col in range(self.N_det)])
            spectral_matrix[n, :, :] = cholesky(submatrix)

        return block_diag(spectral_matrix, format='coo')

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
        white_strain = np.random.randn(self.N_freq*self.N_det) + 1j * \
            np.random.randn(self.N_freq*self.N_det)
        colored_strain = spectral_matrix.dot(white_strain)

        # Split the frequency strain for each detector
        freq_series[:, self.kmin:self.kmax] += np.transpose(
            np.reshape(colored_strain, (self.N_freq, self.N_det)))

        # transform each frequency strain into the time domain
        time_series = np.fft.irfft(freq_series, n=self.N, axis=1)*self.df*self.N

        return time_series

    def next(self) -> np.ndarray:
        """
        Generate a long noise realization in the time domain for each detector.

        Returns:
            np.ndarray: time series for each detector
        """
        N_frame = int(self.duration*self.sampling_frequency)
        if self.seed is not None:
            np.random.seed(int(self.seed))
        else:
            np.random.seed()

        # Generate the initial single noise realisation and apply the final part of the window
        strain_buffer = self.single_noise_realization(self.spectral_matrix)
        strain_buffer[:, -self.N_overlap:] *= self.w0

        # Extend the strain buffer until it has more valid data than a single frame
        while strain_buffer.shape[-1] - self.N_overlap < N_frame:
            np.random.seed(np.random.randint(low=1, high=1e6))
            new_strain = self.single_noise_realization(self.spectral_matrix)
            new_strain[:, :self.N_overlap] *= self.w1
            new_strain[:, -self.N_overlap:] *= self.w0
            strain_buffer[:, -self.N_overlap:] += new_strain[:, :self.N_overlap]
            strain_buffer = np.concatenate(
                (strain_buffer, new_strain[:, self.N_overlap:]), axis=1)

        return strain_buffer[:, :N_frame]

    def save_batch(self, batch: np.ndarray, file_name: str, overwrite: bool = False) -> None:
        """
        Save the noise time series to a Frame file.

        Args:
            batch (np.ndarray): Noise time series with shape (N_det, N), where N_det is the number of detectors and N is the number of data points.
            file_name (str): Name of the output Frame file in the format 'E-{det_name}_STRAIN_CORR_NOISE-{gps_epoch}-{duration}.gwf'.
            overwrite (bool): If True, overwrite existing file. Defaults to False.

        Raises:
            ValueError: If file_name is different from the expected one.
            FileExistsError: If the file already exists and overwrite is False.
        """
        for j, det_name in enumerate(self.detector_names):

            # Check if file_name is correct
            expected_file_name = f"E-{det_name}_STRAIN_CORR_NOISE-{int(gps_epoch)}-{int(duration)}.gwf"
            if Path(file_name).name != expected_file_name:
                raise ValueError(
                    f"file_name must be '{Path(file_name).parent}/{expected_file_name}' for detector {det_name}, "
                    f"got '{Path(file_name).parent}/{file_name}'"
                )

            # Check if the file already exists and overwrite is False
            if file_name.exists() and not overwrite:
                raise FileExistsError(f"File {file_name} already exists and overwrite is False")

            # Save different Frame files for each detector
            channel = Channel(f"{det_name}:CorrNoise")
            data = TimeSeries(batch[j, :],
                              t0=gps_time,
                              dt=self.dt,
                              channel=channel,
                              )
            data.write(file_name, append=False)

    def update_state(self) -> None:
        """Update the internal state for the next batch."""
        if self.seed is not None:
            self.seed += 1
        if self.gps_epoch is not None:
            self.gps_epoch += self.duration
