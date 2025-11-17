from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey

from ..generator.state import StateAttribute
from ..utils.random import get_state
from .base import BaseNoise


class ColoredNoise(BaseNoise):
    """
    Generate colored noise time series for multiple gravitational wave detectors.

    This class generates noise time series with specified power spectral density (PSD)
    """

    previous_strain = StateAttribute()

    def __init__(
        self,
        detector_names: list[str],
        psd: str,
        sampling_frequency: float,
        duration: float,
        flow: float | None = 2,
        fhigh: float | None = None,
        start_time: float = 0,
        previous_strain: np.ndarray | None = None,
        max_samples: int | None = None,
        seed: int | None = None,

    ):
        """
        Initialiser for the ColoredNoise class.

        This class generates noise time series with specified power spectral density (PSD).

        Args:
            detector_names (List[str]): Names of the detectors.
            psd (str): Path to the file containing the Power Spectral Density array, with shape (N, 2), where the first column is frequency (Hz) and the second is PSD values.
            sampling_frequency (float): Sampling frequency in Hz.
            duration (float): Length of the noise time series in seconds.
            flow (float, optional): Lower frequency cut-off in Hz. Defaults to 2.0.
            fhigh (float, optional): Upper frequency cut-off in Hz. Defaults to Nyquist frequency.
            start_time (float, optional): GPS start time for the time series. Defaults to 0.
            previous_strain (np.ndarray, optional): Initial strain buffer for the noise time series, with shape (N_det, N_samples). Initialized to zero. Defaults to None.
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
        self.fhigh = fhigh if (fhigh is not None and fhigh <=
                               sampling_frequency / 2) else sampling_frequency // 2

        # Initialize
        self._initialize_window_properties()
        self._initialize_frequency_properties()
        self._initialize_psd(psd)
        self.previous_strain = np.zeros((self.N_det, self._N_chunk))
        self._temp_strain_buffer = None

    def _initialize_window_properties(self) -> None:
        """
        Initialize window properties for connecting noise realizations

        Raises:
            ValueError: If the duration is smaller than (2 * 100 / flow), raise ValueError
        """
        self._T_window = 2048
        self._f_window = 1.0 / self._T_window
        self._T_overlap = self._T_window / 2.0
        self._N_overlap = int(self._T_overlap * self.sampling_frequency)
        self._w0 = 0.5 + np.cos(2 * np.pi * self._f_window *
                                np.linspace(0, self._T_overlap, self._N_overlap)) / 2
        self._w1 = (
            0.5 + np.sin(2 * np.pi * self._f_window * np.linspace(0,
                         self._T_overlap, self._N_overlap) - np.pi / 2) / 2
        )

        # Safety check to ensure proper noise generation
        if self.duration < self._T_window / 2:
            raise ValueError(
                f"Duration ({self.duration:.1f} seconds) must be at least {self._T_window / 2:.1f} seconds to ensure noise continuity.")

    def _initialize_frequency_properties(self) -> None:
        """
        Initialize frequency and time properties for noise generation
        """
        self._T_chunk = self._T_window
        self._df_chunk = 1.0 / self._T_chunk
        self._N_chunk = int(self._T_chunk * self.sampling_frequency)
        self._kmin_chunk = int(self.flow / self._df_chunk)
        self._kmax_chunk = int(self.fhigh / self._df_chunk) + 1
        self._frequency_chunk = np.arange(0.0, self._N_chunk / 2.0 + 1) * self._df_chunk
        self._N_freq_chunk = len(self._frequency_chunk[self._kmin_chunk: self._kmax_chunk])

        self.dt = 1.0 / self.sampling_frequency

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
                return np.loadtxt(path, delimiter=",")
            else:
                raise ValueError(f"Unsupported file format for {path}")
        else:
            raise TypeError("psd and csd must be a string with path to a file")

    def _initialize_psd(self, psd: str) -> None:
        """
        Initialize PSD interpolations for frequency range

        Args:
            psd (str): Path to the file containing the Power Spectral Density array, with shape (N, 2), where the first column is frequency (Hz) and the second is PSD values.

        Raises:
            ValueError: If the shape of the psd or csd is different form (N, 2), raise ValueError
        """
        # TODO: Allow different PSDs for different detectors

        # Load psd/csd
        psd = self._load_array(psd)

        # Check that PSD has the correct size
        if psd.shape[1] != 2:
            raise ValueError("PSD must have shape (N, 2)")

        # Interpolate the PSD and CSD to the relevant frequencies
        freqs = self._frequency_chunk[self._kmin_chunk: self._kmax_chunk]
        psd_interp = interp1d(psd[:, 0], psd[:, 1], bounds_error=False,
                              fill_value="extrapolate")(freqs)

        # Add a roll-off at the edges
        window = tukey(self._N_freq_chunk, alpha=5e-3)
        self.psd = psd_interp * window

    def single_noise_realization(self, psd: np.ndarray) -> np.ndarray:
        """
        Generate a single noise realization in the frequency domain for each detector, and then transform it into the time domain.

        Args:
            psd (np.ndarray): Power spectral density array

        Returns:
            np.ndarray: time_series
        """
        freq_series = np.zeros((self.N_det, self._frequency_chunk.size), dtype=np.complex128)

        # generate white noise and color it with the PSD
        white_strain = (self.rng.standard_normal((self.N_det, self._N_freq_chunk)) + 1j *
                        self.rng.standard_normal((self.N_det, self._N_freq_chunk))) / np.sqrt(2)
        colored_strain = white_strain[:, :] * np.sqrt(psd * 0.5 / self._df_chunk)
        freq_series[:, self._kmin_chunk: self._kmax_chunk] += colored_strain

        # Transform each frequency strain into the time domain
        time_series = np.fft.irfft(freq_series, n=self._N_chunk, axis=1) * \
            self._df_chunk * self._N_chunk

        return time_series

    def next(self) -> np.ndarray:
        """
        Generate a noise realization in the time domain for each detector.

        Returns:
            np.ndarray: time series for each detector
        """
        # TODO: self.previous_strain should be a list of strains of gwf files. Need to open and read them

        N_frame = int(self.duration * self.sampling_frequency)

        # Load previous strain, or generate new if all zeros
        if self.previous_strain.shape[-1] < self._N_overlap:
            raise ValueError(
                f"Previous_strain has only {self.previous_strain.shape[-1]} points for each detector, but expected at least {self._N_overlap}.")

        strain_buffer = self.previous_strain[:, -self._N_chunk:]
        if np.all(strain_buffer == 0):
            strain_buffer = self.single_noise_realization(self.psd)

        # Apply the final part of the window
        strain_buffer[:, -self._N_overlap:] *= self._w0

        # Extend the strain buffer until it has more valid data than a single frame
        while strain_buffer.shape[-1] - self._N_chunk - self._N_overlap < N_frame:
            new_strain = self.single_noise_realization(self.psd)
            new_strain[:, :self._N_overlap] *= self._w1
            new_strain[:, -self._N_overlap:] *= self._w0
            strain_buffer[:, -self._N_overlap:] += new_strain[:, :self._N_overlap]
            strain_buffer[:, -self._N_overlap:] *= 1 / np.sqrt(self._w0**2 + self._w1**2)
            strain_buffer = np.concatenate((strain_buffer, new_strain[:, self._N_overlap:]), axis=1)

        # Discard the first N points and the excess data
        output_strain = strain_buffer[:, self._N_chunk:(self._N_chunk + N_frame)]

        # Store the strain buffer temporarily
        self._temp_strain_buffer = output_strain

        return output_strain

    def update_state(self) -> None:
        """Update the internal state for the next batch."""
        self.sample_counter += 1
        self.start_time += self.duration
        if self.rng is not None:
            self.rng_state = get_state()
        if self._temp_strain_buffer is not None:
            self.previous_strain = self._temp_strain_buffer
            self._temp_strain_buffer = None

    def save_batch(self, batch: np.ndarray, file_name: str | Path, overwrite: bool = False, **kwargs) -> None:
        """
        Args:
            batch (np.ndarray): One batch of data with shape (D, 2), where D is the number of detectors.
            file_name (str | Path): File name.
            overwrite (bool, optional): If True, overwrite existing file. Defaults to False.
            **kwargs: Optional keyword arguments, e.g., 'dataset_name' for HDF5 or 'channel' for GWF.

        Raises:
            ValueError: If the file suffix is not supported (supported: '.h5', '.hdf5', '.gwf').
        """
        file_name = Path(file_name)

        if batch.shape[0] != self.N_det:
            raise ValueError(
                f"Batch first dimension ({batch.shape[0]}) must match number of detectors ({self.N_det}).")

        for i, det_name in enumerate(self.detector_names):
            # Adjust filename per detector
            det_file_name = self._adjust_filename(file_name=file_name, insert=det_name)

            if file_name.suffix in [".h5", ".hdf5"]:
                # Prepare dataset name
                dataset_name = kwargs.get("dataset_name", "strain")
                det_dataset_name = f"{det_name}:{dataset_name}"
                self._save_batch_hdf5(
                    batch=batch[i, :],
                    file_name=det_file_name,
                    overwrite=overwrite,
                    dataset_name=det_dataset_name
                )
            elif file_name.suffix == ".gwf":
                # Prepare channel
                channel = kwargs.get("channel", "strain")
                det_channel = f"{det_name}:{channel}"
                self._save_batch_gwf(
                    batch=batch[i, :],
                    file_name=det_file_name,
                    overwrite=overwrite,
                    channel=det_channel
                )
            else:
                raise ValueError(
                    f"Suffix of file_name = {file_name} is not supported. Use ['.h5', '.hdf5'] for HDF5 files,"
                    "and '.gwf' for frame files."
                )

    def _adjust_filename(self, file_name: Path, insert: str) -> Path:
        """If the file name contains the keyword `DET`, insert `insert` at its place. Otherwise, insert `insert` at the beginning of the file name."""
        stem = file_name.stem
        suffix = file_name.suffix
        if "DET" in stem:
            new_stem = stem.replace("DET", insert, 1)
        else:
            new_stem = insert + "-" + stem
        return file_name.with_name(new_stem + suffix)
