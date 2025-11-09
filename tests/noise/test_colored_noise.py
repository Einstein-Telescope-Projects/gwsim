from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import kstest
from scipy.linalg import cholesky
from scipy.sparse import block_diag, coo_matrix
from etmdc.noise_curve import load_ET_PSD
from pathlib import Path
import tempfile

import pytest

from gwsim.utils.random import get_state
from gwsim.noise.colored_noise import ColoredNoise


# Fixtures


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture for temporary directory."""
    return tmp_path


@pytest.fixture
def mock_psd(tmp_path):
    """Create PSD array and save it as .npy files."""
    # freqs = np.linspace(0, 6000, 3000)
    # psd = np.column_stack([freqs, np.ones_like(freqs)])
    psd = load_ET_PSD()
    psd_file = tmp_path / "psd.npy"
    np.save(psd_file, psd)
    return str(psd_file)

# Unit-level tests


def test_gaussianity(mock_psd):
    """ Test that whitened noise series in gaussian """
    # Define parameters
    detector_names = ["E1", "E2"]
    N_det = len(detector_names)
    fs = 1024
    duration = 256
    flow = 2
    start_time = 0.0

    # Create colored noise instance
    simulator = ColoredNoise(
        detector_names=detector_names,
        psd=mock_psd,
        sampling_frequency=fs,
        duration=duration,
        flow=flow,
        start_time=start_time,
        seed=123
    )
    noise_ts = simulator.next()

    # Fetch colored noise instance attributes
    dt = simulator.dt
    df = 1 / duration
    frequency = simulator.frequency
    kmin = simulator.kmin
    kmax = simulator.kmax
    N_freq = simulator.N_freq

    # Compute the target PSD
    psd = np.load(mock_psd)
    target_psd = interp1d(psd[:, 0], psd[:, 1], bounds_error=False,
                          fill_value="extrapolate")(frequency[kmin: kmax])
    d1 = np.zeros_like(target_psd)

    # Compute the target spectral matrix
    target_spectral_matrix = np.empty((N_freq, N_det, N_det))
    for n in range(N_freq):
        submatrix = np.array(
            [
                [target_psd[n] if row == col else d1[n] for row in range(N_det)]
                for col in range(N_det)
            ]
        )
        target_spectral_matrix[n, :, :] = cholesky(submatrix, lower=True)
    target_spectral_matrix = block_diag(target_spectral_matrix, format="coo")

    # Transform back time series to frequency domain
    noise_fs = np.fft.rfft(noise_ts, axis=-1) * dt

    # Whiten the noise frequency series with the target spectral matrix
    whitened_noise_fs = np.zeros_like(noise_fs, dtype=np.complex128)
    size = N_det * N_det
    for i in range(N_freq):
        L = np.array(target_spectral_matrix.data[i * size: (i + 1) * size]).reshape(N_det, N_det)
        whitened_noise_fs[:, kmin +
                          i] = np.linalg.inv(L) @ noise_fs[:, kmin + i] / np.sqrt(0.5 / df)

    whitened_noise_fs = whitened_noise_fs[:, kmin: kmax]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.scatter(np.real(whitened_noise_fs.flatten()),
                np.imag(whitened_noise_fs.flatten()), alpha=0.1)
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.title('Scatter plot of whitened frequency domain samples')
    plt.axis('equal')
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.show()

    # Test the gaussianity of the whitened data
    joint_fs = np.concatenate((whitened_noise_fs.real, whitened_noise_fs.imag), axis=None)

    print("Variance is", np.var(joint_fs))

    stats = kstest(joint_fs, cdf="norm", args=(0, np.sqrt(0.5)))

    assert stats.pvalue > 0.05, f'Gaussianity test NOT passed, p-value is: {stats.pvalue:.3f}'
