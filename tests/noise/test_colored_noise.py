from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import kstest, chi2, anderson, distributions
from scipy.signal import windows
from etmdc.noise_curve import load_ET_PSD
from pathlib import Path
import tempfile

import pytest

from gwsim.noise.colored_noise import ColoredNoise


# Fixtures


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture for temporary directory."""
    return tmp_path


@pytest.fixture
def mock_psd(tmp_path):
    """Create PSD array and save it as .npy files."""
    psd = load_ET_PSD()
    psd = np.loadtxt(
        '/home/u0167252@FYSAD.FYS.KULEUVEN.BE/Desktop/gwsim_detectors/noise_curves/ET_10_full_cryo_psd.txt')
    psd_file = tmp_path / "psd.npy"
    np.save(psd_file, psd)
    return str(psd_file)


@pytest.fixture
def whitened_samples(mock_psd):
    """ Generate `N_frames` of colored noise frames and whiten them """
    # Define parameters
    detector_names = ['E1']
    N_det = len(detector_names)
    fs = 4096
    duration = 1024
    flow = 2
    start_time = 0.0
    N_frames = 100

    # Time and frequency properties
    dt = 1 / fs
    df = 1 / duration
    N = int(duration * fs)
    frequency = np.arange(0.0, int(duration * fs) / 2.0 + 1) * df
    kmin = int(flow / df)
    kmax = int((fs / 2) / df) + 1
    N_freq = len(frequency[kmin:kmax])

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

    # Generate several noise frames
    noise_ts = np.zeros((N_frames, N_det, int(duration * fs)))
    for i in range(N_frames):
        noise_ts[i] = simulator.next()
        simulator.update_state()

    # Compute the target PSD for the whitening
    psd = np.load(mock_psd)
    target_psd = interp1d(psd[:, 0], psd[:, 1], bounds_error=False,
                          fill_value="extrapolate")(frequency[kmin: kmax])
    window = windows.tukey(len(frequency[kmin: kmax]), alpha=1e-3)
    target_psd = np.where(target_psd * window > 0, target_psd * window, 1e-60)

    # Whiten the noise frequency series with the target spectral matrix
    whitened_noise_fs = np.zeros((N_frames, N_det, N_freq), dtype=np.complex128)
    for i, ts in enumerate(noise_ts):
        fs = np.fft.rfft(ts, axis=-1) * dt
        w_fs = np.zeros_like(fs, dtype=np.complex128)
        w_fs[:, kmin: kmax] = fs[:, kmin: kmax] / np.sqrt(target_psd * 0.5 / df)
        whitened_noise_fs[i] = w_fs[:, kmin: kmax]

    return whitened_noise_fs


Unit-level tests


def test_gaussianity_KS_test(whitened_samples):
    """ Test that whitened noise series in gaussian per frequency bin with the KS test """

    N_frames, N_det, N_freq = whitened_samples.shape

    # Test the gaussianity of the whitened data for each frequnecy bin
    Pvalue = np.zeros((N_det, N_freq))
    for j in range(N_freq):
        for i in range(N_det):
            joint_fs = np.concatenate(
                (whitened_samples[:, i, j].real, whitened_samples[:, i, j].imag))
            stats = kstest(joint_fs, cdf="norm", args=(0, np.sqrt(0.5)))
            Pvalue[i, j] = stats.pvalue

    # Fisher's combined probability test
    alpha = 0.01
    for i in range(N_det):
        q = -2 * np.sum(np.log(Pvalue[i, :]))
        combined_pvalue = chi2.sf(q, df=2*N_freq)
        assert combined_pvalue > alpha, f"Combined P-value is smaller than threshold {alpha}."


def test_gaussianity_Anderson_Darling(mock_psd):
    """ Generate `N_frames` of colored noise frames and whiten them """
    # Define parameters
    detector_names = ['E1', 'E2']
    N_det = len(detector_names)
    fs = 4096
    duration = 4096
    flow = 2
    start_time = 0.0

    # Time and frequency properties
    dt = 1 / fs
    df = 1 / duration
    N = int(duration * fs)
    frequency = np.arange(0.0, int(duration * fs) / 2.0 + 1) * df
    kmin = int(flow / df)
    kmax = int((fs / 2) / df) + 1
    N_freq = len(frequency[kmin:kmax])

    # Create colored noise instance
    simulator = ColoredNoise(
        detector_names=detector_names,
        psd=mock_psd,
        sampling_frequency=fs,
        duration=duration,
        flow=flow,
        start_time=start_time,
        seed=42
    )

    # Generate one noise frame
    noise_ts = simulator.next()

    # Compute the target PSD for the whitening
    psd = np.load(mock_psd)
    target_psd = interp1d(psd[:, 0], psd[:, 1], bounds_error=False,
                          fill_value="extrapolate")(frequency[kmin: kmax])
    window = windows.tukey(len(frequency[kmin: kmax]), alpha=1e-3)
    target_psd = np.where(target_psd * window > 0, target_psd * window, 1e-60)

    # Whiten the noise frequency series with the target spectral matrix
    fs = np.fft.rfft(noise_ts, axis=-1) * dt
    w_fs = np.zeros_like(fs, dtype=np.complex128)
    w_fs[:, kmin: kmax] = fs[:, kmin: kmax] / np.sqrt(target_psd * 0.5 / df)
    whitened_noise_fs = w_fs[:, kmin: kmax]

    bin_mask = 1024
    for i in range(N_det):
        joint_fs = np.concatenate(
            (whitened_noise_fs[i, bin_mask: -bin_mask].real, whitened_noise_fs[i, bin_mask: -bin_mask].imag))
        # Anderson Darling gaussianity test
        w = np.sort(joint_fs) / np.sqrt(0.5)
        N = len(joint_fs)
        logcdf = distributions.norm.logcdf(w)
        logsf = distributions.norm.logsf(w)
        k = np.arange(1, N + 1)
        A2 = -N - np.sum((2*k - 1.0) / N * (logcdf + logsf[::-1]), axis=0)
        critical_values = anderson(joint_fs).critical_values

        print(A2)

        assert A2 > critical_values[
            1], f"Anderson-Darling statistic is smaller than critical value {critical_values[-1]}."
