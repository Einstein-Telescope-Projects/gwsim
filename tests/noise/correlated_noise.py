from __future__ import annotations

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from gwsim.tools.main import main

from gwsim.noise.correlated_noise import CorrelatedNoise


# Fixtures
@pytest.fixture
def runner():
    """Fixture for Click CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture for temporary directory."""
    return tmp_path


@pytest.fixture
def mock_psd_csd(tmp_path):
    """Create PSD and CSD arrays and save them as .npy files."""
    freqs = np.linspace(0, 32, 100)
    psd = np.column_stack([freqs, np.ones_like(freqs)])
    csd = np.column_stack([freqs, 0.5 * np.ones_like(freqs) + 0.2j * np.ones_like(freqs)])
    psd_file = tmp_path / "psd.npy"
    csd_file = tmp_path / "csd.npy"
    np.save(psd_file, psd)
    np.save(csd_file, csd)
    return str(psd_file), str(csd_file)


@pytest.fixture
def correlated_noise_instance(mock_psd_csd):
    psd, csd = mock_psd_csd
    return CorrelatedNoise(
        detector_names=["ET1", "ET2"],
        psd=psd,
        csd=csd,
        sampling_frequency=64,
        duration=2,
        flow=2,
        fhigh=16,
        start_time=0,
        seed=123,
    )


@pytest.fixture
def mock_config(temp_dir, mock_psd_csd):
    psd, csd = mock_psd_csd
    return {
        "output": {
            "file_name": "E-Correlated_Noise-{{ start_time }}-{{ duration }}.gwf",
            "arguments": {"channel": "STRAIN"},
        },
        "generator": {
            "class": "gwsim.noise.correlated_noise.CorrelatedNoise",
            "arguments": {
                "detector_names": ["ET1", "ET2", "ET3"],
                "psd": psd,
                "csd": csd,
                "sampling_frequency": 64,
                "duration": 4,
                "flow": 2,
                "fhigh": 16,
                "start_time": 123,
                "max_samples": 2,
                "seed": 0,
            },
        },
        "working-directory": str(temp_dir),
    }


# Unit-level tests
def test_initialize_psd_csd_shapes(mock_psd_csd):
    """Check that PSD/CSD interpolation produces arrays with expected shapes"""
    psd, csd = mock_psd_csd
    cn = CorrelatedNoise(
        detector_names=["ET1", "ET2"],
        psd=psd,
        csd=csd,
        sampling_frequency=64,
        duration=2,
    )
    assert cn.psd.shape[0] == cn.N_freq
    assert cn.csd_magnitude.shape == cn.psd.shape
    assert np.all(cn.psd > 0)


def test_spectral_matrix_is_block_diag(correlated_noise_instance):
    """Verify that Cholesky spectral matrix is square, correct size, and numerically valid"""
    sm = correlated_noise_instance.spectral_matrix
    assert sm.shape[0] == correlated_noise_instance.N_freq * correlated_noise_instance.N_det
    assert sm.shape[1] == sm.shape[0]
    # Ensure Hermitian block structure by checking diagonal realness
    assert np.allclose(sm.diagonal().imag, 0, atol=1e-12)


def test_single_noise_realization_shapes(correlated_noise_instance):
    """Check if `single_noise_realization` returns finite data with correct shape"""
    ts = correlated_noise_instance.single_noise_realization(correlated_noise_instance.spectral_matrix)
    # shape: (num_detectors, N)
    assert ts.shape == (2, correlated_noise_instance.N)
    assert np.isfinite(ts).all()


def test_next_output_has_expected_length(correlated_noise_instance):
    """Check that `next()` produces a frame of the right duration and detector count"""
    ts = correlated_noise_instance.next()
    N_frame = int(correlated_noise_instance.duration * correlated_noise_instance.sampling_frequency)
    assert ts.shape == (2, N_frame)


def test_update_state_changes_start_time(correlated_noise_instance):
    """Check that `update_state()` increments start time correctly"""
    old_time = correlated_noise_instance.start_time
    correlated_noise_instance.update_state()
    assert correlated_noise_instance.start_time == old_time + correlated_noise_instance.duration


def test_adjust_filename_valid(tmp_path, correlated_noise_instance):
    """Check that `_adjust_filename` inserts detector name into proper filenames"""
    file_name = tmp_path / "E-test-123.gwf"
    new_name = correlated_noise_instance._adjust_filename(file_name, "ET1")
    assert "ET1" in new_name.stem


def test_adjust_filename_invalid(tmp_path, correlated_noise_instance):
    """Check `_adjust_filename` function"""
    bad_file = tmp_path / "bad-123.gwf"
    with pytest.raises(ValueError):
        correlated_noise_instance._adjust_filename(bad_file, "ET1")


# Integration tests
def test_generate_correlated_noise(runner, mock_config, temp_dir):
    """Run full CLI workflow and checks expected output, metadata, and checkpoint files"""

    # Write the mock config to file
    config_file = temp_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(mock_config, f)

    # Verify config is written/read consistently
    with open(config_file) as f:
        _loaded_config = yaml.safe_load(f)
    assert _loaded_config == mock_config

    # Run gwsim CLI
    result = runner.invoke(main, ["generate", str(config_file), "--metadata"])
    assert result.exit_code == 0

    # Check generated files
    output_files = list((temp_dir / "output").glob("*.gwf"))
    expected_files = mock_config["generator"]["arguments"]["max_samples"] * len(
        mock_config["generator"]["arguments"]["detector_names"]
    )
    assert len(output_files) == expected_files

    metadata_files = list((temp_dir / "metadata").glob("*.json"))
    expected_metadata_files = mock_config["generator"]["arguments"]["max_samples"]
    assert len(metadata_files) == expected_metadata_files

    # Check checkpoint
    assert (temp_dir / "checkpoint.json").exists()
