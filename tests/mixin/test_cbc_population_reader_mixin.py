"""Unit tests for CBCPopulationReaderMixin."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gwsim.mixin.cbc_population_reader import CBCPopulationReaderMixin
from gwsim.simulator.base import Simulator


class MockCBCSimulator(CBCPopulationReaderMixin, Simulator):
    """Mock simulator combining CBCPopulationReaderMixin and Simulator for integration testing."""

    def simulate(self):
        """Mock simulate method."""
        return "mock_signal"

    def _save_data(self, data, file_name, **kwargs):
        """Mock save data method."""
        pass

    @property
    def metadata(self):
        return super().metadata


class TestCBCPopulationReaderMixin:
    """Test suite for CBCPopulationReaderMixin."""

    @pytest.fixture
    def mock_cbc_h5py_data(self):
        """Fixture for mock CBC HDF5 data."""
        data = {
            "tc": [100.0, 50.0, 200.0],
            "m1": [30.0, 25.0, 35.0],  # Will be mapped to mass1
            "m2": [25.0, 20.0, 30.0],  # Will be mapped to mass2
            "z": [0.1, 0.05, 0.2],  # Redshift
        }
        attrs = {"simulation": "cbc_test", "version": 1}
        return data, attrs

    @pytest.fixture
    def mock_cbc_srcmass_data(self):
        """Fixture for mock CBC data with source masses."""
        data = {
            "tc": [100.0, 50.0],
            "m1_source": [28.0, 24.0],  # srcmass1
            "m2_source": [23.0, 19.0],  # srcmass2
            "z": [0.1, 0.05],  # Redshift
        }
        attrs = {"simulation": "cbc_srcmass", "version": 1}
        return data, attrs

    def test_init_success(self, mock_cbc_h5py_data, tmp_path):
        """Test successful initialization with valid CBC file."""
        data, attrs = mock_cbc_h5py_data

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            file_path = tmp_path / "cbc_test.h5"
            file_path.touch()

            simulator = MockCBCSimulator(file_path, start_time=0, duration=100)
            assert simulator.population_data is not None
            # Check sorting by tc
            assert list(simulator.population_data["tc"]) == [50.0, 100.0, 200.0]
            # Check parameter mapping: m1 -> mass1, m2 -> mass2, z -> redshift
            assert "mass1" in simulator.population_data.columns
            assert "mass2" in simulator.population_data.columns
            assert "redshift" in simulator.population_data.columns
            assert "m1" not in simulator.population_data.columns

    def test_parameter_name_mapping(self, mock_cbc_h5py_data, tmp_path):
        """Test CBC parameter name mapping."""
        data, attrs = mock_cbc_h5py_data

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            file_path = tmp_path / "cbc_test.h5"
            file_path.touch()

            simulator = MockCBCSimulator(file_path, start_time=0, duration=100)
            # Verify mappings - first row after sorting by tc (tc=50.0, original index 1)
            expected_mass1 = 25.0
            expected_mass2 = 20.0
            expected_redshift = 0.05
            assert simulator.population_data["mass1"].iloc[0] == expected_mass1  # m1[1] = 25.0
            assert simulator.population_data["mass2"].iloc[0] == expected_mass2  # m2[1] = 20.0
            assert simulator.population_data["redshift"].iloc[0] == expected_redshift  # z[1] = 0.05

    def test_post_process_compute_masses(self, mock_cbc_srcmass_data, tmp_path):
        """Test post-processing computes mass1 and mass2 from source masses and redshift."""
        data, attrs = mock_cbc_srcmass_data

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            file_path = tmp_path / "cbc_srcmass.h5"
            file_path.touch()

            simulator = MockCBCSimulator(file_path, start_time=0, duration=100)
            # mass1 = m1_source * (1 + z) - first after sorting: m1_source=24.0, z=0.05
            expected_mass1 = 24.0 * (1 + 0.05)  # 25.2
            expected_mass2 = 19.0 * (1 + 0.05)  # 19.95
            assert simulator.population_data["mass1"].iloc[0] == pytest.approx(expected_mass1)
            assert simulator.population_data["mass2"].iloc[0] == pytest.approx(expected_mass2)

    def test_post_process_missing_srcmass_raises_error(self, tmp_path):
        """Test that missing source masses raise ValueError."""
        data = {"tc": [100.0], "z": [0.1]}  # Missing m1_source, m2_source
        attrs = {"simulation": "cbc_error", "version": 1}

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            file_path = tmp_path / "cbc_error.h5"
            file_path.touch()

            with pytest.raises(ValueError, match="mass1 is not in population data"):
                MockCBCSimulator(file_path, start_time=0, duration=100)

    def test_url_population_file(self, mock_cbc_h5py_data):
        """Test initialization with a URL CBC population file."""
        data, attrs = mock_cbc_h5py_data
        url = "https://example.com/cbc_population.h5"

        with patch("gwsim.mixin.population_reader.download_file") as mock_download, patch("h5py.File") as mock_file:
            mock_download.return_value = "/tmp/downloaded_cbc.h5"
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            simulator = MockCBCSimulator(url, start_time=0, duration=100)
            assert simulator.population_data is not None
            mock_download.assert_called_once()

    def test_get_injection_parameters(self, mock_cbc_h5py_data, tmp_path):
        """Test getting injection parameters for CBC."""
        data, attrs = mock_cbc_h5py_data

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            file_path = tmp_path / "cbc_test.h5"
            file_path.touch()

            simulator = MockCBCSimulator(file_path, start_time=0, duration=100)
            params = simulator.get_next_injection_parameters()
            assert params is not None
            assert "tc" in params
            expected_tc = 50.0
            assert params["tc"] == expected_tc  # First after sorting
            assert "mass1" in params
            assert "mass2" in params
            assert "redshift" in params
