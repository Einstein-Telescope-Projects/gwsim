"""Unit tests for PopulationReaderMixin."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from gwsim.mixin.population_reader import PopulationReaderMixin
from gwsim.simulator.base import Simulator


class MockPopulationSimulator(PopulationReaderMixin, Simulator):
    """Mock simulator combining PopulationReaderMixin and Simulator for integration testing."""

    def simulate(self):
        """Mock simulate method."""
        return "mock_signal"

    def _save_data(self, data, file_name, **kwargs):
        """Mock save data method."""
        pass

    @property
    def metadata(self):
        return super().metadata


class TestPopulationReaderMixin:
    """Test suite for PopulationReaderMixin."""

    @pytest.fixture
    def mock_h5py_data(self):
        """Fixture for mock HDF5 data."""
        data = {"tc": [100.0, 50.0, 200.0], "mass1": [30.0, 25.0, 35.0], "mass2": [25.0, 20.0, 30.0]}
        attrs = {"simulation": "test", "version": 1}
        return data, attrs

    def test_init_success(self, mock_h5py_data, tmp_path):
        """Test successful initialization with valid file and parameters."""
        data, attrs = mock_h5py_data

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            file_path = tmp_path / "test.h5"
            file_path.touch()  # Create empty file for the check

            simulator = MockPopulationSimulator(file_path, population_sort_by="tc", start_time=0, duration=100)
            assert simulator.population_data is not None
            # Check that data is sorted by 'tc'
            assert list(simulator.population_data["tc"]) == [50.0, 100.0, 200.0]
            assert simulator.population_file == file_path

    def test_init_file_not_found(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match=r"Population file .* does not exist"):
            MockPopulationSimulator("nonexistent.h5", start_time=0, duration=100)

    def test_metadata_property(self, mock_h5py_data, tmp_path):
        """Test metadata property returns correct information."""
        data, attrs = mock_h5py_data

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            file_path = tmp_path / "test.h5"
            file_path.touch()

            simulator = MockPopulationSimulator(file_path, population_sort_by="tc", start_time=0, duration=100)
            metadata = simulator.metadata
            assert metadata["population_reader"]["arguments"]["population_file"] == str(file_path)
            assert metadata["population_reader"]["population_metadata"]["simulation"] == "test"
            assert metadata["population_reader"]["population_metadata"]["version"] == 1

    def test_url_population_file(self, mock_h5py_data):
        """Test initialization with a URL population file."""
        data, attrs = mock_h5py_data
        url = "https://example.com/population.h5"

        with patch("gwsim.mixin.population_reader.download_file") as mock_download, patch("h5py.File") as mock_file:
            mock_download.return_value = "/tmp/downloaded.h5"
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            simulator = MockPopulationSimulator(url, population_sort_by="tc", start_time=0, duration=100)
            assert simulator.population_data is not None
            mock_download.assert_called_once()

    def test_parameter_name_mapping(self, mock_h5py_data, tmp_path):
        """Test parameter name mapping."""
        data, attrs = mock_h5py_data
        data["m1"] = data.pop("mass1")  # Rename for testing

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            file_path = tmp_path / "test.h5"
            file_path.touch()

            mapper = {"m1": "mass1"}
            simulator = MockPopulationSimulator(
                file_path, population_parameter_name_mapper=mapper, population_sort_by="tc", start_time=0, duration=100
            )
            assert "mass1" in simulator.population_data.columns
            assert "m1" not in simulator.population_data.columns

    def test_csv_population_file(self, tmp_path):
        """Test reading CSV population file."""
        csv_data = "tc,mass1,mass2\n100.0,30.0,25.0\n50.0,25.0,20.0\n200.0,35.0,30.0"

        file_path = tmp_path / "test.csv"
        file_path.write_text(csv_data)

        simulator = MockPopulationSimulator(file_path, population_sort_by="tc", start_time=0, duration=100)
        assert simulator.population_data is not None
        assert list(simulator.population_data["tc"]) == [50.0, 100.0, 200.0]

    def test_cache_dir_setting(self, tmp_path):
        """Test setting custom cache directory."""
        custom_cache = tmp_path / "custom_cache"

        file_path = tmp_path / "test.h5"
        file_path.touch()

        with patch.object(MockPopulationSimulator, "_population_read_population_file", return_value=pd.DataFrame()):
            simulator = MockPopulationSimulator(
                file_path, population_cache_dir=custom_cache, start_time=0, duration=100
            )
            assert simulator.population_cache_dir == custom_cache

    def test_population_data_sorting(self, mock_h5py_data, tmp_path):
        """Test population data sorting."""
        data, attrs = mock_h5py_data

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            file_path = tmp_path / "test.h5"
            file_path.touch()

            simulator = MockPopulationSimulator(file_path, population_sort_by="tc", start_time=0, duration=100)
            assert list(simulator.population_data["tc"]) == [50.0, 100.0, 200.0]

    def test_get_injection_parameters(self, mock_h5py_data, tmp_path):
        """Test getting injection parameters."""
        data, attrs = mock_h5py_data

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            file_path = tmp_path / "test.h5"
            file_path.touch()

            simulator = MockPopulationSimulator(file_path, population_sort_by="tc", start_time=0, duration=100)
            params = simulator.get_next_injection_parameters()
            assert params is not None
            assert "tc" in params
            expected_tc = 50.0
            assert params["tc"] == expected_tc  # First after sorting

    def test_get_injection_parameter_keys(self, mock_h5py_data, tmp_path):
        """Test getting injection parameter keys."""
        data, attrs = mock_h5py_data

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            file_path = tmp_path / "test.h5"
            file_path.touch()

            simulator = MockPopulationSimulator(file_path, start_time=0, duration=100)
            keys = simulator.get_injection_parameter_keys()
            assert set(keys) == {"tc", "mass1", "mass2"}
