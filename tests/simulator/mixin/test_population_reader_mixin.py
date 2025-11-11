"""Unit tests for PopulationReaderMixin."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gwsim.simulator.base import Simulator
from gwsim.simulator.mixin.population_reader import PopulationReaderMixin


class MockPopulationSimulator(PopulationReaderMixin, Simulator):
    """Mock simulator combining PopulationReaderMixin and Simulator for integration testing."""

    def simulate(self):
        """Mock simulate method."""
        return "mock_signal"

    def _save_data(self, data, file_name, **kwargs):
        """Mock save data method."""
        pass


class TestPopulationReaderMixin:
    """Test suite for PopulationReaderMixin."""

    @pytest.fixture
    def mock_h5py_data(self):
        """Fixture for mock HDF5 data."""
        data = {"tc": [100.0, 50.0, 200.0], "mass1": [30.0, 25.0, 35.0], "mass2": [25.0, 20.0, 30.0]}
        attrs = {"simulation": "test", "version": 1}
        return data, attrs

    def test_init_success(self, mock_h5py_data):
        """Test successful initialization with valid file and parameters."""
        data, attrs = mock_h5py_data

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                file_path = f.name

            try:
                simulator = MockPopulationSimulator(file_path, population_file_type="pycbc", start_time=0, duration=100)
                assert simulator.population_data is not None
                # Check that data is sorted by 'tc'
                assert list(simulator.population_data["tc"]) == [50.0, 100.0, 200.0]
                assert simulator.population_file == Path(file_path)
                assert simulator.population_file_type == "pycbc"
            finally:
                os.unlink(file_path)

    def test_init_file_not_found(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match=r"Population file .* does not exist"):
            MockPopulationSimulator("nonexistent.h5", population_file_type="pycbc", start_time=0, duration=100)

    def test_metadata_property(self, mock_h5py_data):
        """Test metadata property returns correct information."""
        data, attrs = mock_h5py_data

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                (k, type("MockDataset", (), {"__getitem__": lambda self, idx, k=k: data[k]})()) for k in data
            ]
            mock_f.attrs.items.return_value = attrs.items()

            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                file_path = f.name

            try:
                simulator = MockPopulationSimulator(file_path, population_file_type="pycbc", start_time=0, duration=100)
                metadata = simulator.metadata
                assert metadata["population_file"] == file_path
                assert metadata["population_file_type"] == "pycbc"
                assert metadata["simulation"] == "test"
                assert metadata["version"] == 1
            finally:
                os.unlink(file_path)
