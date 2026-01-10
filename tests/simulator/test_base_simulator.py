"""Unit tests for the Simulator base class."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from gwsim.simulator.base import Simulator


class MockSimulator(Simulator):
    """Mock simulator for testing base functionality."""

    def simulate(self) -> int:
        """Return the current counter value."""
        return self.counter

    def _save_data(self, data: int, file_name: str | Path | np.ndarray[Any, np.dtype[np.object_]], **kwargs) -> None:
        """Save data as JSON."""
        # Handle array of file names for multi-file saves
        if isinstance(file_name, np.ndarray):
            # For arrays, recursively save each element with its corresponding data
            flat_files = file_name.flatten()

            if isinstance(data, np.ndarray):
                flat_data = data.flatten()
                # Ensure data and file counts match
                if len(flat_data) != len(flat_files):
                    raise ValueError(f"Data size ({len(flat_data)}) must match file count ({len(flat_files)})")
            else:
                # Single data value for single file
                if len(flat_files) != 1:
                    raise ValueError(f"Single data value requires single file, got {len(flat_files)} files")
                flat_data = [data]

            for f, d in zip(flat_files, flat_data, strict=False):
                # Convert numpy scalars to Python native types for YAML serialization
                data = d.item() if isinstance(d, np.generic) else d
                with Path(f).open("w", encoding="utf-8") as fp:
                    yaml.safe_dump(data, fp)
        else:
            # Handle single file
            # Convert numpy scalars to Python native types for YAML serialization
            if isinstance(data, np.generic):
                data = data.item()
            with Path(file_name).open("w", encoding="utf-8") as f:
                yaml.safe_dump(data, f)


@pytest.fixture
def simulator() -> MockSimulator:
    """Fixture for a basic simulator instance."""
    return MockSimulator(max_samples=5)


@pytest.fixture
def simulator_with_attrs() -> MockSimulator:
    """Fixture for simulator with additional attributes for template testing."""
    sim = MockSimulator(max_samples=10)
    sim.detector = np.array(["H1", "L1"])
    sim.duration = np.array([4, 8])
    return sim


class TestSimulatorInitialization:
    """Test Simulator initialization."""

    def test_init_with_max_samples(self):
        """Test initialization with max_samples."""
        max_samples = 10
        sim = MockSimulator(max_samples=max_samples)
        assert sim.max_samples == max_samples

    def test_init_with_none_max_samples(self):
        """Test initialization with None max_samples (infinite)."""
        sim = MockSimulator(max_samples=None)
        assert sim.max_samples == np.inf

    def test_init_without_max_samples(self):
        """Test initialization without max_samples defaults to infinite."""
        sim = MockSimulator()
        assert sim.max_samples == np.inf


class TestSimulatorProperties:
    """Test Simulator properties."""

    def test_max_samples_getter_setter(self, simulator: MockSimulator):
        """Test max_samples property."""
        expected_max_samples = 5
        assert simulator.max_samples == expected_max_samples
        new_max_samples = 10
        simulator.max_samples = new_max_samples
        assert simulator.max_samples == new_max_samples

    def test_max_samples_setter_validation(self, simulator: MockSimulator):
        """Test max_samples setter validation."""
        with pytest.raises(ValueError, match="Max samples cannot be negative"):
            simulator.max_samples = -1

    def test_state_property(self, simulator: MockSimulator):
        """Test state property includes counter."""
        state = simulator.state
        assert "counter" in state
        assert state["counter"] == 0

    def test_state_setter(self, simulator: MockSimulator):
        """Test state setter."""
        counter = 5
        simulator.state = {"counter": counter}
        assert simulator.counter == counter

    def test_state_setter_invalid_key(self, simulator: MockSimulator):
        """Test state setter with invalid key."""
        with pytest.raises(ValueError, match="not registered as a state attribute"):
            simulator.state = {"invalid_key": 42}

    def test_metadata_property(self, simulator: MockSimulator):
        """Test metadata property."""
        metadata = simulator.metadata
        expected_max_samples = 5
        assert metadata["max_samples"] == expected_max_samples
        assert metadata["counter"] == 0
        assert "version" in metadata


class TestSimulatorIterator:
    """Test Simulator iterator protocol."""

    def test_iterator_protocol(self, simulator: MockSimulator):
        """Test __iter__ and __next__."""
        iterator = iter(simulator)
        assert iterator is simulator

        expected_number_of_samples = 5

        # Generate samples
        samples = list(simulator)
        assert len(samples) == expected_number_of_samples
        assert samples == [0, 1, 2, 3, 4]

        # Counter should be updated
        assert simulator.counter == expected_number_of_samples

    def test_iterator_stopiteration(self, simulator: MockSimulator):
        """Test StopIteration when max_samples reached."""
        samples = []
        expected_number_of_samples = 5

        for sample in simulator:
            samples.append(sample)
        assert len(samples) == expected_number_of_samples

        # Next call should raise StopIteration
        with pytest.raises(StopIteration):
            next(simulator)

    def test_iterator_infinite(self):
        """Test iterator with infinite max_samples."""
        sim = MockSimulator(max_samples=None)
        iterator = iter(sim)
        # Generate a few samples
        n_iter = 3
        for i in range(n_iter):
            assert next(iterator) == i
        assert sim.counter == n_iter


class TestSimulatorFileIO:
    """Test Simulator file I/O methods."""

    def test_save_state(self, simulator: MockSimulator):
        """Test save_state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "state.yaml"

            simulator.save_state(file_path)
            assert file_path.exists()

            with file_path.open(encoding="utf-8") as f:
                state = yaml.safe_load(f)
            assert state["counter"] == 0

    def test_save_state_invalid_extension(self, simulator: MockSimulator):
        """Test save_state with invalid file extension."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            simulator.save_state("test.txt")

    def test_save_state_overwrite_false(self, simulator: MockSimulator):
        """Test save_state overwrite=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "state.yaml"

            simulator.save_state(file_path)
            with pytest.raises(FileExistsError):
                simulator.save_state(file_path, overwrite=False)

    def test_save_state_overwrite_true(self, simulator: MockSimulator):
        """Test save_state overwrite=True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "state.yaml"

            simulator.save_state(file_path)
            simulator.counter = 1
            simulator.save_state(file_path, overwrite=True)

            with file_path.open(encoding="utf-8") as f:
                state = yaml.safe_load(f)
            assert state["counter"] == 1

    def test_load_state(self, simulator: MockSimulator):
        """Test load_state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "state.yaml"

            # Save state
            simulator.counter = 3
            simulator.save_state(file_path)

            # Load into new simulator
            new_sim = MockSimulator()
            new_sim.load_state(file_path)
            assert new_sim.counter == simulator.counter

    def test_load_state_file_not_found(self, simulator: MockSimulator):
        """Test load_state with non-existent file."""
        with pytest.raises(FileNotFoundError):
            simulator.load_state("nonexistent.yaml")

    def test_load_state_invalid_extension(self, simulator: MockSimulator):
        """Test load_state with invalid extension."""

        with pytest.raises(ValueError, match="Unsupported file format"):
            simulator.load_state("test.txt")

    def test_save_metadata(self, simulator: MockSimulator):
        """Test save_metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "metadata.yaml"

            simulator.save_metadata(file_path)
            assert file_path.exists()

            with file_path.open(encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
            assert metadata["counter"] == 0
            assert "version" in metadata


class TestSimulatorSaveData:
    """Test Simulator save_data method."""

    def test_save_data_single_file(self, simulator: MockSimulator):
        """Test save_data with single file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "data.yaml"

            data = 42
            simulator.save_data(data, file_path)
            assert file_path.exists()

            with file_path.open(encoding="utf-8") as f:
                loaded_data = yaml.safe_load(f)
            assert loaded_data == data

    def test_save_data_with_template(self, simulator: MockSimulator):
        """Test save_data with template resolution."""
        simulator.detector = "H1"
        simulator.duration = 4

        with tempfile.TemporaryDirectory() as temp_dir:
            template = f"{temp_dir}/{{{{detector}}}}-{{{{duration}}}}.yaml"
            data = 100
            simulator.save_data(data, template)

            expected_path = Path(f"{temp_dir}/H1-4.yaml")
            assert expected_path.exists()

            with expected_path.open(encoding="utf-8") as f:
                loaded_data = yaml.safe_load(f)
            assert loaded_data == data

    def test_save_data_array_files(self, simulator_with_attrs: MockSimulator):
        """Test save_data with array of files."""
        sim = simulator_with_attrs
        # data shape should match file_name array shape (2, 2)
        data = np.array([[1, 2], [3, 4]], dtype=object)

        with tempfile.TemporaryDirectory() as temp_dir:
            template = f"{temp_dir}/{{{{detector}}}}-{{{{duration}}}}.yaml"
            sim.save_data(data, template)

            # Check files exist
            assert Path(f"{temp_dir}/H1-4.yaml").exists()
            assert Path(f"{temp_dir}/H1-8.yaml").exists()
            assert Path(f"{temp_dir}/L1-4.yaml").exists()
            assert Path(f"{temp_dir}/L1-8.yaml").exists()

            # Check contents
            with open(f"{temp_dir}/H1-4.yaml", encoding="utf-8") as f:
                assert yaml.safe_load(f) == 1
            with open(f"{temp_dir}/L1-8.yaml", encoding="utf-8") as f:
                assert yaml.safe_load(f) == data[1, 1]

    def test_save_data_array_shape_mismatch(self, simulator_with_attrs: MockSimulator):
        """Test save_data with mismatched data shape raises ValueError."""
        sim = simulator_with_attrs
        # Wrong shape: should be (2, 2) but is (2,)
        data = np.array([1, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            template = f"{temp_dir}/{{{{detector}}}}-{{{{duration}}}}.yaml"
            with pytest.raises(ValueError, match=r"Data size .* must match file count"):
                sim.save_data(data, template)

    def test_save_data_overwrite_false(self, simulator: MockSimulator):
        """Test save_data overwrite=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "data.yaml"

            simulator.save_data(1, file_path)
            with pytest.raises(FileExistsError):
                simulator.save_data(2, file_path, overwrite=False)

    def test_save_data_overwrite_true(self, simulator: MockSimulator):
        """Test save_data overwrite=True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_data = 2

            file_path = Path(temp_dir) / "data.yaml"

            simulator.save_data(1, file_path)
            simulator.save_data(new_data, file_path, overwrite=True)

            with open(file_path, encoding="utf-8") as f:
                loaded_data = yaml.safe_load(f)
            assert loaded_data == new_data
