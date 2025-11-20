"""Unit tests for the simulate command and related functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from gwsim.cli.simulate import (
    execute_plan,
    instantiate_simulator,
    process_batch,
    restore_batch_state,
    retry_with_backoff,
    save_batch_metadata,
    simulate_command,
    update_metadata_index,
    validate_plan,
)
from gwsim.cli.utils.config import (
    Config,
    GlobalsConfig,
    SimulatorConfig,
    SimulatorOutputConfig,
)
from gwsim.cli.utils.simulation_plan import (
    SimulationBatch,
    SimulationPlan,
    create_plan_from_config,
)
from gwsim.simulator.base import Simulator


class MockSimulator(Simulator):
    """Mock simulator for testing, inheriting from Simulator base class.

    This generates simple integer data that increments with each call.
    Useful for testing state management and reproducibility.
    """

    def __init__(self, seed: int = 42, max_samples: int | None = None, **kwargs):
        """Initialize mock simulator with a seed for reproducibility.

        Args:
            seed: Random seed
            max_samples: Maximum number of samples to generate
            **kwargs: Additional arguments (absorbed by base class)
        """
        import numpy as np

        super().__init__(max_samples=max_samples, **kwargs)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._generated_data = []

    def simulate(self) -> int:
        """Generate a mock sample (random integer).

        Returns:
            A random integer
        """
        value = int(self.rng.rand() * 100)
        self._generated_data.append(value)
        return value

    def _save_data(self, data, file_name: str | Path, **kwargs) -> None:
        """Save mock data to a JSON file.

        Args:
            data: Data to save
            file_name: Output file path
            **kwargs: Additional arguments
        """
        file_name = Path(file_name)
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with file_name.open("w") as f:
            json.dump({"data": data, "counter": self.counter}, f)


class TestMockSimulator:
    """Test MockSimulator to verify it works correctly."""

    def test_mock_simulator_instantiation(self):
        """Test that MockSimulator can be instantiated."""
        sim = MockSimulator(seed=42)
        assert sim.seed == 42
        assert sim.counter == 0

    def test_mock_simulator_generate_samples(self):
        """Test that MockSimulator can generate samples."""
        sim = MockSimulator(seed=42, max_samples=3)
        samples = list(sim)
        assert len(samples) == 3
        assert all(isinstance(s, int) for s in samples)
        assert sim.counter == 3

    def test_mock_simulator_state_persistence(self):
        """Test that MockSimulator state persists across generations."""
        sim = MockSimulator(seed=42)
        _sample1 = next(sim)
        state_after_1 = sim.state.copy()

        _sample2 = next(sim)
        state_after_2 = sim.state.copy()

        # Counter should increment
        assert state_after_2["counter"] > state_after_1["counter"]

    def test_mock_simulator_save_data(self):
        """Test that MockSimulator can save data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            output_file = Path(tmpdir) / "output.json"

            sim.save_data(42, output_file)

            assert output_file.exists()
            with output_file.open() as f:
                data = json.load(f)
            assert data["data"] == 42

    def test_mock_simulator_reproducibility_with_seed(self):
        """Test that same seed produces same sequence."""
        sim1 = MockSimulator(seed=42, max_samples=5)
        samples1 = list(sim1)

        sim2 = MockSimulator(seed=42, max_samples=5)
        samples2 = list(sim2)

        assert samples1 == samples2


class TestInstantiateSimulator:
    """Test instantiate_simulator function."""

    def test_instantiate_mock_simulator(self):
        """Test instantiating MockSimulator from config."""
        config = SimulatorConfig(
            class_="tests.cli.test_simulate.MockSimulator",
            arguments={"seed": 42},
        )
        sim = instantiate_simulator(config)
        assert isinstance(sim, MockSimulator)
        assert sim.seed == 42

    def test_instantiate_simulator_invalid_class(self):
        """Test instantiating with invalid class raises error."""
        config = SimulatorConfig(
            class_="nonexistent.Class",
            arguments={},
        )
        with pytest.raises((ImportError, AttributeError)):
            instantiate_simulator(config)


class TestRestoreBatchState:
    """Test restore_batch_state function."""

    def test_restore_state_with_snapshot(self):
        """Test restoring state from batch metadata."""
        sim = MockSimulator(seed=42, max_samples=10)
        next(sim)  # Advance to counter=1

        state_snapshot = {"counter": 5}
        batch = SimulationBatch(
            simulator_name="mock",
            simulator_config=SimulatorConfig(class_="MockSimulator"),
            globals_config=GlobalsConfig(),
            batch_index=0,
            pre_batch_state=state_snapshot,
        )

        restore_batch_state(sim, batch)
        assert sim.counter == 5

    def test_restore_state_without_snapshot(self):
        """Test that missing snapshot doesn't cause error."""
        sim = MockSimulator(seed=42)
        next(sim)

        batch = SimulationBatch(
            simulator_name="mock",
            simulator_config=SimulatorConfig(class_="MockSimulator"),
            globals_config=GlobalsConfig(),
            batch_index=0,
            pre_batch_state=None,
        )

        # Should not raise
        restore_batch_state(sim, batch)
        assert sim.counter == 1  # Unchanged


class TestUpdateMetadataIndex:
    """Test update_metadata_index function."""

    def test_create_new_index(self):
        """Test creating a new metadata index file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)
            output_files = [
                Path(tmpdir) / "H1-1234567890-1024.gwf",
                Path(tmpdir) / "L1-1234567890-1024.gwf",
            ]

            update_metadata_index(metadata_dir, output_files, "signal-0.metadata.yaml")

            index_file = metadata_dir / "index.yaml"
            assert index_file.exists()

            with index_file.open() as f:
                index = yaml.safe_load(f)

            assert "H1-1234567890-1024.gwf" in index
            assert "L1-1234567890-1024.gwf" in index
            assert index["H1-1234567890-1024.gwf"] == "signal-0.metadata.yaml"
            assert index["L1-1234567890-1024.gwf"] == "signal-0.metadata.yaml"

    def test_update_existing_index(self):
        """Test updating an existing metadata index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)

            # Create initial index
            output_files_1 = [Path(tmpdir) / "H1-batch0.gwf"]
            update_metadata_index(metadata_dir, output_files_1, "signal-0.metadata.yaml")

            # Update with more files
            output_files_2 = [Path(tmpdir) / "L1-batch1.gwf"]
            update_metadata_index(metadata_dir, output_files_2, "signal-1.metadata.yaml")

            # Verify both entries exist
            index_file = metadata_dir / "index.yaml"
            with index_file.open() as f:
                index = yaml.safe_load(f)

            assert len(index) == 2
            assert index["H1-batch0.gwf"] == "signal-0.metadata.yaml"
            assert index["L1-batch1.gwf"] == "signal-1.metadata.yaml"

    def test_index_enables_quick_lookup(self):
        """Test that the index enables quick metadata lookup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)

            # Generate metadata for multiple batches
            output_files_batch0 = [
                Path(tmpdir) / "H1-0.gwf",
                Path(tmpdir) / "L1-0.gwf",
                Path(tmpdir) / "V1-0.gwf",
            ]
            update_metadata_index(metadata_dir, output_files_batch0, "detector-0.metadata.yaml")

            output_files_batch1 = [
                Path(tmpdir) / "H1-1.gwf",
                Path(tmpdir) / "L1-1.gwf",
                Path(tmpdir) / "V1-1.gwf",
            ]
            update_metadata_index(metadata_dir, output_files_batch1, "detector-1.metadata.yaml")

            # Load index and verify quick lookup
            index_file = metadata_dir / "index.yaml"
            with index_file.open() as f:
                index = yaml.safe_load(f)

            # Should be able to find metadata for any data file in O(1)
            assert index["H1-0.gwf"] == "detector-0.metadata.yaml"
            assert index["L1-1.gwf"] == "detector-1.metadata.yaml"
            assert index["V1-0.gwf"] == "detector-0.metadata.yaml"


class TestSaveBatchMetadata:
    """Test save_batch_metadata function."""

    def test_save_batch_metadata_creates_file(self):
        """Test that metadata file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            next(sim)

            batch = SimulationBatch(
                simulator_name="mock",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    arguments={"seed": 42},
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            metadata_dir = Path(tmpdir)
            output_files = [Path(tmpdir) / "output.json"]
            save_batch_metadata(sim, batch, metadata_dir, output_files)

            metadata_file = metadata_dir / "mock-0.metadata.yaml"
            assert metadata_file.exists()

    def test_save_batch_metadata_contains_state_and_files(self):
        """Test that metadata contains simulator state and output file list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            next(sim)

            batch = SimulationBatch(
                simulator_name="test",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    arguments={"seed": 42},
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            metadata_dir = Path(tmpdir)
            output_files = [
                Path(tmpdir) / "H1-1234567890-1024.gwf",
                Path(tmpdir) / "L1-1234567890-1024.gwf",
                Path(tmpdir) / "V1-1234567890-1024.gwf",
            ]
            save_batch_metadata(sim, batch, metadata_dir, output_files)

            metadata_file = metadata_dir / "test-0.metadata.yaml"
            with metadata_file.open() as f:
                metadata = yaml.safe_load(f)

            assert "pre_batch_state" in metadata
            assert metadata["simulator_name"] == "test"
            assert metadata["batch_index"] == 0
            assert "output_files" in metadata
            assert len(metadata["output_files"]) == 3
            assert "H1-1234567890-1024.gwf" in metadata["output_files"]
            assert "L1-1234567890-1024.gwf" in metadata["output_files"]
            assert "V1-1234567890-1024.gwf" in metadata["output_files"]

    def test_save_batch_metadata_updates_index(self):
        """Test that save_batch_metadata updates the metadata index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            next(sim)

            batch = SimulationBatch(
                simulator_name="test",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    arguments={"seed": 42},
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            metadata_dir = Path(tmpdir)
            output_files = [
                Path(tmpdir) / "H1-1234567890-1024.gwf",
                Path(tmpdir) / "L1-1234567890-1024.gwf",
            ]
            save_batch_metadata(sim, batch, metadata_dir, output_files)

            # Verify index was created and contains entries
            index_file = metadata_dir / "index.yaml"
            assert index_file.exists()

            with index_file.open() as f:
                index = yaml.safe_load(f)

            assert index["H1-1234567890-1024.gwf"] == "test-0.metadata.yaml"
            assert index["L1-1234567890-1024.gwf"] == "test-0.metadata.yaml"


class TestRetryWithBackoff:
    """Test retry_with_backoff function."""

    def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = 0

        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = retry_with_backoff(success_func, max_retries=3)
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """Test successful execution after retries."""
        call_count = 0

        def retry_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("Simulated I/O error")
            return "success"

        result = retry_with_backoff(retry_func, max_retries=3, initial_delay=0.01)
        assert result == "success"
        assert call_count == 3

    def test_retry_all_attempts_fail(self):
        """Test that exception is raised after all retries fail."""
        call_count = 0

        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            retry_with_backoff(always_fails, max_retries=2, initial_delay=0.01)

        # Should attempt 3 times (initial + 2 retries)
        assert call_count == 3

    def test_retry_exponential_backoff(self):
        """Test that backoff delays increase exponentially."""
        import time

        call_times = []

        def track_calls():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise OSError("Retrying")
            return "success"

        start = time.time()
        result = retry_with_backoff(track_calls, max_retries=2, initial_delay=0.05, backoff_factor=2.0)
        total_time = time.time() - start

        assert result == "success"
        assert len(call_times) == 3
        # Total time should be at least: 0.05 + 0.1 = 0.15 seconds
        assert total_time >= 0.14  # Allow some margin for execution time


class TestValidatePlan:
    """Test validate_plan function."""

    def test_validate_empty_plan_raises_error(self):
        """Test that empty plan fails validation."""
        plan = SimulationPlan()
        with pytest.raises(ValueError, match="no batches"):
            validate_plan(plan)

    def test_validate_plan_with_valid_batch(self):
        """Test that valid plan passes validation."""
        batch = SimulationBatch(
            simulator_name="test",
            simulator_config=SimulatorConfig(
                class_="MockSimulator",
                output=SimulatorOutputConfig(file_name="output.json"),
            ),
            globals_config=GlobalsConfig(),
            batch_index=0,
        )
        plan = SimulationPlan()
        plan.add_batch(batch)

        # Should not raise
        validate_plan(plan)

    def test_validate_plan_missing_file_name(self):
        """Test that batch without file_name fails validation."""
        batch = SimulationBatch(
            simulator_name="test",
            simulator_config=SimulatorConfig(
                class_="MockSimulator",
                output=SimulatorOutputConfig(file_name=""),
            ),
            globals_config=GlobalsConfig(),
            batch_index=0,
        )
        plan = SimulationPlan()
        plan.add_batch(batch)

        with pytest.raises(ValueError, match="file_name"):
            validate_plan(plan)


class TestProcessBatch:
    """Test process_batch function."""

    def test_process_batch_saves_data(self):
        """Test that process_batch saves data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            output_dir = Path(tmpdir)

            batch = SimulationBatch(
                simulator_name="test",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    output=SimulatorOutputConfig(file_name="output.json"),
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            batch_data = 42
            output_files = process_batch(sim, batch_data, batch, output_dir, overwrite=True)

            assert len(output_files) == 1
            assert output_files[0].exists()

    def test_process_batch_respects_overwrite_flag(self):
        """Test that overwrite flag is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            output_dir = Path(tmpdir)

            batch = SimulationBatch(
                simulator_name="test",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    output=SimulatorOutputConfig(file_name="output.json"),
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            # First save
            process_batch(sim, 1, batch, output_dir, overwrite=True)

            # Second save without overwrite should raise
            with pytest.raises(FileExistsError):
                process_batch(sim, 2, batch, output_dir, overwrite=False)

    def test_process_batch_returns_list_of_paths(self):
        """Test that process_batch returns list of Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = MockSimulator(seed=42)
            output_dir = Path(tmpdir)

            batch = SimulationBatch(
                simulator_name="test",
                simulator_config=SimulatorConfig(
                    class_="MockSimulator",
                    output=SimulatorOutputConfig(file_name="output.json"),
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            output_files = process_batch(sim, 42, batch, output_dir, overwrite=True)

            assert isinstance(output_files, list)
            assert all(isinstance(f, Path) for f in output_files)


class TestExecutePlan:
    """Test execute_plan function."""

    def test_execute_plan_single_simulator(self):
        """Test executing a plan with one simulator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            metadata_dir = Path(tmpdir) / "metadata"

            batch = SimulationBatch(
                simulator_name="mock",
                simulator_config=SimulatorConfig(
                    class_="tests.cli.test_simulate.MockSimulator",
                    arguments={"seed": 42},
                    output=SimulatorOutputConfig(file_name="data.json"),
                ),
                globals_config=GlobalsConfig(),
                batch_index=0,
            )

            plan = SimulationPlan()
            plan.add_batch(batch)

            execute_plan(plan, output_dir, metadata_dir, overwrite=True)

            # Verify output file exists
            assert (output_dir / "data.json").exists()
            # Verify metadata file exists
            assert (metadata_dir / "mock-0.metadata.yaml").exists()

    def test_execute_plan_multiple_batches_same_simulator(self):
        """Test executing multiple batches for same simulator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            metadata_dir = Path(tmpdir) / "metadata"

            plan = SimulationPlan()
            for i in range(3):
                batch = SimulationBatch(
                    simulator_name="mock",
                    simulator_config=SimulatorConfig(
                        class_="tests.cli.test_simulate.MockSimulator",
                        arguments={"seed": 42},
                        output=SimulatorOutputConfig(file_name=f"batch_{i}.json"),
                    ),
                    globals_config=GlobalsConfig(),
                    batch_index=i,
                )
                plan.add_batch(batch)

            execute_plan(plan, output_dir, metadata_dir, overwrite=True)

            # Verify all output files exist
            for i in range(3):
                assert (output_dir / f"batch_{i}.json").exists()

            # Verify all metadata files exist
            for i in range(3):
                assert (metadata_dir / f"mock-{i}.metadata.yaml").exists()

    def test_execute_plan_maintains_simulator_state(self):
        """Test that simulator state persists across batches.

        This is critical: the simulator should be created once and
        generate multiple batches, with state accumulating.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            metadata_dir = Path(tmpdir) / "metadata"

            plan = SimulationPlan()
            for i in range(2):
                batch = SimulationBatch(
                    simulator_name="mock",
                    simulator_config=SimulatorConfig(
                        class_="tests.cli.test_simulate.MockSimulator",
                        arguments={"seed": 42},
                        output=SimulatorOutputConfig(file_name=f"batch_{i}.json"),
                    ),
                    globals_config=GlobalsConfig(),
                    batch_index=i,
                )
                plan.add_batch(batch)

            execute_plan(plan, output_dir, metadata_dir, overwrite=True)

            # Load metadata files and check counter progression
            metadata_0 = yaml.safe_load((metadata_dir / "mock-0.metadata.yaml").open())
            metadata_1 = yaml.safe_load((metadata_dir / "mock-1.metadata.yaml").open())

            # Batch 1 should have higher counter than batch 0
            counter_0 = metadata_0["pre_batch_state"]["counter"]
            counter_1 = metadata_1["pre_batch_state"]["counter"]
            assert counter_1 > counter_0


class TestCreateSimulationPlanFromConfig:
    """Test creating simulation plans from configs."""

    def test_create_plan_single_simulator(self):
        """Test creating a plan from config with one simulator."""
        config = Config(
            globals=GlobalsConfig(
                working_directory=".",
                sampling_frequency=4096,
            ),
            simulators={
                "mock": SimulatorConfig(
                    class_="tests.cli.test_simulate.MockSimulator",
                    arguments={"seed": 42},
                    output=SimulatorOutputConfig(file_name="output.json"),
                )
            },
        )

        plan = create_plan_from_config(config, Path("checkpoints"))

        assert plan.total_batches == 1
        assert len(plan.batches) == 1
        assert plan.batches[0].simulator_name == "mock"

    def test_create_plan_multiple_simulators(self):
        """Test creating a plan with multiple simulators."""
        config = Config(
            globals=GlobalsConfig(),
            simulators={
                "mock1": SimulatorConfig(
                    class_="tests.cli.test_simulate.MockSimulator",
                    arguments={"seed": 1},
                    output=SimulatorOutputConfig(file_name="out1.json"),
                ),
                "mock2": SimulatorConfig(
                    class_="tests.cli.test_simulate.MockSimulator",
                    arguments={"seed": 2},
                    output=SimulatorOutputConfig(file_name="out2.json"),
                ),
            },
        )

        plan = create_plan_from_config(config, Path("checkpoints"))

        assert plan.total_batches == 2
        simulator_names = {b.simulator_name for b in plan.batches}
        assert simulator_names == {"mock1", "mock2"}


class TestSimulateCommandIntegration:
    """Integration tests for the simulate_command."""

    def test_simulate_command_with_config_file(self):
        """Test simulate command with a real config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create config file
            config = Config(
                globals=GlobalsConfig(
                    working_directory=str(tmpdir_path),
                    output_directory="output",
                    metadata_directory="metadata",
                ),
                simulators={
                    "mock": SimulatorConfig(
                        class_="tests.cli.test_simulate.MockSimulator",
                        arguments={"seed": 42},
                        output=SimulatorOutputConfig(file_name="data.json"),
                    )
                },
            )

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                # Convert config to dict with aliases for YAML
                config_dict = config.model_dump(by_alias=True, exclude_none=True)
                yaml.safe_dump(config_dict, f)

            # Run simulate command
            simulate_command(str(config_file), overwrite=True, metadata=True)

            # Verify output structure
            # Paths should be resolved relative to working_directory
            output_dir = tmpdir_path / "output"
            metadata_dir = tmpdir_path / "metadata"

            assert (output_dir / "data.json").exists(), f"Output file not found at {output_dir / 'data.json'}"
            assert (
                metadata_dir / "mock-0.metadata.yaml"
            ).exists(), f"Metadata file not found at {metadata_dir / 'mock-0.metadata.yaml'}"
            assert (metadata_dir / "index.yaml").exists(), f"Index file not found at {metadata_dir / 'index.yaml'}"

    def test_simulate_command_data_correctness(self):
        """Test that simulate command produces correct data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            config = Config(
                globals=GlobalsConfig(
                    working_directory=str(tmpdir_path),
                    output_directory="output",
                ),
                simulators={
                    "mock": SimulatorConfig(
                        class_="tests.cli.test_simulate.MockSimulator",
                        arguments={"seed": 42, "max_samples": 1},
                        output=SimulatorOutputConfig(file_name="data.json"),
                    )
                },
            )

            config_file = tmpdir_path / "config.yaml"
            with config_file.open("w") as f:
                config_dict = config.model_dump(by_alias=True, exclude_none=True)
                yaml.safe_dump(config_dict, f)

            simulate_command(str(config_file), overwrite=True, metadata=False)

            # Verify data format
            output_file = tmpdir_path / "output" / "data.json"
            with output_file.open() as f:
                data = json.load(f)

            assert "data" in data
            assert "counter" in data
