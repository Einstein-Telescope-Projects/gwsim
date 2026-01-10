"""Unit tests for simulation_plan module."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from gwsim.cli.utils.config import Config, GlobalsConfig, SimulatorConfig
from gwsim.cli.utils.simulation_plan import (
    SimulationBatch,
    SimulationPlan,
    create_batch_metadata,
    create_plan_from_config,
    create_plan_from_metadata,
    merge_plans,
    parse_batch_metadata,
)

# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def globals_config() -> GlobalsConfig:
    """Create a basic GlobalsConfig for testing."""
    return GlobalsConfig(
        sampling_frequency=16384,
        duration=1.0,
    )


@pytest.fixture
def simulator_config() -> SimulatorConfig:
    """Create a basic SimulatorConfig for testing."""
    return SimulatorConfig(
        **{"class": "WhiteNoise"},
        arguments={"seed": 42},
    )


@pytest.fixture
def simulation_batch(
    globals_config: GlobalsConfig,
    simulator_config: SimulatorConfig,
) -> SimulationBatch:
    """Create a basic SimulationBatch for testing."""
    return SimulationBatch(
        simulator_name="noise",
        simulator_config=simulator_config,
        globals_config=globals_config,
        batch_index=0,
        source="config",
    )


# ============================================================================
# SimulationBatch Tests
# ============================================================================


class TestSimulationBatchCreation:
    """Tests for SimulationBatch creation and validation."""

    def test_create_batch_with_valid_params(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test creating a batch with valid parameters."""
        batch = SimulationBatch(
            simulator_name="noise",
            simulator_config=simulator_config,
            globals_config=globals_config,
            batch_index=0,
        )
        assert batch.simulator_name == "noise"
        assert batch.batch_index == 0
        assert batch.source == "config"
        assert batch.metadata_file is None
        assert batch.pre_batch_state is None

    def test_create_batch_with_state_snapshot(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test creating a batch with a state snapshot."""
        state = {"rng_state": [1, 2, 3], "counter": 5}
        batch = SimulationBatch(
            simulator_name="signal",
            simulator_config=simulator_config,
            globals_config=globals_config,
            batch_index=2,
            pre_batch_state=state,
            source="metadata_state",
        )
        assert batch.has_state_snapshot()
        assert batch.pre_batch_state == state
        assert batch.source == "metadata_state"

    def test_batch_invalid_simulator_name(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that empty simulator_name raises ValueError."""
        with pytest.raises(ValueError, match="simulator_name must not be empty"):
            SimulationBatch(
                simulator_name="",
                simulator_config=simulator_config,
                globals_config=globals_config,
                batch_index=0,
            )

    def test_batch_negative_batch_index(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that negative batch_index raises ValueError."""
        with pytest.raises(ValueError, match="batch_index must be non-negative"):
            SimulationBatch(
                simulator_name="noise",
                simulator_config=simulator_config,
                globals_config=globals_config,
                batch_index=-1,
            )


class TestSimulationBatchMethods:
    """Tests for SimulationBatch methods."""

    def test_is_metadata_based_config_source(self, simulation_batch: SimulationBatch):
        """Test is_metadata_based with config source."""
        simulation_batch.source = "config"
        assert not simulation_batch.is_metadata_based()

    def test_is_metadata_based_metadata_config_source(self, simulation_batch: SimulationBatch):
        """Test is_metadata_based with metadata_config source."""
        simulation_batch.source = "metadata_config"
        assert simulation_batch.is_metadata_based()

    def test_is_metadata_based_metadata_state_source(self, simulation_batch: SimulationBatch):
        """Test is_metadata_based with metadata_state source."""
        simulation_batch.source = "metadata_state"
        assert simulation_batch.is_metadata_based()

    def test_has_state_snapshot_without_state(self, simulation_batch: SimulationBatch):
        """Test has_state_snapshot returns False when state is None."""
        assert not simulation_batch.has_state_snapshot()

    def test_has_state_snapshot_with_state(self, simulation_batch: SimulationBatch):
        """Test has_state_snapshot returns True when state is present."""
        simulation_batch.pre_batch_state = {"rng_state": [1, 2, 3]}
        assert simulation_batch.has_state_snapshot()


# ============================================================================
# SimulationPlan Tests
# ============================================================================


class TestSimulationPlanCreation:
    """Tests for SimulationPlan creation and basic operations."""

    def test_create_empty_plan(self):
        """Test creating an empty simulation plan."""
        plan = SimulationPlan()
        assert plan.total_batches == 0
        assert len(plan.batches) == 0
        assert plan.checkpoint_directory == Path("checkpoints")

    def test_create_plan_with_checkpoint_dir(self):
        """Test creating a plan with custom checkpoint directory."""
        checkpoint_dir = Path("checkpoints") / "test"
        plan = SimulationPlan(checkpoint_directory=checkpoint_dir)
        assert plan.checkpoint_directory == checkpoint_dir

    def test_add_batch_to_plan(self, simulation_batch: SimulationBatch):
        """Test adding a batch to a plan."""
        plan = SimulationPlan()
        plan.add_batch(simulation_batch)
        assert plan.total_batches == 1
        assert len(plan.batches) == 1
        assert plan.batches[0] == simulation_batch

    def test_add_multiple_batches(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test adding multiple batches to a plan."""
        plan = SimulationPlan()
        for i in range(3):
            batch = SimulationBatch(
                simulator_name="noise",
                simulator_config=simulator_config,
                globals_config=globals_config,
                batch_index=i,
            )
            plan.add_batch(batch)
        expected_total_batches = 3
        assert plan.total_batches == expected_total_batches
        assert all(b.batch_index == i for i, b in enumerate(plan.batches))

    def test_get_batches_for_simulator(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test retrieving batches for a specific simulator."""
        plan = SimulationPlan()

        # Add noise batches
        for i in range(2):
            batch = SimulationBatch(
                simulator_name="noise",
                simulator_config=simulator_config,
                globals_config=globals_config,
                batch_index=i,
            )
            plan.add_batch(batch)

        # Add signal batches
        for i in range(3):
            batch = SimulationBatch(
                simulator_name="signal",
                simulator_config=simulator_config,
                globals_config=globals_config,
                batch_index=i,
            )
            plan.add_batch(batch)

        noise_batches = plan.get_batches_for_simulator("noise")
        signal_batches = plan.get_batches_for_simulator("signal")

        expected_noise_batches = 2
        assert len(noise_batches) == expected_noise_batches
        expected_signal_batches = 3
        assert len(signal_batches) == expected_signal_batches
        assert all(b.simulator_name == "noise" for b in noise_batches)
        assert all(b.simulator_name == "signal" for b in signal_batches)

    def test_get_batches_for_nonexistent_simulator(self):
        """Test retrieving batches for a simulator that doesn't exist."""
        plan = SimulationPlan()
        batches = plan.get_batches_for_simulator("nonexistent")
        assert len(batches) == 0


# ============================================================================
# Metadata Functions Tests
# ============================================================================


class TestParseAndCreateMetadata:
    """Tests for metadata parsing and creation functions."""

    def test_create_batch_metadata_without_state(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test creating metadata without pre-batch state."""
        metadata = create_batch_metadata(
            simulator_name="noise",
            batch_index=0,
            simulator_config=simulator_config,
            globals_config=globals_config,
        )
        assert metadata["simulator_name"] == "noise"
        assert metadata["batch_index"] == 0
        assert metadata["source"] == "config"
        assert "pre_batch_state" not in metadata
        assert "simulator_config" in metadata
        assert "globals_config" in metadata
        # Check new fields
        assert "author" in metadata
        assert "email" in metadata
        assert "timestamp" in metadata
        assert isinstance(metadata["author"], str)
        assert "+" in metadata["timestamp"] or "Z" in metadata["timestamp"]  # ISO format with timezone

    def test_create_batch_metadata_with_state(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test creating metadata with pre-batch state."""
        state = {"rng_state": [1, 2, 3], "counter": 5}
        batch_index = 2
        metadata = create_batch_metadata(
            simulator_name="signal",
            batch_index=batch_index,
            simulator_config=simulator_config,
            globals_config=globals_config,
            pre_batch_state=state,
        )
        assert metadata["simulator_name"] == "signal"
        assert metadata["batch_index"] == batch_index
        assert metadata["source"] == "config"
        assert metadata["pre_batch_state"] == state
        # Check new fields
        assert "author" in metadata
        assert "email" in metadata
        assert "timestamp" in metadata

    def test_parse_batch_metadata_valid_file(self, tmp_path: Path):
        """Test parsing a valid metadata YAML file."""
        metadata_file = tmp_path / "test.metadata.yaml"
        metadata_content = {
            "simulator_name": "noise",
            "batch_index": 0,
            "simulator_config": {"class": "WhiteNoise", "arguments": {"seed": 42}},
            "globals_config": {
                "sampling_frequency": 16384,
                "duration": 1.0,
            },
        }
        with metadata_file.open("w") as f:
            yaml.safe_dump(metadata_content, f)

        parsed = parse_batch_metadata(metadata_file)
        assert parsed["simulator_name"] == "noise"
        assert parsed["batch_index"] == 0

    def test_parse_batch_metadata_file_not_found(self):
        """Test parsing a non-existent metadata file."""
        with pytest.raises(FileNotFoundError):
            parse_batch_metadata(Path("/nonexistent/file.yaml"))

    def test_parse_batch_metadata_invalid_yaml(self, tmp_path: Path):
        """Test parsing an invalid YAML file."""
        metadata_file = tmp_path / "invalid.yaml"
        metadata_file.write_text("invalid: yaml: content: ][")

        with pytest.raises(ValueError, match="Failed to parse metadata YAML"):
            parse_batch_metadata(metadata_file)

    def test_parse_batch_metadata_not_dict(self, tmp_path: Path):
        """Test parsing a YAML file that doesn't contain a dictionary."""
        metadata_file = tmp_path / "list.yaml"
        with metadata_file.open("w") as f:
            yaml.safe_dump([1, 2, 3], f)

        with pytest.raises(ValueError, match="Metadata must be a dictionary"):
            parse_batch_metadata(metadata_file)


class TestMetadataAuthorEmailTimestamp:
    """Tests for author, email, and timestamp functionality in metadata."""

    def test_create_batch_metadata_with_explicit_author_email(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test creating metadata with explicit author and email."""
        metadata = create_batch_metadata(
            simulator_name="noise",
            batch_index=0,
            simulator_config=simulator_config,
            globals_config=globals_config,
            author="john.doe",
            email="john.doe@example.com",
        )
        assert metadata["author"] == "john.doe"
        assert metadata["email"] == "john.doe@example.com"
        assert "timestamp" in metadata

    def test_create_batch_metadata_default_author(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
        monkeypatch,
    ):
        """Test that author defaults to getpass.getuser() when not provided."""
        # Mock getpass.getuser to return a predictable value
        monkeypatch.setattr("gwsim.cli.utils.simulation_plan.getpass.getuser", lambda: "testuser")

        metadata = create_batch_metadata(
            simulator_name="noise",
            batch_index=0,
            simulator_config=simulator_config,
            globals_config=globals_config,
        )
        assert metadata["author"] == "testuser"
        assert metadata["email"] is None  # email defaults to None

    def test_create_batch_metadata_explicit_timestamp(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test creating metadata with explicit timestamp."""
        custom_timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        metadata = create_batch_metadata(
            simulator_name="noise",
            batch_index=0,
            simulator_config=simulator_config,
            globals_config=globals_config,
            timestamp=custom_timestamp,
        )
        assert metadata["timestamp"] == "2024-01-01T12:00:00+00:00"

    def test_create_batch_metadata_default_timestamp(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that timestamp defaults to current UTC time."""
        # Capture time before creating metadata
        before = datetime.datetime.now(datetime.timezone.utc)

        metadata = create_batch_metadata(
            simulator_name="noise",
            batch_index=0,
            simulator_config=simulator_config,
            globals_config=globals_config,
        )

        # Capture time after
        after = datetime.datetime.now(datetime.timezone.utc)

        # Parse the timestamp from metadata
        timestamp = datetime.datetime.fromisoformat(metadata["timestamp"])

        # Should be between before and after (within a reasonable tolerance)
        assert before <= timestamp <= after

    def test_create_batch_metadata_timestamp_format(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that timestamp is in ISO format with timezone."""
        metadata = create_batch_metadata(
            simulator_name="noise",
            batch_index=0,
            simulator_config=simulator_config,
            globals_config=globals_config,
        )

        timestamp_str = metadata["timestamp"]
        # Should be ISO format like "2024-01-01T12:00:00+00:00"
        assert "T" in timestamp_str
        assert "+" in timestamp_str or "Z" in timestamp_str

        # Should be parseable back to datetime
        parsed = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        assert isinstance(parsed, datetime.datetime)

    def test_create_batch_metadata_none_email(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that email can be None."""
        metadata = create_batch_metadata(
            simulator_name="noise",
            batch_index=0,
            simulator_config=simulator_config,
            globals_config=globals_config,
            email=None,
        )
        assert metadata["email"] is None


# ============================================================================
# create_plan_from_config Tests
# ============================================================================


class TestCreatePlanFromConfig:
    """Tests for create_plan_from_config function."""

    @pytest.fixture
    def mock_config(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Create a mock Config object."""
        config = MagicMock(spec=Config)
        config.globals = globals_config
        config.simulators = {"noise": simulator_config}
        return config

    def test_create_plan_single_simulator_single_batch(
        self,
        mock_config: Any,
    ):
        """Test creating a plan with one simulator and one batch."""
        checkpoint_dir = Path("checkpoints")
        plan = create_plan_from_config(mock_config, checkpoint_dir)

        assert plan.total_batches == 1
        assert len(plan.batches) == 1
        assert plan.batches[0].simulator_name == "noise"
        assert plan.batches[0].batch_index == 0
        assert plan.batches[0].source == "config"
        assert plan.source_config == mock_config

    def test_create_plan_multiple_simulators(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test creating a plan with multiple simulators."""
        config = MagicMock(spec=Config)
        config.globals = globals_config
        config.simulators = {
            "noise": simulator_config,
            "signal": simulator_config,
        }

        plan = create_plan_from_config(config, Path("checkpoints"))

        expected_total_batches = 2
        assert plan.total_batches == expected_total_batches
        simulator_names = {batch.simulator_name for batch in plan.batches}
        assert simulator_names == {"noise", "signal"}

    def test_create_plan_simulator_with_multiple_batches(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test creating a plan where a simulator generates multiple batches via global simulator_arguments."""
        # Create a globals_config with max_samples in simulator_arguments
        globals_config.simulator_arguments = {"max_samples": 3}

        config = MagicMock(spec=Config)
        config.globals = globals_config
        config.simulators = {"noise": simulator_config}

        plan = create_plan_from_config(config, Path("checkpoints"))

        expected_total_batches = 3
        assert plan.total_batches == expected_total_batches
        noise_batches = plan.get_batches_for_simulator("noise")
        expected_noise_batches = 3
        assert len(noise_batches) == expected_noise_batches
        assert all(b.batch_index == i for i, b in enumerate(noise_batches))

    def test_create_plan_max_samples_from_simulator_arguments(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that max_samples from globals.simulator_arguments is used for plan creation."""
        # Set max_samples in global simulator_arguments
        globals_config.simulator_arguments = {"max_samples": 5}
        simulator_config.arguments = {}

        config = MagicMock(spec=Config)
        config.globals = globals_config
        config.simulators = {"signal": simulator_config}

        plan = create_plan_from_config(config, Path("checkpoints"))

        # Should create 5 batches based on global max_samples
        expected_total_batches = 5
        assert plan.total_batches == expected_total_batches
        signal_batches = plan.get_batches_for_simulator("signal")
        expected_signal_batches = 5
        assert len(signal_batches) == expected_signal_batches

    def test_create_plan_max_samples_simulator_override(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that simulator-specific max_samples overrides global max_samples."""
        # Set global max_samples to 3, but simulator-specific to 2
        globals_config.simulator_arguments = {"max_samples": 3}
        simulator_config.arguments = {"max_samples": 2}

        config = MagicMock(spec=Config)
        config.globals = globals_config
        config.simulators = {"noise": simulator_config}

        plan = create_plan_from_config(config, Path("checkpoints"))

        # Should create 2 batches (simulator-specific overrides global)
        expected_total_batches = 2
        assert plan.total_batches == expected_total_batches
        noise_batches = plan.get_batches_for_simulator("noise")
        expected_noise_batches = 2
        assert len(noise_batches) == expected_noise_batches

    def test_create_plan_max_samples_with_hyphenated_keys(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that hyphenated keys in simulator_arguments (YAML style) are normalized."""
        # Use hyphenated keys like they come from YAML parsing
        globals_config.simulator_arguments = {"max-samples": 4, "sampling-frequency": 2048}
        simulator_config.arguments = {}

        config = MagicMock(spec=Config)
        config.globals = globals_config
        config.simulators = {"noise": simulator_config}

        plan = create_plan_from_config(config, Path("checkpoints"))

        # Should create 4 batches (hyphenated "max-samples" should be normalized)
        expected_total_batches = 4
        assert plan.total_batches == expected_total_batches
        noise_batches = plan.get_batches_for_simulator("noise")
        expected_noise_batches = 4
        assert len(noise_batches) == expected_noise_batches


# ============================================================================
# create_plan_from_metadata Tests
# ============================================================================


class TestCreatePlanFromMetadata:
    """Tests for create_plan_from_metadata function."""

    @pytest.fixture
    def metadata_directory(self, tmp_path: Path) -> Path:
        """Create a temporary directory with metadata files."""
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        return metadata_dir

    def test_create_plan_from_metadata_single_batch(
        self,
        metadata_directory: Path,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test creating a plan from a single metadata file."""
        metadata = create_batch_metadata(
            simulator_name="noise",
            batch_index=0,
            simulator_config=simulator_config,
            globals_config=globals_config,
            author="metadata_author",
            email="metadata@example.com",
        )
        metadata_file = metadata_directory / "noise-0.metadata.yaml"
        with metadata_file.open("w") as f:
            yaml.safe_dump(metadata, f)

        plan = create_plan_from_metadata(metadata_directory, Path("checkpoints"))

        assert plan.total_batches == 1
        assert plan.batches[0].simulator_name == "noise"
        assert plan.batches[0].batch_index == 0
        assert plan.batches[0].source == "metadata_config"
        # Check metadata fields
        assert plan.batches[0].batch_metadata["author"] == "metadata_author"
        assert plan.batches[0].batch_metadata["email"] == "metadata@example.com"

    def test_create_plan_from_metadata_multiple_batches(
        self,
        metadata_directory: Path,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test creating a plan from multiple metadata files."""
        for sim_name in ["noise", "signal"]:
            for batch_idx in range(2):
                metadata = create_batch_metadata(
                    simulator_name=sim_name,
                    batch_index=batch_idx,
                    simulator_config=simulator_config,
                    globals_config=globals_config,
                )
                metadata_file = metadata_directory / f"{sim_name}-{batch_idx}.metadata.yaml"
                with metadata_file.open("w") as f:
                    yaml.safe_dump(metadata, f)

        plan = create_plan_from_metadata(metadata_directory, Path("checkpoints"))

        expected_total_batches = 4  # 2 simulators * 2 batches each
        assert plan.total_batches == expected_total_batches
        noise_batches = plan.get_batches_for_simulator("noise")
        signal_batches = plan.get_batches_for_simulator("signal")
        expected_noise_batches = 2
        expected_signal_batches = 2
        assert len(noise_batches) == expected_noise_batches
        assert len(signal_batches) == expected_signal_batches

    def test_create_plan_from_metadata_with_state(
        self,
        metadata_directory: Path,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test creating a plan from metadata with state snapshots."""
        state = {"rng_state": [1, 2, 3], "counter": 5}
        metadata = create_batch_metadata(
            simulator_name="noise",
            batch_index=0,
            simulator_config=simulator_config,
            globals_config=globals_config,
            pre_batch_state=state,
            author="test_user",
            email="test@example.com",
        )
        metadata_file = metadata_directory / "noise-0.metadata.yaml"
        with metadata_file.open("w") as f:
            yaml.safe_dump(metadata, f)

        plan = create_plan_from_metadata(metadata_directory, Path("checkpoints"))

        assert plan.batches[0].has_state_snapshot()
        assert plan.batches[0].pre_batch_state == state
        assert plan.batches[0].source == "metadata_state"
        # Check metadata fields
        assert plan.batches[0].batch_metadata["author"] == "test_user"
        assert plan.batches[0].batch_metadata["email"] == "test@example.com"

    def test_create_plan_from_metadata_directory_not_found(self):
        """Test that non-existent metadata directory raises error."""
        with pytest.raises(FileNotFoundError):
            create_plan_from_metadata(Path("/nonexistent"), Path("checkpoints"))

    def test_create_plan_from_metadata_empty_directory(self, tmp_path: Path):
        """Test creating a plan from an empty metadata directory."""
        metadata_dir = tmp_path / "empty"
        metadata_dir.mkdir()

        plan = create_plan_from_metadata(metadata_dir, Path("checkpoints"))

        assert plan.total_batches == 0

    def test_create_plan_from_metadata_invalid_metadata_file(
        self,
        metadata_directory: Path,
    ):
        """Test that invalid metadata file raises error."""
        metadata_file = metadata_directory / "invalid.metadata.yaml"
        with metadata_file.open("w") as f:
            yaml.safe_dump({"incomplete": "metadata"}, f)

        with pytest.raises(ValueError, match="Invalid metadata"):
            create_plan_from_metadata(metadata_directory, Path("checkpoints"))

    def test_create_plan_from_metadata_missing_simulator_name(
        self,
        metadata_directory: Path,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that metadata without simulator_name raises error."""
        metadata = {
            "batch_index": 0,
            "simulator_config": simulator_config.model_dump(mode="python"),
            "globals_config": globals_config.model_dump(mode="python"),
        }
        metadata_file = metadata_directory / "invalid.metadata.yaml"
        with metadata_file.open("w") as f:
            yaml.safe_dump(metadata, f)

        with pytest.raises(ValueError, match="missing simulator_name or batch_index"):
            create_plan_from_metadata(metadata_directory, Path("checkpoints"))


# ============================================================================
# Integration Tests
# ============================================================================


class TestSimulationPlanIntegration:
    """Integration tests for simulation plan workflow."""

    def test_full_workflow_config_to_metadata(
        self,
        tmp_path: Path,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test full workflow: create plan from config, save metadata, recreate from metadata."""
        # Step 1: Create plan from config
        config = MagicMock(spec=Config)
        config.globals = globals_config
        config.simulators = {"noise": simulator_config}

        plan1 = create_plan_from_config(config, Path("checkpoints"))
        assert plan1.total_batches == 1

        # Step 2: Save metadata
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        batch = plan1.batches[0]
        counter = 10
        metadata = create_batch_metadata(
            simulator_name=batch.simulator_name,
            batch_index=batch.batch_index,
            simulator_config=batch.simulator_config,
            globals_config=batch.globals_config,
            pre_batch_state={"counter": counter},
            author="test_author",
            email="test@example.com",
        )
        metadata_file = metadata_dir / "noise-0.metadata.yaml"
        with metadata_file.open("w") as f:
            yaml.safe_dump(metadata, f)

        # Step 3: Recreate plan from metadata
        plan2 = create_plan_from_metadata(metadata_dir, Path("checkpoints"))
        assert plan2.total_batches == 1
        assert plan2.batches[0].simulator_name == "noise"
        assert plan2.batches[0].has_state_snapshot()
        assert plan2.batches[0].pre_batch_state["counter"] == counter
        # Check that metadata fields are preserved
        assert plan2.batches[0].batch_metadata["author"] == "test_author"
        assert plan2.batches[0].batch_metadata["email"] == "test@example.com"

    def test_batch_ordering_preserved(
        self,
        tmp_path: Path,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that batch ordering is preserved when recreating from metadata."""
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        # Create metadata files in non-alphabetical order
        for sim_name in ["signal", "noise", "glitch"]:
            for batch_idx in range(2):
                metadata = create_batch_metadata(
                    simulator_name=sim_name,
                    batch_index=batch_idx,
                    simulator_config=simulator_config,
                    globals_config=globals_config,
                )
                metadata_file = metadata_dir / f"{sim_name}-{batch_idx}.metadata.yaml"
                with metadata_file.open("w") as f:
                    yaml.safe_dump(metadata, f)

        plan = create_plan_from_metadata(metadata_dir, Path("checkpoints"))

        # Should be sorted by filename
        expected_order = [
            ("glitch", 0),
            ("glitch", 1),
            ("noise", 0),
            ("noise", 1),
            ("signal", 0),
            ("signal", 1),
        ]
        actual_order = [(b.simulator_name, b.batch_index) for b in plan.batches]
        assert actual_order == expected_order


# ============================================================================
# merge_plans Tests
# ============================================================================


class TestMergePlans:
    """Tests for merge_plans function."""

    def test_merge_empty_plans(self):
        """Test merging empty plans."""
        plan1 = SimulationPlan()
        plan2 = SimulationPlan()

        merged = merge_plans(plan1, plan2)

        assert merged.total_batches == 0
        assert len(merged.batches) == 0

    def test_merge_single_plan(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test merging a single plan (identity operation)."""
        plan = SimulationPlan()
        batch = SimulationBatch(
            simulator_name="noise",
            simulator_config=simulator_config,
            globals_config=globals_config,
            batch_index=0,
        )
        plan.add_batch(batch)

        merged = merge_plans(plan)

        assert merged.total_batches == 1
        assert merged.batches[0].simulator_name == "noise"

    def test_merge_two_plans_no_overlap(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test merging two plans with different simulators."""
        plan1 = SimulationPlan()
        plan2 = SimulationPlan()

        # Add noise batches to plan1
        for i in range(2):
            batch = SimulationBatch(
                simulator_name="noise",
                simulator_config=simulator_config,
                globals_config=globals_config,
                batch_index=i,
            )
            plan1.add_batch(batch)

        # Add signal batches to plan2
        for i in range(2):
            batch = SimulationBatch(
                simulator_name="signal",
                simulator_config=simulator_config,
                globals_config=globals_config,
                batch_index=i,
            )
            plan2.add_batch(batch)

        merged = merge_plans(plan1, plan2)

        expected_total_batches = 2 + 2  # 2 from plan1 and 2 from plan2
        assert merged.total_batches == expected_total_batches
        # Check batch indices are reassigned sequentially
        expected_batch_indices = [0, 1, 2, 3]
        assert merged.batches[0].batch_index == expected_batch_indices[0]
        assert merged.batches[1].batch_index == expected_batch_indices[1]
        assert merged.batches[2].batch_index == expected_batch_indices[2]
        assert merged.batches[3].batch_index == expected_batch_indices[3]

    def test_merge_multiple_plans(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test merging three or more plans."""
        plans = []
        for plan_idx in range(3):
            plan = SimulationPlan()
            for batch_idx in range(2):
                batch = SimulationBatch(
                    simulator_name=f"sim{plan_idx}",
                    simulator_config=simulator_config,
                    globals_config=globals_config,
                    batch_index=batch_idx,
                )
                plan.add_batch(batch)
            plans.append(plan)

        merged = merge_plans(*plans)

        expected_total_batches = 3 * 2  # 3 plans, each with 2 batches
        assert merged.total_batches == expected_total_batches
        # Verify sequential batch indices across all merged batches
        for i, batch in enumerate(merged.batches):
            assert batch.batch_index == i

    def test_merge_preserves_batch_content(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that merging preserves batch configuration and metadata."""
        plan1 = SimulationPlan()
        plan2 = SimulationPlan()

        counter = 5
        batch1 = SimulationBatch(
            simulator_name="noise",
            simulator_config=simulator_config,
            globals_config=globals_config,
            batch_index=0,
            source="config",
            pre_batch_state={"counter": counter},
        )
        plan1.add_batch(batch1)

        batch2 = SimulationBatch(
            simulator_name="signal",
            simulator_config=simulator_config,
            globals_config=globals_config,
            batch_index=0,
            source="metadata_state",
            pre_batch_state={"rng_state": [1, 2, 3]},
        )
        plan2.add_batch(batch2)

        merged = merge_plans(plan1, plan2)

        # Original batch content should be preserved
        assert merged.batches[0].simulator_name == "noise"
        assert merged.batches[0].source == "config"
        assert merged.batches[0].pre_batch_state["counter"] == counter

        assert merged.batches[1].simulator_name == "signal"
        assert merged.batches[1].source == "metadata_state"
        assert merged.batches[1].pre_batch_state["rng_state"] == [1, 2, 3]

    def test_merge_checkpoint_directory_from_first_plan(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that merged plan uses checkpoint directory from first plan."""
        checkpoint_dir1 = Path("checkpoints") / "plan1"
        checkpoint_dir2 = Path("checkpoints") / "plan2"

        plan1 = SimulationPlan(checkpoint_directory=checkpoint_dir1)
        plan2 = SimulationPlan(checkpoint_directory=checkpoint_dir2)

        batch1 = SimulationBatch(
            simulator_name="noise",
            simulator_config=simulator_config,
            globals_config=globals_config,
            batch_index=0,
        )
        plan1.add_batch(batch1)

        merged = merge_plans(plan1, plan2)

        assert merged.checkpoint_directory == checkpoint_dir1

    def test_merge_empty_plans_list(self):
        """Test merging with no plans."""
        merged = merge_plans()

        assert merged.total_batches == 0
        assert merged.checkpoint_directory == Path("checkpoints")

    def test_merge_batch_indices_reset(
        self,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test that batch indices are properly reset in merged plan."""
        plan1 = SimulationPlan()
        plan2 = SimulationPlan()

        # Both plans have batches with same indices (0, 1)
        for i in range(2):
            batch = SimulationBatch(
                simulator_name="noise",
                simulator_config=simulator_config,
                globals_config=globals_config,
                batch_index=i,
            )
            plan1.add_batch(batch)

        for i in range(2):
            batch = SimulationBatch(
                simulator_name="signal",
                simulator_config=simulator_config,
                globals_config=globals_config,
                batch_index=i,
            )
            plan2.add_batch(batch)

        merged = merge_plans(plan1, plan2)

        # Indices should be 0, 1, 2, 3 in merged plan, not 0, 1, 0, 1
        expected_indices = [0, 1, 2, 3]
        actual_indices = [b.batch_index for b in merged.batches]
        assert actual_indices == expected_indices

    def test_merge_plans_with_metadata_state(
        self,
        tmp_path: Path,
        globals_config: GlobalsConfig,
        simulator_config: SimulatorConfig,
    ):
        """Test merging plans that include metadata with state and author/email info."""
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        # Create metadata with state and author info
        state = {"rng_state": [1, 2, 3], "counter": 5}
        metadata = create_batch_metadata(
            simulator_name="noise",
            batch_index=0,
            simulator_config=simulator_config,
            globals_config=globals_config,
            pre_batch_state=state,
            author="merge_author",
            email="merge@example.com",
        )
        metadata_file = metadata_dir / "noise-0.metadata.yaml"
        with metadata_file.open("w") as f:
            yaml.safe_dump(metadata, f)

        # Create plan from metadata
        plan_metadata = create_plan_from_metadata(metadata_dir, Path("checkpoints"))

        # Create another plan from config
        config = MagicMock(spec=Config)
        config.globals = globals_config
        config.simulators = {"signal": simulator_config}
        plan_config = create_plan_from_config(config, Path("checkpoints"))

        # Merge the plans
        merged_plan = merge_plans(plan_metadata, plan_config)
        expected_total_batches = 2
        assert merged_plan.total_batches == expected_total_batches
        # Check that metadata fields are preserved in merged plan
        noise_batch = merged_plan.get_batches_for_simulator("noise")[0]
        assert noise_batch.has_state_snapshot()
        assert noise_batch.batch_metadata["author"] == "merge_author"
        assert noise_batch.batch_metadata["email"] == "merge@example.com"
