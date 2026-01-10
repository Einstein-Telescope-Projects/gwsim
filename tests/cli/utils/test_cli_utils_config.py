"""
Unit tests for configuration utilities.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gwsim.cli.utils.config import (
    Config,
    GlobalsConfig,
    SimulatorConfig,
    SimulatorOutputConfig,
    get_examples_dir,
    get_output_directories,
    load_config,
    merge_parameters,
    resolve_class_path,
    save_config,
    validate_config,
)

# Constants for test assertions to avoid magic values
SEED = 42
SAMPLING_FREQUENCY_2048 = 2048
SAMPLING_FREQUENCY_4096 = 4096
DURATION_4 = 4
DURATION_64 = 64
SAMPLE_RATE_16384 = 16384
NUM_SIMULATORS = 2


class TestSimulatorOutputConfig:
    """Test SimulatorOutputConfig dataclass."""

    def test_valid_output_config(self):
        """Test creating a valid output config."""
        config = SimulatorOutputConfig(file_name="output.gwf")
        assert config.file_name == "output.gwf"
        assert config.arguments == {}
        assert config.output_directory is None
        assert config.metadata_directory is None

    def test_output_config_with_arguments(self):
        """Test output config with arguments."""
        config = SimulatorOutputConfig(
            file_name="{{ detector }}-{{ start_time }}.gwf",
            arguments={"channel": "STRAIN"},
        )
        assert config.arguments == {"channel": "STRAIN"}

    def test_output_config_with_directories(self):
        """Test output config with directory overrides."""
        config = SimulatorOutputConfig(
            file_name="output.gwf",
            output_directory="/custom/output",
            metadata_directory="/custom/metadata",
        )
        assert config.output_directory == "/custom/output"
        assert config.metadata_directory == "/custom/metadata"

    def test_output_config_missing_file_name(self):
        """Test that file_name is required."""
        with pytest.raises(ValidationError):
            SimulatorOutputConfig()


class TestSimulatorConfig:
    """Test SimulatorConfig dataclass."""

    def test_valid_simulator_config(self):
        """Test creating a valid simulator config."""
        config = SimulatorConfig(class_="WhiteNoise")
        assert config.class_ == "WhiteNoise"
        assert config.arguments == {}

    def test_simulator_config_with_arguments(self):
        """Test simulator config with arguments."""
        config = SimulatorConfig(
            class_="WhiteNoise",
            arguments={"seed": 42, "detectors": ["H1", "L1"]},
        )
        assert config.arguments["seed"] == SEED
        assert config.arguments["detectors"] == ["H1", "L1"]

    def test_simulator_config_with_output(self):
        """Test simulator config with output configuration."""
        config = SimulatorConfig(
            class_="WhiteNoise",
            output=SimulatorOutputConfig(file_name="noise.gwf"),
        )
        assert config.output.file_name == "noise.gwf"

    def test_simulator_config_class_alias(self):
        """Test that 'class' alias works (YAML compatibility)."""
        # Using alias directly (as would come from YAML)
        config = SimulatorConfig.model_validate({"class": "WhiteNoise"})
        assert config.class_ == "WhiteNoise"

    def test_simulator_config_missing_class(self):
        """Test that class is required."""
        with pytest.raises(ValidationError):
            SimulatorConfig()


class TestGlobalsConfig:
    """Test GlobalsConfig dataclass."""

    def test_default_globals_config(self):
        """Test GlobalsConfig with default values."""
        config = GlobalsConfig()
        assert config.working_directory == "."
        assert config.output_directory is None
        assert config.metadata_directory is None
        assert config.simulator_arguments == {}
        assert config.output_arguments == {}

    def test_globals_config_with_simulator_arguments(self):
        """Test GlobalsConfig with simulator arguments."""
        config = GlobalsConfig(
            working_directory="/data",
            simulator_arguments={"sampling_frequency": 4096, "duration": 64, "seed": 42},
        )
        assert config.working_directory == "/data"
        assert config.simulator_arguments["sampling_frequency"] == SAMPLING_FREQUENCY_4096
        assert config.simulator_arguments["duration"] == DURATION_64
        assert config.simulator_arguments["seed"] == SEED

    def test_globals_config_with_output_arguments(self):
        """Test GlobalsConfig with output arguments."""
        config = GlobalsConfig(
            output_arguments={"channel": "H1:STRAIN", "sample_rate": 16384},
        )
        assert config.output_arguments["channel"] == "H1:STRAIN"
        assert config.output_arguments["sample_rate"] == SAMPLE_RATE_16384

    def test_globals_config_with_aliases(self):
        """Test that YAML aliases work (snake-case keys)."""
        config = GlobalsConfig.model_validate(
            {
                "working-directory": "/work",
                "output-directory": "/out",
                "metadata-directory": "/meta",
                "simulator-arguments": {"sampling-frequency": 2048, "duration": 4},
                "output-arguments": {"channel": "STRAIN"},
            }
        )
        assert config.working_directory == "/work"
        assert config.output_directory == "/out"
        assert config.metadata_directory == "/meta"
        assert config.simulator_arguments["sampling-frequency"] == SAMPLING_FREQUENCY_2048
        assert config.simulator_arguments["duration"] == DURATION_4
        assert config.output_arguments["channel"] == "STRAIN"

    def test_globals_config_serialization(self):
        """Test YAML serialization with aliases."""
        config = GlobalsConfig(
            working_directory="/data",
            simulator_arguments={"sampling_frequency": 4096},
            output_arguments={"channel": "H1:STRAIN"},
        )
        # Serialize with aliases for YAML export
        data = config.model_dump(by_alias=True, exclude_none=True)
        assert data["working-directory"] == "/data"
        assert data["simulator-arguments"]["sampling_frequency"] == SAMPLING_FREQUENCY_4096
        assert data["output-arguments"]["channel"] == "H1:STRAIN"
        assert "output-directory" not in data  # None values excluded


class TestConfig:
    """Test Config (top-level) dataclass."""

    def test_valid_config(self):
        """Test creating a valid Config."""
        config = Config(simulators={"noise": SimulatorConfig(class_="WhiteNoise", arguments={"seed": 42})})
        assert "noise" in config.simulators
        assert config.simulators["noise"].class_ == "WhiteNoise"
        assert isinstance(config.globals, GlobalsConfig)

    def test_config_with_globals(self):
        """Test Config with explicit globals."""
        config = Config(
            globals=GlobalsConfig(working_directory="/data"),
            simulators={"noise": SimulatorConfig(class_="WhiteNoise")},
        )
        assert config.globals.working_directory == "/data"

    def test_config_missing_simulators(self):
        """Test that simulators field is required."""
        with pytest.raises(ValidationError):
            Config()

    def test_config_empty_simulators(self):
        """Test that simulators cannot be empty."""
        with pytest.raises(ValidationError, match=r"simulators.*cannot be empty"):
            Config(simulators={})

    def test_config_multiple_simulators(self):
        """Test Config with multiple simulators."""
        config = Config(
            simulators={
                "noise": SimulatorConfig(class_="WhiteNoise"),
                "signal": SimulatorConfig(class_="SignalSimulator"),
            }
        )
        assert len(config.simulators) == NUM_SIMULATORS
        assert "noise" in config.simulators
        assert "signal" in config.simulators

    def test_config_serialization(self):
        """Test Config serialization for YAML export."""
        config = Config(
            globals=GlobalsConfig(
                working_directory=".",
                simulator_arguments={"sampling_frequency": 2048},
            ),
            simulators={"noise": SimulatorConfig(class_="WhiteNoise", arguments={"seed": 42})},
        )
        # Serialize with aliases
        data = config.model_dump(by_alias=True, exclude_none=True)
        assert data["globals"]["simulator-arguments"]["sampling_frequency"] == SAMPLING_FREQUENCY_2048
        assert data["simulators"]["noise"]["class"] == "WhiteNoise"


class TestLoadConfig:
    """Test load_config function."""

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_yaml = """\
globals:
  working-directory: .
  simulator-arguments:
    sampling-frequency: 2048
  output-arguments: {}
simulators:
  noise:
    class: WhiteNoise
    arguments:
      seed: 42
      detectors:
        - H1
        - L1
    output:
      file_name: "{{ detectors }}-{{ start_time }}-{{ duration }}.gwf"
      arguments:
        channel: STRAIN
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_config(config_path)
            assert isinstance(config, Config)
            assert config.globals.simulator_arguments["sampling-frequency"] == SAMPLING_FREQUENCY_2048
            assert "noise" in config.simulators
            assert config.simulators["noise"].class_ == "WhiteNoise"
        finally:
            config_path.unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Failed to parse YAML"):
                load_config(config_path)
        finally:
            config_path.unlink()

    def test_load_config_missing_simulators(self):
        """Test loading config without simulators raises error."""
        config_yaml = "globals:\n  working-directory: .\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Configuration validation failed"):
                load_config(config_path)
        finally:
            config_path.unlink()


class TestSaveConfig:
    """Test save_config function."""

    def test_save_config_creates_file(self):
        """Test that save_config creates a new file."""
        config = Config(
            globals=GlobalsConfig(working_directory="."),
            simulators={"noise": SimulatorConfig(class_="WhiteNoise")},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(config_path, config)
            assert config_path.exists()

    def test_save_config_contains_correct_data(self):
        """Test that saved config contains correct YAML data."""
        config = Config(
            globals=GlobalsConfig(
                working_directory="/data",
                simulator_arguments={"sampling_frequency": 4096},
            ),
            simulators={
                "noise": SimulatorConfig(
                    class_="WhiteNoise",
                    arguments={"seed": 42},
                )
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(config_path, config)

            # Load and verify
            with open(config_path) as f:
                data = yaml.safe_load(f)
            assert data["globals"]["working-directory"] == "/data"
            assert data["globals"]["simulator-arguments"]["sampling_frequency"] == SAMPLING_FREQUENCY_4096
            assert data["simulators"]["noise"]["class"] == "WhiteNoise"

    def test_save_config_file_exists_without_overwrite(self):
        """Test that save_config raises error if file exists and overwrite=False."""
        config = Config(simulators={"noise": SimulatorConfig(class_="WhiteNoise")})
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.touch()
            with pytest.raises(FileExistsError):
                save_config(config_path, config, overwrite=False)

    def test_save_config_overwrites_with_flag(self):
        """Test that save_config overwrites when overwrite=True."""
        config = Config(simulators={"noise": SimulatorConfig(class_="WhiteNoise")})
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("old content")
            save_config(config_path, config, overwrite=True)
            # Verify new content
            content = config_path.read_text()
            assert "WhiteNoise" in content
            assert "old content" not in content

    def test_save_config_creates_backup(self):
        """Test that save_config creates backup when overwriting."""
        config = Config(simulators={"noise": SimulatorConfig(class_="WhiteNoise")})
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("old content")
            save_config(config_path, config, overwrite=True, backup=True)
            backup_path = config_path.with_suffix(".yaml.backup")
            assert backup_path.exists()
            assert backup_path.read_text() == "old content"


class TestResolveClassPath:
    """Test resolve_class_path function."""

    def test_resolve_class_path_simple_name(self):
        """Test resolving simple class name."""
        path = resolve_class_path("WhiteNoise", "noise")
        assert path == "gwsim.noise.WhiteNoise"

    def test_resolve_class_path_simple_signal(self):
        """Test resolving simple class name in signal section."""
        path = resolve_class_path("SignalSimulator", "signal")
        assert path == "gwsim.signal.SignalSimulator"

    def test_resolve_class_path_full_path(self):
        """Test resolving full import path."""
        path = resolve_class_path("numpy.random.Generator", "noise")
        assert path == "numpy.random.Generator"

    def test_resolve_class_path_third_party(self):
        """Test resolving third-party class path."""
        path = resolve_class_path("scipy.stats.norm", "noise")
        assert path == "scipy.stats.norm"


class TestMergeParameters:
    """Test merge_parameters function."""

    def test_merge_parameters_empty_global(self):
        """Test merging with empty global config."""
        globals_cfg = GlobalsConfig()
        seed = 42
        sim_args = {"seed": seed}
        merged = merge_parameters(globals_cfg, sim_args)
        assert merged["seed"] == seed
        assert merged["working-directory"] == "."  # Default from globals

    def test_merge_parameters_with_simulator_arguments(self):
        """Test merging with simulator arguments in global config."""
        working_directory = "/data"
        sampling_frequency = 4096
        global_seed = 0
        local_seed = 42

        globals_cfg = GlobalsConfig(
            working_directory=working_directory,
            simulator_arguments={"sampling_frequency": sampling_frequency, "seed": global_seed},
        )
        sim_args = {"seed": local_seed}  # Override seed
        merged = merge_parameters(globals_cfg, sim_args)
        assert merged["seed"] == local_seed  # Simulator overrides
        assert merged["sampling_frequency"] == sampling_frequency  # From globals
        assert merged["working-directory"] == working_directory  # From globals

    def test_merge_parameters_simulator_wins(self):
        """Test that simulator parameters take precedence."""
        global_sampling_frequency = 2048
        global_duration = 64
        local_sampling_frequency = 4096
        globals_cfg = GlobalsConfig(
            simulator_arguments={"sampling_frequency": global_sampling_frequency, "duration": global_duration},
        )
        sim_args = {"sampling_frequency": local_sampling_frequency}
        merged = merge_parameters(globals_cfg, sim_args)
        # Simulator value should win
        assert merged["sampling_frequency"] == local_sampling_frequency
        assert merged["duration"] == global_duration


class TestValidateConfig:
    """Test validate_config function."""

    def test_validate_config_valid(self):
        """Test validation of a valid configuration."""
        config = {
            "globals": {"simulator-arguments": {"sampling_frequency": 4096}},
            "simulators": {
                "noise": {
                    "class": "WhiteNoise",
                    "arguments": {"seed": 42},
                    "output": {"file_name": "noise.gwf", "arguments": {"channel": "H1:STRAIN"}},
                }
            },
        }
        # Should not raise
        validate_config(config)

    def test_validate_config_missing_simulators(self):
        """Test validation fails when simulators section is missing."""
        config = {"globals": {}}
        with pytest.raises(ValueError, match="Must contain 'simulators' section"):
            validate_config(config)

    def test_validate_config_empty_simulators(self):
        """Test validation fails when simulators section is empty."""
        config = {"simulators": {}}
        with pytest.raises(ValueError, match="'simulators' section cannot be empty"):
            validate_config(config)

    def test_validate_config_invalid_simulators_type(self):
        """Test validation fails when simulators is not a dict."""
        config = {"simulators": ["noise"]}
        with pytest.raises(ValueError, match="'simulators' must be a dictionary"):
            validate_config(config)

    def test_validate_config_invalid_simulator_config_type(self):
        """Test validation fails when simulator config is not a dict."""
        config = {"simulators": {"noise": "invalid"}}
        with pytest.raises(ValueError, match="configuration must be a dictionary"):
            validate_config(config)

    def test_validate_config_missing_class_field(self):
        """Test validation fails when class field is missing."""
        config = {"simulators": {"noise": {"arguments": {}}}}
        with pytest.raises(ValueError, match="missing required 'class' field"):
            validate_config(config)

    def test_validate_config_invalid_class_field(self):
        """Test validation fails when class field is invalid."""
        config = {"simulators": {"noise": {"class": ""}}}
        with pytest.raises(ValueError, match="'class' must be a non-empty string"):
            validate_config(config)

    def test_validate_config_invalid_arguments_field(self):
        """Test validation fails when arguments field is invalid."""
        config = {"simulators": {"noise": {"class": "WhiteNoise", "arguments": "invalid"}}}
        with pytest.raises(ValueError, match="'arguments' must be a dictionary"):
            validate_config(config)

    def test_validate_config_invalid_output_field(self):
        """Test validation fails when output field is invalid."""
        config = {"simulators": {"noise": {"class": "WhiteNoise", "output": "invalid"}}}
        with pytest.raises(ValueError, match="'output' must be a dictionary"):
            validate_config(config)

    def test_validate_config_invalid_globals_field(self):
        """Test validation fails when globals field is invalid."""
        config = {"globals": "invalid", "simulators": {"noise": {"class": "WhiteNoise"}}}
        with pytest.raises(ValueError, match="'globals' must be a dictionary"):
            validate_config(config)


class TestGetOutputDirectories:
    """Test get_output_directories function."""

    def test_get_output_directories_defaults(self):
        """Test directory resolution with all defaults."""
        globals_cfg = GlobalsConfig(working_directory="/work")
        sim_cfg = SimulatorConfig(class_="Noise")
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/work/output/noise")
        assert meta_dir == Path("/work/metadata/noise")

    def test_get_output_directories_global_override(self):
        """Test directory resolution with global override."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="/global/output",
            metadata_directory="/global/metadata",
        )
        sim_cfg = SimulatorConfig(class_="Noise")
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/global/output")
        assert meta_dir == Path("/global/metadata")

    def test_get_output_directories_simulator_override(self):
        """Test directory resolution with simulator-specific override."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="/global/output",
        )
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="/custom/noise",
                metadata_directory="/custom/noise/meta",
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/custom/noise")
        assert meta_dir == Path("/custom/noise/meta")

    def test_get_output_directories_priority_order(self):
        """Test three-level priority: simulator > global > default."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="/global/output",
        )
        # Simulator only overrides metadata, not output
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                metadata_directory="/custom/metadata",
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/global/output")  # From global
        assert meta_dir == Path("/custom/metadata")  # From simulator

    def test_get_output_directories_relative_simulator_path(self):
        """Test that relative paths in simulator config are prepended with working_dir."""
        globals_cfg = GlobalsConfig(working_directory="/work")
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="data/output",  # Relative path
                metadata_directory="data/metadata",  # Relative path
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/work/data/output")
        assert meta_dir == Path("/work/data/metadata")

    def test_get_output_directories_relative_global_path(self):
        """Test that relative paths in global config are prepended with working_dir."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="output",  # Relative path
            metadata_directory="metadata",  # Relative path
        )
        sim_cfg = SimulatorConfig(class_="Noise")
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/work/output")
        assert meta_dir == Path("/work/metadata")

    def test_get_output_directories_absolute_paths_not_prepended(self):
        """Test that absolute paths are NOT prepended with working_dir."""
        globals_cfg = GlobalsConfig(working_directory="/work")
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="/absolute/output",  # Absolute path
                metadata_directory="/absolute/metadata",  # Absolute path
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/absolute/output")  # Not prepended
        assert meta_dir == Path("/absolute/metadata")  # Not prepended

    def test_get_output_directories_mixed_absolute_relative(self):
        """Test mixed absolute and relative paths."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="./output",  # Relative
        )
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                metadata_directory="/var/data/metadata",  # Absolute
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/work/./output")  # Relative prepended
        assert meta_dir == Path("/var/data/metadata")  # Absolute not prepended

    def test_get_output_directories_relative_default_working_dir(self):
        """Test relative paths when working_directory itself is relative."""
        globals_cfg = GlobalsConfig(working_directory=".")
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="output",  # Relative to working_dir
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("./output")
        assert meta_dir == Path("./metadata/noise")  # Default falls back to working_dir

    def test_get_output_directories_simulator_relative_overrides_global_absolute(self):
        """Test that simulator relative path is resolved relative to working_dir."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="/global/output",  # Absolute
        )
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="custom/output",  # Relative - takes precedence
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        # Simulator relative path is resolved relative to working_dir, not global path
        assert out_dir == Path("/work/custom/output")
        assert meta_dir == Path("/work/metadata/noise")  # Falls back to default

    def test_get_output_directories_custom_working_dir_parameter(self):
        """Test using custom working_directory parameter."""
        globals_cfg = GlobalsConfig(working_directory="/default/work")
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="output",  # Relative path
            ),
        )
        custom_work_dir = Path("/custom/work")
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise", working_directory=custom_work_dir)
        # Should use custom working_dir, not globals_cfg.working_directory
        assert out_dir == Path("/custom/work/output")
        assert meta_dir == Path("/custom/work/metadata/noise")


class TestConfigRoundTrip:
    """Integration tests for loading and saving configs."""

    def test_round_trip_config(self):
        """Test load -> save -> load round trip."""
        config_yaml = """\
globals:
  working-directory: /data
  simulator-arguments:
    sampling-frequency: 4096
  output-directory: /output
simulators:
  noise:
    class: WhiteNoise
    arguments:
      seed: 42
      detectors:
        - H1
        - L1
    output:
      file_name: "{{ detectors }}-{{ start_time }}-{{ duration }}.gwf"
      arguments:
        channel: STRAIN
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Load
            config_path = Path(tmpdir) / "config1.yaml"
            config_path.write_text(config_yaml)
            config1 = load_config(config_path)

            # Save
            config_path2 = Path(tmpdir) / "config2.yaml"
            save_config(config_path2, config1)

            # Load again
            config2 = load_config(config_path2)

            # Verify equivalence
            assert config1.globals.simulator_arguments == config2.globals.simulator_arguments
            assert config1.simulators["noise"].class_ == config2.simulators["noise"].class_
            assert config1.simulators["noise"].arguments["seed"] == config2.simulators["noise"].arguments["seed"]

    def test_multiple_simulators_round_trip(self):
        """Test round trip with multiple simulators."""
        sampling_frequency = 2048
        config = Config(
            globals=GlobalsConfig(
                working_directory="/data",
                simulator_arguments={"sampling_frequency": sampling_frequency, "duration": 8},
            ),
            simulators={
                "noise": SimulatorConfig(
                    class_="WhiteNoise",
                    arguments={"seed": 42},
                ),
                "signal": SimulatorConfig(
                    class_="SignalSimulator",
                    arguments={"mass1": 10, "mass2": 10},
                ),
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(config_path, config)
            loaded = load_config(config_path)

            expected_num_of_simulators = NUM_SIMULATORS
            assert len(loaded.simulators) == expected_num_of_simulators
            assert "noise" in loaded.simulators
            assert "signal" in loaded.simulators
            assert loaded.globals.simulator_arguments["sampling_frequency"] == sampling_frequency


def test_get_examples_dir_discovers_examples():
    """Test that get_examples_dir() finds the examples directory with YAML files."""
    examples_dir = get_examples_dir()

    # Assert it's a Path object
    assert isinstance(examples_dir, Path)

    # Assert the directory exists
    assert examples_dir.exists(), f"Examples directory not found: {examples_dir}"

    # Assert it's a directory
    assert examples_dir.is_dir(), f"Examples path is not a directory: {examples_dir}"

    # Assert it contains at least one YAML file (to verify discovery works)
    yaml_files = list(examples_dir.rglob("*.yaml"))
    assert len(yaml_files) > 0, f"No YAML files found in examples directory: {examples_dir}"

    # Optional: Check for a known example file (adjust based on your examples/)
    # This ensures the structure is as expected
    expected_files = ["noise/uncorrelated_gaussian/et_triangle_emr/config.yaml"]  # Example; update as needed
    for rel_path in expected_files:
        full_path = examples_dir / rel_path
        assert full_path.exists(), f"Expected example file not found: {full_path}"
