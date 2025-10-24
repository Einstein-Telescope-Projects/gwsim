"""
Unit tests for configuration utilities.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from gwsim.cli.utils.config import (
    expand_config_templates,
    expand_detector_templates,
    expand_templates,
    load_config,
    merge_parameters,
    normalize_config,
    process_config,
    resolve_class_path,
    save_config,
    validate_config,
)


class TestValidateConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test validation of a valid configuration."""
        config = {
            "globals": {"sampling_frequency": 4096},
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

    def test_missing_simulators_section(self):
        """Test validation fails when simulators section is missing."""
        config = {"globals": {}}
        with pytest.raises(ValueError, match="Must contain 'simulators' section"):
            validate_config(config)

    def test_empty_simulators_section(self):
        """Test validation fails when simulators section is empty."""
        config = {"simulators": {}}
        with pytest.raises(ValueError, match="'simulators' section cannot be empty"):
            validate_config(config)

    def test_invalid_simulators_type(self):
        """Test validation fails when simulators is not a dict."""
        config = {"simulators": ["noise"]}
        with pytest.raises(ValueError, match="'simulators' must be a dictionary"):
            validate_config(config)

    def test_invalid_simulator_config_type(self):
        """Test validation fails when simulator config is not a dict."""
        config = {"simulators": {"noise": "invalid"}}
        with pytest.raises(ValueError, match="configuration must be a dictionary"):
            validate_config(config)

    def test_missing_class_field(self):
        """Test validation fails when class field is missing."""
        config = {"simulators": {"noise": {"arguments": {}}}}
        with pytest.raises(ValueError, match="missing required 'class' field"):
            validate_config(config)

    def test_invalid_class_field(self):
        """Test validation fails when class field is invalid."""
        config = {"simulators": {"noise": {"class": ""}}}
        with pytest.raises(ValueError, match="'class' must be a non-empty string"):
            validate_config(config)

    def test_invalid_arguments_field(self):
        """Test validation fails when arguments field is invalid."""
        config = {"simulators": {"noise": {"class": "WhiteNoise", "arguments": "invalid"}}}
        with pytest.raises(ValueError, match="'arguments' must be a dictionary"):
            validate_config(config)

    def test_invalid_output_field(self):
        """Test validation fails when output field is invalid."""
        config = {"simulators": {"noise": {"class": "WhiteNoise", "output": "invalid"}}}
        with pytest.raises(ValueError, match="'output' must be a dictionary"):
            validate_config(config)

    def test_invalid_globals_field(self):
        """Test validation fails when globals field is invalid."""
        config = {"globals": "invalid", "simulators": {"noise": {"class": "WhiteNoise"}}}
        with pytest.raises(ValueError, match="'globals' must be a dictionary"):
            validate_config(config)


class TestLoadConfig:
    """Test configuration loading."""

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_data = {
            "globals": {"sampling_frequency": 4096},
            "simulators": {"noise": {"class": "WhiteNoise"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            config_path = Path(f.name)

        try:
            loaded = load_config(config_path)
            assert loaded == config_data
        finally:
            config_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading fails for nonexistent file."""
        config_path = Path("nonexistent.yaml")
        with pytest.raises(FileNotFoundError):
            load_config(config_path)

    def test_load_invalid_yaml(self):
        """Test loading fails for invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [\n")
            config_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            config_path.unlink()

    def test_load_invalid_config(self):
        """Test loading fails for invalid configuration structure."""
        invalid_config = {"invalid": "config"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(invalid_config, f)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ValueError, match="Invalid configuration: Must contain 'simulators' section with simulator definitions"
            ):
                load_config(config_path)
        finally:
            config_path.unlink()


class TestSaveConfig:
    """Test configuration saving."""

    def test_save_new_file(self):
        """Test saving configuration to a new file."""
        config = {"globals": {}, "simulators": {"noise": {"class": "WhiteNoise"}}}

        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = Path(temp_dir) / "config.yaml"
            save_config(file_name=file_name, config=config, overwrite=False)
            assert file_name.exists()

            # Verify content
            with open(file_name) as f:
                saved = yaml.safe_load(f)
            assert saved == config

    def test_save_overwrite_existing(self):
        """Test overwriting an existing configuration file."""
        config1 = {"globals": {}, "simulators": {"noise": {"class": "WhiteNoise"}}}
        config2 = {"globals": {}, "simulators": {"signal": {"class": "BBHInspiral"}}}

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"

            # Save first config
            save_config(file_name=config_path, config=config1, overwrite=False)
            assert config_path.exists()

            # Overwrite with second config
            save_config(file_name=config_path, config=config2, overwrite=True)

            # Verify content
            with open(config_path) as f:
                saved = yaml.safe_load(f)
            assert saved == config2

    def test_save_no_overwrite_existing(self):
        """Test saving fails when file exists and overwrite=False."""
        config = {"globals": {}, "simulators": {"noise": {"class": "WhiteNoise"}}}

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"

            # Save first time
            save_config(file_name=config_path, config=config, overwrite=False)
            assert config_path.exists()

            # Try to save again without overwrite
            with pytest.raises(FileExistsError):
                save_config(file_name=config_path, config=config, overwrite=False)

    def test_save_with_backup(self):
        """Test saving with backup creation."""
        config1 = {"globals": {}, "simulators": {"noise": {"class": "WhiteNoise"}}}
        config2 = {"globals": {}, "simulators": {"signal": {"class": "BBHInspiral"}}}

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"

            # Save first config
            save_config(file_name=config_path, config=config1, overwrite=False)
            backup_path = config_path.with_suffix(f"{config_path.suffix}.backup")
            assert not backup_path.exists()

            # Overwrite with backup
            save_config(file_name=config_path, config=config2, overwrite=True, backup=True)
            assert backup_path.exists()

            # Verify backup content
            with open(backup_path) as f:
                backup_content = yaml.safe_load(f)
            assert backup_content == config1

            # Verify new content
            with open(config_path) as f:
                saved = yaml.safe_load(f)
            assert saved == config2


class TestResolveClassPath:
    """Test class path resolution."""

    @pytest.mark.parametrize(
        ("class_spec", "section_name", "expected"),
        [
            ("WhiteNoise", "noise", "gwsim.noise.WhiteNoise"),
            ("BBHInspiral", "signal", "gwsim.signal.BBHInspiral"),
            ("numpy.random.Generator", "noise", "numpy.random.Generator"),
            ("scipy.stats.norm", "noise", "scipy.stats.norm"),
        ],
    )
    def test_resolve_class_path(self, class_spec, section_name, expected):
        """Test class path resolution for various inputs."""
        result = resolve_class_path(class_spec, section_name)
        assert result == expected


class TestMergeParameters:
    """Test parameter merging."""

    def test_merge_empty_globals(self):
        """Test merging with empty globals."""
        globals_config = {}
        simulator_config = {"seed": 42, "duration": 4}
        result = merge_parameters(globals_config, simulator_config)
        assert result == {"seed": 42, "duration": 4}

    def test_merge_empty_simulator_config(self):
        """Test merging with empty simulator config."""
        globals_config = {"sampling_frequency": 4096, "duration": 4}
        simulator_config = {}
        result = merge_parameters(globals_config, simulator_config)
        assert result == {"sampling_frequency": 4096, "duration": 4}

    def test_merge_overlapping_keys(self):
        """Test merging when keys overlap (simulator takes precedence)."""
        globals_config = {"sampling_frequency": 4096, "duration": 4, "seed": 0}
        simulator_config = {"duration": 8, "amplitude": 1.0}
        result = merge_parameters(globals_config, simulator_config)
        expected = {"sampling_frequency": 4096, "duration": 8, "seed": 0, "amplitude": 1.0}
        assert result == expected

    def test_merge_no_overlap(self):
        """Test merging with no overlapping keys."""
        globals_config = {"sampling_frequency": 4096, "start_time": 0}
        simulator_config = {"duration": 4, "seed": 42}
        result = merge_parameters(globals_config, simulator_config)
        expected = {"sampling_frequency": 4096, "start_time": 0, "duration": 4, "seed": 42}
        assert result == expected


class TestExpandTemplates:
    """Test template expansion."""

    def test_expand_simple_variable(self):
        """Test expanding a simple template variable."""
        text = "file-{{ duration }}.gwf"
        context = {"duration": 4}
        result = expand_templates(text, context)
        assert result == "file-4.gwf"

    def test_expand_multiple_variables(self):
        """Test expanding multiple template variables."""
        text = "{{ detector }}-{{ start_time }}-{{ duration }}.gwf"
        context = {"detector": "H1", "start_time": 1234567890, "duration": 4}
        result = expand_templates(text, context)
        assert result == "H1-1234567890-4.gwf"

    def test_expand_nested_variable(self):
        """Test expanding nested template variables."""
        text = "{{ globals.duration }}s-{{ detector }}.gwf"
        context = {"globals": {"duration": 4}, "detector": "H1"}
        result = expand_templates(text, context)
        assert result == "4s-H1.gwf"

    def test_expand_missing_variable(self):
        """Test handling of missing template variables."""
        text = "file-{{ missing_var }}.gwf"
        context = {"duration": 4}
        with patch("gwsim.cli.utils.config.logger") as mock_logger:
            result = expand_templates(text, context)
            assert result == "file-{{ missing_var }}.gwf"
            mock_logger.warning.assert_called_once()

    def test_expand_no_templates(self):
        """Test text with no template variables."""
        text = "simple-filename.gwf"
        context = {"duration": 4}
        result = expand_templates(text, context)
        assert result == "simple-filename.gwf"


class TestExpandDetectorTemplates:
    """Test detector template expansion."""

    def test_expand_string_no_detector_placeholder(self):
        """Test string without detector placeholder is unchanged."""
        config = "file-{{ duration }}.gwf"
        detectors = ["H1", "L1"]
        result = expand_detector_templates(config, detectors)
        assert result == "file-{{ duration }}.gwf"

    def test_expand_dict_with_detector_placeholder(self):
        """Test dict with detector placeholder is preserved."""
        config = {"channel": "{detector}:STRAIN", "duration": 4}
        detectors = ["H1", "L1"]
        result = expand_detector_templates(config, detectors)
        assert result == {"channel": "{detector}:STRAIN", "duration": 4}

    def test_expand_list_with_detector_placeholder(self):
        """Test list with detector placeholder is preserved."""
        config = ["{detector}:STRAIN", "H2:STRAIN"]
        detectors = ["H1", "L1"]
        result = expand_detector_templates(config, detectors)
        assert result == ["{detector}:STRAIN", "H2:STRAIN"]

    def test_expand_nested_structure(self):
        """Test nested dict/list structures."""
        config = {
            "channels": ["{detector}:STRAIN", "H2:STRAIN"],
            "metadata": {"detector": "{detector}"},
        }
        detectors = ["H1", "L1"]
        result = expand_detector_templates(config, detectors)
        expected = {
            "channels": ["{detector}:STRAIN", "H2:STRAIN"],
            "metadata": {"detector": "{detector}"},
        }
        assert result == expected

    def test_expand_no_detectors(self):
        """Test expansion with no detectors provided."""
        config = {"channel": "{detector}:STRAIN"}
        result = expand_detector_templates(config, None)
        assert result == {"channel": "{detector}:STRAIN"}


class TestExpandConfigTemplates:
    """Test configuration template expansion."""

    def test_expand_string(self):
        """Test expanding templates in string."""
        config = "file-{{ duration }}.gwf"
        context = {"duration": 4}
        result = expand_config_templates(config, context)
        assert result == "file-4.gwf"

    def test_expand_dict(self):
        """Test expanding templates in dict."""
        config = {"file_name": "data-{{ duration }}.gwf", "duration": "{{ duration }}"}
        context = {"duration": 4}
        result = expand_config_templates(config, context)
        expected = {"file_name": "data-4.gwf", "duration": "4"}
        assert result == expected

    def test_expand_list(self):
        """Test expanding templates in list."""
        config = ["file-{{ duration }}.gwf", "{{ detector }}"]
        context = {"duration": 4, "detector": "H1"}
        result = expand_config_templates(config, context)
        assert result == ["file-4.gwf", "H1"]

    def test_expand_nested_structure(self):
        """Test expanding templates in nested structures."""
        config = {
            "output": {"file_name": "{{ detector }}-{{ duration }}.gwf"},
            "arguments": ["--duration={{ duration }}", "--seed={{ seed }}"],
        }
        context = {"detector": "H1", "duration": 4, "seed": 42}
        result = expand_config_templates(config, context)
        expected = {
            "output": {"file_name": "H1-4.gwf"},
            "arguments": ["--duration=4", "--seed=42"],
        }
        assert result == expected

    def test_expand_no_templates(self):
        """Test config with no templates."""
        config = {"file_name": "data.gwf", "duration": 4}
        context = {"detector": "H1"}
        result = expand_config_templates(config, context)
        assert result == config


class TestNormalizeConfig:
    """Test configuration normalization."""

    def test_normalize_complete_config(self):
        """Test normalizing a complete configuration."""
        config = {
            "globals": {"sampling_frequency": 4096},
            "simulators": {
                "noise": {
                    "class": "WhiteNoise",
                    "arguments": {"seed": 42},
                    "output": {"file_name": "noise.gwf", "arguments": {"channel": "H1:STRAIN"}},
                }
            },
        }
        result = normalize_config(config)
        assert result == config

    def test_normalize_missing_globals(self):
        """Test normalizing config missing globals section."""
        config = {
            "simulators": {
                "noise": {
                    "class": "WhiteNoise",
                    "arguments": {"seed": 42},
                }
            },
        }
        result = normalize_config(config)
        expected = {
            "globals": {},
            "simulators": {
                "noise": {
                    "class": "WhiteNoise",
                    "arguments": {"seed": 42},
                    "output": {"file_name": "noise-{{ counter }}.hdf5", "arguments": {}},
                }
            },
        }
        assert result == expected

    def test_normalize_missing_simulator_fields(self):
        """Test normalizing config with missing simulator fields."""
        config = {
            "simulators": {"noise": {"class": "WhiteNoise"}},
        }
        result = normalize_config(config)
        expected = {
            "globals": {},
            "simulators": {
                "noise": {
                    "class": "WhiteNoise",
                    "arguments": {},
                    "output": {"file_name": "noise-{{ counter }}.hdf5", "arguments": {}},
                }
            },
        }
        assert result == expected

    def test_normalize_invalid_config(self):
        """Test normalizing invalid config raises error."""
        config = {"globals": {}}
        with pytest.raises(ValueError, match="must contain 'simulators' section"):
            normalize_config(config)


class TestProcessConfig:
    """Test configuration processing."""

    def test_process_complete_config(self):
        """Test processing a complete configuration."""
        config = {
            "globals": {"sampling_frequency": 4096, "duration": 4},
            "simulators": {
                "noise": {
                    "class": "WhiteNoise",
                    "arguments": {"seed": 42, "duration": 8},  # duration overrides global
                    "output": {"file_name": "noise.gwf", "arguments": {"channel": "H1:STRAIN"}},
                }
            },
        }
        result = process_config(config)
        expected = {
            "globals": {"sampling_frequency": 4096, "duration": 4},
            "simulators": {
                "noise": {
                    "class": "WhiteNoise",
                    "arguments": {"sampling_frequency": 4096, "duration": 8, "seed": 42},
                    "output": {
                        "file_name": "noise.gwf",
                        "arguments": {"sampling_frequency": 4096, "duration": 4, "channel": "H1:STRAIN"},
                    },
                }
            },
        }
        assert result == expected

    def test_process_parameter_inheritance(self):
        """Test that global parameters are inherited by simulators."""
        config = {
            "globals": {"sampling_frequency": 4096, "seed": 0},
            "simulators": {
                "noise": {
                    "class": "WhiteNoise",
                    "arguments": {"duration": 4},
                    "output": {"arguments": {"channel": "H1:STRAIN"}},
                }
            },
        }
        result = process_config(config)
        simulator_args = result["simulators"]["noise"]["arguments"]
        output_args = result["simulators"]["noise"]["output"]["arguments"]

        assert simulator_args["sampling_frequency"] == 4096
        assert simulator_args["seed"] == 0
        assert simulator_args["duration"] == 4
        assert output_args["sampling_frequency"] == 4096
        assert output_args["seed"] == 0
        assert output_args["channel"] == "H1:STRAIN"

    def test_process_simulator_override(self):
        """Test that simulator parameters override globals."""
        config = {
            "globals": {"duration": 4, "seed": 0},
            "simulators": {
                "noise": {
                    "class": "WhiteNoise",
                    "arguments": {"duration": 8, "seed": 42},
                }
            },
        }
        result = process_config(config)
        simulator_args = result["simulators"]["noise"]["arguments"]

        assert simulator_args["duration"] == 8  # overridden
        assert simulator_args["seed"] == 42  # overridden
