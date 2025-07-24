from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from gwsim.tools.main import main

logger = logging.getLogger("gwsim")


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
def mock_config(temp_dir):
    """Fixture for mock configuration."""
    return {
        "output": {"file_name": "E1-test-{{ start_time }}-{{ duration }}.gwf", "arguments": {"channel": "STRAIN"}},
        "generator": {
            "class": "gwsim.noise.white_noise.WhiteNoise",
            "arguments": {
                "loc": 0.0,
                "scale": 1.0,
                "sampling_frequency": 16,
                "duration": 4,
                "start_time": 123,
                "batch_size": 1,
                "max_samples": 10,
                "seed": 0,
            },
        },
        "working-directory": str(temp_dir),
    }


def test_generate_white_noise(
    runner,
    mock_config,
    temp_dir,
):
    logger.info("Generating data in %s", temp_dir)

    # Write the mock config to file
    config_file = temp_dir / "config.yaml"

    with open(config_file, "w") as f:
        yaml.safe_dump(mock_config, f)

    # Check the written config file is the same as the mock config
    with open(config_file) as f:
        _loaded_config = yaml.safe_load(f)

    assert _loaded_config == mock_config

    result = runner.invoke(main, ["generate", str(config_file), "--metadata"])

    assert result.exit_code == 0

    # Count the number of generated files.
    output_files = list((temp_dir / "output").glob("*.gwf"))
    expected_files = mock_config["generator"]["arguments"]["max_samples"]
    assert len(output_files) == expected_files, f"Expected {expected_files} .gwf files, got {len(output_files)}"

    metadata_files = list((temp_dir / "metadata").glob("*.json"))
    expected_files = mock_config["generator"]["arguments"]["max_samples"]
    assert len(metadata_files) == expected_files, f"Expected {expected_files} .gwf files, got {len(metadata_files)}"

    # Check whether a checkpoint file exists.
    assert (temp_dir / "checkpoint.json").exists()
