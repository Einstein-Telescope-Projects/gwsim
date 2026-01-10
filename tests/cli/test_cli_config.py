"""Unit tests for CLI config command."""

from __future__ import annotations

import contextlib

import pytest
from typer.testing import CliRunner

from gwsim.cli.main import app


@pytest.fixture
def runner():
    """Create a CliRunner instance for testing CLI commands."""
    return CliRunner()


@pytest.fixture
def temp_examples_dir(tmp_path):
    """Create a temporary examples directory with mock config files."""
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()

    # Create default config with valid YAML
    (examples_dir / "default_config").mkdir()
    (examples_dir / "default_config" / "config.yaml").write_text("simulators:\n  noise:\n    class: default")

    # Create mock example directories with valid YAML config
    (examples_dir / "example1").mkdir()
    (examples_dir / "example1" / "config.yaml").write_text("simulators:\n  noise:\n    class: example1")

    (examples_dir / "example2").mkdir()
    (examples_dir / "example2" / "config.yaml").write_text("simulators:\n  noise:\n    class: example2")

    # Create a nested example
    (examples_dir / "nested" / "example3").mkdir(parents=True)
    (examples_dir / "nested" / "example3" / "config.yaml").write_text("simulators:\n  noise:\n    class: example3")

    return examples_dir


@pytest.fixture(autouse=True)
def mock_examples_dir(monkeypatch, temp_examples_dir):
    """Mock get_examples_dir to return our temp directory."""
    # Import and patch at the module level where it's used
    monkeypatch.setattr("gwsim.cli.config.get_examples_dir", lambda: temp_examples_dir, raising=False)
    # Also patch in utils.config in case it's imported there
    with contextlib.suppress(ImportError, AttributeError):
        monkeypatch.setattr("gwsim.cli.utils.config.get_examples_dir", lambda: temp_examples_dir)


def test_config_list(runner, temp_examples_dir):
    """Test --list command lists available labels."""
    result = runner.invoke(app, ["config", "--list"])
    assert result.exit_code == 0
    assert "Available example configuration labels:" in result.output
    assert "example1" in result.output
    assert "example2" in result.output
    assert "example3" in result.output


def test_config_init_default(runner, tmp_path):
    """Test --init generates default config with default name."""
    output_file = tmp_path / "config.yaml"
    result = runner.invoke(app, ["config", "--init", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()


def test_config_init_custom_name(runner, tmp_path):
    """Test --init with custom filename."""
    output_file = tmp_path / "my_config.yaml"
    result = runner.invoke(app, ["config", "--init", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()


def test_config_get_valid_label(runner, temp_examples_dir, tmp_path):
    """Test --get with valid label copies config."""
    output_file = tmp_path / "config.yaml"
    result = runner.invoke(app, ["config", "--get", "example1", "--output", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()
    content = output_file.read_text()
    assert "example1" in content


def test_config_get_invalid_label(runner):
    """Test --get with invalid label fails."""
    result = runner.invoke(app, ["config", "--get", "nonexistent"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_config_no_flags(runner):
    """Test running without flags shows error."""
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 1
    assert "No action specified" in result.output


def test_config_overwrite(runner, temp_examples_dir, tmp_path):
    """Test --overwrite flag works."""
    # Create existing file
    existing = tmp_path / "config.yaml"
    existing.write_text("existing content")

    result = runner.invoke(app, ["config", "--get", "example1", "--output", str(existing), "--overwrite"])
    assert result.exit_code == 0
    content = existing.read_text()
    assert "example1" in content


def test_config_output_directory(runner, temp_examples_dir, tmp_path):
    """Test --output specifies directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()  # Create the directory so it's recognized as a directory
    result = runner.invoke(app, ["config", "--get", "example1", "--output", str(output_dir)])
    assert result.exit_code == 0
    assert (output_dir / "config.yaml").exists()


def test_config_output_file(runner, temp_examples_dir, tmp_path):
    """Test --output specifies file."""
    output_file = tmp_path / "custom.yaml"
    result = runner.invoke(app, ["config", "--get", "example1", "--output", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()
    content = output_file.read_text()
    print(content)
    assert "simulators:\n  noise:\n    class: example1" in content
