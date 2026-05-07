"""Smoke tests before publishing to verify the wheel and source distribution."""

from __future__ import annotations

import subprocess
import sys

import gwmock

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - used in isolated wheel smoke runs.
    pytestmark = ()
else:
    pytestmark = pytest.mark.integration


def test_basic_import() -> None:
    """Test basic import."""
    print(f"Python version: {sys.version}")
    print(f"Package version: {gwmock.__version__}")

    # Ensure it's not importing the local folder
    assert "site-packages" in gwmock.__file__ or "dist" in gwmock.__file__, (
        f"Package imported from unexpected location: {gwmock.__file__}"
    )


def test_cli_help() -> None:
    """Test CLI help."""
    # Ensure the 'gwmock' command was registered and runs
    result = subprocess.run(["gwmock", "--help"], capture_output=True, text=True, check=False)  # noqa: S607
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


if __name__ == "__main__":
    test_basic_import()
    print("Smoke test passed: Package is importable.")

    test_cli_help()
    print("Smoke test passed: The CLI is executable.")
