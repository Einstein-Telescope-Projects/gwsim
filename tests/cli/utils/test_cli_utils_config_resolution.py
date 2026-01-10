"""Unit tests for CLI utils config resolution."""

from __future__ import annotations

from gwsim.cli.utils.config_resolution import resolve_max_samples


def test_computed_from_total_duration_takes_priority():
    """Computed total_duration takes priority over explicit max_samples."""
    result = resolve_max_samples(
        {"max_samples": 5, "total_duration": 100, "duration": 4},
        {"max_samples": 10},
    )
    expected_result = 100 / 4  # 25
    assert result == expected_result


def test_computed_from_total_duration():
    """Compute max_samples from total_duration and duration."""
    result = resolve_max_samples(
        {"total_duration": 3600, "duration": 4},
        {},
    )
    expected_result = 3600 / 4  # 900
    assert result == expected_result  # 3600 / 4


def test_string_duration_parsing():
    """Parse string duration like '1h'."""
    result = resolve_max_samples(
        {"total_duration": "1 hour", "duration": 4},
        {},
    )
    expected_result = 3600 / 4  # 1 hour = 3600 seconds
    assert result == expected_result


def test_fallback_to_global_max_samples():
    """Use global max_samples if total_duration not set."""
    max_samples = 7
    result = resolve_max_samples(
        {},
        {"max_samples": max_samples},
    )
    assert result == max_samples


def test_default_to_one():
    """Default to 1 if nothing specified."""
    result = resolve_max_samples({}, {})
    assert result == 1
