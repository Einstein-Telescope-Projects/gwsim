from __future__ import annotations

import pytest

from gwsim.utils.datetime_parser import parse_duration_to_seconds


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("1 second", 1),
        ("2 minutes", 2 * 60),
        ("1 hour", 3_600),
        ("1 day", 86_400),
        ("2 weeks", 2 * 604_800),
        ("1 month", 2_592_000),
        ("0.5 year", 0.5 * 31_536_000),
        ("  3 days  ", 3 * 86_400),
        ("1 Hours", 3_600),
    ],
)
def test_parse_duration_to_seconds_valid(input_str: str, expected: float) -> None:
    assert parse_duration_to_seconds(input_str) == expected


def test_parse_duration_to_seconds_invalid_unit() -> None:
    with pytest.raises(ValueError, match="Unsupported duration unit"):
        parse_duration_to_seconds("1 fortnight")


def test_parse_duration_to_seconds_invalid_format() -> None:
    with pytest.raises(ValueError, match="Duration must be of the form"):
        parse_duration_to_seconds("one day")
