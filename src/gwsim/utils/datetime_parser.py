"""Utilities for parsing human-readable time durations."""

from __future__ import annotations

import re

SECOND_UNITS = {
    "second": 1,
    "minute": 60,
    "hour": 3_600,
    "day": 86_400,
    "week": 604_800,
    # approximate values for longer units
    "month": 2_592_000,  # 30 days
    "year": 31_536_000,  # 365 days
}


def parse_duration_to_seconds(duration: str) -> float:
    """Convert a human-friendly duration like "1 day" into seconds.

    Args:
        duration: A string such as "1 week", "2 days", "1.5 hours".

    Returns:
        Number of seconds represented by the duration.

    Raises:
        ValueError: If the string cannot be parsed or the unit is unsupported.
    """

    pattern = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>\w+)s?\s*$", re.IGNORECASE)
    match = pattern.match(duration)
    if not match:
        raise ValueError("Duration must be of the form '<number> <unit>' (e.g. '1 week').")

    value = float(match.group("value"))
    unit = match.group("unit").lower().rstrip("s")

    seconds = SECOND_UNITS.get(unit)
    if seconds is None:
        raise ValueError(f"Unsupported duration unit '{unit}'. Supported units: {', '.join(SECOND_UNITS)}.")

    return value * seconds
