"""Configuration resolution utilities for pre-instantiation parameter calculations."""

from __future__ import annotations

import logging
from typing import Any

from gwsim.utils.datetime_parser import parse_duration_to_seconds

logger = logging.getLogger("gwsim")


def resolve_max_samples(
    simulator_args: dict[str, Any],
    global_args: dict[str, Any],
) -> int:
    """Resolve max_samples from simulator and global arguments.

    This function replicates the logic in TimeSeriesMixin to compute max_samples
    from total_duration and duration, allowing plan creation to determine batch counts
    without instantiating simulators.

    Priority order:
    1. Computed from total_duration / duration (simulator or global)
    2. Explicit max_samples in simulator_args
    3. Explicit max_samples in global_args
    4. Default to 1

    Args:
        simulator_args: Simulator-specific arguments (normalized with underscores)
        global_args: Global simulator arguments (normalized with underscores)

    Returns:
        Resolved max_samples value

    Example:
        >>> resolve_max_samples(
        ...     {"total_duration": "1h", "duration": 4},
        ...     {"max_samples": 10}
        ... )
        900  # 3600 seconds / 4 seconds
    """
    # Try to compute from total_duration and duration (highest priority)
    total_duration = simulator_args.get("total_duration") or global_args.get("total_duration")
    duration = simulator_args.get("duration") or global_args.get("duration", 4)

    if total_duration is not None:
        # Parse total_duration (may be string like "1h" or numeric)
        if isinstance(total_duration, str):
            total_duration_seconds = parse_duration_to_seconds(total_duration)
        else:
            total_duration_seconds = float(total_duration)

        duration_seconds = float(duration)

        if total_duration_seconds < duration_seconds:
            logger.warning(
                "total_duration (%s) < duration (%s), setting max_samples=1",
                total_duration_seconds,
                duration_seconds,
            )
            return 1

        # Round to nearest integer to match TimeSeriesMixin behavior
        computed_samples = round(total_duration_seconds / duration_seconds)
        logger.debug(
            "Computed max_samples=%d from total_duration=%s, duration=%s",
            computed_samples,
            total_duration_seconds,
            duration_seconds,
        )
        return computed_samples

    # Fall back to explicit max_samples in simulator args
    if "max_samples" in simulator_args:
        return int(simulator_args["max_samples"])

    # Fall back to explicit max_samples in global args
    if "max_samples" in global_args:
        return int(global_args["max_samples"])

    # Default
    logger.debug("No max_samples or total_duration specified, defaulting to 1")
    return 1
