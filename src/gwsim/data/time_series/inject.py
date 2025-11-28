"""Module to handle injection of one TimeSeries into another, with support for time offsets."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from astropy.units import second  # pylint: disable=no-name-in-module
from gwpy.timeseries import TimeSeries
from scipy.interpolate import interp1d

logger = logging.getLogger("gwsim")


def inject(timeseries: TimeSeries, other: TimeSeries, interpolate_if_offset: bool = True) -> TimeSeries:
    """Inject one TimeSeries into another, handling time offsets.

    Args:
        timeseries: The target TimeSeries to inject into.
        other: The TimeSeries to be injected.
        interpolate_if_offset: Whether to interpolate if there is a non-integer sample offset.

    Returns:
        TimeSeries: The resulting TimeSeries after injection.
    """
    # Check whether timeseries is compatible with other
    timeseries.is_compatible(other)

    # crop to fit
    if (timeseries.xunit == second) and (other.xspan[0] < timeseries.xspan[0]):
        other = cast(TimeSeries, other.crop(start=timeseries.xspan[0]))
    if (timeseries.xunit == second) and (other.xspan[1] > timeseries.xspan[1]):
        other = cast(TimeSeries, other.crop(end=timeseries.xspan[1]))

    # Check if other is empty after cropping
    if len(other.times) == 0:
        logger.debug("Other TimeSeries is empty after cropping to fit; returning original timeseries")
        return timeseries

    target_times = timeseries.times.value
    other_times = other.times.value
    sample_spacing = float(timeseries.dt.value)

    # Calculate offset between start times
    offset = (other_times[0] - target_times[0]) / sample_spacing

    # Check if offset is aligned (integer number of samples)
    if not np.isclose(offset, round(offset)):
        if not interpolate_if_offset:
            logger.debug("Non-integer offset of %s samples; not interpolating, returning original timeseries", offset)
            return timeseries

        # Interpolate to align grids
        logger.debug("Injecting with interpolation due to non-integer offset of %s samples", offset)

        # Determine overlap range in target time grid
        start_idx = int(np.searchsorted(target_times, other_times[0], side="left"))
        end_idx = int(np.searchsorted(target_times, other_times[-1], side="right")) - 1

        if start_idx >= len(target_times) or end_idx < 0 or start_idx > end_idx:
            logger.debug("No overlap between timeseries and other after searching; returning original timeseries")
            return timeseries

        interp_func = interp1d(other_times, other.value, kind="cubic", axis=0, bounds_error=False, fill_value=0.0)
        resampled = interp_func(target_times[start_idx : end_idx + 1])

        # Create a new TimeSeries with explicit parameters to avoid floating-point precision issues
        injected_data = timeseries.value.copy()
        injected_data[start_idx : end_idx + 1] += resampled
        injected = TimeSeries(
            injected_data,
            t0=timeseries.t0,
            dt=timeseries.dt,
            unit=timeseries.unit,
        )
        return injected

    # Aligned case: offset is integer
    logger.debug("Injecting with aligned grids (offset: %s samples)", round(offset))
    start_idx = round(offset)
    end_idx = start_idx + len(other.value) - 1

    # Bounds check
    if start_idx < 0 or end_idx >= len(target_times) or start_idx >= len(target_times):
        logger.warning(
            "Injection range [%s:%s] out of bounds for timeseries of length %s; skipping injection",
            start_idx,
            end_idx,
            len(target_times),
        )
        return timeseries

    # Create a new TimeSeries with explicit parameters to avoid floating-point precision issues
    injected_data = timeseries.value.copy()
    inject_len = min(len(other.value), end_idx - start_idx + 1)
    injected_data[start_idx : start_idx + inject_len] += other.value[:inject_len]
    injected = TimeSeries(
        injected_data,
        t0=timeseries.t0,
        dt=timeseries.dt,
        unit=timeseries.unit,
    )
    return injected
