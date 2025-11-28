"""Time_series module."""

from __future__ import annotations

from gwsim.data.time_series import inject
from gwsim.data.time_series.time_series import TimeSeries
from gwsim.data.time_series.time_series_list import TimeSeriesList

__all__ = ["TimeSeries", "TimeSeriesList", "inject"]
