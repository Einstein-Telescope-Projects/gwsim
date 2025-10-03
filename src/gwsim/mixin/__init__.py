"""Mixins for GW simulation classes."""

from __future__ import annotations

from .randomness import RandomnessMixin
from .time_series import TimeSeriesMixin

__all__ = ["RandomnessMixin", "TimeSeriesMixin"]
