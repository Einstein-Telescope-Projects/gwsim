"""Initialization for the signal module."""

from __future__ import annotations

from gwsim.signal.base import SignalSimulator
from gwsim.signal.cbc import CBCSignalSimulator

__all__ = [
    "CBCSignalSimulator",
    "SignalSimulator",
]
