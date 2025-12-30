# pylint: disable=too-few-public-methods
"""
Frequency-domain calibration models for gravitational-wave detectors.

This module defines calibration utilities used to model fixed,
frequency-dependent amplitude and phase errors applied to strain
data in the frequency domain.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


class CalibrationModel:
    """Frequency-domain detector calibration model.

    This model represents fixed, frequency-dependent amplitude and phase
    calibration errors and provides a transfer function C(f) that multiplies
    the strain in the frequency domain.

    The phase response uses the standard LIGO/Bilby rational form:
        (2 + i*delta_phi) / (2 - i*delta_phi)
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        delta_amplitude: np.ndarray,
        delta_phase: np.ndarray,
    ):
        self._amp_interp = interp1d(
            frequencies,
            delta_amplitude,
            bounds_error=False,
            fill_value=0.0,
        )
        self._phase_interp = interp1d(
            frequencies,
            delta_phase,
            bounds_error=False,
            fill_value=0.0,
        )

    def transfer_function(self, frequency_array: np.ndarray) -> np.ndarray:
        """Return the complex calibration transfer function C(f)."""

        amp = self._amp_interp(frequency_array)
        phase = self._phase_interp(frequency_array)

        phase_factor = (2.0 + 1j * phase) / (2.0 - 1j * phase)
        return (1.0 + amp) * phase_factor
