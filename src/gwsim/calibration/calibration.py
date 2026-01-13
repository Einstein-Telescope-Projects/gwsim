# pylint: disable=too-few-public-methods
"""
Bilby-compatible cubic spline calibration model.
"""

from __future__ import annotations

import numpy as np


class CalibrationModel:
    """Bilby-equivalent cubic spline frequency-domain calibration."""

    def __init__(
        self,
        recalibration_parameters: dict[str, float],
        detector_name: str,
        minimum_frequency: float,
        maximum_frequency: float,
        n_points: int,
    ):
        if n_points < 4:
            raise ValueError("Cubic spline calibration requires at least 4 nodes.")
        if minimum_frequency <= 0:
            raise ValueError("minimum_frequency must be positive.")
        if maximum_frequency <= minimum_frequency:
            raise ValueError("maximum_frequency must be greater than minimum_frequency.")
        self.detector_name = detector_name
        self.n_points = n_points

        # Log-spaced spline nodes (Bilby convention)
        self.log_spline_points = np.linspace(
            np.log10(minimum_frequency),
            np.log10(maximum_frequency),
            n_points,
        )

        self.delta_log_spline_points = self.log_spline_points[1] - self.log_spline_points[0]

        # Store parameters
        self.params = recalibration_parameters

        # Precompute node → spline coefficient matrix
        self._setup_spline_coefficients()

    def _setup_spline_coefficients(self):
        n = self.n_points

        tmp1 = np.zeros((n, n))
        tmp2 = np.zeros((n, n))

        tmp1[0, 0] = -1
        tmp1[0, 1] = 2
        tmp1[0, 2] = -1
        tmp1[-1, -3] = -1
        tmp1[-1, -2] = 2
        tmp1[-1, -1] = -1

        for i in range(1, n - 1):
            tmp1[i, i - 1] = 1 / 6
            tmp1[i, i] = 2 / 3
            tmp1[i, i + 1] = 1 / 6
            tmp2[i, i - 1] = 1
            tmp2[i, i] = -2
            tmp2[i, i + 1] = 1

        self.nodes_to_spline_coefficients = np.linalg.solve(tmp1, tmp2)

    def _evaluate_spline(self, kind, a, b, c, d, previous_nodes):
        parameters = np.array([self.params[f"recalib_{self.detector_name}_{kind}_{i}"] for i in range(self.n_points)])

        next_nodes = previous_nodes + 1
        spline_coeffs = self.nodes_to_spline_coefficients @ parameters

        return (
            a * parameters[previous_nodes]
            + b * parameters[next_nodes]
            + c * spline_coeffs[previous_nodes]
            + d * spline_coeffs[next_nodes]
        )

    def transfer_function(self, frequency_array: np.ndarray) -> np.ndarray:
        """Bilby-equivalent get_calibration_factor()."""

        log10f = np.log10(frequency_array)

        x = (log10f - self.log_spline_points[0]) / self.delta_log_spline_points
        previous_nodes = np.clip(
            np.floor(x).astype(int),
            0,
            self.n_points - 2,
        )

        b = x - previous_nodes
        a = 1 - b
        c = (a**3 - a) / 6
        d = (b**3 - b) / 6

        delta_amp = self._evaluate_spline("amplitude", a, b, c, d, previous_nodes)
        delta_phase = self._evaluate_spline("phase", a, b, c, d, previous_nodes)

        return (1 + delta_amp) * (2 + 1j * delta_phase) / (2 - 1j * delta_phase)
