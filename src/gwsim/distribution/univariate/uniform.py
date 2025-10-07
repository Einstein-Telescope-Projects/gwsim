"""A uniform distribution."""

from __future__ import annotations

import numpy as np

from ...utils.random import get_rng
from .univariate import UnivariateDistribution


class Uniform(UnivariateDistribution):
    """A uniform distribution."""

    def __init__(self, x_min: float = 0.0, x_max: float = 1.0, name: str = "uniform"):
        """Uniform distribution.

        Args:
            x_min (float, optional): Minimum. Defaults to 0..
            x_max (float, optional): Maximum. Defaults to 1..
            name (str, optional): Name of the distribution. Defaults to 'uniform'.
        """
        super().__init__(x_min=x_min, x_max=x_max, name=name)

    def __repr__(self) -> str:
        """Printable representation of the object.

        Returns:
            str: Printable representation of the object.
        """
        return f"Uniform({self.name})"

    def calculate_log_norm(self) -> float:
        """Calculate the log normalization constant.

        Returns:
            float: Log normalization constant.
        """
        return -np.log(self._x_max - self._x_min)

    def log_prob(self, samples: np.ndarray, given: np.ndarray | None = None) -> np.ndarray:
        """Log probability density.

        Args:
            samples (np.ndarray): Samples
            given (Union[np.ndarray, None], optional): Conditioned samples. Defaults to None.

        Returns:
            np.ndarray: Log probability density.
        """
        return np.log(samples) + self.log_norm

    def sample(self, number: int = 1, given=None) -> np.ndarray:
        """Draw samples.

        Args:
            number (int, optional): Number of samples. Defaults to 1.
            given (None, optional): A placeholder. Defaults to None.

        Returns:
            np.ndarray: Samples.
        """
        return get_rng().uniform(low=self.x_min, high=self.x_max, size=(number, 1))
