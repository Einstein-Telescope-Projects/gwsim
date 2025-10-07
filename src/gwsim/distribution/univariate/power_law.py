"""A power law distribution."""

from __future__ import annotations

import numpy as np

from ...utils.random import get_rng
from .uniform import Uniform
from .univariate import UnivariateDistribution


class PowerLaw(UnivariateDistribution):
    """
    A class that represents a power law distribution.

    The probability density function for a power law distribution is given by:

    .. math::
        p(x | \alpha, x_{\text{min}}, x_{\text{max}}) =
        \frac{\alpha - 1}{x_{\text{min}}} \\left( \frac{x}{x_{\text{min}}} \right)^{-\alpha}

    for :math:`\alpha > 0`, and

    .. math::
        p(x | \alpha = 0, x_{\text{min}}, x_{\text{max}}) =
        \frac{1}{x_{\text{min}}} \\quad \text{(uniform distribution for } \alpha = 0 \text{)}

    where:
    - :math:`\alpha` is the exponent parameter (typically greater than 1).
    - :math:`x_{\text{min}}` is the minimum value of `x` (the cutoff below which the distribution is zero).
    - :math:`x_{\text{max}}` is the maximum value of `x`.
    - :math:`x` is the variable of interest.
    """

    def __new__(cls, alpha: float, x_min: float, x_max: float | None = None, name: str | None = None):
        """Return a PowerLaw instance, or a Uniform instance if alpha == 0.

        Args:
            alpha (float): Spectral index.
            x_min (float): Minimum.
            x_max (Union[float, None], optional): Maximum. Defaults to None.
            name (Union[str, None], optional): Name. Defaults to None.

        Raises:
            ValueError: If alpha is 0, x_max must be provided.

        Returns:
            Union[Uniform, PowerLaw]: The appropriate distribution class.
        """
        if alpha == 0:
            if x_max is None:
                raise ValueError(f"x_max = {x_max} must be provided for a uniform distribution.")
            if name is None:
                name = "uniform"
            return Uniform(x_min=x_min, x_max=x_max, name=name)
        return super().__new__(cls)

    def __init__(self, alpha: float, x_min: float, x_max: float = np.inf, name: str = "power_law"):
        """Power law distribution.

        Args:
            alpha (float): Spectral index.
            x_min (float): Minimum.
            x_max (float, optional): Maximum. Defaults to inf.
            name (str, optional): Name. Defaults to 'power_law'.

        Raises:
            ValueError: x_min must be non-negative.
            ValueError: alpha is negative and x_min cannot be zero.
            ValueError: alpha >= -1 but x_max is not provided.
        """
        super().__init__(x_min=x_min, x_max=x_max, name=name)
        if x_min < 0:
            raise ValueError(f"x_min = {x_min} must be non-negative.")
        if alpha < 0.0 and x_min == 0.0:
            raise ValueError(f"alpha = {alpha} is negative and x_min = {x_min} cannot be zero.")
        if alpha >= -1.0 and x_max is None:
            raise ValueError(f"alpha = {alpha} >= -1 but x_max = {x_max} is not provided.")
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        """Spectral index.

        Returns:
            float: Spectral index.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        """Set the spectral index.

        Args:
            value (float): Spectral index.
        """
        self._alpha = value
        self._log_norm = None

    def calculate_log_norm(self) -> float:
        """Log normalization constant.

        Raises:
            Exception: Uncatched exceptions in __init__.

        Returns:
            float: Log normalization constant.
        """
        if self.alpha != -1 and self.x_max is not None:
            return np.log(1 + self.alpha) - np.log(self.x_max ** (1 + self.alpha) - self.x_min ** (1 + self.alpha))
        if self.alpha < -1 and self.x_max is None:
            return np.log(-1 - self.alpha) - (1 + self.alpha) * np.log(self.x_min)
        if self.alpha == -1:
            return -np.log(np.log(self.x_max) - np.log(self.x_min))
        raise ValueError(
            f"Unexpected error. alpha = {self.alpha}, x_min = {self.x_min}, and x_max = {self.x_max}"
            "miss the conditions to compute the log normalization constant."
        )

    def log_prob(self, samples: np.ndarray, given: np.ndarray | None = None) -> np.ndarray:
        """Log probability density.

        Args:
            samples (np.ndarray): Samples.

        Returns:
            np.ndarray: Log probability density.
        """
        return self.alpha * np.log(samples) + self.log_norm

    def sample(self, number: int = 1, given: np.ndarray | None = None) -> np.ndarray:
        """Draw samples.

        Args:
            number (int, optional): Number of samples. Defaults to 1.

        Returns:
            np.ndarray: An array of samples.
        """
        if self.alpha != -1 and self.x_max is not None:
            a = self.x_max ** (1 + self.alpha)
            b = self.x_min ** (1 + self.alpha)
            return (get_rng().random((number, 1)) * (a - b) + b) ** (1 / (1 + self.alpha))
        if self.alpha == -1:
            log_x_min = np.log(self.x_min)
            log_x_max = np.log(self.x_max)
            return np.exp(get_rng().random((number, 1)) * (log_x_max - log_x_min) + log_x_min)
        return (1.0 - get_rng().random((number, 1))) ** (1 / (1 + self.alpha)) * self.x_min
