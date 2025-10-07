"""A parent class for all univariate distribution classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class UnivariateDistribution(ABC):
    """A univariate distribution."""

    def __init__(self, x_min: float = -np.inf, x_max: float = np.inf, name: str | None = None):
        """Distribution.

        Args:
            x_min (float, optional): Minimum. Defaults to -inf.
            x_max (float, optional): Maximum. Defaults to inf.
            name (Union[str, None], optional): Name. Defaults to None.

        Raises:
            ValueError: x_min must be smaller than x_max.
            ValueError: If name == 'mass_1', x_min must be positive.
        """
        self._x_min = x_min
        self._x_max = x_max
        self._name = name
        self._log_norm = None
        if x_min >= x_max:
            raise ValueError(f"x_min = {x_min} must be smaller than x_max = {x_max}.")
        if name == "mass_1":
            if x_min <= 0:
                raise ValueError(f"For {name} distribution, x_min = {x_min} must be positive.")

    def __repr__(self) -> str:
        """Printable representation of the object.

        Returns:
            str: Printable representation of the object.
        """
        return f"UnivariateDistribution({self.name})"

    @property
    def x_min(self) -> float:
        """Minimum.

        Returns:
            Union[float, None]: Minimum.
        """
        return self._x_min

    @x_min.setter
    def x_min(self, value: float):
        """Set the minimum.

        Resetting the minimum will force recalculating the log normalization constant.

        Args:
            value (float): Minimum.

        Raises:
            ValueError: value must be smaller than x_max.
        """
        if value >= self.x_max:
            raise ValueError(f"value = {value} must be smaller than x_max = {self.x_max}.")
        self._x_min = value
        self._log_norm = None

    @property
    def x_max(self) -> float:
        """Maximum.

        Returns:
            Union[float, None]: Maximum
        """
        return self._x_max

    @x_max.setter
    def x_max(self, value: float):
        """Set the maximum.

        Resetting the maximum will force recalculating the log normalization constant.

        Args:
            value (float): Maximum.

        Raises:
            ValueError: value must be greater than x_min.
        """
        if value <= self.x_min:
            raise ValueError(f"value = {value} must be greater than x_min = {self.x_min}.")
        self._x_max = value
        self._log_norm = None

    @property
    def name(self) -> str | None:
        """Name of the distribution.

        Returns:
            str: Name of the distribution.
        """
        return self._name

    @name.setter
    def name(self, value: str | None):
        """Set the

        Args:
            value (str): _description_
        """
        self._name = value

    @property
    def log_norm(self) -> float:
        """Log normalization constant.

        Returns:
            float: Log normalization constant.
        """
        if self._log_norm is None:
            self._log_norm = self.calculate_log_norm()
        return self._log_norm

    @abstractmethod
    def calculate_log_norm(self) -> float:
        """Calculate the log normalization constant.

        Raises:
            NotImplementedError: This method should be implemented in a subclass.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def log_prob(self, samples: np.ndarray, given: np.ndarray | None) -> np.ndarray:
        """Log probability density for continuous distributions,
        of log probability for discrete distributions.

        Args:
            samples (np.ndarray): Samples.
            given (Union[np.ndarray, None]): Conditioned samples.

        Raises:
            NotImplementedError: This method should be implemented in a subclass.

        Returns:
            np.ndarray: Log probability density for continuous distributions,
            or log probability for discrete distributions.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def prob(self, samples: np.ndarray, given: np.ndarray | None) -> np.ndarray:
        """Probability density for continuous distributions or probability for discrete distribution.

        Args:
            samples (np.ndarray): Samples.
            given (Union[np.ndarray, None]): Conditioned samples.

        Returns:
            np.ndarray: Probability density for continuous distributions or probability for discrete distribution.
        """
        return np.exp(self.log_prob(samples=samples, given=given))

    @abstractmethod
    def sample(self, number: int = 1, given: np.ndarray | None = None) -> np.ndarray:
        """Draw samples from the distribution $p(x | y)$.

        Args:
            number (int, optional): Number of samples. Defaults to 1.
            given (Union[np.ndarray,None], optional): Values y to be conditioned on. Defaults to None.

        Raises:
            NotImplementedError: This method should be implemented in a subclass.

        Returns:
            np.ndarray: An array of samples.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")
