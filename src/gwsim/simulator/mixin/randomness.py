"""Mixins for GW simulation classes for randomness handling."""

from __future__ import annotations

import logging

from gwsim.simulator.state import StateAttribute
from gwsim.utils.random import Generator, get_rng, get_state, set_seed, set_state

logger = logging.getLogger("gwsim")


class RandomnessMixin:
    """Mixin providing random number generation capabilities.

    This mixin should be used by simulators that require randomness.
    It provides RNG state management, seed handling, and state persistence.

    Example:
        >>> class MySimulator(RandomnessMixin, Simulator):
        ...     def __init__(self, seed=None, **kwargs):
        ...         super().__init__(**kwargs)
        ...         self.seed = seed
        ...
        ...     def simulate(self):
        ...         # Use self.rng for random operations
        ...         return self.rng.random()
    """

    # State attributes for RNG persistence
    rng_state = StateAttribute(
        lambda self: get_state() if self.rng else None, post_set_hook=lambda self, state: self.init_rng(state)
    )

    def __init__(self, seed: int | None = None, **kwargs):
        """Initialize the randomness mixin.

        Args:
            seed: Random seed for reproducibility. If None, no RNG is created.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(**kwargs)
        self.seed = seed

    @property
    def seed(self) -> int | None:
        """Get the random seed.

        Returns:
            The random seed used for initialization.
        """
        return self._seed

    @seed.setter
    def seed(self, value: int | None) -> None:
        """Set the random seed and reinitialize RNG.

        Args:
            value: New random seed. If None, RNG is disabled.
        """
        self._seed = value
        if value is not None:
            set_seed(value)
            self.rng = get_rng()
            self.rng_state = get_state()
        else:
            self.rng = None

    @property
    def rng(self) -> Generator | None:
        """Get the random number generator.

        Returns:
            Random number generator instance or None if no seed was set.
        """
        return self._rng

    @rng.setter
    def rng(self, value: Generator | None) -> None:
        """Set the random number generator.

        Args:
            value: Random number generator instance.
        """
        self._rng = value

    def init_rng(self, state: dict | None) -> None:
        """Initialize RNG from saved state.

        Args:
            state: Saved RNG state dictionary.
        """
        if state is not None and self.rng is not None:
            set_state(state)
            self._rng = get_rng()
        else:
            logger.debug("init_rng called but state is %s and self.rng is %s.", state, self.rng)

    @property
    def metadata(self) -> dict:
        """Get metadata including seed information.

        Returns:
            Dictionary containing seed and other metadata.
        """
        metadata = {"seed": self.seed}
        return metadata
