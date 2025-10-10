"""Base class for noise simulators."""

from __future__ import annotations

from typing import Any

from gwsim.simulator.base import Simulator
from gwsim.simulator.mixin.gwf import GWFOutputMixin
from gwsim.simulator.mixin.randomness import RandomnessMixin
from gwsim.simulator.mixin.time_series import TimeSeriesMixin
from gwsim.simulator.state import StateAttribute


class NoiseSimulator(Simulator, RandomnessMixin, TimeSeriesMixin, GWFOutputMixin):
    """Base class for noise simulators."""

    start_time = StateAttribute(0)

    def __init__(
        self,
        sampling_frequency: float,
        duration: float,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize the base noise simulator.

        Args:
            sampling_frequency: Sampling frequency of the noise in Hz.
            duration: Duration of each noise segment in seconds.
            start_time: Start time of the first noise segment in GPS seconds. Default is 0
            max_samples: Maximum number of samples to generate. None means infinite.
            seed: Seed for the random number generator. If None, the RNG is not initialized.
            **kwargs: Additional arguments absorbed by subclasses and mixins.
        """
        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
            **kwargs,
        )

    @property
    def metadata(self) -> dict:
        """Get a dictionary of metadata.
        This can be overridden by the subclass.

        Returns:
            dict: A dictionary of metadata.
        """
        # Get metadata from all parent classes using cooperative inheritance
        metadata = super().metadata

        return metadata

    def next(self) -> Any:
        """Next noise segment."""
        raise NotImplementedError("Not implemented.")
