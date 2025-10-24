"""Base class for noise simulators."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

from gwsim.cli.utils.utils import get_file_name_from_template_with_dict
from gwsim.simulator.base import Simulator
from gwsim.simulator.mixin.detector import DetectorMixin
from gwsim.simulator.mixin.gwf import GWFOutputMixin
from gwsim.simulator.mixin.randomness import RandomnessMixin
from gwsim.simulator.mixin.time_series import TimeSeriesMixin
from gwsim.simulator.state import StateAttribute
from gwsim.utils.random import get_state


class NoiseSimulator(
    Simulator, RandomnessMixin, DetectorMixin, TimeSeriesMixin, GWFOutputMixin
):  # pylint: disable=duplicate-code
    """Base class for noise simulators."""

    start_time = StateAttribute(0)

    def __init__(
        self,
        sampling_frequency: float,
        duration: float,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
        detectors: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the base noise simulator.

        Args:
            sampling_frequency: Sampling frequency of the noise in Hz.
            duration: Duration of each noise segment in seconds.
            start_time: Start time of the first noise segment in GPS seconds. Default is 0
            max_samples: Maximum number of samples to generate. None means infinite.
            seed: Seed for the random number generator. If None, the RNG is not initialized.
            detectors: List of detector names. Default is None.
            **kwargs: Additional arguments absorbed by subclasses and mixins.
        """
        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
            detectors=detectors,
            **kwargs,
        )

    def save_batch(self, batch: np.ndarray, file_name: str | Path, overwrite: bool = False, **kwargs) -> None:
        """Save a batch of noise data to a file.

        Args:
            batch: Batch of noise data to save.
            file_name: Name of the output file.
            overwrite: Whether to overwrite existing files. Default is False.
            **kwargs: Additional arguments for the output mixin.

        Raises:
            NotImplementedError: If the output mixin does not implement this method.
        """
        suffix = Path(file_name).suffix.lower()
        if suffix == ".gwf":
            save_function = self.save_batch_to_gwf
        else:
            raise NotImplementedError(f"Output format {suffix} not supported by the output mixin.")

        # Check whether the file_name contains the {detector} placeholder
        if "{detector}" in str(file_name).replace(" ", ""):
            # Check whether self.detectors is set
            if self.detectors is None:
                raise ValueError(
                    "The file_name contains the {detector} placeholder, but the simulator does not have detectors set."
                )
            # Check whether the dimension of batch matches number of detectors
            if len(batch.shape) == 1:
                batch = batch[None, :]
            # Check whether the length of batch matches number of detectors
            if batch.shape[0] != len(self.detectors):
                raise ValueError(
                    f"The batch has {batch.shape[0]} channels, but the simulator has {len(self.detectors)} detectors."
                )
            # Save each detector's data separately
            for i, detector in enumerate(self.detectors):
                detector_file_name = get_file_name_from_template_with_dict(
                    template=str(file_name),
                    values={
                        "detector": detector,
                    },
                )
                self.save_batch_to_gwf(
                    batch=batch[i, :],
                    file_path=detector_file_name,
                    overwrite=overwrite,
                    **kwargs,
                )
        else:
            save_function(
                batch=batch,
                file_path=file_name,
                overwrite=overwrite,
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

    def update_state(self) -> None:
        """Update internal state after each sample generation.

        This method can be overridden by subclasses to update any internal state
        after generating a sample. The default implementation does nothing.
        """
        self.counter = cast(int, self.counter) + 1
        self.start_time += self.duration
        self.rng_state = get_state()
