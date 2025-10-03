"""Base class for noise simulators."""

from __future__ import annotations

from typing import Any

from ..mixin.randomness import RandomnessMixin
from ..simulator.base import Simulator
from ..simulator.state import StateAttribute
from ..version import __version__


class NoiseSimulator(Simulator, RandomnessMixin):
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
        super().__init__(max_samples=max_samples, seed=seed, **kwargs)
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.start_time = start_time

    @property
    def metadata(self) -> dict:
        """Get a dictionary of metadata.
        This can be overridden by the subclass.

        Returns:
            dict: A dictionary of metadata.
        """
        return {
            "max_samples": self.max_samples,
            "rng_state": self.rng_state,
            "sampling_frequency": self.sampling_frequency,
            "duration": self.duration,
            "start_time": self.start_time,
            "version": __version__,
        }

    def next(self) -> Any:
        """Next noise segment."""
        raise NotImplementedError("Not implemented.")

    # def update_state(self) -> None:
    #     """Update the state of the simulator after generating a noise segment."""
    #     # Type ignore needed due to StateAttribute descriptor behavior
    #     self.counter += 1
    #     self.start_time += self.duration
    #     if self.rng is not None:
    #         self.rng_state = get_state()

    # def save_batch(self, batch: np.ndarray, file_name: str | Path, overwrite: bool = False, **kwargs) -> None:
    #     """Save a batch of noise segments to a file."""
    #     file_name = Path(file_name)
    #     if file_name.suffix in [".h5", ".hdf5"]:
    #         self._save_batch_hdf5(batch=batch, file_name=file_name, overwrite=overwrite, **kwargs)
    #     elif file_name.suffix == ".gwf":
    #         self._save_batch_gwf(batch=batch, file_name=file_name, overwrite=overwrite, **kwargs)
    #     else:
    #         raise ValueError(
    #             f"Suffix of file_name = {file_name} is not supported. Use ['.h5', '.hdf5'] for HDF5 files,"
    #             "and '.gwf' for frame files."
    #         )

    # @check_file_overwrite()
    # def _save_batch_hdf5(
    #     self,
    #     batch: np.ndarray,
    #     file_name: str | Path,
    #     overwrite: bool = False,
    #     dataset_name: str = "strain",
    # ) -> None:
    #     """Save a batch of noise segments to an HDF5 file."""
    #     with h5py.File(file_name, "w") as f:
    #         # Add dataset.
    #         f.create_dataset(dataset_name, data=batch)

    # @check_file_overwrite()
    # def _save_batch_gwf(
    #     self, batch: np.ndarray, file_name: str | Path, overwrite: bool = False, channel: str = "strain"
    # ) -> None:
    #     """Save a batch of noise segments to a GWF frame file."""
    #     # Create a pycbc TimeSeries instance.
    #     time_series = TimeSeries(initial_array=batch, delta_t=1 / self.sampling_frequency, epoch=self.start_time)

    #     # Write to frame file.
    #     write_frame(location=str(file_name), channels=channel, timeseries=time_series)
