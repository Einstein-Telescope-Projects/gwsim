from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pycbc.frame import write_frame
from pycbc.types.timeseries import TimeSeries

from gwsim.simulator.base import Simulator
from gwsim.simulator.mixin.detector import DetectorMixin
from gwsim.simulator.mixin.population_reader import PopulationReaderMixin
from gwsim.simulator.mixin.time_series import TimeSeriesMixin

from ..generator.base import Generator
from ..generator.state import StateAttribute
from ..utils.io import check_file_overwrite
from ..version import __version__


class BaseSignal(Generator):
    start_time = StateAttribute(0)

    def __init__(
        self,
        sampling_frequency: float,
        duration: float,
        start_time: float = 0,
        max_samples: int | None = None,
    ) -> None:
        super().__init__(max_samples=max_samples)
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
        raise NotImplementedError("Not implemented.")

    def update_state(self) -> None:
        self.sample_counter += 1
        self.start_time += self.duration

    def save_batch(self, batch: np.ndarray, file_name: str | Path, overwrite: bool = False, **kwargs) -> None:
        file_name = Path(file_name)
        if file_name.suffix in [".h5", ".hdf5"]:
            self._save_batch_hdf5(batch=batch, file_name=file_name, overwrite=overwrite, **kwargs)
        elif file_name.suffix == ".gwf":
            self._save_batch_gwf(batch=batch, file_name=file_name, overwrite=overwrite, **kwargs)
        else:
            raise ValueError(
                f"Suffix of file_name = {file_name} is not supported. Use ['.h5', '.hdf5'] for HDF5 files,"
                "and '.gwf' for frame files."
            )

    @check_file_overwrite()
    def _save_batch_hdf5(
        self,
        batch: np.ndarray,
        file_name: str | Path,
        overwrite: bool = False,
        dataset_name: str = "strain",
    ) -> None:
        with h5py.File(file_name, "w") as f:
            # Add dataset.
            f.create_dataset(dataset_name, data=batch)

    @check_file_overwrite()
    def _save_batch_gwf(
        self, batch: np.ndarray, file_name: str | Path, overwrite: bool = False, channel: str = "strain"
    ) -> None:
        # Create a pycbc TimeSeries instance.
        time_series = TimeSeries(initial_array=batch, delta_t=1 / self.sampling_frequency, epoch=self.start_time)

        # Write to frame file.
        write_frame(location=str(file_name), channels=channel, timeseries=time_series)


class SignalSimulator(PopulationReaderMixin, TimeSeriesMixin, DetectorMixin, Simulator):
    def __init__(
        self,
        population_file: str | Path,
        start_time: int = 0,
        duration: float = 1024,
        sampling_frequency: float = 4096,
        max_samples: int | None = None,
        dtype: type = np.float64,
        **kwargs,
    ) -> None:
        pass
