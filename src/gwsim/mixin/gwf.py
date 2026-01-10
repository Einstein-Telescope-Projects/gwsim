"""GWF (Gravitational Wave Frame) file output utilities using gwpy."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from gwpy.io.gwf import write_frames
from gwpy.timeseries import TimeSeries


def save_timeseries_to_gwf(
    data: np.ndarray,
    file_path: str | Path,
    channel: str = "H1:STRAIN",
    sampling_frequency: float = 4096,
    start_time: float = 0,
    overwrite: bool = False,
) -> None:
    """Save time series data to GWF frame file using gwpy.

    Args:
        data: Time series data array.
        file_path: Output GWF file path.
        channel: Channel name (e.g., "H1:STRAIN"). Default is "H1:STRAIN".
        sampling_frequency: Sampling rate in Hz. Default is 4096.
        start_time: GPS start time. Default is 0.
        overwrite: Whether to overwrite existing files. Default is False.

    Raises:
        ImportError: If gwpy is not available.
        FileExistsError: If file exists and overwrite=False.
        ValueError: If data is empty or parameters are invalid.
    """

    if len(data) == 0:
        raise ValueError("Data array cannot be empty")

    if sampling_frequency <= 0:
        raise ValueError("Sample rate must be positive")

    # Convert to Path object
    file_path = Path(file_path)

    # Check if file exists
    if file_path.exists() and not overwrite:
        raise FileExistsError(f"File {file_path} already exists and overwrite=False")

    # Create gwpy TimeSeries
    timeseries = TimeSeries(data=data, t0=start_time, sample_rate=sampling_frequency, channel=channel, name=channel)

    # Ensure output directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to GWF file
    timeseries.write(str(file_path), format="gwf")


def combine_timeseries_to_gwf(
    timeseries_list: list[tuple[np.ndarray, str]],
    file_path: str | Path,
    sampling_frequency: float = 4096,
    start_time: float = 0,
    overwrite: bool = False,
) -> None:
    """Combine multiple time series into a single GWF file.

    Args:
        timeseries_list: List of (data, channel_name) tuples.
        file_path: Output GWF file path.
        sampling_frequency: Sampling rate in Hz. Default is 4096.
        start_time: GPS start time. Default is 0.
        overwrite: Whether to overwrite existing files. Default is False.

    Raises:
        ImportError: If gwpy is not available.
        ValueError: If timeseries_list is empty or data shapes don't match.
    """
    if not timeseries_list:
        raise ValueError("timeseries_list cannot be empty")

    # Convert to Path object
    file_path = Path(file_path)

    # Check if file exists
    if file_path.exists() and not overwrite:
        raise FileExistsError(f"File {file_path} already exists and overwrite=False")

    # Validate all data arrays have the same length
    lengths = [len(data) for data, _ in timeseries_list]
    if not all(length == lengths[0] for length in lengths):
        raise ValueError("All time series must have the same length")

    # Create gwpy TimeSeries objects
    gwpy_timeseries = []
    for data, channel in timeseries_list:
        ts = TimeSeries(data, sampling_frequency=sampling_frequency, epoch=start_time, name=channel, channel=channel)
        gwpy_timeseries.append(ts)

    # Ensure output directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write multiple channels to GWF file
    # gwpy handles multiple timeseries automatically
    write_frames(str(file_path), gwpy_timeseries)


class GWFOutputMixin:
    """Mixin to add GWF file output capabilities to simulators.

    This mixin provides methods for saving simulator output to GWF frame files
    using gwpy. It assumes the simulator has sampling_frequency, start_time, and
    generates numpy arrays.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize GWFOutputMixin.

        Args:
            **kwargs: Additional arguments absorbed by subclasses.
        """
        super().__init__(**kwargs)

    def save_to_gwf(
        self,
        data: np.ndarray,
        file_path: str | Path,
        channel: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Save simulator data to GWF file.

        Args:
            data: Time series data from simulator.
            file_path: Output GWF file path.
            channel: Channel name. If None, uses a default based on class name.
            overwrite: Whether to overwrite existing files.
        """
        # Get parameters from simulator
        sampling_frequency = getattr(self, "sampling_frequency", None)
        start_time = getattr(self, "start_time", 0)

        if sampling_frequency is None:
            raise ValueError("Simulator must have sampling_frequency attribute")

        # Generate default channel name if not provided
        if channel is None:
            class_name = self.__class__.__name__
            # Convert CamelCase to UPPER_CASE
            channel = re.sub("([a-z0-9])([A-Z])", r"\1_\2", class_name).upper()
            channel = f"SIM:{channel}"

        save_timeseries_to_gwf(
            data=data,
            file_path=file_path,
            channel=channel,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            overwrite=overwrite,
        )

    def save_batch_to_gwf(  # pylint: disable=unused-argument
        self,
        batch: list[np.ndarray] | np.ndarray,
        file_path: str | Path,
        channel: str | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """Save a batch of simulator data to GWF file.

        Args:
            batch: Batch of time series data (list of arrays or 2D array).
            file_path: Output GWF file path.
            channel: Channel name. If None, uses a default based on class name.
            overwrite: Whether to overwrite existing files.
        """
        # Flatten batch if needed
        if isinstance(batch, list):
            concatenated = np.concatenate(batch)
        elif isinstance(batch, np.ndarray) and batch.ndim == 2:  # noqa: PLR2004
            concatenated = batch.flatten()
        else:
            concatenated = batch

        self.save_to_gwf(
            data=concatenated,
            file_path=file_path,
            channel=channel,
            overwrite=overwrite,
        )
