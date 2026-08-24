"""Mixins for simulator classes providing optional functionality."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy.units.quantity import Quantity
from gwpy.timeseries import TimeSeries as GWPyTimeSeries

from gwmock.cli.utils.config_resolution import resolve_max_samples
from gwmock.cli.utils.template import expand_template_variables
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.simulator.state import StateAttribute
from gwmock.utils.datetime_parser import parse_duration_to_seconds

logger = logging.getLogger("gwmock")


def _injection_record(chunk: TimeSeries) -> dict[str, Any] | None:
    """Return the injection record a chunk carries, or ``None`` if it carries none.

    Both are stamped at generation and copied onto a tail when a chunk crosses a segment boundary,
    which is what lets a carried-forward chunk still say what it is. A chunk with parameters but no
    ``event_id`` is recorded with ``event_id`` ``None`` rather than dropped: the parameters are still
    the provenance, and dropping it would silently lose a signal from the record.
    """
    parameters = chunk.metadata.get("injection_parameters")
    if parameters is None:
        return None
    event_id = chunk.metadata.get("event_id")
    return {"event_id": event_id, "parameters": dict(parameters)}


def _contributing_injections(segment: TimeSeries, chunks: Iterable[TimeSeries]) -> list[dict[str, Any]]:
    """Return injection records for the chunks that place at least one sample in *segment*."""
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        if not segment.contributes_samples(chunk):
            continue
        record = _injection_record(chunk)
        if record is not None:
            records.append(record)
    return records


def _merge_injection_records(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatenate injection records, keeping the first of each ``event_id`` and order stable.

    Deduplicated because one signal can reach a segment through several chunks -- a multi-detector
    generation emits one chunk per detector, all carrying the same record -- and a provenance list
    naming the same event three times would read as three injections.

    Records whose ``event_id`` is ``None`` are never merged together: without an id there is nothing
    to say two chunks are the same signal, so collapsing them would drop real injections.
    """
    merged: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for group in groups:
        for record in group:
            event_id = record.get("event_id")
            if event_id is not None:
                if event_id in seen:
                    continue
                seen.add(event_id)
            merged.append(record)
    return merged


class TimeSeriesMixin:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Mixin providing timing and duration management.

    This mixin adds time-based parameters commonly used
    in gravitational wave simulations.
    """

    start_time = StateAttribute(Quantity(0, unit="s"))
    #: Spillover: the part of a chunk that extends past the segment being built, waiting for the
    #: next one.
    #:
    #: **Not a `StateAttribute`, deliberately, and it is still persisted.** It used to be neither,
    #: and a resumed run started with none of it: the tail of any signal crossing the resume point
    #: was never placed and the following segment lost that content silently -- 7.280e-23 to exactly
    #: 0.0, the merger absent, with the frames before it bit-identical.
    #:
    #: The obvious fix, making this stateful, does not work: `state` is serialized into every *batch
    #: metadata record* as well as into the checkpoint, so it would write spillover samples into a
    #: provenance document meant to stay small and readable, and those records are dumped with plain
    #: `json`, which has no encoder for a `TimeSeriesList`. So it travels as its own checkpoint
    #: field, in the entry `simulator_tails` keeps for the simulator that produced it, and is handed
    #: back only to that simulator and only to the batch immediately after the one it came from.
    cached_data_chunks = TimeSeriesList()
    #: Injection records for every signal that reaches the segment currently being built, including
    #: signals generated for an *earlier* segment whose content extends into this one. Rebuilt per
    #: segment by :meth:`simulate`; a subclass writing provenance should union this with whatever it
    #: generated itself. Empty for simulators that do not inject.
    #:
    #: Survives a checkpoint resume, because the chunks carrying these records do: see
    #: ``cached_data_chunks`` above. It did not until the spillover was persisted -- a resumed run
    #: lost both the samples and their provenance at the resume boundary.
    carried_injections: list[dict[str, Any]]

    def __init__(
        self,
        start_time: int = 0,
        duration: float = 4,
        total_duration: float | str | None = None,
        sampling_frequency: float = 4096,
        num_of_channels: int | None = None,
        dtype: type = np.float64,
        **kwargs,
    ):
        """Initialize timing parameters.

        Args:
            start_time: Start time in GPS seconds. Default is 0.
            duration: Duration of simulation in seconds. Default is 4.
            total_duration
            sampling_frequency: Sampling frequency in Hz. Default is 4096.
            dtype: Data type for the time series data. Default is np.float64.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(**kwargs)
        # TimeSeriesMixin is the last mixin in the hierarchy, so no super().__init__() call needed
        self.start_time = Quantity(start_time, unit="s")
        # Per instance, not a class attribute: two simulators in one process must not share the
        # list of signals reaching the segment each is building.
        self.carried_injections = []
        self.duration = duration
        self.total_duration = total_duration
        self.sampling_frequency = sampling_frequency
        self.dtype = dtype

        # Get the number of channels.
        if num_of_channels is not None:
            self.num_of_channels = num_of_channels
            if (
                "detectors" in kwargs
                and kwargs["detectors"] is not None
                and len(kwargs["detectors"]) != num_of_channels
            ):
                raise ValueError("Number of detectors does not match num_of_channels.")
        elif "detectors" in kwargs and kwargs["detectors"] is not None:
            self.num_of_channels = len(kwargs["detectors"])
        else:
            self.num_of_channels = 1

    @property
    def duration(self) -> Quantity:
        """Get the duration of each simulation segment.

        Returns:
            Duration in seconds.
        """
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        """Set the duration of each simulation segment.

        Args:
            value: Duration in seconds.
        """
        if value <= 0:
            raise ValueError("duration must be positive.")
        self._duration = Quantity(value, unit="s")

    @property
    def sampling_frequency(self) -> Quantity:
        """Get the sampling frequency.

        Returns:
            Sampling frequency in Hz.
        """
        return self._sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, value: float) -> None:
        """Set the sampling frequency.

        Args:
            value: Sampling frequency in Hz.
        """
        if value <= 0:
            raise ValueError("sampling_frequency must be positive.")
        self._sampling_frequency = Quantity(value, unit="Hz")

    @property
    def total_duration(self) -> Quantity:
        """Get the total duration of the simulation.

        Returns:
            Total duration in seconds.
        """
        return self._total_duration

    @total_duration.setter
    def total_duration(self, value: float | str | None) -> None:
        """Set the total duration of the simulation.

        Args:
            value: Total duration in seconds.
        """
        if value is not None:
            if isinstance(value, (float, int)):
                self._total_duration = Quantity(value, unit="s")
            elif isinstance(value, str):
                self._total_duration = Quantity(parse_duration_to_seconds(value), unit="s")
            else:
                raise ValueError("total_duration must be a float, int, or str representing duration.")

            if self.total_duration < 0:
                raise ValueError("total_duration must be non-negative.")

            if self.total_duration < self.duration:
                raise ValueError("total_duration must be greater than or equal to duration.")

            # Round the total_duration to the nearest multiple of duration
            num_segments = round(self.total_duration.value / self.duration.value)
            self._total_duration = Quantity(num_segments * self.duration, unit="s")

            logger.info("Total duration set to %s seconds.", self.total_duration)

            # Set the max_samples based on total_duration and duration
            self.max_samples = resolve_max_samples(
                {"total_duration": self.total_duration.value, "duration": self.duration.value}, {}
            )
            logger.info("Setting max_samples to %s based on total_duration and duration.", self.max_samples)
        else:
            self._total_duration = self.duration * self.max_samples
            # total_duration was not passed to this simulator directly. That does
            # not mean the user never set it: the orchestrator resolves a config
            # total_duration into max_samples upstream and forwards only that, so
            # claiming "total_duration not set" here misleads. Report the derived
            # value factually instead.
            logger.info(
                "Resolved total_duration to %s seconds (duration * max_samples).",
                self.total_duration.value,
            )

    @property
    def end_time(self) -> Quantity:
        """Calculate the end time of the current segment.

        Returns:
            End time in GPS seconds.
        """
        return cast(Quantity, self.start_time + self.duration)

    @property
    def final_end_time(self) -> Quantity:
        """Calculate the final end time of the entire simulation.

        Returns:
            Final end time in GPS seconds.
        """
        return cast(Quantity, self.start_time + self.total_duration)

    def _simulate(self, *args, **kwargs) -> TimeSeriesList:
        """Generate time series data chunks.

        This method should be implemented by subclasses to generate
        the actual time series data.
        """
        raise NotImplementedError("Subclasses must implement the _simulate method.")

    def simulate(self, *args: Any, **kwargs: Any) -> TimeSeries:
        """
        Simulate a segment of time series data.

        Args:
            *args: Positional arguments for the _simulate method.
            **kwargs: Keyword arguments for the _simulate method.

        Returns:
            TimeSeries: Simulated time series segment.
        """
        # First create a new segment
        segment = TimeSeries(
            data=np.zeros(
                (self.num_of_channels, int(self.duration.value * self.sampling_frequency.value)), dtype=self.dtype
            ),
            start_time=self.start_time,
            sampling_frequency=self.sampling_frequency,
        )

        # Which carried-forward chunks reach this segment, recorded *before* injecting them.
        # Injection sums into shared channels, so after this line there is no way to ask which
        # signal contributed to the segment -- and a signal long enough to cross a boundary is
        # exactly the case where the answer is not the segment it was generated in.
        self.carried_injections = _contributing_injections(segment, self.cached_data_chunks)

        # Inject cached data chunks into the segment
        self.cached_data_chunks = segment.inject_from_list(self.cached_data_chunks)

        # Generate new chunks of data
        new_chunks = self._simulate(*args, **kwargs)

        # Chunks generated for this segment that do not actually reach it are excluded for the same
        # reason the carried ones are included: the record names the frames a signal is *in*.
        self.carried_injections = _merge_injection_records(
            self.carried_injections, _contributing_injections(segment, new_chunks)
        )

        # Add the new chunks to the segment
        remaining_chunks = segment.inject_from_list(new_chunks)

        # Add the remaining chunks to the cache
        self.cached_data_chunks.extend(remaining_chunks)

        # Check whether there are chunks that are outside the whole dataset duration
        # Remove the chunks that are outside the total duration
        for i in reversed(range(len(self.cached_data_chunks))):
            chunk = self.cached_data_chunks[i]
            if chunk.start_time >= self.final_end_time:
                logger.info(
                    "Removing cached chunk starting at %s which is outside the total duration ending at %s.",
                    chunk.start_time,
                    self.final_end_time,
                )
                self.cached_data_chunks.pop(i)
            elif chunk.end_time <= self.start_time:
                logger.info(
                    "Removing cached chunk ending at %s which is before the current segment starting at %s.",
                    chunk.end_time,
                    self.start_time,
                )
                self.cached_data_chunks.pop(i)

        return segment

    @property
    def metadata(self) -> dict:
        """Get metadata including timing information.

        Returns:
            Dictionary containing timing parameters and other metadata.
        """
        metadata = {
            "time_series": {
                "arguments": {
                    "start_time": self.start_time,
                    "duration": self.duration,
                    "sampling_frequency": self.sampling_frequency,
                    "num_of_channels": self.num_of_channels,
                    "dtype": str(self.dtype),
                }
            }
        }
        return metadata

    def _save_data(  # pylint: disable=unused-argument
        self,
        data: TimeSeries,
        file_name: str | Path | np.ndarray[Any, np.dtype[np.object_]],
        **kwargs,
    ) -> None:
        """Save time series data to a file.

        Args:
            data: Time series data to save.
            file_name: Path to the output file.
            **kwargs: Additional arguments for the saving function.
        """
        if "channel" in kwargs:
            channel = kwargs.pop("channel")
            channel = expand_template_variables(value=channel, simulator_instance=self)
            if isinstance(channel, str):
                channel = [channel] * data.num_of_channels
            elif isinstance(channel, list):
                if len(channel) != data.num_of_channels:
                    raise ValueError("Length of channel list must match number of channels in data.")
            else:
                raise ValueError("channel must be a string or list of strings.")
        else:
            channel = [None] * data.num_of_channels
        if data.num_of_channels == 1 and isinstance(file_name, (str, Path)):
            self._save_gwf_data(data=data[0], file_name=file_name, channel=channel[0], **kwargs)
        elif (
            data.num_of_channels > 1
            and isinstance(file_name, np.ndarray)
            and len(file_name.shape) == 1
            and file_name.shape[0] == data.num_of_channels
        ):
            for i in range(data.num_of_channels):
                single_file_name = cast(Path, file_name[i])
                single_channel = channel[i]
                self._save_gwf_data(data=data[i], file_name=single_file_name, channel=single_channel, **kwargs)
        else:
            raise ValueError(
                "file_name must be a single path for single-channel data or an array of paths for multi-channel data."
            )

    def _save_gwf_data(  # pylint: disable=unused-argument
        self, data: GWPyTimeSeries, file_name: str | Path, channel: str | None = None, **kwargs
    ) -> None:
        """Save GWPy TimeSeries data to a GWF file.

        Args:
            data: GWPy TimeSeries data to save.
            file_name: Path to the output GWF file.
            channel: Optional channel name to set in the data.
            **kwargs: Additional arguments for the write function.
        """
        if channel is not None:
            data.channel = channel
        data.write(str(file_name))
