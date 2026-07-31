"""Converting batched device output into gwmock chunks.

The conversion is pure host-side arithmetic on placement and ordering, so almost all of it is tested
against a stub batch rather than a real device run: that keeps the guards covered in an installation
without the ``[jax]`` extra, and it lets the failure cases be constructed deliberately instead of
hoped for. One test at the end runs the real batched path to check the stub has the right shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from gwmock.signal.device_chunks import batched_strain_to_chunks, per_event_injections

_SAMPLING_FREQUENCY = 1024.0
_EPOCH = 1577491296.0


class _Grid:
    """Stand-in for ``gwmock_signal.SamplingGrid`` with the two methods the conversion uses."""

    def __init__(self, epoch: float = _EPOCH, sampling_frequency: float = _SAMPLING_FREQUENCY) -> None:
        self.epoch = epoch
        self.sampling_frequency = sampling_frequency

    def time_of(self, index: int) -> float:
        return self.epoch + index / self.sampling_frequency

    def index_of(self, gps_time):
        return (np.asarray(gps_time, dtype=float) - self.epoch) * self.sampling_frequency

    def is_on_lattice(self, gps_time):
        exact = self.index_of(gps_time)
        return np.abs(exact - np.round(exact)) <= 1e-3


@dataclass
class _Batch:
    """Stand-in for ``BatchedDetectorStrain``, carrying only the fields the conversion reads."""

    strain: np.ndarray
    detector_names: tuple[str, ...]
    start_index: Any
    grid: Any
    sampling_frequency: float = _SAMPLING_FREQUENCY
    coa_time: Any = None
    epoch: float = 0.0


def _batch(n_events: int = 2, n_detectors: int = 3, n_samples: int = 64, start_indices=(10, 500)) -> _Batch:
    """Return a stub batch whose samples identify their own event and detector.

    Each row is filled with ``event + detector / 10``, so a chunk that ends up in the wrong place is
    identifiable from its contents rather than only from a length.
    """
    strain = np.zeros((n_events, n_detectors, n_samples))
    for event in range(n_events):
        for detector in range(n_detectors):
            strain[event, detector, :] = event + detector / 10.0
    return _Batch(
        strain=strain,
        detector_names=tuple(f"ET{index + 1}_SARD" for index in range(n_detectors)),
        start_index=np.asarray(start_indices[:n_events]),
        grid=_Grid(),
    )


class TestConversion:
    """What the chunks look like when everything is well formed."""

    def test_one_chunk_per_event_in_order(self):
        chunks = batched_strain_to_chunks(_batch())

        assert len(chunks) == 2
        assert np.asarray(chunks[0])[0, 0] == pytest.approx(0.0)
        assert np.asarray(chunks[1])[0, 0] == pytest.approx(1.0), "chunks are not in event order"

    def test_each_chunk_keeps_every_detector_row(self):
        """Channel *k* becomes detector *k* downstream, so the rows must survive intact."""
        chunks = batched_strain_to_chunks(_batch())

        first = np.asarray(chunks[0])
        assert first.shape == (3, 64)
        assert [first[row, 0] for row in range(3)] == pytest.approx([0.0, 0.1, 0.2])

    def test_start_times_come_from_the_grid(self):
        """Placement is the whole point: a wrong start time misplaces the signal in time."""
        chunks = batched_strain_to_chunks(_batch(start_indices=(10, 500)))

        assert float(chunks[0].start_time.value) == pytest.approx(_EPOCH + 10 / _SAMPLING_FREQUENCY)
        assert float(chunks[1].start_time.value) == pytest.approx(_EPOCH + 500 / _SAMPLING_FREQUENCY)

    def test_the_chunks_land_on_whole_samples_of_a_matching_segment(self):
        """The property that makes injection an integer-offset add rather than a resample.

        Asserted through the real injection path, since that is what consumes these chunks: an
        exactly-placed chunk is added verbatim, so its constant value survives.
        """
        from gwmock.data.time_series.time_series import TimeSeries

        chunks = batched_strain_to_chunks(_batch(start_indices=(10, 500)))
        segment = TimeSeries(
            data=np.zeros((3, 1024)),
            start_time=_EPOCH,
            sampling_frequency=_SAMPLING_FREQUENCY,
        )

        remaining = segment.inject_from_list(chunks)

        assert len(remaining) == 0, "no chunk should overflow this segment"
        values = np.asarray(segment)
        assert values[0, 10] == pytest.approx(0.0)
        assert values[1, 500] == pytest.approx(1.1), (
            "the second event's second detector was not placed verbatim at its grid index"
        )

    def test_a_scalar_start_index_is_accepted(self):
        """A one-event batch may carry a scalar rather than a length-one array."""
        batch = _batch(n_events=1, start_indices=(7,))
        batch.start_index = 7

        chunks = batched_strain_to_chunks(batch)

        assert len(chunks) == 1
        assert float(chunks[0].start_time.value) == pytest.approx(_EPOCH + 7 / _SAMPLING_FREQUENCY)


class TestGuards:
    """Each failure this refuses to let through quietly."""

    def test_an_unaligned_batch_is_refused(self):
        """Without a grid every chunk would be resampled onto the segment, losing accuracy."""
        batch = _batch()
        batch.grid = None
        batch.start_index = None

        with pytest.raises(ValueError, match="without an output grid"):
            batched_strain_to_chunks(batch)

    def test_a_buffer_off_the_lattice_is_refused(self):
        """A half-sample start would be interpolated instead of placed, silently."""
        batch = _batch()

        class _OffByHalf(_Grid):
            def time_of(self, index: int) -> float:
                return self.epoch + (index + 0.5) / self.sampling_frequency

        batch.grid = _OffByHalf()

        with pytest.raises(ValueError, match="not on the output grid"):
            batched_strain_to_chunks(batch)

    def test_reordered_detectors_are_refused(self):
        """Channel order carries detector identity, so a permutation misattributes signals."""
        batch = _batch()

        with pytest.raises(ValueError, match="expects"):
            batched_strain_to_chunks(batch, expected_detector_names=("ET3_SARD", "ET2_SARD", "ET1_SARD"))

    def test_matching_detectors_are_accepted(self):
        """The complement, so the check above cannot pass by rejecting everything."""
        chunks = batched_strain_to_chunks(_batch(), expected_detector_names=("ET1_SARD", "ET2_SARD", "ET3_SARD"))

        assert len(chunks) == 2

    def test_a_start_index_per_event_is_required(self):
        batch = _batch()
        batch.start_index = np.asarray([10])

        with pytest.raises(ValueError, match="start indices for 2 events"):
            batched_strain_to_chunks(batch)

    def test_detector_rows_must_match_the_names(self):
        batch = _batch()
        batch.detector_names = ("ET1_SARD", "ET2_SARD")

        with pytest.raises(ValueError, match="detector rows but 2 detector names"):
            batched_strain_to_chunks(batch)

    def test_a_non_batched_shape_is_refused(self):
        batch = _batch()
        batch.strain = np.zeros((3, 64))

        with pytest.raises(ValueError, match="Expected strain of shape"):
            batched_strain_to_chunks(batch)


class TestProvenance:
    """Per-event records, so the device path does not thin out provenance."""

    def test_each_event_gets_its_own_parameters(self):
        records = per_event_injections(
            {"detector_frame_mass_1": [30.0, 25.0], "coa_time": [1.0, 2.0], "f_ref": 20.0},
            [7, 8],
        )

        assert [record["event_id"] for record in records] == [7, 8]
        assert records[0]["parameters"]["detector_frame_mass_1"] == pytest.approx(30.0)
        assert records[1]["parameters"]["coa_time"] == pytest.approx(2.0)

    def test_a_scalar_parameter_is_shared_by_every_event(self):
        """A fixed value applies to all events rather than being indexed per event."""
        records = per_event_injections({"f_ref": 20.0}, [0, 1])

        assert all(record["parameters"]["f_ref"] == pytest.approx(20.0) for record in records)

    def test_a_short_column_is_refused(self):
        """Silently recording the wrong parameters would be worse than failing."""
        with pytest.raises(ValueError, match="has 1 values but the batch has 2 events"):
            per_event_injections({"coa_time": [1.0]}, [0, 1])


class TestAgainstTheRealDeviceOutput:
    """That the stub above matches what gwmock-signal actually returns."""

    def test_a_real_batch_converts_and_places(self):
        """Runs the batched path for real, so the stub cannot drift from the true contract.

        Without this the guards could all pass against a stub whose field names or shapes no longer
        match ``BatchedDetectorStrain``.
        """
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
        from gwmock_signal import Network, SamplingGrid, simulate_cbc_batch

        from gwmock.data.time_series.time_series import TimeSeries

        detectors = list(Network.from_preset("ET-Triangle-Sardinia").detector_names)
        grid = SamplingGrid(_EPOCH, _SAMPLING_FREQUENCY)
        batch = simulate_cbc_batch(
            "IMRPhenomD",
            detectors,
            sampling_frequency=_SAMPLING_FREQUENCY,
            minimum_frequency=30.0,
            parameters={
                "detector_frame_mass_1": [30.0, 25.0],
                "detector_frame_mass_2": [28.0, 24.0],
                "luminosity_distance": [400.0, 500.0],
                "inclination": [0.4, 0.2],
                "coa_phase": [0.0, 0.1],
                "right_ascension": [1.0, 1.2],
                "declination": [0.5, 0.3],
                "polarization_angle": [0.2, 0.1],
                "coa_time": [_EPOCH + 4.0, _EPOCH + 9.0],
                "spin_1z": [0.0, 0.0],
                "spin_2z": [0.0, 0.0],
            },
            output_grid=grid,
        )

        chunks = batched_strain_to_chunks(batch, expected_detector_names=("ET1_SARD", "ET2_SARD", "ET3_SARD"))

        assert len(chunks) == 2
        segment = TimeSeries(
            data=np.zeros((3, int(16 * _SAMPLING_FREQUENCY))),
            start_time=_EPOCH,
            sampling_frequency=_SAMPLING_FREQUENCY,
        )
        segment.inject_from_list(chunks)
        values = np.asarray(segment)
        assert np.count_nonzero(values), "the converted chunks put no signal into the segment"
        assert np.all(np.isfinite(values))
