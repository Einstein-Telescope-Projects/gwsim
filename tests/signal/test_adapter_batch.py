"""Tests for the batched (on-device) entry point on the gwmock-side adapter.

gwmock's orchestration generates one event at a time: ``cli/adapter_orchestration.py`` loops over
population events calling ``simulate``, producing a strain per event. That is the only shape the
LAL and PyCBC backends offer, and it is why gwmock could not run a GPU simulation at all — the
batched path in gwmock-signal had no caller here.

``simulate_segments`` is the batched counterpart: catalogue in as a struct-of-arrays, assembled
segments out. It is deliberately a *second* entry point rather than a change to ``simulate_stack``,
because the per-event path has to keep working unchanged for backends with no batched form.

The tests that actually generate waveforms need the ``[jax]`` extra and are skipped without it; the
input-handling tests do not, and run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock.signal.adapter import SignalAdapter

_FS = 2048.0
_START = 1400000000.0
_SEGMENT = 64.0
_SPAN = 256.0


def _events(n: int = 4) -> list[dict[str, float]]:
    """Return *n* BNS events with every canonical parameter the device path requires."""
    rng = np.random.default_rng(0)
    return [
        {
            "detector_frame_mass_1": 1.4,
            "detector_frame_mass_2": 1.35,
            "luminosity_distance": float(distance),
            "inclination": float(inclination),
            "coa_phase": float(phase),
            "right_ascension": float(ascension),
            "declination": float(declination),
            "polarization_angle": float(polarization),
            "coa_time": float(coalescence),
            "spin_1z": 0.0,
            "spin_2z": 0.0,
        }
        for distance, inclination, phase, ascension, declination, polarization, coalescence in zip(
            rng.uniform(200.0, 2000.0, n),
            rng.uniform(0.0, np.pi, n),
            rng.uniform(0.0, 2 * np.pi, n),
            rng.uniform(0.0, 2 * np.pi, n),
            np.arcsin(rng.uniform(-1.0, 1.0, n)),
            rng.uniform(0.0, np.pi, n),
            _START + np.sort(rng.uniform(0.0, _SPAN, n)),
            strict=True,
        )
    ]


def _adapter() -> SignalAdapter:
    """An adapter on the bundled ET triangle preset, which resolves to CustomDetector instances."""
    return SignalAdapter.from_source_type(
        source_type="bns", waveform_model="IMRPhenomD", detectors=["ET-Triangle-Sardinia"]
    )


class TestEventsToStructOfArrays:
    """Transposing per-event mappings into the columns the batched path takes."""

    def test_transposes_shared_keys_in_event_order(self):
        """Each column holds one key's value across events, in the order given."""
        events = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        assert SignalAdapter.events_to_struct_of_arrays(events) == {"a": [1.0, 3.0], "b": [2.0, 4.0]}

    def test_drops_keys_missing_from_any_event(self):
        """A key some events lack cannot become a column.

        Filling a default would put a fabricated value into a simulation, which is worse than the
        clear failure the caller gets downstream when a required parameter turns out to be absent.
        """
        events = [{"a": 1.0, "only_first": 9.0}, {"a": 2.0}]
        assert SignalAdapter.events_to_struct_of_arrays(events) == {"a": [1.0, 2.0]}

    def test_rejects_an_empty_catalogue(self):
        """No events means no batch, and saying so beats returning an empty mapping."""
        with pytest.raises(ValueError, match="non-empty"):
            SignalAdapter.events_to_struct_of_arrays([])

    def test_rejects_events_with_no_key_in_common(self):
        """With nothing shared there is no column to build, which is a caller error."""
        with pytest.raises(ValueError, match="no parameter is present in every event"):
            SignalAdapter.events_to_struct_of_arrays([{"a": 1.0}, {"b": 2.0}])


class TestSimulateSegmentsValidation:
    """Input checks that run before any device work, so they need no extra installed."""

    def test_missing_canonical_parameters_are_named(self):
        """The message must list what is absent and say aliases are not accepted.

        The batched path reads canonical names straight from the struct-of-arrays, so an alias such
        as ``mass1`` would otherwise surface as a ``KeyError`` from inside a jitted kernel rather
        than as a statement about the input.
        """
        parameters = {"mass1": [1.4], "mass2": [1.35]}
        with pytest.raises(ValueError, match="detector_frame_mass_1") as raised:
            _adapter().simulate_segments(
                parameters,
                sampling_frequency=_FS,
                minimum_frequency=30.0,
                segment_duration=_SEGMENT,
                start_time=_START,
                end_time=_START + _SPAN,
            )
        message = str(raised.value)
        assert "coa_time" in message
        assert "alias" in message.lower()

    def test_ragged_columns_are_rejected(self):
        """Columns of different lengths cannot describe one catalogue."""
        events = _events(3)
        parameters = SignalAdapter.events_to_struct_of_arrays(events)
        parameters["coa_time"] = parameters["coa_time"][:2]
        with pytest.raises(ValueError, match="disagree in length"):
            _adapter().simulate_segments(
                parameters,
                sampling_frequency=_FS,
                minimum_frequency=30.0,
                segment_duration=_SEGMENT,
                start_time=_START,
                end_time=_START + _SPAN,
            )

    def test_waveform_arguments_are_merged_and_overridden_by_the_catalogue(self):
        """Fixed arguments fill gaps; per-event values win, as on the per-event path.

        Checked through the failure message rather than by generating: supplying the missing
        parameters via ``waveform_arguments`` must satisfy the same validation.
        """
        adapter = _adapter()
        parameters = {"coa_time": [_START + 1.0], "right_ascension": [0.3]}
        with pytest.raises(ValueError, match="declination"):
            adapter.simulate_segments(
                parameters,
                sampling_frequency=_FS,
                minimum_frequency=30.0,
                segment_duration=_SEGMENT,
                start_time=_START,
                end_time=_START + _SPAN,
                waveform_arguments={"detector_frame_mass_1": [1.4], "detector_frame_mass_2": [1.35]},
            )


class TestSimulateSegmentsOnDevice:
    """End-to-end through the real batched path. Requires the ``[jax]`` extra."""

    @pytest.fixture(autouse=True)
    def _require_device_stack(self):
        pytest.importorskip("jax", reason="the [jax] extra is not installed")
        pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")

    def test_returns_one_stack_per_segment(self):
        """The span is tiled into fixed-duration segments, each a full detector stack."""
        adapter = _adapter()
        segments = adapter.simulate_segments(
            SignalAdapter.events_to_struct_of_arrays(_events()),
            sampling_frequency=_FS,
            minimum_frequency=30.0,
            segment_duration=_SEGMENT,
            start_time=_START,
            end_time=_START + _SPAN,
        )
        assert len(segments) == int(_SPAN // _SEGMENT)
        assert segments[0].detector_names == ("ET1_SARD", "ET2_SARD", "ET3_SARD")
        assert segments[0].data.shape == (3, round(_SEGMENT * _FS))

    def test_the_segments_contain_signal(self):
        """A catalogue that overlaps the span must leave non-zero strain in it.

        Asserted on finite, non-zero content rather than on exact values: this test exists to show
        the device path is reached and produces a waveform, not to re-verify gwmock-signal's physics,
        which its own suite covers.
        """
        adapter = _adapter()
        segments = adapter.simulate_segments(
            SignalAdapter.events_to_struct_of_arrays(_events()),
            sampling_frequency=_FS,
            minimum_frequency=30.0,
            segment_duration=_SEGMENT,
            start_time=_START,
            end_time=_START + _SPAN,
        )
        channel = segments[0].detector_names[0]
        occupied = sum(int(np.count_nonzero(segment.to_dict()[channel].value)) for segment in segments)
        peak = max(float(np.max(np.abs(segment.to_dict()[channel].value))) for segment in segments)
        assert occupied > 0, "no segment contains signal, so the device path produced nothing"
        assert np.isfinite(peak)
        assert peak > 0.0

    def test_every_detector_in_the_network_differs(self):
        """Three ET channels must carry three different projections, not one repeated.

        atol=0 because strain is ~1e-23: the default absolute tolerance would call any two strain
        arrays equal and the assertion could not fail.
        """
        adapter = _adapter()
        segments = adapter.simulate_segments(
            SignalAdapter.events_to_struct_of_arrays(_events()),
            sampling_frequency=_FS,
            minimum_frequency=30.0,
            segment_duration=_SEGMENT,
            start_time=_START,
            end_time=_START + _SPAN,
        )
        with_signal = max(segments, key=lambda segment: np.count_nonzero(segment.data))
        first, second, third = with_signal.data
        assert not np.allclose(first, second, rtol=1e-6, atol=0.0)
        assert not np.allclose(second, third, rtol=1e-6, atol=0.0)
