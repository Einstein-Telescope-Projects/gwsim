"""Tests for the gwmock-side gwmock_signal adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gwmock_signal import DetectorStrainStack
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock.signal.adapter import SignalAdapter


class FakeBackend:
    """Minimal backend exposing the public gwmock_signal ``simulate`` interface."""

    def __init__(self, *, waveform_model: str) -> None:
        self.waveform_model = waveform_model
        self.required_params = frozenset({"coa_time"})
        self.simulate_calls: list[dict[str, object]] = []

    def register_waveform_model(self, name: str, factory) -> None:
        self.waveform_model = name
        self._registered = (name, factory)

    def simulate(
        self,
        params: dict,
        detector_names,
        background=None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        """Record the call and return simple per-detector strains."""
        names = tuple(detector if isinstance(detector, str) else detector.name for detector in detector_names)
        self.simulate_calls.append(
            {
                "params": params,
                "detector_names": tuple(detector_names),
                "background": background,
                "sampling_frequency": sampling_frequency,
                "minimum_frequency": minimum_frequency,
                "earth_rotation": earth_rotation,
                "interpolate_if_offset": interpolate_if_offset,
            }
        )
        return DetectorStrainStack.from_mapping(
            names,
            {
                name: GWpyTimeSeries(
                    [10.0 + index, 20.0 + index, 30.0 + index, 40.0 + index], t0=8.0, sample_rate=sampling_frequency
                )
                for index, name in enumerate(names)
            },
        )


class NoRegistrationBackend:
    """Backend without waveform registration support."""

    def __init__(self, *, waveform_model: str) -> None:
        self.waveform_model = waveform_model


class TestSignalAdapter:
    """Test the gwmock-side signal adapter."""

    def test_from_source_type_merges_fixed_waveform_arguments(self):
        """The adapter should merge waveform args and call the public backend ``simulate``."""
        reference_frequency = 50.0
        detector_frame_mass_1 = 30.0
        expected = {
            "H1": [10.0, 20.0, 30.0, 40.0],
            "L1": [11.0, 21.0, 31.0, 41.0],
        }
        parameters = {
            "detector_frame_mass_1": detector_frame_mass_1,
            "detector_frame_mass_2": 25.0,
            "coa_time": 12.0,
            "distance": 800.0,
            "inclination": 0.3,
            "right_ascension": 1.1,
            "declination": -0.8,
            "polarization_angle": 0.2,
        }

        with (
            patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=FakeBackend),
        ):
            adapter = SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model="IMRPhenomD",
                detectors=["H1", "L1"],
            )
            strain = adapter.simulate(
                parameters,
                sampling_frequency=4.0,
                minimum_frequency=5.0,
                waveform_arguments={"reference_frequency": reference_frequency},
            )

        backend = adapter._backend
        assert adapter.detector_names == ("H1", "L1")
        assert backend.simulate_calls[0]["params"]["reference_frequency"] == reference_frequency
        assert backend.simulate_calls[0]["params"]["detector_frame_mass_1"] == detector_frame_mass_1
        assert backend.simulate_calls[0]["detector_names"] == ("H1", "L1")
        assert backend.simulate_calls[0]["background"] is None
        assert backend.simulate_calls[0]["sampling_frequency"] == 4.0
        assert backend.simulate_calls[0]["minimum_frequency"] == 5.0
        assert backend.simulate_calls[0]["earth_rotation"] is True
        assert strain.shape == (2, 4)
        np.testing.assert_allclose(strain[0].value, expected["H1"])

    def test_from_source_type_loads_detector_config_files_via_network_from_file(self, tmp_path):
        """Detector config paths should be loaded through ``Network.from_file``."""
        config_path = tmp_path / "custom.interferometer"
        config_path.write_text("# dummy\n")
        sentinel_detector = SimpleNamespace(name="custom")

        with (
            patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=FakeBackend),
            patch(
                "gwmock.signal.adapter.Network.from_file",
                return_value=SimpleNamespace(detector_names=(sentinel_detector,)),
            ) as mock_from_file,
            patch(
                "gwmock.signal.adapter.Network.from_detectors",
                return_value=SimpleNamespace(detector_names=(sentinel_detector,)),
            ) as mock_from_detectors,
        ):
            adapter = SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model="IMRPhenomD",
                detectors=[str(config_path)],
            )

        assert adapter.detector_names == ("custom",)
        mock_from_file.assert_called_once_with(config_path)
        mock_from_detectors.assert_called_once_with((sentinel_detector,))

    def test_from_source_type_resolves_named_networks_via_public_network_api(self):
        """Named detector presets should be resolved through ``Network.from_preset``."""
        sentinel_detector = SimpleNamespace(name="ET1_SARD")

        with (
            patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=FakeBackend),
            patch(
                "gwmock.signal.adapter.Network.from_preset",
                return_value=SimpleNamespace(detector_names=(sentinel_detector,)),
            ) as mock_from_preset,
            patch(
                "gwmock.signal.adapter.Network.from_detectors",
                return_value=SimpleNamespace(detector_names=(sentinel_detector,)),
            ) as mock_from_detectors,
        ):
            adapter = SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model="IMRPhenomD",
                detectors=["ET-Triangle-Sardinia"],
            )

        assert adapter.detector_names == ("ET1_SARD",)
        mock_from_preset.assert_called_once_with("ET-Triangle-Sardinia")
        mock_from_detectors.assert_called_once_with((sentinel_detector,))

    def test_from_source_type_resolves_single_public_detector_aliases_via_public_presets(self):
        """Single ET-detector aliases should resolve through the public preset catalog."""
        with patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=FakeBackend):
            adapter = SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model="IMRPhenomD",
                detectors=["ET1_SARD"],
            )

        assert adapter.detector_names == ("ET1_SARD",)

    def test_from_source_type_callable_requires_public_waveform_registration(self):
        """Backends without ``register_waveform_model`` cannot host a callable ``waveform_model``."""

        def _dummy(**_kwargs):
            return {"plus": None, "cross": None}

        with (
            patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=NoRegistrationBackend),
            pytest.raises(AttributeError, match="register_waveform_model"),
        ):
            SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model=_dummy,
                detectors=["H1"],
            )

    def test_from_source_type_callable_waveform_with_real_backend(self):
        """A callable should be registered and invoked on the public CBC backend."""
        from gwpy.timeseries import TimeSeries

        n = 32
        merged_calls: list[dict] = []

        def custom_waveform(*, waveform_model, tc, sampling_frequency, minimum_frequency, **params):
            merged_calls.append(
                {"waveform_model": waveform_model, "tc": tc, "minimum_frequency": minimum_frequency, **params}
            )
            t0 = float(tc) - (n / 2) / sampling_frequency
            hp = TimeSeries(np.linspace(0, 1, n), t0=t0, sample_rate=sampling_frequency)
            hc = TimeSeries(np.linspace(1, 0, n), t0=t0, sample_rate=sampling_frequency)
            return {"plus": hp, "cross": hc}

        parameters = {
            "detector_frame_mass_1": 30.0,
            "detector_frame_mass_2": 25.0,
            "coa_time": 12.0,
            "distance": 800.0,
            "inclination": 0.3,
            "right_ascension": 1.1,
            "declination": -0.8,
            "polarization_angle": 0.2,
        }

        adapter = SignalAdapter.from_source_type(
            source_type="bbh",
            waveform_model=custom_waveform,
            detectors=["H1", "L1"],
        )
        strain = adapter.simulate(
            parameters,
            sampling_frequency=4.0,
            minimum_frequency=5.0,
            waveform_arguments={"reference_frequency": 50.0},
        )

        assert strain.shape == (2, n)
        backend = adapter._backend
        assert isinstance(backend.waveform_model, str)
        assert backend.waveform_model.startswith("__gwmock_custom__")
        assert merged_calls[0]["reference_frequency"] == 50.0
        assert merged_calls[0]["detector_frame_mass_1"] == 30.0


class FailingBackend:
    """Backend that raises 'Unsupported LAL waveform parameters' on the first call."""

    def __init__(self, *, waveform_model: str) -> None:
        self.waveform_model = waveform_model
        self.required_params = frozenset({"coa_time"})
        self.call_count = 0
        self.received_params: list[dict] = []

    def simulate(
        self,
        params,
        detector_names,
        background=None,
        *,
        sampling_frequency,
        minimum_frequency,
        earth_rotation=True,
        interpolate_if_offset=True,
    ):
        self.call_count += 1
        self.received_params.append(dict(params))
        if "redshift" in params or "lambda_1" in params:
            raise ValueError("Unsupported LAL waveform parameters: lambda_1, redshift")
        names = tuple(d if isinstance(d, str) else d.name for d in detector_names)
        from gwpy.timeseries import TimeSeries as GWpyTimeSeries

        return DetectorStrainStack.from_mapping(
            names,
            {name: GWpyTimeSeries([1.0, 2.0], t0=0.0, sample_rate=sampling_frequency) for name in names},
        )


def _make_network(detector_names=("H1",)):
    """Return a minimal Network-like stub."""
    return SimpleNamespace(detector_names=detector_names)


def _make_adapter_with_backend(backend):
    """Build a SignalAdapter using an already-instantiated backend stub."""
    network = _make_network()
    return SignalAdapter(source_type="bns", backend=backend, network=network)


class TestSimulateStackUnsupportedParams:
    """Verify catch-parse-retry for 'Unsupported LAL waveform parameters' errors."""

    def test_unsupported_params_trigger_warning_and_retry(self):
        """On first failure the adapter logs a WARNING and retries without bad keys."""
        backend = FailingBackend(waveform_model="IMRPhenomXPHM")
        adapter = _make_adapter_with_backend(backend)

        params = {"mass_1": 1.4, "redshift": 0.05, "lambda_1": 300.0, "coa_time": 0.0}

        with patch("gwmock.signal.adapter.logger") as mock_logger:
            adapter.simulate_stack(params, sampling_frequency=4.0, minimum_frequency=20.0)

        # Warning was emitted listing the ignored keys
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0] % mock_logger.warning.call_args[0][1:]
        assert "lambda_1" in warning_msg
        assert "redshift" in warning_msg
        assert "injection_parameters" in warning_msg
        # Two backend calls: initial (failing) + retry (filtered)
        assert backend.call_count == 2
        assert "redshift" not in backend.received_params[1]
        assert "lambda_1" not in backend.received_params[1]
        assert "mass_1" in backend.received_params[1]

    def test_unsupported_params_cached_on_second_call(self):
        """Subsequent calls pre-filter and do NOT trigger a second backend failure."""
        backend = FailingBackend(waveform_model="IMRPhenomXPHM")
        adapter = _make_adapter_with_backend(backend)

        params = {"mass_1": 1.4, "redshift": 0.05, "lambda_1": 300.0, "coa_time": 0.0}

        with patch("gwmock.signal.adapter.logger") as mock_logger:
            adapter.simulate_stack(params, sampling_frequency=4.0, minimum_frequency=20.0)
            first_warning_count = mock_logger.warning.call_count
            adapter.simulate_stack(params, sampling_frequency=4.0, minimum_frequency=20.0)
            second_warning_count = mock_logger.warning.call_count

        # No additional warning on the second call
        assert second_warning_count == first_warning_count
        # Second call: only one backend call (pre-filtered, no failure)
        assert backend.call_count == 3  # 2 from first simulate_stack + 1 from second
        assert "redshift" not in backend.received_params[2]
        assert "lambda_1" not in backend.received_params[2]

    def test_genuine_value_error_is_reraised(self):
        """A ValueError not matching the prefix must propagate unchanged."""
        backend = MagicMock()
        backend.simulate.side_effect = ValueError("mass_1 must be positive")
        network = _make_network()
        adapter = SignalAdapter(source_type="bbh", backend=backend, network=network)

        with pytest.raises(ValueError, match="mass_1 must be positive"):
            adapter.simulate_stack({"mass_1": -1.0}, sampling_frequency=4.0, minimum_frequency=20.0)
