"""Tests for partial OrchestrationConfig modes (noise-only, population-only, combinations)."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from gwmock.cli.adapter_orchestration import AdapterOrchestrationResult, AdapterOrchestrator
from gwmock.cli.utils.config import (
    NoiseAdapterConfig,
    OrchestrationConfig,
    PopulationConfig,
    SignalConfig,
    SimulatorOutputConfig,
)

# ---------------------------------------------------------------------------
# Minimal fake adapters reused across tests
# ---------------------------------------------------------------------------


class _FakeNoiseBackend:
    metadata: ClassVar[dict] = {"kind": "fake"}

    def __init__(
        self,
        duration: float = 4.0,
        sampling_frequency: float = 4.0,
        detectors: list[str] | None = None,
        seed: int | None = None,
        **_kwargs,
    ) -> None:
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = ["H1"] if detectors is None else list(detectors)
        self.seed = seed

    def generate(self, duration, sampling_frequency, detectors, seed=None):
        return {d: np.zeros(round(duration * sampling_frequency)) for d in detectors}

    def generate_stream(self, chunk_duration, sampling_frequency, detectors, seed=None):
        while True:
            yield {d: np.zeros(round(chunk_duration * sampling_frequency)) for d in detectors}

    @property
    def backend(self):
        return self


class _FakePopulationBackend:
    parameter_names: ClassVar[tuple] = ("coa_time",)
    metadata: ClassVar[dict] = {}
    source_type = "bbh"

    def __init__(self, **_kwargs):
        pass

    def simulate(self, n_samples, **_kwargs):
        return {"coa_time": np.array([100.0] * n_samples)}


# ---------------------------------------------------------------------------
# Validation tests (model construction only — no I/O)
# ---------------------------------------------------------------------------


class TestOrchestrationConfigValidation:
    def test_noise_only_config_validates(self):
        cfg = OrchestrationConfig(
            noise=NoiseAdapterConfig(
                arguments={"detectors": ["H1"]},
                output=SimulatorOutputConfig(file_name="noise-{{ counter }}.gwf"),
            )
        )
        assert cfg.noise is not None
        assert cfg.population is None
        assert cfg.signal is None

    def test_population_only_config_validates(self):
        cfg = OrchestrationConfig(population=PopulationConfig(backend="fake", n_samples=1))
        assert cfg.population is not None
        assert cfg.noise is None
        assert cfg.signal is None

    def test_population_signal_no_noise_validates(self):
        cfg = OrchestrationConfig(
            population=PopulationConfig(backend="fake", n_samples=1),
            signal=SignalConfig(
                detectors=["H1"],
                output=SimulatorOutputConfig(file_name="signal-{{ counter }}.gwf"),
            ),
        )
        assert cfg.signal is not None
        assert cfg.noise is None

    def test_full_config_still_validates(self):
        cfg = OrchestrationConfig(
            noise=NoiseAdapterConfig(
                arguments={"detectors": ["H1"]},
                output=SimulatorOutputConfig(file_name="noise-{{ counter }}.gwf"),
            ),
            population=PopulationConfig(backend="fake", n_samples=1),
            signal=SignalConfig(
                detectors=["H1"],
                output=SimulatorOutputConfig(file_name="signal-{{ counter }}.gwf"),
            ),
        )
        assert cfg.noise is not None
        assert cfg.population is not None
        assert cfg.signal is not None

    def test_signal_without_population_rejected(self):
        with pytest.raises(ValidationError, match="population"):
            OrchestrationConfig(
                signal=SignalConfig(
                    detectors=["H1"],
                    output=SimulatorOutputConfig(file_name="signal.gwf"),
                )
            )

    def test_empty_orchestration_rejected(self):
        with pytest.raises(ValidationError, match="at least one"):
            OrchestrationConfig()


# ---------------------------------------------------------------------------
# Orchestrator construction tests (mock out real adapters)
# ---------------------------------------------------------------------------


def _make_noise_config(detectors=None):
    return NoiseAdapterConfig(
        backend="tests.cli.test_partial_orchestration:_FakeNoiseBackend",
        arguments={"detectors": detectors or ["H1"], "seed": 1},
        output=SimulatorOutputConfig(file_name="noise-{{ counter }}.npy", output_directory="noise"),
    )


def _make_population_config():
    return PopulationConfig(
        backend="tests.cli.test_partial_orchestration:_FakePopulationBackend",
        n_samples=1,
        arguments={},
    )


_GLOBAL_ARGS = {
    "sampling-frequency": 4,
    "duration": 4,
    "start-time": 0,
    "max-samples": 2,
}


class TestAdapterOrchestratorFromConfig:
    def test_noise_only_from_config(self):
        cfg = OrchestrationConfig(noise=_make_noise_config())
        orch = AdapterOrchestrator.from_config(cfg, _GLOBAL_ARGS)
        assert orch.signal_adapter is None
        assert orch.noise_adapter is not None
        assert orch.detectors == ["H1"]

    def test_population_only_from_config(self):
        cfg = OrchestrationConfig(population=_make_population_config())
        orch = AdapterOrchestrator.from_config(cfg, _GLOBAL_ARGS)
        assert orch.signal_adapter is None
        assert orch.noise_adapter is None
        assert orch.detectors == []
        assert len(orch._population_events) == 1


# ---------------------------------------------------------------------------
# Simulate result tests
# ---------------------------------------------------------------------------


class TestAdapterOrchestratorSimulate:
    def test_noise_only_simulate_result(self, tmp_path):
        cfg = OrchestrationConfig(noise=_make_noise_config())
        orch = AdapterOrchestrator.from_config(cfg, _GLOBAL_ARGS)

        batch_mock = MagicMock()
        batch_mock.simulator_config = cfg
        orch.set_batch_context(
            batch=batch_mock,
            output_directory=tmp_path,
            overwrite=True,
        )

        result = orch.simulate()
        assert isinstance(result, AdapterOrchestrationResult)
        assert result.signal_segment is None
        assert result.noise_result is not None

    def test_population_only_simulate_result(self):
        cfg = OrchestrationConfig(population=_make_population_config())
        orch = AdapterOrchestrator.from_config(cfg, _GLOBAL_ARGS)

        batch_mock = MagicMock()
        batch_mock.simulator_config = cfg
        orch.set_batch_context(
            batch=batch_mock,
            output_directory=MagicMock(),
            overwrite=True,
        )

        result = orch.simulate()
        assert isinstance(result, AdapterOrchestrationResult)
        assert result.signal_segment is None
        assert result.noise_result is None
