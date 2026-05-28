"""Focused tests for the adapter-backed CLI orchestration path."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import yaml
from gwmock_signal import DetectorStrainStack
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock.cli.adapter_orchestration import AdapterOrchestrator
from gwmock.cli.simulate import _simulate_impl
from gwmock.cli.utils.config import (
    Config,
    GlobalsConfig,
    NoiseAdapterConfig,
    OrchestrationConfig,
    PopulationConfig,
    SignalConfig,
    SimulatorOutputConfig,
)
from gwmock.cli.utils.simulation_plan import create_plan_from_config
from gwmock.simulator.seeds import derive_seed

EXPECTED_BATCHES = 2
FAKE_POPULATION_BACKEND = "tests.cli.test_cli_orchestration:FakePopulationBackend"
FAKE_SIGNAL_BACKEND = "tests.cli.test_cli_orchestration:FakeSignalAdapter"
FAKE_NOISE_BACKEND = "tests.cli.test_cli_orchestration:FakeNoiseAdapter"


class FakePopulationBackend:
    """Minimal public-style population backend for orchestration tests."""

    parameter_names: ClassVar[tuple[str, ...]] = ("detector_frame_mass_1", "detector_frame_mass_2", "coa_time")
    metadata: ClassVar[dict[str, object]] = {
        "fetch": {"scheme": "https"},
        "resolved_path": str(Path(tempfile.gettempdir()) / "catalog.h5"),
    }

    def __init__(self, path: str, source_type: str = "bbh") -> None:
        self.path = path
        self.source_type = source_type

    def simulate(self, n_samples: int, **_kwargs):
        if n_samples != EXPECTED_BATCHES:
            raise AssertionError("Unexpected population sample count for test.")
        return {
            "detector_frame_mass_1": np.array([30.0, 31.0]),
            "detector_frame_mass_2": np.array([20.0, 21.0]),
            "coa_time": np.array([100.5, 104.5]),
        }


class FakeSignalAdapter:
    """Minimal signal backend returning deterministic strain stacks."""

    required_params = frozenset({"detector_frame_mass_1", "coa_time"})

    def __init__(self, waveform_model: str = "IMRPhenomXPHM") -> None:
        self.waveform_model = waveform_model

    def simulate(
        self,
        parameters: dict,
        detector_names: tuple[str, ...],
        background=None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        _ = background, minimum_frequency, earth_rotation, interpolate_if_offset
        return DetectorStrainStack.from_mapping(
            detector_names,
            {
                detector: GWpyTimeSeries(
                    np.full(4, parameters["detector_frame_mass_1"]),
                    t0=parameters["coa_time"],
                    sample_rate=sampling_frequency,
                )
                for detector in detector_names
            },
        )


class FakeNoiseAdapter:
    """Minimal noise protocol backend that materializes deterministic arrays."""

    stream_open_calls: ClassVar[list[dict[str, object]]] = []

    def __init__(
        self,
        duration: float = 4.0,
        sampling_frequency: float = 4.0,
        detectors: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = ["H1"] if detectors is None else detectors
        self.seed = seed
        self._chunk_index = 0

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        _ = duration, sampling_frequency, seed
        return {detector: np.zeros(4) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ):
        type(self).stream_open_calls.append(
            {
                "chunk_duration": chunk_duration,
                "sampling_frequency": sampling_frequency,
                "detectors": list(detectors),
                "seed": seed,
            }
        )
        while True:
            yield {
                detector: np.full(round(chunk_duration * sampling_frequency), self._chunk_index, dtype=float)
                for detector in detectors
            }
            self._chunk_index += 1

    @property
    def metadata(self) -> dict[str, object]:
        return {"kind": "fake-noise"}


def _write_signal_file(self, path, **kwargs):
    _ = kwargs
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("STRAIN")


def _fake_orchestration_config(tmp_path: Path, *, source_type: str) -> Config:
    return Config(
        globals=GlobalsConfig(
            working_directory=str(tmp_path),
            output_directory="output",
            metadata_directory="metadata",
            simulator_arguments={
                "sampling-frequency": 4,
                "duration": 4,
                "start-time": 100,
                "max-samples": 2,
            },
        ),
        orchestration=OrchestrationConfig(
            population=PopulationConfig(
                backend=FAKE_POPULATION_BACKEND,
                source_type=source_type,
                n_samples=2,
                arguments={"path": str(tmp_path / "population.h5"), "source_type": source_type},
            ),
            signal=SignalConfig(
                backend=FAKE_SIGNAL_BACKEND,
                waveform_model="IMRPhenomD",
                detectors=["H1"],
                output=SimulatorOutputConfig(
                    file_name="signal-{{ counter }}.gwf",
                    output_directory="signal",
                    arguments={"channel": "H1:STRAIN"},
                ),
            ),
            noise=NoiseAdapterConfig(
                backend=FAKE_NOISE_BACKEND,
                arguments={"seed": 7, "detectors": ["H1"], "duration": 4.0, "sampling_frequency": 4.0},
                output=SimulatorOutputConfig(
                    file_name="noise-{{ counter }}.npy",
                    output_directory="noise",
                ),
            ),
        ),
    )


def _assert_noise_outputs_exist(output_directory: Path) -> None:
    for counter in range(EXPECTED_BATCHES):
        for detector in ["H1"]:
            assert (output_directory / f"noise-{counter}_{detector}.npy").exists()


def test_create_plan_from_orchestration_config(tmp_path: Path):
    """Batch planning should respect the new orchestration config surface."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")

    plan = create_plan_from_config(config, tmp_path / "checkpoints")

    assert plan.total_batches == EXPECTED_BATCHES
    assert all(batch.simulator_name == "orchestration" for batch in plan.batches)
    assert all(isinstance(batch.simulator_config, OrchestrationConfig) for batch in plan.batches)


@pytest.mark.parametrize("source_type", ["bbh", "bns", "nsbh", "gengli"])
def test_simulate_command_runs_adapter_orchestration(monkeypatch, tmp_path: Path, source_type: str):
    """The CLI should execute the adapter-backed orchestration path end to end."""
    FakeNoiseAdapter.stream_open_calls.clear()
    config = _fake_orchestration_config(tmp_path, source_type=source_type)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))

    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    assert (tmp_path / "output" / "signal" / "signal-0.gwf").exists()
    assert (tmp_path / "output" / "signal" / "signal-1.gwf").exists()
    _assert_noise_outputs_exist(tmp_path / "output" / "noise")
    metadata = yaml.safe_load((tmp_path / "metadata" / "orchestration-0.metadata.json").read_text())
    assert metadata["schema_version"] == "1.0.0"
    assert metadata["config"]["orchestration"]["population"]["backend"] == FAKE_POPULATION_BACKEND
    assert metadata["config"]["orchestration"]["signal"]["backend"] == FAKE_SIGNAL_BACKEND
    assert metadata["config"]["orchestration"]["noise"]["backend"] == FAKE_NOISE_BACKEND
    assert metadata["population"]["source_type"] == source_type
    assert metadata["signal"]["detector_network"] == ["H1"]
    assert {output["kind"] for output in metadata["outputs"]} == {"signal", "noise"}
    assert metadata["segment_seeds"] == [derive_seed(7, "signal", 0)]
    assert metadata["simulator_config"]["population"]["backend"] == FAKE_POPULATION_BACKEND
    assert metadata["simulator_config"]["signal"]["detectors"] == ["H1"]
    assert metadata["simulator_metadata"]["orchestration"]["population"]["metadata"] == FakePopulationBackend.metadata
    assert metadata["simulator_metadata"]["orchestration"]["population"]["seed"] == derive_seed(7, "population")
    assert metadata["simulator_metadata"]["orchestration"]["signal"]["network_resolution"] == {
        "inputs": ["H1"],
        "detector_names": ["H1"],
        "steps": [{"input": "H1", "resolver": "detector", "detector_names": ["H1"]}],
    }
    assert metadata["simulator_metadata"]["orchestration"]["signal"]["segment_seed"] == derive_seed(7, "signal", 0)
    assert metadata["simulator_metadata"]["orchestration"]["noise"]["stream_seed"] == derive_seed(7, "noise", "stream")
    assert FakeNoiseAdapter.stream_open_calls == [
        {
            "chunk_duration": 4.0,
            "sampling_frequency": 4.0,
            "detectors": ["H1"],
            "seed": derive_seed(7, "noise", "stream"),
        }
    ]


def test_orchestrator_restores_noise_stream_from_committed_cursor(tmp_path: Path):
    """Restart should fast-forward noise stream from persisted committed cursor."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)

    # Simulate restored state where one chunk was already committed.
    orchestrator.noise_stream_committed_count = 1
    orchestrator.counter = 0
    orchestrator._noise_stream = None
    orchestrator._noise_stream_position = 0
    orchestrator._pending_noise_chunk = None

    chunk = orchestrator._next_noise_chunk()
    assert chunk["H1"][0] == 1.0


def test_orchestrator_records_preset_network_resolution(tmp_path: Path):
    """Named detector presets should be resolved once and reflected into metadata."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.signal.detectors = ["ET-Triangle-Sardinia"]
    config.orchestration.noise.arguments = {"seed": 7, "duration": 4.0, "sampling_frequency": 4.0}

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    signal_metadata = orchestrator.metadata["orchestration"]["signal"]

    assert orchestrator.detectors == ["ET1_SARD", "ET2_SARD", "ET3_SARD"]
    assert orchestrator.noise_arguments["detectors"] == ["ET1_SARD", "ET2_SARD", "ET3_SARD"]
    assert signal_metadata["network_resolution"] == {
        "inputs": ["ET-Triangle-Sardinia"],
        "detector_names": ["ET1_SARD", "ET2_SARD", "ET3_SARD"],
        "steps": [
            {
                "input": "ET-Triangle-Sardinia",
                "resolver": "preset",
                "detector_names": ["ET1_SARD", "ET2_SARD", "ET3_SARD"],
            }
        ],
    }


def test_orchestrator_resolves_noise_detector_alias(tmp_path: Path):
    """Detector aliases in noise.arguments.detectors must be resolved to sub-detectors."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.signal.detectors = ["ET-Triangle-Sardinia"]
    config.orchestration.noise.arguments = {
        "seed": 7,
        "duration": 4.0,
        "sampling_frequency": 4.0,
        "detectors": ["ET-Triangle-Sardinia"],
    }

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)

    assert orchestrator.noise_arguments["detectors"] == ["ET1_SARD", "ET2_SARD", "ET3_SARD"]
    assert orchestrator.detectors == ["ET1_SARD", "ET2_SARD", "ET3_SARD"]


def test_orchestrator_records_single_detector_preset_resolution(tmp_path: Path):
    """Single public ET-detector aliases should resolve via the preset-backed detector catalog."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.signal.detectors = ["ET1_SARD"]
    config.orchestration.noise.arguments = {"seed": 7, "duration": 4.0, "sampling_frequency": 4.0}

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    signal_metadata = orchestrator.metadata["orchestration"]["signal"]

    assert orchestrator.detectors == ["ET1_SARD"]
    assert orchestrator.noise_arguments["detectors"] == ["ET1_SARD"]
    assert signal_metadata["network_resolution"] == {
        "inputs": ["ET1_SARD"],
        "detector_names": ["ET1_SARD"],
        "steps": [
            {
                "input": "ET1_SARD",
                "resolver": "preset-detector",
                "detector_names": ["ET1_SARD"],
            }
        ],
    }
