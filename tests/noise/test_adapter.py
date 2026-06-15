"""Tests for the gwmock-side gwmock_noise adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gwmock.noise import NoiseAdapter

TEST_DURATION = 8.0
TEST_SAMPLING_FREQUENCY = 256.0
TEST_GPS_START = 1234.5
TEST_SEED = 11


class FakeNoiseBackend:
    """Minimal run-style gwmock-noise backend for direct adapter tests."""

    def __init__(self) -> None:
        self.run_calls = []

    def run(self, config):
        """Record the config and materialize the declared outputs."""
        self.run_calls.append(config)
        config.output.directory.mkdir(parents=True, exist_ok=True)
        output_paths = {}
        for detector in config.detectors:
            artifact_path = config.output.directory / f"{config.output.prefix}_{detector}.npy"
            artifact_path.write_text(f"{detector}:{config.output.format}")
            output_paths[detector] = artifact_path
        return type("SimulationResultStub", (), {"output_paths": output_paths, "config": config})()


class FakeStreamNoiseBackend:
    """Protocol-style backend that exposes one stateful chunk iterator."""

    def __init__(self) -> None:
        self.duration = 0.0
        self.sampling_frequency = 0.0
        self.detectors = ["H1", "L1"]
        self.seed = None
        self.stream_open_calls = []
        self.chunk_index = 0

    def generate(self, duration: float, sampling_frequency: float, detectors: list[str], seed: int | None = None):
        """Return one deterministic chunk."""
        _ = seed
        n_samples = round(duration * sampling_frequency)
        return {detector: np.full(n_samples, self.chunk_index, dtype=float) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ):
        """Yield deterministic chunks while recording one stream-open event."""
        self.stream_open_calls.append(
            {
                "chunk_duration": chunk_duration,
                "sampling_frequency": sampling_frequency,
                "detectors": list(detectors),
                "seed": seed,
            }
        )
        while True:
            n_samples = round(chunk_duration * sampling_frequency)
            value = self.chunk_index
            self.chunk_index += 1
            yield {detector: np.full(n_samples, value, dtype=float) for detector in detectors}

    @property
    def metadata(self) -> dict[str, object]:
        """Return fake backend metadata."""
        return {"kind": "fake-stream-noise"}


def _psd_file(tmp_path: Path) -> Path:
    """Create a simple PSD file for reproducibility tests."""
    freqs = np.linspace(1, 64, 64)
    psd_values = np.ones_like(freqs)
    psd_path = tmp_path / "psd.txt"
    np.savetxt(psd_path, np.column_stack([freqs, psd_values]))
    return psd_path


class TestNoiseAdapter:
    """Tests for direct adapter behavior."""

    def test_run_builds_public_noise_config(self, tmp_path: Path):
        """The adapter should pass gwmock orchestration inputs through NoiseConfig."""
        backend = FakeNoiseBackend()
        adapter = NoiseAdapter.from_backend(backend)
        psd_path = _psd_file(tmp_path)

        result = adapter.run(
            detectors=["H1", "L1"],
            duration=TEST_DURATION,
            sampling_frequency=TEST_SAMPLING_FREQUENCY,
            output_directory=tmp_path,
            output_prefix="segment-0",
            output_format="npy",
            gps_start=TEST_GPS_START,
            channel="TEST",
            seed=TEST_SEED,
            psd_file=psd_path,
            low_frequency_cutoff=10.0,
            high_frequency_cutoff=100.0,
        )

        config = backend.run_calls[0]
        assert config.detectors == ["H1", "L1"]
        assert config.duration == TEST_DURATION
        assert config.sampling_frequency == TEST_SAMPLING_FREQUENCY
        assert config.output.directory == tmp_path
        assert config.output.prefix == "segment-0"
        assert config.output.format == "npy"
        assert config.output.gps_start == TEST_GPS_START
        assert config.output.channel == "TEST"
        assert config.seed == TEST_SEED
        assert len(config.components) == 1
        assert config.components[0].simulator == "colored"
        assert config.components[0].options["psd_file"] == psd_path
        assert config.components[0].options["low_frequency_cutoff"] == 10.0
        assert config.components[0].options["high_frequency_cutoff"] == 100.0
        assert result.output_paths["H1"] == tmp_path / "segment-0_H1.npy"

    def test_open_stream_uses_one_upstream_iterator(self):
        """The adapter should open one shared stream and consume it chunk by chunk."""
        backend = FakeStreamNoiseBackend()
        adapter = NoiseAdapter.from_backend(backend)

        stream = adapter.open_stream(
            chunk_duration=4.0,
            sampling_frequency=8.0,
            detectors=["H1", "L1"],
            seed=7,
        )

        first_chunk = next(stream)
        second_chunk = next(stream)

        assert backend.stream_open_calls == [
            {
                "chunk_duration": 4.0,
                "sampling_frequency": 8.0,
                "detectors": ["H1", "L1"],
                "seed": 7,
            }
        ]
        assert np.all(first_chunk["H1"] == 0.0)
        assert np.all(second_chunk["L1"] == 1.0)

    def test_write_chunk_persists_numpy_outputs(self, tmp_path: Path):
        """The adapter should let gwmock own NumPy chunk output writing."""
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="noise-0",
            output_format="npy",
            gps_start=100.0,
            channel="MOCK_NOISE",
            seed=7,
        )

        result = adapter.write_chunk(
            config=config,
            chunk={
                "H1": np.arange(32, dtype=float),
                "L1": np.arange(32, dtype=float) + 1.0,
            },
        )

        assert adapter.expected_output_paths(config=config) == [
            tmp_path / "noise-0_H1.npy",
            tmp_path / "noise-0_L1.npy",
        ]
        assert np.array_equal(np.load(result.output_paths["H1"]), np.arange(32, dtype=float))
        assert np.array_equal(np.load(result.output_paths["L1"]), np.arange(32, dtype=float) + 1.0)

    def test_expected_output_paths_for_gwf(self, tmp_path: Path):
        """GWF expected paths should match gwmock-noise FrameWriter naming."""
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="noise-0",
            output_format="gwf",
            gps_start=100.5,
            channel="MOCK_NOISE",
            seed=7,
        )

        assert adapter.expected_output_paths(config=config) == [
            tmp_path / "noise-0_H-H1:MOCK_NOISE_100p5-4.gwf",
            tmp_path / "noise-0_L-L1:MOCK_NOISE_100p5-4.gwf",
        ]

    def test_multisegment_outputs_match_single_long_run_with_stateful_backend(self, tmp_path: Path):
        """Concatenated chunk outputs should match one long protocol run for the same stream."""
        backend = FakeStreamNoiseBackend()
        adapter = NoiseAdapter.from_backend(backend)
        stream = adapter.open_stream(
            chunk_duration=2.0,
            sampling_frequency=4.0,
            detectors=["H1"],
            seed=13,
        )

        segments = []
        for index in range(5):
            config = adapter.build_config(
                detectors=["H1"],
                duration=2.0,
                sampling_frequency=4.0,
                output_directory=tmp_path,
                output_prefix=f"noise-{index}",
                output_format="npy",
                gps_start=100.0 + index * 2.0,
                channel="MOCK_NOISE",
                seed=13,
            )
            result = adapter.write_chunk(config=config, chunk=next(stream))
            segments.append(np.load(result.output_paths["H1"]))

        concatenated = np.concatenate(segments)
        expected = np.concatenate([np.full(8, value, dtype=float) for value in range(5)])
        assert concatenated.tobytes() == expected.tobytes()

    def test_per_detector_channels_config_and_expected_paths(self, tmp_path: Path):
        """build_config stores channels dict and expected_output_paths uses per-detector names."""
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="noise-0",
            output_format="gwf",
            gps_start=100.5,
            channel="STRAIN_NOISE",
            channels={"H1": "H1:STRAIN_NOISE", "L1": "L1:STRAIN_NOISE"},
            seed=7,
        )

        assert config.output.channel == "STRAIN_NOISE"
        assert config.output.channels == {"H1": "H1:STRAIN_NOISE", "L1": "L1:STRAIN_NOISE"}
        assert adapter.expected_output_paths(config=config) == [
            tmp_path / "noise-0_H-H1:STRAIN_NOISE_100p5-4.gwf",
            tmp_path / "noise-0_L-L1:STRAIN_NOISE_100p5-4.gwf",
        ]


def _blip_glitch_dict() -> dict[str, object]:
    """Return a valid dict-form blip glitch config."""
    return {
        "kind": "blip",
        "rate": 1.0,
        "amplitude_distribution": {"distribution": "lognormal", "mean": 1e-21, "std": 0.0},
        "width": 0.01,
    }


class TestDefaultStreamBackendComponentNormalization:
    """Regression tests for ISS-011.

    The default stream backend builds ``SpectralLineSimulator`` / ``AddLines`` /
    ``InjectGlitches`` directly, so the adapter must normalize dict-form config
    entries into ``SpectralLine`` / ``GlitchModel`` instances first. Before the
    fix these raised ``AttributeError: 'dict' object has no attribute ...`` on the
    first chunk.
    """

    def test_open_stream_accepts_dict_form_spectral_lines(self):
        """A dict-form spectral_lines entry generates a chunk (no AttributeError)."""
        adapter = NoiseAdapter.from_backend()  # DefaultNoiseSimulator
        stream = adapter.open_stream(
            chunk_duration=TEST_DURATION,
            sampling_frequency=TEST_SAMPLING_FREQUENCY,
            detectors=["H1"],
            seed=TEST_SEED,
            spectral_lines=[{"frequency": 60.0, "amplitude": 1e-23}],
        )

        chunk = next(stream)

        n_samples = round(TEST_DURATION * TEST_SAMPLING_FREQUENCY)
        assert chunk["H1"].shape == (n_samples,)
        assert np.all(np.isfinite(chunk["H1"]))
        assert np.any(chunk["H1"] != 0.0)

    def test_open_stream_accepts_dict_form_glitches(self):
        """A dict-form glitches entry generates a chunk (no AttributeError)."""
        adapter = NoiseAdapter.from_backend()  # DefaultNoiseSimulator
        stream = adapter.open_stream(
            chunk_duration=TEST_DURATION,
            sampling_frequency=TEST_SAMPLING_FREQUENCY,
            detectors=["H1"],
            seed=TEST_SEED,
            glitches=[_blip_glitch_dict()],
        )

        chunk = next(stream)

        n_samples = round(TEST_DURATION * TEST_SAMPLING_FREQUENCY)
        assert chunk["H1"].shape == (n_samples,)
        assert np.all(np.isfinite(chunk["H1"]))

    def test_open_stream_accepts_dataclass_instances(self):
        """Passing SpectralLine/GlitchModel instances still works (normalize is idempotent)."""
        from gwmock_noise.gaussian import SpectralLine
        from gwmock_noise.glitches.models import BlipGlitch, LogNormalAmplitudeDistribution

        adapter = NoiseAdapter.from_backend()  # DefaultNoiseSimulator
        stream = adapter.open_stream(
            chunk_duration=TEST_DURATION,
            sampling_frequency=TEST_SAMPLING_FREQUENCY,
            detectors=["H1"],
            seed=TEST_SEED,
            spectral_lines=[SpectralLine(frequency=60.0, amplitude=1e-23)],
            glitches=[BlipGlitch(rate=1.0, amplitude_distribution=LogNormalAmplitudeDistribution(mean=1e-21))],
        )

        chunk = next(stream)

        n_samples = round(TEST_DURATION * TEST_SAMPLING_FREQUENCY)
        assert chunk["H1"].shape == (n_samples,)
        assert np.all(np.isfinite(chunk["H1"]))
