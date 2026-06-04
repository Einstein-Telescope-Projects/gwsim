"""Adapter-backed orchestration for the primary gwmock CLI path."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from gwmock_noise import SimulationResult
from gwmock_pop import GWPopSimulator
from gwmock_signal import DetectorStrainStack, Network
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock.cli.utils.backend_resolver import instantiate_backend, resolve_backend_class, validate_backend
from gwmock.cli.utils.config import OrchestrationConfig
from gwmock.cli.utils.config_resolution import resolve_max_samples
from gwmock.cli.utils.template import expand_template_variables
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.mixin.time_series import TimeSeriesMixin
from gwmock.noise import NoiseAdapter
from gwmock.population import PopulationAdapter
from gwmock.signal import SignalAdapter
from gwmock.simulator.base import Simulator
from gwmock.simulator.seeds import derive_seed
from gwmock.simulator.state import StateAttribute


@dataclass(slots=True)
class AdapterOrchestrationResult:
    """Artifacts produced by one adapter-backed orchestration batch."""

    signal_segment: TimeSeries | None
    noise_result: SimulationResult | None


def _normalize_keys(values: dict[str, Any]) -> dict[str, Any]:
    """Convert YAML-style hyphenated keys into Python identifiers."""
    return {key.replace("-", "_"): value for key, value in values.items()}


class _NoiseTemplateProxy:
    """Forward attribute access to an orchestrator, substituting noise detectors.

    ``expand_template_variables`` resolves ``{{ detectors }}`` via ``getattr``.
    This proxy redirects that lookup to the noise detector list instead of the
    signal detector list stored on the real orchestrator.
    """

    __slots__ = ("_wrapped", "detectors")

    def __init__(self, wrapped: Any, noise_detectors: list[str]) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "detectors", noise_detectors)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped"), name)


class AdapterOrchestrator(TimeSeriesMixin, Simulator):
    """Compose population, signal, and noise adapters inside gwmock."""

    population_index = StateAttribute(0)
    noise_stream_committed_count = StateAttribute(0)

    def __init__(  # noqa: PLR0913
        self,
        *,
        population_events: list[dict[str, Any]],
        population_metadata: dict[str, Any],
        source_type: str,
        source_detector_specs: list[str],
        detector_network: Network,
        detector_resolution: dict[str, Any],
        waveform_model: str | None,
        waveform_arguments: dict[str, Any],
        signal_parameters: dict[str, Any],
        detectors: list[str],
        duration: float,
        sampling_frequency: float,
        start_time: float,
        max_samples: int,
        minimum_frequency: float,
        earth_rotation: bool,
        noise_arguments: dict[str, Any],
        orchestration_config: OrchestrationConfig,
        population_seed: int | None = None,
        signal_adapter: SignalAdapter | None = None,
        noise_adapter: NoiseAdapter | None = None,
    ) -> None:
        self._population_events = tuple(population_events)
        self._population_metadata = dict(population_metadata)
        self._source_type = source_type
        self._source_detector_specs = tuple(source_detector_specs)
        self._signal_network = detector_network
        self._detector_resolution = detector_resolution
        self._population_seed = population_seed
        self.waveform_model = waveform_model
        self.waveform_arguments = waveform_arguments
        self.signal_parameters = signal_parameters
        self.minimum_frequency = minimum_frequency
        self.earth_rotation = earth_rotation
        self.orchestration_config = orchestration_config
        if signal_adapter is not None:
            self.signal_adapter = signal_adapter
        elif orchestration_config.signal is not None:
            self.signal_adapter = SignalAdapter.from_source_type(
                source_type=source_type,
                waveform_model=waveform_model,
                backend_arguments={"duration": duration} if source_type == "sgwb" else None,
                network=detector_network,
            )
        else:
            self.signal_adapter = None
        self.detectors = (
            list(self.signal_adapter.detector_names) if self.signal_adapter is not None else list(detectors)
        )
        self.noise_arguments = noise_arguments
        if noise_adapter is not None:
            self.noise_adapter = noise_adapter
        elif orchestration_config.noise is not None:
            self.noise_adapter = NoiseAdapter.from_backend()
        else:
            self.noise_adapter = None
        self._active_signal_output_directory = Path("signal")
        self._active_noise_output_directory = Path("noise")
        self._active_noise_output_arguments: dict[str, Any] = {}
        self._active_overwrite = False
        self._noise_stream: Iterator[dict[str, Any]] | None = None
        self._noise_stream_position = 0
        self._pending_noise_chunk: dict[str, Any] | None = None

        super().__init__(
            max_samples=max_samples,
            start_time=start_time,
            duration=duration,
            sampling_frequency=sampling_frequency,
            num_of_channels=len(self.detectors),
        )

    @classmethod
    def from_config(
        cls,
        orchestration_config: OrchestrationConfig,
        global_simulator_arguments: dict[str, Any] | None = None,
    ) -> AdapterOrchestrator:
        """Instantiate the composite adapter-backed orchestration path."""
        global_args = _normalize_keys(global_simulator_arguments or {})
        max_samples = resolve_max_samples(simulator_args={}, global_args=global_args)
        duration = float(global_args.get("duration", 4.0))
        sampling_frequency = float(global_args.get("sampling_frequency", 4096.0))
        start_time = float(global_args.get("start_time", 0.0))

        has_population = orchestration_config.population is not None
        has_signal = orchestration_config.signal is not None

        noise_arguments, detector_network, detector_resolution, resolved_detectors = (
            cls._resolve_detectors_and_noise_args(orchestration_config, global_args)
        )
        top_level_seed = int(noise_arguments["seed"]) if noise_arguments.get("seed") is not None else None
        population_seed = derive_seed(top_level_seed, "population") if top_level_seed is not None else None

        population_events: list[dict[str, Any]] = []
        population_metadata: dict[str, Any] = {}
        source_type = ""
        source_detector_specs: list[str] = []
        waveform_model: str | None = None
        waveform_arguments: dict[str, Any] = {}
        signal_parameters: dict[str, Any] = {}
        minimum_frequency = 5.0
        earth_rotation = True
        signal_adapter: SignalAdapter | None = None
        noise_adapter: NoiseAdapter | None = None

        if has_population:
            population_events, population_metadata, source_type = cls._build_population_context(
                orchestration_config.population,  # type: ignore[arg-type]
                population_seed,
            )
        elif has_signal:
            source_type = str(orchestration_config.signal.source_type)  # type: ignore[union-attr]

        if has_signal:
            signal_adapter = cls._instantiate_signal_adapter(
                orchestration_config.signal,
                source_type=source_type,
                detector_network=detector_network,
                duration=duration,
            )
            source_detector_specs = list(orchestration_config.signal.detectors)  # type: ignore[union-attr]
            waveform_model = orchestration_config.signal.waveform_model  # type: ignore[union-attr]
            waveform_arguments = orchestration_config.signal.waveform_arguments  # type: ignore[union-attr]
            signal_parameters = orchestration_config.signal.parameters  # type: ignore[union-attr]
            minimum_frequency = orchestration_config.signal.minimum_frequency  # type: ignore[union-attr]
            earth_rotation = orchestration_config.signal.earth_rotation  # type: ignore[union-attr]
            if orchestration_config.noise is None:
                noise_arguments["detectors"] = resolved_detectors

        if orchestration_config.noise is not None:
            noise_adapter = cls._instantiate_noise_adapter(orchestration_config.noise)

        return cls(
            population_events=population_events,
            population_metadata=population_metadata,
            source_type=source_type,
            source_detector_specs=source_detector_specs,
            detector_network=detector_network,
            detector_resolution=detector_resolution,
            waveform_model=waveform_model,
            waveform_arguments=waveform_arguments,
            signal_parameters=signal_parameters,
            detectors=resolved_detectors,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            max_samples=max_samples,
            minimum_frequency=minimum_frequency,
            earth_rotation=earth_rotation,
            noise_arguments=noise_arguments,
            orchestration_config=orchestration_config,
            population_seed=population_seed,
            signal_adapter=signal_adapter,
            noise_adapter=noise_adapter,
        )

    @classmethod
    def _resolve_detectors_and_noise_args(
        cls,
        orchestration_config: OrchestrationConfig,
        global_args: dict[str, Any],
    ) -> tuple[dict[str, Any], Network | None, dict[str, Any], list[str]]:
        """Resolve noise arguments and detector lists for all active adapters."""
        has_noise = orchestration_config.noise is not None
        has_signal = orchestration_config.signal is not None

        noise_arguments: dict[str, Any] = (
            _normalize_keys(orchestration_config.noise.arguments)  # type: ignore[union-attr]
            if has_noise
            else {}
        )
        if global_args.get("seed") is not None:
            noise_arguments.setdefault("seed", int(global_args["seed"]))

        detector_network: Network | None = None
        detector_resolution: dict[str, Any] = {}
        resolved_detectors: list[str] = []

        if has_signal:
            detector_network, detector_resolution = cls._resolve_detector_network(
                orchestration_config.signal.detectors  # type: ignore[union-attr]
            )
            resolved_detectors = cls._network_detector_names(detector_network)

        if has_noise:
            if "detectors" in noise_arguments:
                noise_network, _ = cls._resolve_detector_network(noise_arguments["detectors"])
                noise_arguments["detectors"] = cls._network_detector_names(noise_network)
                if not has_signal:
                    resolved_detectors = noise_arguments["detectors"]
            elif has_signal:
                noise_arguments["detectors"] = resolved_detectors
            else:
                raise ValueError(
                    "noise orchestration without a signal section must specify detectors under noise.arguments.detectors"
                )

        return noise_arguments, detector_network, detector_resolution, resolved_detectors

    @classmethod
    def _build_population_context(
        cls,
        population_config: Any,
        population_seed: int | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        """Materialise population events and return (events, metadata, source_type)."""
        population_backend = cls._instantiate_population_backend(population_config)
        population_adapter = PopulationAdapter.from_backend(
            population_backend,
            n_samples=population_config.n_samples,
            **({"seed": population_seed} if population_seed is not None else {}),
        )
        population_events = list(population_adapter.iter_event_parameters())
        sort_key = population_config.sort_by
        if sort_key:
            if population_events and any(sort_key not in event for event in population_events):
                raise ValueError(f"Population event ordering key '{sort_key}' is missing from one or more events.")
            population_events.sort(key=lambda event: event[sort_key])
        return population_events, population_adapter.metadata, population_adapter.source_type

    @staticmethod
    def _instantiate_population_backend(population_config) -> GWPopSimulator:
        backend_arguments = dict(population_config.arguments)
        if population_config.source_type is not None:
            backend_arguments.setdefault("source_type", population_config.source_type)

        return instantiate_backend(
            "population",
            population_config.backend,
            init_kwargs=backend_arguments,
        )

    @staticmethod
    def _instantiate_signal_adapter(
        signal_config,
        *,
        source_type: str,
        detector_network: Network,
        duration: float,
    ) -> SignalAdapter:
        backend_name = signal_config.backend or source_type
        backend_class = resolve_backend_class("signal", backend_name)
        backend_arguments = _normalize_keys(dict(signal_config.arguments))
        if source_type == "sgwb":
            backend_arguments.setdefault("duration", duration)
        backend_instance = SignalAdapter.instantiate_backend(
            backend_class,
            waveform_model=signal_config.waveform_model,
            backend_arguments=backend_arguments,
        )
        validate_backend("signal", backend_name, backend_class, backend_instance)
        return SignalAdapter.from_backend(
            source_type=source_type,
            backend=backend_instance,
            network=detector_network,
        )

    @staticmethod
    def _instantiate_noise_adapter(noise_config) -> NoiseAdapter:
        if noise_config.backend is None:
            return NoiseAdapter.from_backend()
        backend_instance = instantiate_backend(
            "noise",
            noise_config.backend,
            init_kwargs=dict(noise_config.arguments),
        )
        return NoiseAdapter.from_backend(backend_instance)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return orchestration metadata for reproducibility."""
        signal_segment_seed = self._signal_segment_seed()
        return {
            **super().metadata,
            "orchestration": {
                "source_type": self._source_type,
                "population_events_total": len(self._population_events),
                "population_events_remaining": len(self._population_events) - int(self.population_index),
                "population": {
                    "metadata": self._population_metadata,
                    "seed": self._population_seed,
                },
                "signal": {
                    "waveform_model": self.waveform_model,
                    "waveform_arguments": self.waveform_arguments,
                    "parameters": self.signal_parameters,
                    "minimum_frequency": self.minimum_frequency,
                    "earth_rotation": self.earth_rotation,
                    "detector_specs": list(self._source_detector_specs),
                    "detectors": self.detectors,
                    "network_resolution": self._detector_resolution,
                    "segment_seed": signal_segment_seed,
                },
                "noise": {
                    "arguments": self.noise_arguments,
                    "stream_seed": self._noise_stream_seed(),
                    "state_model": "gwmock consumes one shared gwmock_noise.open_stream() iterator across batches.",
                },
                "segment_seeds": self.segment_seeds(),
            },
        }

    def set_batch_context(self, *, batch: Any, output_directory: Path, overwrite: bool) -> None:
        """Resolve per-batch output directories and runtime arguments."""
        if batch.simulator_config.signal is not None:
            signal_output = batch.simulator_config.signal.output
            self._active_signal_output_directory = self._resolve_output_directory(
                output_directory,
                signal_output.output_directory,
                fallback_subdir="signal",
            )
        if batch.simulator_config.noise is not None:
            noise_output = batch.simulator_config.noise.output
            self._active_noise_output_directory = self._resolve_output_directory(
                output_directory,
                noise_output.output_directory,
                fallback_subdir="noise",
            )
            noise_detectors = list(self.noise_arguments.get("detectors", []))
            proxy = _NoiseTemplateProxy(self, noise_detectors)
            self._active_noise_output_arguments = expand_template_variables(noise_output.arguments or {}, proxy)
        self._active_overwrite = overwrite

    def simulate(self) -> AdapterOrchestrationResult:
        """Generate signal and/or noise for the current batch depending on active adapters."""
        signal_segment = TimeSeriesMixin.simulate(self) if self.signal_adapter is not None else None
        noise_result = self._run_noise_batch() if self.noise_adapter is not None else None
        return AdapterOrchestrationResult(signal_segment=signal_segment, noise_result=noise_result)

    def _simulate(self) -> TimeSeriesList:
        """Generate signal chunks for the current segment from population events."""
        if not self._population_events and self._source_type == "sgwb":
            return self._simulate_stationary_signal_segment()

        chunks = TimeSeriesList()
        while self.population_index < len(self._population_events):
            parameters = dict(self._population_events[int(self.population_index)])
            coa_time = parameters.get("coa_time")
            end_time_value = float(getattr(self.end_time, "value", self.end_time))
            if coa_time is not None and float(coa_time) >= end_time_value:
                break
            strain = self.signal_adapter.simulate(
                parameters,
                sampling_frequency=float(self.sampling_frequency.value),
                minimum_frequency=self.minimum_frequency,
                waveform_arguments=self.waveform_arguments,
                earth_rotation=self.earth_rotation,
            )
            strain.metadata.update({"injection_parameters": dict(parameters)})
            chunks.append(strain)
            self.population_index = cast(int, self.population_index) + 1
            if strain.start_time >= self.end_time:
                break
        return chunks

    def _simulate_stationary_signal_segment(self) -> TimeSeriesList:
        """Generate one stationary signal chunk spanning the active segment."""
        if self.signal_adapter is None:
            return TimeSeriesList()
        segment_seed = self._signal_segment_seed()
        self.signal_adapter.set_seed(segment_seed)
        n_samples = round(float(self.duration.value) * float(self.sampling_frequency.value))
        background = {
            detector: GWpyTimeSeries(
                np.zeros(n_samples, dtype=float),
                t0=float(self.start_time.value),
                sample_rate=float(self.sampling_frequency.value),
                unit="strain",
            )
            for detector in self.signal_adapter.detector_names
        }
        parameters = {**self.waveform_arguments, **self.signal_parameters}
        strain = self.signal_adapter.simulate(
            parameters,
            sampling_frequency=float(self.sampling_frequency.value),
            minimum_frequency=self.minimum_frequency,
            background=background,
            earth_rotation=self.earth_rotation,
        )
        strain.metadata.update({"signal_parameters": dict(parameters), "segment_seed": segment_seed})
        return TimeSeriesList([strain])

    def update_state(self) -> None:
        """Advance to the next segment."""
        self.noise_stream_committed_count = max(
            int(self.noise_stream_committed_count), int(self._noise_stream_position)
        )
        self.counter = cast(int, self.counter) + 1
        self.start_time += self.duration
        self._pending_noise_chunk = None

    def signal_output_directory(self) -> Path:
        """Return the active signal output directory."""
        return self._active_signal_output_directory

    def signal_output_arguments(self) -> dict[str, Any]:
        """Return the active signal output keyword arguments."""
        if self.orchestration_config.signal is None:
            return {}
        return dict(self.orchestration_config.signal.output.arguments or {})

    def _save_data(
        self,
        data: TimeSeries,
        file_name: str | Path | np.ndarray[Any, np.dtype[np.object_]],
        **kwargs,
    ) -> None:
        """Persist orchestration signal output through ``DetectorStrainStack.write``."""
        if not isinstance(data, TimeSeries):
            raise TypeError(f"AdapterOrchestrator can only save TimeSeries signal data, got {type(data)}.")

        channel_names = self._resolve_signal_channels(data=data, channel_spec=kwargs.pop("channel", None))
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(
                f"Signal orchestration output arguments only support channel. Unsupported keys: {unsupported}."
            )

        if isinstance(file_name, (str, Path)):
            output_path = Path(file_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._build_signal_stack(data=data, channel_names=channel_names).write(
                output_path,
                format=self._infer_signal_output_format(output_path),
            )
            return

        if len(file_name.shape) != 1 or file_name.shape[0] != data.num_of_channels:
            raise ValueError(
                "Resolved signal output paths must be a single path or a one-dimensional array "
                "matching the number of detector channels."
            )

        for index in range(data.num_of_channels):
            detector_name = self.detectors[index]
            output_path = Path(file_name[index])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._build_signal_stack(
                data=data,
                detector_names=[detector_name],
                channel_names=channel_names[index : index + 1],
                indices=[index],
            ).write(
                output_path,
                format=self._infer_signal_output_format(output_path),
            )

    def _run_noise_batch(self) -> SimulationResult:
        output_format = self._infer_noise_output_format(self.orchestration_config.noise.output.file_name)
        output_paths = self._expand_noise_output_paths(self.orchestration_config.noise.output.file_name)
        output_arguments = dict(self._active_noise_output_arguments)
        if "channel_prefix" in output_arguments:
            raise ValueError(
                "Noise output argument 'channel_prefix' is no longer supported. "
                "Rename it to 'channel' (e.g. channel: MOCK_NOISE)."
            )
        channel_raw = output_arguments.pop("channel", "MOCK_NOISE")
        gps_start = float(output_arguments.pop("gps_start", float(self.start_time.value)))
        if output_arguments:
            unsupported = ", ".join(sorted(output_arguments))
            raise ValueError(
                "Noise orchestration output arguments only support channel and gps_start. "
                f"Unsupported keys: {unsupported}."
            )

        noise_detectors = list(self.noise_arguments["detectors"])
        if isinstance(channel_raw, list):
            if len(channel_raw) != len(noise_detectors):
                raise ValueError(
                    f"Noise channel list expanded to {len(channel_raw)} entries "
                    f"but there are {len(noise_detectors)} noise detectors."
                )
            channels_dict: dict[str, str] | None = dict(
                zip(noise_detectors, [str(c) for c in channel_raw], strict=True)
            )
            first = str(channel_raw[0])
            channel_str = first.split(":", 1)[1] if ":" in first else first
        else:
            channel_str = str(channel_raw)
            channels_dict = None

        if not self._active_overwrite:
            existing = [path for path in dict.fromkeys(output_paths) if path.exists()]
            if existing:
                raise FileExistsError(
                    f"Noise adapter output(s) already exist: {', '.join(str(path) for path in existing)}. "
                    "Use overwrite=True to overwrite them."
                )

        chunk = self._next_noise_chunk()
        output_paths_by_detector: dict[str, Path] = {}

        for i, detector in enumerate(noise_detectors):
            output_path = output_paths[i] if len(output_paths) > 1 else output_paths[0]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_format == "gwf":
                channel_id = channels_dict[detector] if channels_dict is not None else f"{detector}:{channel_str}"
                gwpy_ts = GWpyTimeSeries(
                    np.asarray(chunk[detector], dtype=float),
                    t0=gps_start,
                    sample_rate=float(self.sampling_frequency.value),
                )
                DetectorStrainStack.from_mapping([channel_id], {channel_id: gwpy_ts}).write(output_path, format="gwf")
            else:
                np.save(output_path, chunk[detector])
            output_paths_by_detector[detector] = output_path

        self.noise_stream_committed_count = max(
            int(self.noise_stream_committed_count), int(self._noise_stream_position)
        )
        config = self.noise_adapter.build_config(
            detectors=noise_detectors,
            duration=float(self.duration.value),
            sampling_frequency=float(self.sampling_frequency.value),
            output_directory=self._active_noise_output_directory,
            output_prefix="",
            output_format=output_format,
            gps_start=gps_start,
            channel=channel_str,
            channels=channels_dict,
            seed=self._noise_stream_seed(),
            psd_file=self.noise_arguments.get("psd_file"),
            psd_schedule=self.noise_arguments.get("psd_schedule"),
            psd_files=self.noise_arguments.get("psd_files"),
            csd_files=self.noise_arguments.get("csd_files"),
            low_frequency_cutoff=self.noise_arguments.get("low_frequency_cutoff", 2.0),
            high_frequency_cutoff=self.noise_arguments.get("high_frequency_cutoff"),
            spectral_lines=self.noise_arguments.get("spectral_lines"),
            glitches=self.noise_arguments.get("glitches"),
        )
        return SimulationResult(output_paths=output_paths_by_detector, config=config)

    def segment_seeds(self) -> list[int]:
        """Return the deterministic per-segment seeds derived locally by gwmock."""
        return [seed for seed in (self._signal_segment_seed(),) if seed is not None]

    def _root_seed(self) -> int | None:
        base_seed = self.noise_arguments.get("seed")
        if base_seed is None:
            return None
        return int(base_seed)

    def _signal_segment_seed(self) -> int | None:
        root_seed = self._root_seed()
        if root_seed is None:
            return None
        return derive_seed(root_seed, "signal", int(self.counter))

    def _noise_stream_seed(self) -> int | None:
        root_seed = self._root_seed()
        if root_seed is None:
            return None
        return derive_seed(root_seed, "noise", "stream")

    @staticmethod
    def _infer_signal_output_format(path: Path) -> Literal["gwf", "hdf5", "npy", "txt"]:
        suffix = path.suffix.lower().lstrip(".")
        if suffix == "h5":
            suffix = "hdf5"
        if suffix not in {"gwf", "hdf5", "npy", "txt"}:
            raise ValueError("Signal output files must end with .gwf, .hdf5, .h5, .npy, or .txt.")
        return cast(Literal["gwf", "hdf5", "npy", "txt"], suffix)

    def _infer_noise_output_format(self, file_name_template: str) -> Literal["npy", "gwf"]:
        suffix = Path(file_name_template).suffix.lower().lstrip(".")
        if suffix not in {"npy", "gwf"}:
            raise ValueError("Noise output templates must end with .npy or .gwf.")
        return cast(Literal["npy", "gwf"], suffix)

    def _expand_noise_output_paths(self, file_name_template: str) -> list[Path]:
        """Expand the noise file_name template to one output path per detector."""
        noise_detectors = list(self.noise_arguments["detectors"])
        proxy = _NoiseTemplateProxy(self, noise_detectors)
        expanded = expand_template_variables(file_name_template, proxy)
        if isinstance(expanded, list):
            if len(expanded) != len(noise_detectors):
                raise ValueError(
                    f"Noise file_name template expanded to {len(expanded)} paths "
                    f"but there are {len(noise_detectors)} noise detectors."
                )
            paths = [self._active_noise_output_directory / str(p) for p in expanded]
        else:
            paths = [self._active_noise_output_directory / str(expanded)] * len(noise_detectors)
        if len(set(paths)) < len(paths):
            raise ValueError(
                "Noise file_name template produces identical paths for multiple detectors. "
                "Include {{ detectors }} in the template to generate per-detector paths."
            )
        return paths

    @staticmethod
    def _resolve_output_directory(
        base_output_directory: Path, configured_directory: str | None, fallback_subdir: str
    ) -> Path:
        if configured_directory is None:
            return base_output_directory / fallback_subdir
        configured_path = Path(configured_directory)
        return configured_path if configured_path.is_absolute() else base_output_directory / configured_path

    def _next_noise_chunk(self) -> dict[str, Any]:
        """Return the chunk for the current batch, reusing it across retries."""
        if self._pending_noise_chunk is not None:
            return self._pending_noise_chunk

        self._ensure_noise_stream()
        if self._noise_stream is None:
            raise RuntimeError("Noise stream was not initialized.")
        try:
            self._pending_noise_chunk = next(self._noise_stream)
        except StopIteration as error:
            raise ValueError("Noise stream ended before all orchestration batches were generated.") from error
        self._noise_stream_position += 1
        return self._pending_noise_chunk

    def _ensure_noise_stream(self) -> None:
        """Open or realign the shared upstream noise stream to the current batch index."""
        target_position = max(int(self.counter), int(self.noise_stream_committed_count))
        if self._noise_stream is not None and self._noise_stream_position == target_position:
            return

        if int(self.counter) > 0 and self._root_seed() is None:
            raise ValueError(
                "Cannot resume an unseeded noise stream from a non-zero batch index; "
                "the upstream stream is non-deterministic without a seed."
            )
        self._pending_noise_chunk = None

        self._noise_stream = self.noise_adapter.open_stream(
            chunk_duration=float(self.duration.value),
            sampling_frequency=float(self.sampling_frequency.value),
            detectors=list(self.noise_arguments["detectors"]),
            seed=self._noise_stream_seed(),
            psd_file=self.noise_arguments.get("psd_file"),
            psd_schedule=self.noise_arguments.get("psd_schedule"),
            psd_files=self.noise_arguments.get("psd_files"),
            csd_files=self.noise_arguments.get("csd_files"),
            low_frequency_cutoff=self.noise_arguments.get("low_frequency_cutoff", 2.0),
            high_frequency_cutoff=self.noise_arguments.get("high_frequency_cutoff"),
            spectral_lines=self.noise_arguments.get("spectral_lines"),
            glitches=self.noise_arguments.get("glitches"),
        )
        self._noise_stream_position = 0
        for _ in range(target_position):
            try:
                next(self._noise_stream)
            except StopIteration as error:
                raise ValueError(
                    "Noise stream ended before the saved orchestration state could be restored."
                ) from error
            self._noise_stream_position += 1

    @classmethod
    def _resolve_detector_network(cls, detector_specs: Sequence[str]) -> tuple[Network, dict[str, Any]]:
        resolved_detectors: list[str | Any] = []
        resolution_steps: list[dict[str, Any]] = []
        for detector_spec in detector_specs:
            resolved, resolution_step = SignalAdapter.resolve_detector_spec(str(detector_spec))
            resolved_detectors.extend(resolved)
            resolution_steps.append(resolution_step)

        network = Network.from_detectors(tuple(resolved_detectors))
        return network, {
            "inputs": [str(detector_spec) for detector_spec in detector_specs],
            "detector_names": cls._network_detector_names(network),
            "steps": resolution_steps,
        }

    @staticmethod
    def _network_detector_names(network: Network) -> list[str]:
        return [detector if isinstance(detector, str) else detector.name for detector in network.detector_names]

    def _resolve_signal_channels(self, *, data: TimeSeries, channel_spec: Any) -> list[str | None]:
        if channel_spec is None:
            return [None] * data.num_of_channels

        channel_value = expand_template_variables(channel_spec, self)
        if isinstance(channel_value, str):
            return [channel_value] * data.num_of_channels

        channel_names = [str(channel) for channel in list(channel_value)]
        if len(channel_names) != data.num_of_channels:
            raise ValueError("Length of channel list must match number of channels in data.")
        return channel_names

    def _build_signal_stack(
        self,
        *,
        data: TimeSeries,
        channel_names: list[str | None],
        detector_names: list[str] | None = None,
        indices: list[int] | None = None,
    ) -> DetectorStrainStack:
        active_detector_names = self.detectors if detector_names is None else detector_names
        active_indices = list(range(data.num_of_channels)) if indices is None else indices
        if len(active_detector_names) != len(active_indices) or len(channel_names) != len(active_indices):
            raise ValueError("Signal detector, channel, and data selections must have matching lengths.")

        mapping = {}
        effective_names = []
        for detector_name, channel_name, index in zip(
            active_detector_names, channel_names, active_indices, strict=True
        ):
            series = data[index].copy()
            effective_name = channel_name if channel_name is not None else detector_name
            if effective_name in mapping:
                effective_name = f"{effective_name}__{detector_name}"
            effective_names.append(effective_name)
            mapping[effective_name] = series
        return DetectorStrainStack.from_mapping(effective_names, mapping)
