"""Adapter from gwmock orchestration to public ``gwmock_noise`` APIs."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np
from gwmock_noise import (
    AddLines,
    BaseNoiseSimulator,
    ColoredNoiseSimulator,
    CorrelatedNoiseSimulator,
    DefaultNoiseSimulator,
    FrameWriter,
    InjectGlitches,
    NoiseComponentConfig,
    NoiseConfig,
    NoiseSimulator,
    OutputConfig,
    SimulationResult,
    SpectralLineSimulator,
)
from gwmock_noise import (
    open_stream as upstream_open_stream,
)
from gwmock_noise.gaussian import normalize_spectral_lines
from gwmock_noise.glitches import normalize_glitch_models

from gwmock.utils.download import download_file

_SUPPORTED_OUTPUT_FORMATS = {"npy", "gwf"}
_DETECTOR_PAIR_SIZE = 2


def _format_frame_time_token(value: float) -> str:
    """Match gwmock-noise frame filename token formatting.

    Args:
        value: The value to format.

    Returns:
        The formatted value.
    """
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _frame_channel_name(detector: str, channel: str) -> str:
    """Return detector channel name used by gwmock-noise frame outputs.

    Args:
        detector: The detector name.
        channel: The channel name suffix.

    Returns:
        The detector channel name.
    """
    return f"{detector}:{channel}"


def _frame_artifact_name(
    *,
    detector: str,
    channel: str,
    prefix: str,
    gps_start: float,
    duration: float,
) -> str:
    """Return the GWF filename used by gwmock-noise ``FrameWriter``.

    Args:
        detector: The detector name.
        channel: The full channel name (e.g. ``H1:MOCK_NOISE``).
        prefix: The prefix.
        gps_start: The GPS start time.
        duration: The duration.

    Returns:
        The GWF filename.
    """
    start_token = _format_frame_time_token(gps_start)
    duration_token = _format_frame_time_token(duration)
    name = f"{detector[0]}-{channel}_{start_token}-{duration_token}.gwf"
    return f"{prefix}_{name}" if prefix else name


def _coerce_path(value: str | Path | None) -> Path | None:
    """Normalize a path-like input.

    Args:
        value: The value to coerce.

    Returns:
        The coerced path.
    """
    if value is None:
        return None
    return Path(value)


def _coerce_path_mapping(values: dict[str, str | Path] | None) -> dict[str, Path] | None:
    """Normalize mapping values to ``Path`` objects.

    Args:
        values: The values to coerce.

    Returns:
        The coerced path mapping.
    """
    if values is None:
        return None
    return {key: Path(value) for key, value in values.items()}


def _coerce_path_schedule(values: list[tuple[float, str | Path]] | None) -> list[tuple[float, Path]] | None:
    """Normalize scheduled path values to ``Path`` objects.

    Args:
        values: The values to coerce.

    Returns:
        The coerced path schedule.
    """
    if values is None:
        return None
    return [(offset, Path(path)) for offset, path in values]


def _parse_csd_file_map(csd_files: dict[str, Path] | None) -> dict[tuple[str, str], Path]:
    """Convert ``DET1-DET2`` mapping keys into detector-pair tuples.

    Args:
        csd_files: The CSD files to parse.

    Returns:
        The parsed CSD file map.
    """
    if not csd_files:
        return {}

    parsed: dict[tuple[str, str], Path] = {}
    for pair_key, file_path in csd_files.items():
        detectors = pair_key.split("-")
        if len(detectors) != _DETECTOR_PAIR_SIZE or not all(detectors):
            raise ValueError("csd_files keys must use the 'DET1-DET2' format.")

        detector_a, detector_b = tuple(sorted(detectors))
        if detector_a == detector_b:
            raise ValueError("csd_files keys must reference two distinct detectors.")

        normalized_key = (detector_a, detector_b)
        if normalized_key in parsed:
            raise ValueError(f"Duplicate CSD file mapping for detector pair {detector_a}-{detector_b}.")
        parsed[normalized_key] = Path(file_path)
    return parsed


_REMOTE_GLITCH_FILE_KEYS = ("population_file", "psd_file")


def _is_remote_url(value: Any) -> bool:
    """Return True if ``value`` is an ``http``/``https`` URL string.

    Args:
        value: The candidate file reference.

    Returns:
        Whether the value is a remote URL (named assets and local paths are not).
    """
    return isinstance(value, str) and urlparse(value).scheme in {"http", "https"}


def _glitch_population_cache_directory() -> Path:
    """Return the cache directory for downloaded glitch population/PSD files."""
    return Path(tempfile.gettempdir()) / "gwmock" / "noise_glitches"


def _resolve_glitch_file_urls(glitches: list[Any] | None) -> list[Any] | None:
    """Download URL-valued glitch file references to a local cache.

    gwmock-noise glitch models (e.g. ``GengliBlipGlitch``) read ``population_file``
    and ``psd_file`` with ``Path``/``h5py``, which cannot handle URLs. Resolve any
    ``http(s)`` URL to a cached local path so remote files declared in configs work
    transparently. Named bundled assets (e.g. ``ET_10_full_cryo_psd``) and local
    paths are returned unchanged. The input list and its entries are not mutated.

    Args:
        glitches: The raw glitch-model configuration entries.

    Returns:
        A copy of ``glitches`` with remote file references replaced by local paths.
    """
    if not glitches:
        return glitches

    resolved: list[Any] = []
    for entry in glitches:
        if not isinstance(entry, Mapping):
            resolved.append(entry)
            continue
        updated = dict(entry)
        for key in _REMOTE_GLITCH_FILE_KEYS:
            value = updated.get(key)
            if _is_remote_url(value):
                updated[key] = str(
                    download_file(
                        value,
                        outdir=_glitch_population_cache_directory(),
                        allow_existing=True,
                        dest_path_from_hashed_url=True,
                    )
                )
        resolved.append(updated)
    return resolved


def _build_components(
    *,
    psd_file: Path | None,
    psd_schedule: list[tuple[float, Path]] | None,
    psd_files: dict[str, Path] | None,
    csd_files: dict[str, Path] | None,
    low_frequency_cutoff: float,
    high_frequency_cutoff: float | None,
    spectral_lines: list[Any] | None,
    glitches: list[Any] | None,
) -> list[NoiseComponentConfig]:
    """Translate legacy flat noise fields into the v0.3 component list."""
    components: list[NoiseComponentConfig] = []

    if psd_files is not None or csd_files is not None:
        options: dict[str, Any] = {}
        if psd_files:
            options["psd_files"] = psd_files
        if csd_files:
            options["csd_files"] = csd_files
        options["low_frequency_cutoff"] = low_frequency_cutoff
        if high_frequency_cutoff is not None:
            options["high_frequency_cutoff"] = high_frequency_cutoff
        components.append(NoiseComponentConfig(simulator="correlated", options=options))
    elif psd_file is not None or psd_schedule is not None:
        options = {}
        if psd_file is not None:
            options["psd_file"] = psd_file
        if psd_schedule is not None:
            options["psd_schedule"] = psd_schedule
        options["low_frequency_cutoff"] = low_frequency_cutoff
        if high_frequency_cutoff is not None:
            options["high_frequency_cutoff"] = high_frequency_cutoff
        components.append(NoiseComponentConfig(simulator="colored", options=options))

    if spectral_lines:
        components.append(NoiseComponentConfig(simulator="spectral_lines", options={"lines": spectral_lines}))

    if glitches:
        components.append(NoiseComponentConfig(simulator="glitches", options={"models": glitches}))

    return components


class NoiseAdapter:
    """Bridge gwmock orchestration state to public ``gwmock_noise`` APIs."""

    def __init__(self, *, backend: Any) -> None:
        """Store the resolved public gwmock-noise backend.

        Args:
            backend: The backend to use.
        """
        self._backend = backend

    @classmethod
    def from_backend(cls, backend: BaseNoiseSimulator | NoiseSimulator | Any | None = None) -> NoiseAdapter:
        """Build an adapter from a public gwmock-noise backend.

        Args:
            backend: The backend to use.

        Returns:
            A NoiseAdapter instance.
        """
        if backend is None:
            resolved_backend = DefaultNoiseSimulator()
        elif isinstance(backend, (BaseNoiseSimulator, NoiseSimulator)) or callable(getattr(backend, "run", None)):
            resolved_backend = backend
        else:
            raise TypeError("backend must satisfy BaseNoiseSimulator or NoiseSimulator.")
        return cls(backend=resolved_backend)

    @property
    def backend(self) -> Any:
        """Return the public backend used by the adapter.

        Returns:
            The public backend used by the adapter.
        """
        return self._backend

    def run(  # noqa: PLR0913
        self,
        *,
        detectors: list[str],
        duration: float,
        sampling_frequency: float,
        output_directory: str | Path,
        output_prefix: str,
        output_format: Literal["npy", "gwf"],
        gps_start: float,
        channel: str = "MOCK_NOISE",
        channels: dict[str, str] | None = None,
        seed: int | None = None,
        psd_file: str | Path | None = None,
        psd_schedule: list[tuple[float, str | Path]] | None = None,
        psd_files: dict[str, str | Path] | None = None,
        csd_files: dict[str, str | Path] | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        spectral_lines: list[Any] | None = None,
        glitches: list[Any] | None = None,
    ) -> SimulationResult:
        """Run one noise batch through the public gwmock-noise boundary.

        Args:
            detectors: The detectors to use.
            duration: The duration.
            sampling_frequency: The sampling frequency.
            output_directory: The output directory.
            output_prefix: The output prefix.
            output_format: The output format.
            gps_start: The GPS start time.
            channel: The channel name suffix assembled as ``{detector}:{channel}``.
            channels: Optional per-detector full channel names, e.g. ``{"H1": "H1:STRAIN_NOISE"}``.
            seed: The seed.
            psd_file: The PSD file.
            psd_schedule: The PSD schedule.
            psd_files: The PSD files.
            csd_files: The CSD files.
            low_frequency_cutoff: The low frequency cutoff.
            high_frequency_cutoff: The high frequency cutoff.
            spectral_lines: The spectral lines.
            glitches: The glitches.

        Returns:
            The simulation result.
        """
        config = self.build_config(
            detectors=detectors,
            duration=duration,
            sampling_frequency=sampling_frequency,
            output_directory=output_directory,
            output_prefix=output_prefix,
            output_format=output_format,
            gps_start=gps_start,
            channel=channel,
            channels=channels,
            seed=seed,
            psd_file=psd_file,
            psd_schedule=psd_schedule,
            psd_files=psd_files,
            csd_files=csd_files,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            spectral_lines=spectral_lines,
            glitches=glitches,
        )
        if callable(getattr(self._backend, "run", None)):
            return self._backend.run(config)

        if not isinstance(self._backend, NoiseSimulator):
            raise TypeError("Noise backend must expose run() or satisfy the gwmock_noise NoiseSimulator protocol.")

        chunk = self._backend.generate(
            duration=duration,
            sampling_frequency=sampling_frequency,
            detectors=list(detectors),
            seed=seed,
        )
        return self.write_chunk(config=config, chunk=chunk)

    def open_stream(  # noqa: PLR0913
        self,
        *,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: Sequence[str],
        seed: int | None = None,
        psd_file: str | Path | None = None,
        psd_schedule: list[tuple[float, str | Path]] | None = None,
        psd_files: dict[str, str | Path] | None = None,
        csd_files: dict[str, str | Path] | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        spectral_lines: list[Any] | None = None,
        glitches: list[Any] | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Open one stateful upstream stream and consume it chunk-by-chunk.

        Args:
            chunk_duration: The chunk duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.
            psd_file: The PSD file.
            psd_schedule: The PSD schedule.
            psd_files: The PSD files.
            csd_files: The CSD files.
            low_frequency_cutoff: The low frequency cutoff.
            high_frequency_cutoff: The high frequency cutoff.
            spectral_lines: The spectral lines.
            glitches: The glitches.

        Returns:
            An iterator over the chunks.
        """
        simulator = self._resolve_stream_backend(
            chunk_duration=chunk_duration,
            sampling_frequency=sampling_frequency,
            detectors=list(detectors),
            seed=seed,
            psd_file=psd_file,
            psd_schedule=psd_schedule,
            psd_files=psd_files,
            csd_files=csd_files,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            spectral_lines=spectral_lines,
            glitches=glitches,
        )
        return upstream_open_stream(
            simulator,
            chunk_duration=chunk_duration,
            sampling_frequency=sampling_frequency,
            detectors=list(detectors),
            seed=seed,
        )

    def build_config(  # noqa: PLR0913
        self,
        *,
        detectors: Sequence[str],
        duration: float,
        sampling_frequency: float,
        output_directory: str | Path,
        output_prefix: str,
        output_format: Literal["npy", "gwf"],
        gps_start: float,
        channel: str = "MOCK_NOISE",
        channels: dict[str, str] | None = None,
        seed: int | None = None,
        psd_file: str | Path | None = None,
        psd_schedule: list[tuple[float, str | Path]] | None = None,
        psd_files: dict[str, str | Path] | None = None,
        csd_files: dict[str, str | Path] | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        spectral_lines: list[Any] | None = None,
        glitches: list[Any] | None = None,
    ) -> NoiseConfig:
        """Construct the public gwmock-noise config model for one output chunk.

        Args:
            detectors: The detectors to use.
            duration: The duration.
            sampling_frequency: The sampling frequency.
            output_directory: The output directory.
            output_prefix: The output prefix.
            output_format: The output format.
            gps_start: The GPS start time.
            channel: The channel name suffix assembled as ``{detector}:{channel}``.
            channels: Optional per-detector full channel names, e.g. ``{"H1": "H1:STRAIN_NOISE"}``.
            seed: The seed.
            psd_file: The PSD file.
            psd_schedule: The PSD schedule.
            psd_files: The PSD files.
            csd_files: The CSD files.
            low_frequency_cutoff: The low frequency cutoff.
            high_frequency_cutoff: The high frequency cutoff.
            spectral_lines: The spectral lines.
            glitches: The glitches.

        Returns:
            The noise config.
        """
        components = _build_components(
            psd_file=_coerce_path(psd_file),
            psd_schedule=_coerce_path_schedule(psd_schedule),
            psd_files=_coerce_path_mapping(psd_files),
            csd_files=_coerce_path_mapping(csd_files),
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            spectral_lines=spectral_lines,
            glitches=glitches,
        )
        kwargs: dict[str, Any] = {
            "detectors": list(detectors),
            "duration": duration,
            "sampling_frequency": sampling_frequency,
            "output": OutputConfig(
                directory=Path(output_directory),
                prefix=output_prefix,
                format=output_format,
                gps_start=gps_start,
                channel=channel,
                channels=channels,
            ),
            "seed": seed,
        }
        if components:
            kwargs["components"] = components
        return NoiseConfig(**kwargs)

    def expected_output_paths(self, *, config: NoiseConfig) -> list[Path]:
        """Return the artifact paths gwmock will write for one chunk.

        Args:
            config: The noise config.

        Returns:
            The expected output paths.
        """
        if config.output.format == "npy":
            return [
                config.output.directory
                / (f"{config.output.prefix}_{detector}.npy" if config.output.prefix else f"{detector}.npy")
                for detector in config.detectors
            ]
        return [
            config.output.directory
            / _frame_artifact_name(
                detector=detector,
                channel=(
                    config.output.channels[detector]
                    if config.output.channels and detector in config.output.channels
                    else _frame_channel_name(detector, config.output.channel)
                ),
                prefix=config.output.prefix,
                gps_start=config.output.gps_start,
                duration=config.duration,
            )
            for detector in config.detectors
        ]

    def write_chunk(self, *, config: NoiseConfig, chunk: Mapping[str, np.ndarray]) -> SimulationResult:
        """Write one chunk returned by ``open_stream`` to gwmock-owned outputs.

        Args:
            config: The noise config.
            chunk: The chunk to write.

        Returns:
            The simulation result.
        """
        chunk_by_detector = self._normalize_chunk(chunk=chunk, detectors=config.detectors)
        config.output.directory.mkdir(parents=True, exist_ok=True)
        if config.output.format == "gwf":
            output_paths = FrameWriter(
                _ChunkNoiseSimulator(chunk_by_detector),
                gps_start=config.output.gps_start,
                output_dir=config.output.directory,
                channel=config.output.channel,
                channels=config.output.channels,
                prefix=config.output.prefix,
            ).write(
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                detectors=config.detectors,
                seed=None,
            )
        else:
            output_paths = {}
            for detector, strain in chunk_by_detector.items():
                file_name = f"{config.output.prefix}_{detector}.npy" if config.output.prefix else f"{detector}.npy"
                output_path = config.output.directory / file_name
                np.save(output_path, strain)
                output_paths[detector] = output_path
        return SimulationResult(output_paths=output_paths, config=config)

    def _normalize_chunk(self, *, chunk: Mapping[str, np.ndarray], detectors: Sequence[str]) -> dict[str, np.ndarray]:
        """Validate and normalize one upstream chunk.

        Args:
            chunk: The chunk to normalize.
            detectors: The detectors to use.

        Returns:
            The normalized chunk.
        """
        normalized: dict[str, np.ndarray] = {}
        for detector in detectors:
            if detector not in chunk:
                raise ValueError(f"Noise stream did not produce detector '{detector}'.")
            normalized[detector] = np.asarray(chunk[detector])
        return normalized

    def _resolve_stream_backend(  # noqa: PLR0913
        self,
        *,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None,
        psd_file: str | Path | None,
        psd_schedule: list[tuple[float, str | Path]] | None,
        psd_files: dict[str, str | Path] | None,
        csd_files: dict[str, str | Path] | None,
        low_frequency_cutoff: float,
        high_frequency_cutoff: float | None,
        spectral_lines: list[Any] | None,
        glitches: list[Any] | None,
    ) -> NoiseSimulator:
        """Return the protocol-compatible backend for ``open_stream``.

        Args:
            chunk_duration: The chunk duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.
            psd_file: The PSD file.
            psd_schedule: The PSD schedule.
            psd_files: The PSD files.
            csd_files: The CSD files.
            low_frequency_cutoff: The low frequency cutoff.
            high_frequency_cutoff: The high frequency cutoff.
            spectral_lines: The spectral lines.
            glitches: The glitches.

        Returns:
            The protocol-compatible backend.
        """
        if isinstance(self._backend, DefaultNoiseSimulator):
            protocol_backend = self._configure_default_stream_backend(
                chunk_duration=chunk_duration,
                sampling_frequency=sampling_frequency,
                detectors=detectors,
                seed=seed,
                psd_file=psd_file,
                psd_schedule=psd_schedule,
                psd_files=psd_files,
                csd_files=csd_files,
                low_frequency_cutoff=low_frequency_cutoff,
                high_frequency_cutoff=high_frequency_cutoff,
                spectral_lines=spectral_lines,
                glitches=glitches,
            )
            if protocol_backend is not None:
                return protocol_backend

        if isinstance(self._backend, NoiseSimulator):
            return self._backend

        raise TypeError("Noise backend must satisfy the gwmock_noise NoiseSimulator protocol to open a stream.")

    def _configure_default_stream_backend(  # noqa: PLR0913
        self,
        *,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None,
        psd_file: str | Path | None,
        psd_schedule: list[tuple[float, str | Path]] | None,
        psd_files: dict[str, str | Path] | None,
        csd_files: dict[str, str | Path] | None,
        low_frequency_cutoff: float,
        high_frequency_cutoff: float | None,
        spectral_lines: list[Any] | None,
        glitches: list[Any] | None,
    ) -> NoiseSimulator | None:
        """Mirror the default gwmock-noise backend selection with protocol simulators.

        Args:
            chunk_duration: The chunk duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.
            psd_file: The PSD file.
            psd_schedule: The PSD schedule.
            psd_files: The PSD files.
            csd_files: The CSD files.
            low_frequency_cutoff: The low frequency cutoff.
            high_frequency_cutoff: The high frequency cutoff.
            spectral_lines: The spectral lines.
            glitches: The glitches.

        Returns:
            The protocol-compatible backend.
        """
        normalized_psd_files = _coerce_path_mapping(psd_files)
        normalized_csd_files = _coerce_path_mapping(csd_files)
        normalized_psd_schedule = _coerce_path_schedule(psd_schedule)
        normalized_psd_file = _coerce_path(psd_file)

        simulator: NoiseSimulator | None = None
        if normalized_psd_files is not None or normalized_csd_files is not None:
            simulator = CorrelatedNoiseSimulator(
                psd_files=normalized_psd_files or {},
                csd_files=_parse_csd_file_map(normalized_csd_files),
                detectors=detectors,
                duration=chunk_duration,
                sampling_frequency=sampling_frequency,
                seed=seed,
                low_frequency_cutoff=low_frequency_cutoff,
                high_frequency_cutoff=high_frequency_cutoff,
            )
        elif normalized_psd_file is not None or normalized_psd_schedule is not None:
            simulator = ColoredNoiseSimulator(
                psd_file=normalized_psd_file,
                psd_schedule=normalized_psd_schedule,
                detectors=detectors,
                duration=chunk_duration,
                sampling_frequency=sampling_frequency,
                seed=seed,
                low_frequency_cutoff=low_frequency_cutoff,
                high_frequency_cutoff=high_frequency_cutoff,
            )

        if spectral_lines is not None:
            if not spectral_lines:
                raise ValueError("spectral_lines must contain at least one spectral line.")
            normalized_lines = normalize_spectral_lines(spectral_lines)
            simulator = (
                SpectralLineSimulator(
                    lines=normalized_lines,
                    detectors=detectors,
                    duration=chunk_duration,
                    sampling_frequency=sampling_frequency,
                    seed=seed,
                )
                if simulator is None
                else AddLines(simulator, normalized_lines)
            )

        if glitches is not None:
            if not glitches:
                raise ValueError("glitches must contain at least one glitch model.")
            resolved_glitches = _resolve_glitch_file_urls(glitches)
            if simulator is None:
                simulator = _ZeroNoiseSimulator(
                    detectors=detectors,
                    duration=chunk_duration,
                    sampling_frequency=sampling_frequency,
                    seed=seed,
                )
            simulator = InjectGlitches(simulator, normalize_glitch_models(resolved_glitches))

        return simulator


class _ChunkNoiseSimulator:
    """Protocol adapter that replays one already-generated chunk."""

    def __init__(self, chunk: Mapping[str, np.ndarray]) -> None:
        """Initialize the chunk noise simulator.

        Args:
            chunk: The chunk to replay.
        """
        self._chunk = {detector: np.asarray(strain) for detector, strain in chunk.items()}
        self.detectors = list(self._chunk)
        self.duration = 0.0
        self.sampling_frequency = 0.0
        self.seed = None

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return the stored chunk after validating the requested shape.

        Args:
            duration: The duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.

        Returns:
            The generated chunk.

        Raises:
            ValueError: If the noise stream did not produce the requested detector.
            ValueError: If the noise chunk for the requested detector has the wrong number of samples.
        """
        _ = seed
        expected_samples = round(duration * sampling_frequency)
        generated: dict[str, np.ndarray] = {}
        for detector in detectors:
            if detector not in self._chunk:
                raise ValueError(f"Noise stream did not produce detector '{detector}'.")
            strain = self._chunk[detector]
            if strain.shape[0] != expected_samples:
                raise ValueError(
                    f"Noise chunk for detector '{detector}' has {strain.shape[0]} samples; expected {expected_samples}."
                )
            generated[detector] = strain
        self.detectors = list(detectors)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        return generated

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield the stored chunk once.

        Args:
            chunk_duration: The chunk duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.

        Returns:
            An iterator over the generated chunk.
        """
        yield self.generate(chunk_duration, sampling_frequency, detectors, seed)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return adapter metadata layered on the chunk replay backend.

        Returns:
            The adapter metadata.
        """
        return {"adapter": "chunk-replay"}


class _ZeroNoiseSimulator:
    """Protocol-compatible zero-noise backend used for glitches-only streams."""

    def __init__(
        self,
        *,
        detectors: list[str],
        duration: float,
        sampling_frequency: float,
        seed: int | None,
    ) -> None:
        """Initialize the zero noise simulator.

        Args:
            detectors: The detectors to use.
            duration: The duration.
            sampling_frequency: The sampling frequency.
            seed: The seed.
        """
        self.detectors = list(detectors)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.seed = seed

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return zeros with the requested runtime shape.

        Args:
            duration: The duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.
        """
        _ = seed
        n_samples = round(duration * sampling_frequency)
        self.detectors = list(detectors)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        return {detector: np.zeros(n_samples, dtype=float) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield zero-noise chunks lazily.

        Args:
            chunk_duration: The chunk duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.

        Returns:
            An iterator over the generated chunk.
        """
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata for the zero-noise helper backend.

        Returns:
            The metadata.
        """
        return {"kind": "zero-noise"}
