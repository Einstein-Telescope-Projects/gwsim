"""Adapter from gwmock orchestration to public ``gwmock_signal`` APIs."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from gwmock_signal import CustomDetector, DetectorStrainStack, Network, resolve_simulator_backend

from gwmock.data.time_series.time_series import TimeSeries

logger = logging.getLogger("gwmock")

_DEFAULT_WAVEFORM_MODEL = "IMRPhenomXPHM"
_LEGACY_SINGLE_DETECTOR_ALIASES = {
    "E1_triangle_sardinia": ("ET-Triangle-Sardinia", "ET1_SARD"),
    "E2_triangle_sardinia": ("ET-Triangle-Sardinia", "ET2_SARD"),
    "E3_triangle_sardinia": ("ET-Triangle-Sardinia", "ET3_SARD"),
    "E1_triangle_emr": ("ET-Triangle-EMR", "ET1_EMR"),
    "E2_triangle_emr": ("ET-Triangle-EMR", "ET2_EMR"),
    "E3_triangle_emr": ("ET-Triangle-EMR", "ET3_EMR"),
    "E1_2L_aligned_sardinia": ("ET-2L-Aligned", "ET1_2L_ALIGNED_SARD"),
    "E2_2L_aligned_emr": ("ET-2L-Aligned", "ET2_2L_ALIGNED_EMR"),
    "E1_2L_misaligned_sardinia": ("ET-2L-Misaligned", "ET1_2L_MISALIGNED_SARD"),
    "E2_2L_misaligned_emr": ("ET-2L-Misaligned", "ET2_2L_MISALIGNED_EMR"),
}
DetectorSpec = str | CustomDetector


def _callable_waveform_registry_key(func: Callable[..., Any]) -> str:
    """Return a unique registry name for *func* on a ``WaveformFactory`` instance."""
    qual = getattr(func, "__qualname__", type(func).__name__)
    mod = getattr(func, "__module__", "")
    return f"__gwmock_custom__{mod}:{qual}__{id(func):#x}"


def _register_callable_waveform(backend: Any, registry_key: str, factory: Callable[..., Any]) -> None:
    """Register *factory* under *registry_key* through the public backend API."""
    backend.register_waveform_model(registry_key, factory)


def _resolve_detector_path(detector_spec: str) -> Path | None:
    """Resolve a detector spec to an on-disk network file when one exists."""
    detector_path = Path(detector_spec)
    if detector_path.is_file():
        return detector_path

    return None


def _network_detector_names(detector_specs: Sequence[DetectorSpec]) -> list[str]:
    """Normalize detector specs to their output detector-name strings."""
    return [detector if isinstance(detector, str) else detector.name for detector in detector_specs]


@lru_cache(maxsize=1)
def _single_detector_catalog() -> dict[str, CustomDetector]:
    """Build a catalog of public single-detector aliases backed by public presets."""
    catalog: dict[str, CustomDetector] = {}
    for preset_name in Network.list_names():
        try:
            detector_specs = Network.from_preset(preset_name).detector_names
        except ValueError:
            continue
        for detector in detector_specs:
            if isinstance(detector, str):
                continue
            catalog.setdefault(detector.name, detector)
            catalog.setdefault(detector.name.lower(), detector)

    for alias, (preset_name, detector_name) in _LEGACY_SINGLE_DETECTOR_ALIASES.items():
        detector = next(
            spec
            for spec in Network.from_preset(preset_name).detector_names
            if not isinstance(spec, str) and spec.name == detector_name
        )
        catalog[alias] = detector

    return catalog


def _resolve_single_detector_alias(detector_spec: str) -> tuple[CustomDetector, ...] | None:
    """Resolve one public or legacy single-detector alias via public preset geometry."""
    detector = _single_detector_catalog().get(detector_spec) or _single_detector_catalog().get(detector_spec.lower())
    if detector is None:
        return None
    return (detector,)


def _resolve_detector_spec(detector_spec: str) -> tuple[tuple[DetectorSpec, ...], dict[str, Any]]:
    """Resolve one detector spec and return both the detector specs and metadata."""
    detector_alias = str(detector_spec)
    detector_path = _resolve_detector_path(detector_alias)
    if detector_path is not None:
        resolved = tuple(Network.from_file(detector_path).detector_names)
        return resolved, {
            "input": detector_alias,
            "resolver": "file",
            "source": str(detector_path),
            "detector_names": _network_detector_names(resolved),
        }

    try:
        resolved = tuple(Network.from_preset(detector_alias).detector_names)
        return resolved, {
            "input": detector_alias,
            "resolver": "preset",
            "detector_names": _network_detector_names(resolved),
        }
    except ValueError:
        pass

    try:
        resolved = tuple(Network.from_name(detector_alias).detector_names)
        return resolved, {
            "input": detector_alias,
            "resolver": "name",
            "detector_names": _network_detector_names(resolved),
        }
    except ValueError:
        pass

    resolved_single = _resolve_single_detector_alias(detector_alias)
    if resolved_single is not None:
        return resolved_single, {
            "input": detector_alias,
            "resolver": "preset-detector",
            "detector_names": _network_detector_names(resolved_single),
        }

    return (detector_alias,), {
        "input": detector_alias,
        "resolver": "detector",
        "detector_names": [detector_alias],
    }


class SignalAdapter:
    """Bridge gwmock population/orchestration state to public gwmock-signal APIs."""

    def __init__(
        self,
        *,
        source_type: str,
        backend: Any,
        network: Network,
    ) -> None:
        """Store the resolved backend and detector network.

        Args:
            source_type: The source type to use for the backend.
            backend: The backend to use.
            network: The network to use.
        """
        self._source_type = source_type
        self._backend = backend
        self._network = network
        self._detector_names = tuple(
            detector if isinstance(detector, str) else detector.name for detector in self._network.detector_names
        )
        self._unsupported_params: set[str] = set()

    @classmethod
    def from_source_type(
        cls,
        *,
        source_type: str,
        waveform_model: str | Callable[..., Any] | None,
        backend_arguments: Mapping[str, Any] | None = None,
        detectors: Sequence[str] | None = None,
        network: Network | None = None,
    ) -> SignalAdapter:
        """Resolve the public gwmock-signal backend for one source type.

        Args:
            source_type: The source type to use for the backend.
            waveform_model: The waveform model to use.
            backend_arguments: Constructor arguments for the backend.
            detectors: The detectors to use.
            network: The network to use.

        Returns:
            A SignalAdapter instance.

        Raises:
            ValueError: If detectors is not a non-empty sequence.
        """
        backend_class = resolve_simulator_backend(source_type)
        backend = cls.instantiate_backend(
            backend_class,
            waveform_model=waveform_model,
            backend_arguments=backend_arguments,
        )
        return cls(
            source_type=source_type,
            backend=backend,
            network=cls._require_network(detectors=detectors, network=network),
        )

    @classmethod
    def from_backend(
        cls,
        *,
        source_type: str,
        backend: Any,
        detectors: Sequence[str] | None = None,
        network: Network | None = None,
    ) -> SignalAdapter:
        """Build an adapter from an already-instantiated backend.

        Args:
            source_type: The source type to use for the backend.
            backend: The backend to use.
            detectors: The detectors to use.
            network: The network to use.

        Returns:
            A SignalAdapter instance.
        """
        return cls(
            source_type=source_type,
            backend=backend,
            network=cls._require_network(detectors=detectors, network=network),
        )

    @staticmethod
    def instantiate_backend(
        backend_class: type[Any],
        *,
        waveform_model: str | Callable[..., Any] | None,
        backend_arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        """Instantiate a signal backend class while preserving callable waveform support.

        Args:
            backend_class: The backend class to instantiate.
            waveform_model: The waveform model to use.
            backend_arguments: Constructor arguments for the backend.

        Returns:
            An instantiated backend.
        """
        init_kwargs = dict(backend_arguments or {})
        if waveform_model is None:
            try:
                return backend_class(waveform_model=_DEFAULT_WAVEFORM_MODEL, **init_kwargs)
            except TypeError:
                return backend_class(**init_kwargs)

        if callable(waveform_model):
            registry_key = _callable_waveform_registry_key(waveform_model)
            backend = backend_class(waveform_model=registry_key, **init_kwargs)
            _register_callable_waveform(backend, registry_key, waveform_model)
            return backend

        return backend_class(waveform_model=waveform_model, **init_kwargs)

    @classmethod
    def _require_network(cls, *, detectors: Sequence[str] | None, network: Network | None) -> Network:
        if network is not None:
            return network
        if not detectors:
            raise ValueError("detectors must be a non-empty sequence.")
        return cls.resolve_detector_network(detectors)

    @staticmethod
    def resolve_detector_path(detector_spec: str) -> Path | None:
        """Resolve a detector spec to an on-disk network file when one exists.

        Args:
            detector_spec: The detector spec to resolve.

        Returns:
            The resolved detector path or None if not found.
        """
        return _resolve_detector_path(detector_spec)

    @staticmethod
    def resolve_detector_spec(detector_spec: str) -> tuple[tuple[DetectorSpec, ...], dict[str, Any]]:
        """Resolve one detector spec into public detector objects or built-in detector names.

        Args:
            detector_spec: The detector spec to resolve.

        Returns:
            A tuple of the resolved detector specs and metadata.
        """
        return _resolve_detector_spec(detector_spec)

    @staticmethod
    def resolve_detector_network(detector_specs: Sequence[str]) -> Network:
        """Resolve detector specs into one public ``Network`` instance.

        Args:
            detector_specs: The detector specs to resolve.

        Returns:
            A resolved detector network.
        """
        resolved_detectors: list[DetectorSpec] = []
        for detector_spec in detector_specs:
            resolved, _ = _resolve_detector_spec(str(detector_spec))
            resolved_detectors.extend(resolved)

        return Network.from_detectors(tuple(resolved_detectors))

    @property
    def source_type(self) -> str:
        """Return the source-type routing key used for backend resolution.

        Returns:
            The source-type routing key used for backend resolution.
        """
        return self._source_type

    @property
    def detector_names(self) -> tuple[str, ...]:
        """Return the ordered detector names used for output stacking.

        Returns:
            The ordered detector names used for output stacking.
        """
        return self._detector_names

    @property
    def network(self) -> Network:
        """Return the resolved public detector network.

        Returns:
            The resolved public detector network.
        """
        return self._network

    def simulate_stack(
        self,
        parameters: Mapping[str, Any],
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        waveform_arguments: Mapping[str, Any] | None = None,
        background: Mapping[str, Any] | None = None,
        earth_rotation: bool = True,
    ) -> DetectorStrainStack:
        """Generate one detector strain stack via the public ``gwmock_signal`` API.

        Args:
            parameters: The parameters to use for the simulation.
            sampling_frequency: The sampling frequency to use for the simulation.
            minimum_frequency: The minimum frequency to use for the simulation.
            waveform_arguments: The waveform arguments to use for the simulation.
            background: Optional background mapping forwarded to the signal backend.
            earth_rotation: Whether to include earth rotation in the simulation.

        Returns:
            A DetectorStrainStack instance.
        """
        backend_parameters = {**(waveform_arguments or {}), **dict(parameters)}
        if self._unsupported_params:
            backend_parameters = {k: v for k, v in backend_parameters.items() if k not in self._unsupported_params}
        try:
            return self._backend.simulate(
                backend_parameters,
                self._network.detector_names,
                background=background,
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
                earth_rotation=earth_rotation,
            )
        except ValueError as exc:
            _prefix = "Unsupported LAL waveform parameters:"
            msg = str(exc)
            if not msg.startswith(_prefix):
                raise
            extras = {k.strip() for k in msg[len(_prefix) :].split(",")}
            new = extras - self._unsupported_params
            if new:
                logger.warning(
                    "Waveform backend does not accept the following population parameters "
                    "and they will be ignored: %s. "
                    "They are still recorded in injection_parameters metadata.",
                    ", ".join(sorted(new)),
                )
                self._unsupported_params |= new
            filtered = {k: v for k, v in backend_parameters.items() if k not in self._unsupported_params}
            return self._backend.simulate(
                filtered,
                self._network.detector_names,
                background=background,
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
                earth_rotation=earth_rotation,
            )

    def simulate(
        self,
        parameters: Mapping[str, Any],
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        waveform_arguments: Mapping[str, Any] | None = None,
        background: Mapping[str, Any] | None = None,
        earth_rotation: bool = True,
    ) -> TimeSeries:
        """Generate one signal chunk via the public gwmock-signal ``simulate`` API.

        Args:
            parameters: The parameters to use for the simulation.
            sampling_frequency: The sampling frequency to use for the simulation.
            minimum_frequency: The minimum frequency to use for the simulation.
            waveform_arguments: The waveform arguments to use for the simulation.
            background: Optional background mapping forwarded to the signal backend.
            earth_rotation: Whether to include earth rotation in the simulation.

        Returns:
            A TimeSeries instance.
        """
        strain_stack = self.simulate_stack(
            parameters,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            waveform_arguments=waveform_arguments,
            background=background,
            earth_rotation=earth_rotation,
        )
        return TimeSeries(
            data=strain_stack.data,
            start_time=strain_stack.t0,
            sampling_frequency=strain_stack.sample_rate,
        )

    def set_seed(self, seed: int | None) -> None:
        """Set a backend seed attribute when the backend exposes one."""
        if hasattr(self._backend, "seed"):
            self._backend.seed = seed
