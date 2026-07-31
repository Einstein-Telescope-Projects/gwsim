"""Resolve orchestration backends from aliases, entry points, or import paths."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping, Sequence
from importlib import import_module, metadata
from typing import Any, Literal

from gwmock_noise import BaseNoiseSimulator, DefaultNoiseSimulator
from gwmock_noise.simulators.protocol import NoiseSimulator
from gwmock_pop import (
    BBHSimulator,
    BNSSimulator,
    CBCSimulator,
    FilePopulationLoader,
    GWPopSimulator,
    NSBHSimulator,
)
from gwmock_signal import GWSimulator, resolve_simulator_backend

BackendKind = Literal["population", "signal", "noise", "waveform"]

_ENTRY_POINT_GROUPS: dict[BackendKind, str] = {
    "population": "gwmock.population",
    "signal": "gwmock.signal",
    "noise": "gwmock.noise",
    "waveform": "gwmock.waveform",
}

#: Waveform backends, as ``module:Class`` paths rather than imported classes.
#:
#: Deliberately unimported here, so that resolving ``lal`` or ``pycbc`` does not require the
#: optional stack. What the ``gwmock[jax]`` extra actually supplies is **ripplegw** -- JAX
#: itself arrives unconditionally as a hard dependency of gwmock-pop -- and instantiating
#: ``RippleBackend`` without ripplegw raises gwmock-signal's own ``ImportError``, which names
#: what to install. Importing these at module scope would turn that into an import-time
#: failure of the whole CLI for users who selected a backend they do have.
_WAVEFORM_BACKEND_PATHS: dict[str, str] = {
    "lal": "gwmock_signal.waveform.backends.lal:LALSimulationBackend",
    "lalsimulation": "gwmock_signal.waveform.backends.lal:LALSimulationBackend",
    "LALSimulationBackend": "gwmock_signal.waveform.backends.lal:LALSimulationBackend",
    "pycbc": "gwmock_signal.waveform.backends.pycbc:PyCBCBackend",
    "PyCBCBackend": "gwmock_signal.waveform.backends.pycbc:PyCBCBackend",
    "ripple": "gwmock_signal.waveform.backends.ripple:RippleBackend",
    "RippleBackend": "gwmock_signal.waveform.backends.ripple:RippleBackend",
    "gwsignal": "gwmock_signal.waveform.backends.gwsignal:GWSignalBackend",
    "GWSignalBackend": "gwmock_signal.waveform.backends.gwsignal:GWSignalBackend",
}

_POPULATION_BACKENDS: dict[str, type[Any]] = {
    "BBHSimulator": BBHSimulator,
    "bbh": BBHSimulator,
    "CBCSimulator": CBCSimulator,
    "CBCPriorSimulator": CBCSimulator,  # legacy alias
    "cbc_prior": CBCSimulator,
    "BNSSimulator": BNSSimulator,
    "BNSPriorSimulator": BNSSimulator,  # legacy alias
    "bns_prior": BNSSimulator,
    "NSBHSimulator": NSBHSimulator,
    "NSBHPriorSimulator": NSBHSimulator,  # legacy alias
    "nsbh_prior": NSBHSimulator,
    "FilePopulationLoader": FilePopulationLoader,
    "file": FilePopulationLoader,
}

_NOISE_BACKENDS: dict[str, type[Any]] = {
    "DefaultNoiseSimulator": DefaultNoiseSimulator,
    "default": DefaultNoiseSimulator,
}

_LEGACY_PATH_WARNINGS: set[str] = set()


def resolve_backend_class(kind: BackendKind, backend_name: str) -> type[Any]:
    """Resolve *backend_name* to a backend class for the given *kind*."""
    normalized_name = _normalize_backend_name(kind, backend_name)

    backend_class = _resolve_builtin_backend(kind, normalized_name)
    if backend_class is not None:
        return backend_class

    backend_class = _resolve_entry_point_backend(kind, normalized_name)
    if backend_class is not None:
        return backend_class

    if ":" in normalized_name:
        return _load_backend_class(normalized_name)

    if "." in normalized_name:
        _warn_legacy_dotted_path(normalized_name)
        return _load_backend_class(normalized_name, separator=".")

    available = ", ".join(sorted(_builtin_aliases(kind)))
    group = _ENTRY_POINT_GROUPS[kind]
    raise ValueError(
        f"Unknown {kind} backend '{normalized_name}'. "
        f"Available built-in aliases: {available}. "
        f"Expected a built-in alias, an entry point in '{group}', or a 'module:Class' reference."
    )


def instantiate_backend(
    kind: BackendKind,
    backend_name: str,
    *,
    init_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Resolve, instantiate, and validate a backend."""
    backend_class = resolve_backend_class(kind, backend_name)
    kwargs = dict(init_kwargs or {})
    try:
        backend_instance = backend_class(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate {kind} backend '{backend_name}' with arguments {sorted(kwargs)}: {exc}"
        ) from exc

    validate_backend(kind, backend_name, backend_class, backend_instance)
    return backend_instance


def validate_backend(
    kind: BackendKind,
    backend_name: str,
    backend_class: type[Any],
    backend_instance: Any,
) -> None:
    """Validate a resolved backend instance against the expected public contract."""
    if kind == "population":
        _validate_population_backend(backend_name, backend_instance)
        return
    if kind == "signal":
        _validate_signal_backend(backend_name, backend_class, backend_instance)
        return
    if kind == "waveform":
        _validate_waveform_backend(backend_name, backend_instance)
        return
    _validate_noise_backend(backend_name, backend_instance)


def _normalize_backend_name(kind: BackendKind, backend_name: str) -> str:
    if not isinstance(backend_name, str) or not backend_name.strip():
        raise ValueError(f"{kind} backend must be a non-empty string.")
    return backend_name.strip()


def _resolve_builtin_backend(kind: BackendKind, backend_name: str) -> type[Any] | None:
    if kind == "population":
        return _POPULATION_BACKENDS.get(backend_name)
    if kind == "signal":
        try:
            return resolve_simulator_backend(backend_name)
        except (KeyError, ValueError):
            return None
    if kind == "waveform":
        path = _WAVEFORM_BACKEND_PATHS.get(backend_name)
        return None if path is None else _load_backend_class(path)
    return _NOISE_BACKENDS.get(backend_name)


def _resolve_entry_point_backend(kind: BackendKind, backend_name: str) -> type[Any] | None:
    group = _ENTRY_POINT_GROUPS[kind]
    entry_points = metadata.entry_points(group=group)
    matches = [entry_point for entry_point in entry_points if entry_point.name == backend_name]
    if not matches:
        return None
    if len(matches) > 1:
        import warnings  # noqa: PLC0415

        warnings.warn(
            f"Multiple {kind} entry points named '{backend_name}' found; "
            f"using '{matches[-1].value}'. Others: {[ep.value for ep in matches[:-1]]}",
            UserWarning,
            stacklevel=4,
        )
    return _load_entry_point_backend(kind, backend_name, matches[-1])


def _load_entry_point_backend(kind: BackendKind, backend_name: str, entry_point: metadata.EntryPoint) -> type[Any]:
    try:
        loaded = entry_point.load()
    except Exception as exc:  # pragma: no cover - exercised by importlib internals
        raise ImportError(
            f"Failed to load {kind} backend entry point '{backend_name}' from '{entry_point.value}': {exc}"
        ) from exc
    return _ensure_backend_class(loaded, description=f"{kind} backend entry point '{backend_name}'")


def _load_backend_class(backend_path: str, *, separator: str = ":") -> type[Any]:
    try:
        module_name, class_name = backend_path.rsplit(separator, 1)
    except ValueError as exc:
        legacy_hint = " or 'module.Class'" if separator == ":" else ""
        raise ValueError(
            f"Backend reference '{backend_path}' must use the format 'module{separator}Class'{legacy_hint}."
        ) from exc

    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"Failed to import module '{module_name}' while resolving '{backend_path}'.") from exc

    try:
        loaded = getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' does not define '{class_name}' for '{backend_path}'.") from exc

    return _ensure_backend_class(loaded, description=f"backend '{backend_path}'")


def _ensure_backend_class(loaded: Any, *, description: str) -> type[Any]:
    if not inspect.isclass(loaded):
        raise TypeError(f"Resolved {description} to a {type(loaded).__name__}; expected a class.")
    return loaded


def _warn_legacy_dotted_path(backend_name: str) -> None:
    if backend_name in _LEGACY_PATH_WARNINGS:
        return
    _LEGACY_PATH_WARNINGS.add(backend_name)
    module_name, class_name = backend_name.rsplit(".", 1)
    warnings.warn(
        f"Legacy dotted backend path '{backend_name}' is deprecated; use '{module_name}:{class_name}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _builtin_aliases(kind: BackendKind) -> tuple[str, ...]:
    if kind == "population":
        return tuple(_POPULATION_BACKENDS)
    if kind == "signal":
        return ("bbh", "bns", "nsbh", "sgwb")
    if kind == "waveform":
        return tuple(_WAVEFORM_BACKEND_PATHS)
    return tuple(_NOISE_BACKENDS)


def _validate_population_backend(backend_name: str, backend_instance: Any) -> None:
    if isinstance(backend_instance, GWPopSimulator):
        return

    raise TypeError(f"Resolved population backend '{backend_name}' does not satisfy GWPopSimulator.")


def _validate_signal_backend(backend_name: str, backend_class: type[Any], backend_instance: Any) -> None:
    if issubclass(backend_class, GWSimulator):
        return

    issues = _collect_protocol_issues(
        backend_instance,
        required_attributes={"required_params": lambda value: isinstance(value, frozenset)},
        required_callables=("simulate",),
    )
    if not issues:
        return
    raise TypeError(f"Resolved signal backend '{backend_name}' does not satisfy GWSimulator: {', '.join(issues)}.")


def _validate_waveform_backend(backend_name: str, backend_instance: Any) -> None:
    """Require a resolved waveform backend to satisfy gwmock-signal's ``WaveformBackend``.

    ``WaveformFactory`` calls ``available_approximants`` on whatever it is handed, so a wrong
    type surfaces as ``AttributeError: 'str' object has no attribute 'available_approximants'``
    -- which names neither the setting at fault nor what it should have been. Checking here
    turns that into a statement about the configuration.
    """
    from gwmock_signal.waveform.backends.base import WaveformBackend  # noqa: PLC0415

    if isinstance(backend_instance, WaveformBackend):
        return

    # Matching by public surface, as the population, signal and noise validators do. A
    # third-party backend reached through an entry point should not be forced to subclass
    # gwmock-signal's ABC in order to be usable, and ``WaveformFactory`` itself only ever calls
    # these two methods.
    issues = _collect_protocol_issues(
        backend_instance,
        required_attributes={},
        required_callables=("available_approximants", "generate_td_waveform"),
    )
    if not issues:
        return
    raise TypeError(
        f"Resolved waveform backend '{backend_name}' does not satisfy WaveformBackend: {', '.join(issues)}."
    )


def _validate_noise_backend(backend_name: str, backend_instance: Any) -> None:
    if isinstance(backend_instance, (BaseNoiseSimulator, NoiseSimulator)):
        return

    issues = _collect_protocol_issues(
        backend_instance,
        required_attributes={
            "duration": lambda value: isinstance(value, (int, float)),
            "sampling_frequency": lambda value: isinstance(value, (int, float)),
            "detectors": lambda value: isinstance(value, list),
            "seed": lambda value: value is None or isinstance(value, int),
            "metadata": lambda value: isinstance(value, Mapping),
        },
        required_callables=("generate", "generate_stream"),
    )
    if not issues:
        return
    raise TypeError(f"Resolved noise backend '{backend_name}' does not satisfy NoiseSimulator: {', '.join(issues)}.")


def _collect_protocol_issues(
    backend_instance: Any,
    *,
    required_attributes: Mapping[str, Any],
    required_callables: Sequence[str],
) -> list[str]:
    issues: list[str] = []

    for attribute_name, validator in required_attributes.items():
        if not hasattr(backend_instance, attribute_name):
            issues.append(f"missing attribute '{attribute_name}'")
            continue
        value = getattr(backend_instance, attribute_name)
        if not validator(value):
            issues.append(f"attribute '{attribute_name}' is mismatched")

    for method_name in required_callables:
        if not hasattr(backend_instance, method_name):
            issues.append(f"missing method '{method_name}'")
            continue
        if not callable(getattr(backend_instance, method_name)):
            issues.append(f"member '{method_name}' is not callable")

    return issues
