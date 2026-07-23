"""Look up which frame file(s) contain a given simulated signal.

Signals are recorded per batch in the metadata files (``signal.injections``,
the source of truth) and mirrored into ``signal_index.yaml`` for O(1) lookup by
``event_id``. Parameter-based lookup scans the injections in the metadata files.
"""

from __future__ import annotations

import json
import math
import operator
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

# Relative tolerance for numeric == / != filters, so representation noise (e.g.
# a float round-tripped through JSON) does not cause a scientifically-equal
# value to be missed. Ordering operators stay exact.
_EQUALITY_REL_TOL = 1.0e-9

# Two-character operators must be tried before one-character ones.
_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "!=": operator.ne,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,
}
_FILTER_RE = re.compile(r"^\s*(?P<key>[^<>=!\s]+)\s*(?P<op>>=|<=|!=|==|>|<|=)\s*(?P<value>.+?)\s*$")


def parse_param_filter(spec: str) -> tuple[str, str, Any]:
    """Parse a ``key OP value`` filter such as ``mass_1>=30`` or ``approximant==IMRPhenomXPHM``."""
    match = _FILTER_RE.match(spec)
    if not match:
        raise ValueError(f"Invalid parameter filter '{spec}'; expected key OP value (e.g. mass_1>=30).")
    return match.group("key"), match.group("op"), _coerce(match.group("value"))


def _coerce(raw: str) -> Any:
    """Coerce a filter value to float when possible, otherwise keep it as a string."""
    try:
        return float(raw)
    except ValueError:
        return raw


def _matches(parameters: dict[str, Any], filters: list[tuple[str, str, Any]]) -> bool:
    """Return whether an event's parameters satisfy every filter predicate."""
    for key, op, value in filters:
        if key not in parameters:
            return False
        try:
            if not _compare(parameters[key], op, value):
                return False
        except TypeError:
            # Mismatched types (e.g. numeric op on a string parameter) never match.
            return False
    return True


def _compare(actual: Any, op: str, value: Any) -> bool:
    """Apply one filter operator, using a tolerance for numeric equality tests."""
    if op in ("==", "=", "!=") and isinstance(actual, (int, float)) and isinstance(value, (int, float)):
        close = math.isclose(actual, value, rel_tol=_EQUALITY_REL_TOL, abs_tol=0.0)
        return not close if op == "!=" else close
    return _OPERATORS[op](actual, value)


def _iter_metadata_files(metadata_directory: Path) -> Iterator[Path]:
    yield from sorted(metadata_directory.glob("*.metadata.json"))


def find_signals(
    metadata_directory: Path | str,
    *,
    event_id: int | None = None,
    param_filters: list[tuple[str, str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return the signals matching an id and/or parameter filters with their frame file(s).

    With only ``event_id`` set, the fast ``signal_index.yaml`` path is used. When
    parameter filters are supplied, the batch metadata files are scanned (their
    ``signal.injections`` are the source of truth). Each result is a mapping with
    ``event_id``, ``frames`` (signal frame paths), ``metadata`` (batch metadata
    file name), and ``coa_time``; parameter-filtered results also carry
    ``parameters``.
    """
    metadata_directory = Path(metadata_directory)
    param_filters = param_filters or []
    results: list[dict[str, Any]] = []

    if event_id is not None and not param_filters:
        index_file = metadata_directory / "signal_index.yaml"
        if not index_file.exists():
            return results
        with index_file.open(encoding="utf-8") as f:
            index = yaml.safe_load(f) or {}
        entry = index.get(str(event_id))
        if entry:
            results.append(
                {
                    "event_id": event_id,
                    "frames": entry.get("frames", []),
                    "metadata": entry.get("metadata"),
                    "coa_time": entry.get("coa_time"),
                }
            )
        return results

    for meta_path in _iter_metadata_files(metadata_directory):
        try:
            with meta_path.open(encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        injections = (metadata.get("signal") or {}).get("injections") or []
        if not injections:
            continue
        frames = [
            output["path"]
            for output in metadata.get("outputs", [])
            if output.get("kind") == "signal" and "path" in output
        ]
        for injection in injections:
            if event_id is not None and injection.get("event_id") != event_id:
                continue
            parameters = injection.get("parameters") or {}
            if not _matches(parameters, param_filters):
                continue
            results.append(
                {
                    "event_id": injection.get("event_id"),
                    "frames": frames,
                    "metadata": meta_path.name,
                    "parameters": parameters,
                    "coa_time": parameters.get("coa_time"),
                }
            )
    return results
