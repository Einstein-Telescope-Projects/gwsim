"""
Utilities for executing simulation plans via CLI.
"""

from __future__ import annotations

import atexit
import copy
import json
import logging
import platform
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from gwmock_noise import SimulationResult
from tqdm import tqdm

from gwmock.cli.adapter_orchestration import AdapterOrchestrationResult, AdapterOrchestrator
from gwmock.cli.utils.checkpoint import CheckpointManager, require_matching_config, run_fingerprint, spillover_applies
from gwmock.cli.utils.config import OrchestrationConfig, SimulatorConfig, resolve_class_path
from gwmock.cli.utils.environment import capture_environment
from gwmock.cli.utils.hash import compute_content_hash, compute_file_hash
from gwmock.cli.utils.metadata import save_metadata_record
from gwmock.cli.utils.simulation_plan import (
    SimulationBatch,
    SimulationPlan,
    create_batch_metadata,
)
from gwmock.cli.utils.template import expand_template_variables
from gwmock.cli.utils.utils import handle_signal
from gwmock.simulator.base import Simulator

logger = logging.getLogger("gwmock")

# A full git commit SHA (SHA-1): the only revision form that immutably pins a
# downloaded dataset. Branches, tags, and None can all move upstream.
_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
logger.setLevel(logging.DEBUG)


def _backend_path_from_object(obj: Any) -> str:
    """Return a stable ``module:qualname`` identifier for an object or class."""
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _flatten_to_strings(value: Any) -> list[str]:
    """Flatten template-expanded values into a simple ordered list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.flatten().tolist()]
    if isinstance(value, (list, tuple)):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_to_strings(item))
        return flattened
    return [str(value)]


def _to_path_string(path: Path, working_directory: str | None) -> str:
    """Prefer working-directory-relative paths for portable metadata."""
    if working_directory:
        base = Path(working_directory)
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)
    return str(path)


def _to_plain_number(value: Any) -> float | int | None:
    """Convert quantities and numpy scalars to native numbers."""
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if float(value).is_integer() else float(value)
    return value


def _get_host_metadata() -> dict[str, Any]:
    """Collect stable host metadata for provenance reporting."""
    git_sha = _get_distribution_git_sha() or _get_source_tree_git_sha()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine() or "unknown",
        "git_sha": git_sha,
    }


def _get_distribution_git_sha() -> str | None:
    """Return the gwmock distribution VCS commit hash when explicitly available."""
    try:
        distribution = importlib_metadata.distribution("gwmock")
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None

    for source in (distribution.read_text("direct_url.json"), distribution.metadata.get("Direct-URL")):
        git_sha = _extract_git_sha_from_direct_url(source)
        if git_sha is not None:
            return git_sha
    return None


def _get_source_tree_git_sha() -> str | None:
    """Return the working-tree git commit for a source checkout, else ``None``.

    Complements :func:`_get_distribution_git_sha`: editable/source installs carry no
    PEP 610 VCS metadata, so read the commit directly from the repository that
    contains this module. A ``-dirty`` suffix marks an uncommitted working tree, so a
    downstream lineage system can tell the output was not built from a clean commit.
    Returns ``None`` when git is unavailable or the source is not a repository (e.g. a
    released wheel unpacked into site-packages).
    """
    git_exe = shutil.which("git")
    if git_exe is None:
        return None
    repo_dir = str(Path(__file__).resolve().parent)
    try:
        head = subprocess.run(  # noqa: S603
            [git_exe, "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode != 0 or not head.stdout.strip():
            return None
        sha = head.stdout.strip()
        status = subprocess.run(  # noqa: S603
            [git_exe, "-C", repo_dir, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status.returncode == 0 and status.stdout.strip():
            sha = f"{sha}-dirty"
        return sha
    except (OSError, subprocess.SubprocessError):
        return None


def _extract_git_sha_from_direct_url(direct_url: str | None) -> str | None:
    """Parse a commit hash from PEP 610 Direct URL metadata content."""
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
    except (TypeError, json.JSONDecodeError):
        return None
    vcs_info = payload.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit_id = vcs_info.get("commit_id")
    if isinstance(commit_id, str):
        commit_id = commit_id.strip()
        return commit_id or None
    return None


def _build_config_payload(batch: SimulationBatch, simulator: Simulator) -> dict[str, Any]:
    """Build the resolved config snapshot stored in metadata."""
    base_payload = (
        copy.deepcopy(batch.config_payload)
        if batch.config_payload is not None
        else {
            "globals": batch.globals_config.model_dump(by_alias=True, exclude_none=True),
        }
    )

    if isinstance(batch.simulator_config, OrchestrationConfig):
        base_payload["orchestration"] = batch.simulator_config.model_dump(by_alias=True, exclude_none=True)
    else:
        simulators = base_payload.setdefault("simulators", {})
        simulators[batch.simulator_name] = batch.simulator_config.model_dump(by_alias=True, exclude_none=True)

    return cast(dict[str, Any], expand_template_variables(base_payload, simulator))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge ``override`` into ``base`` in place.

    Nested mappings merge key-by-key; every other value (including lists)
    replaces wholesale, so a resolved ``glitches`` list supersedes the input
    one rather than being concatenated with it.
    """
    for key, value in override.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(existing, value)
        else:
            base[key] = value


def _unresolved_external_inputs(fragment: dict[str, Any]) -> list[str]:
    """Return labels for resolved entries not pinned to an immutable version.

    A glitch entry that carries a ``revision`` key names a dataset-backed model.
    It is only reproducible when that revision is a full commit SHA: ``None``
    means resolution failed (e.g. an offline Hub with no cache), and a symbolic
    ref such as a branch or tag (``"main"``) still moves upstream. Both are
    reported as unresolved so the run is marked non-replayable.
    """
    unresolved: list[str] = []
    noise_arguments = fragment.get("noise", {}).get("arguments", {})
    for entry in noise_arguments.get("glitches", []) or []:
        if isinstance(entry, dict) and "revision" in entry and not _is_pinned_revision(entry["revision"]):
            unresolved.append(f"glitch:{entry.get('kind', 'unknown')}")
    return unresolved


def _is_pinned_revision(revision: Any) -> bool:
    """Return whether a dataset revision is an immutable full commit SHA.

    A 40-character hex string is a git commit SHA and cannot move; anything else
    — ``None`` or a symbolic ref like a branch or tag — can point at different
    content later, so it does not pin the run.
    """
    return isinstance(revision, str) and _COMMIT_SHA_RE.fullmatch(revision) is not None


def _build_resolved_config(
    simulator: Simulator,
    input_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    """Build the fully-resolved, replayable config for this batch.

    Overlays each adapter's runtime-resolved values (e.g. a pinned dataset
    revision) onto the template-expanded input config. Returns
    ``(resolved_payload, replayable)`` — ``resolved_payload`` is ``None`` when
    nothing needed resolving (a purely parametric run), and ``replayable`` is
    ``False`` when a declared external-mutable input could not be pinned.
    """
    resolved_config_fn = getattr(simulator, "resolved_config", None)
    if not callable(resolved_config_fn):
        return None, True
    fragment = cast(dict[str, Any], resolved_config_fn())
    if not fragment:
        return None, True

    orchestration = input_payload.get("orchestration")
    if not isinstance(orchestration, dict):
        return None, True

    unresolved = _unresolved_external_inputs(fragment)
    if unresolved:
        logger.warning(
            "Could not pin external-mutable input(s) %s to an immutable version; "
            "this run's metadata is marked non-replayable and is not bit-reproducible.",
            ", ".join(unresolved),
        )

    resolved_payload = copy.deepcopy(input_payload)
    _deep_merge(resolved_payload["orchestration"], fragment)
    return resolved_payload, not unresolved


def _resolve_seed(simulator: Simulator, batch: SimulationBatch) -> int | None:
    """Resolve the top-level seed recorded for this batch."""
    if isinstance(simulator, AdapterOrchestrator):
        seed = simulator.noise_arguments.get("seed")
        return int(seed) if seed is not None else None

    seed = getattr(simulator, "seed", None)
    if seed is not None:
        return int(seed)

    global_seed = batch.globals_config.simulator_arguments.get("seed")
    if global_seed is not None:
        return int(global_seed)

    local_seed = getattr(batch.simulator_config, "arguments", {}).get("seed")
    return int(local_seed) if local_seed is not None else None


def _resolve_segment_seeds(simulator: Simulator, batch: SimulationBatch, seed: int | None) -> list[int]:
    """Resolve per-segment seeds for this batch."""
    if seed is None:
        return []
    if isinstance(simulator, AdapterOrchestrator):
        return simulator.segment_seeds()
    return [seed + batch.batch_index]


def _build_population_section(simulator: Simulator, batch: SimulationBatch) -> dict[str, Any] | None:
    """Build the population section for the metadata schema."""
    simulator_metadata = simulator.metadata
    if isinstance(simulator, AdapterOrchestrator):
        if batch.simulator_config.population is None:
            return None
        return {
            "backend": batch.simulator_config.population.backend,
            "source_type": simulator_metadata["orchestration"]["source_type"],
            "n_events": len(simulator._population_events),
            "parameter_names": list(simulator._population_events[0].keys()) if simulator._population_events else [],
            "metadata": simulator_metadata["orchestration"]["population"]["metadata"],
        }

    signal_metadata = simulator_metadata.get("signal", {}).get("arguments", {})
    source_type = signal_metadata.get("source_type")
    return {
        "backend": resolve_class_path(batch.simulator_config.class_, batch.simulator_name),
        "source_type": source_type,
        "n_events": None,
        "parameter_names": [],
        "metadata": {},
    }


def _build_signal_section(simulator: Simulator, batch: SimulationBatch) -> dict[str, Any] | None:
    """Build the signal section for the metadata schema."""
    simulator_metadata = simulator.metadata
    if isinstance(simulator, AdapterOrchestrator):
        if batch.simulator_config.signal is None or simulator.signal_adapter is None:
            return None
        return {
            "backend": _backend_path_from_object(simulator.signal_adapter._backend),
            "waveform_model": simulator.waveform_model,
            "detector_network": list(simulator.detectors),
            # Source parameters of the signals that merge in this batch's frame(s),
            # in injection order (empty for stationary/SGWB segments). This makes each
            # frame self-describing and backs the signal->frame lookup.
            "injections": list(simulator_metadata["orchestration"]["signal"].get("injections", [])),
            "metadata": simulator_metadata["orchestration"]["signal"],
        }

    signal_metadata = simulator_metadata.get("signal", {}).get("arguments", {})
    detectors = signal_metadata.get("detectors", getattr(simulator, "detectors", []))
    return {
        "backend": resolve_class_path(batch.simulator_config.class_, batch.simulator_name),
        "waveform_model": signal_metadata.get("waveform_model"),
        "detector_network": [str(detector) for detector in detectors],
        "metadata": simulator_metadata,
    }


def _build_noise_section(simulator: Simulator, batch: SimulationBatch) -> dict[str, Any] | None:
    """Build the noise section for the metadata schema."""
    simulator_metadata = simulator.metadata
    if isinstance(simulator, AdapterOrchestrator):
        if batch.simulator_config.noise is None or simulator.noise_adapter is None:
            return None
        psd_value = simulator.noise_arguments.get("psd_file")
        if psd_value is None and simulator.noise_arguments.get("psd_files"):
            psd_value = "multiple"
        return {
            "backend": _backend_path_from_object(simulator.noise_adapter.backend),
            "psd": None if psd_value is None else str(psd_value),
            "metadata": simulator_metadata["orchestration"]["noise"],
        }

    noise_metadata = simulator_metadata.get("colored_noise", {}).get("arguments", {})
    return {
        "backend": resolve_class_path(batch.simulator_config.class_, batch.simulator_name),
        "psd": noise_metadata.get("psd_file"),
        "metadata": simulator_metadata,
    }


def _build_output_records(
    simulator: Simulator,
    batch: SimulationBatch,
    batch_data: object,
    output_files: list[Path],
) -> list[dict[str, Any]]:
    """Build output descriptors for the versioned metadata schema."""
    working_directory = batch.globals_config.working_directory
    output_records: list[dict[str, Any]] = []

    if isinstance(batch_data, AdapterOrchestrationResult):
        if batch.simulator_config.signal is not None and batch_data.signal_segment is not None:
            signal_files = _resolve_output_paths(
                file_name_template=batch.simulator_config.signal.output.file_name,
                simulator=simulator,
                output_directory=cast(AdapterOrchestrator, simulator).signal_output_directory(),
            )
            signal_channels = _flatten_to_strings(
                expand_template_variables(batch.simulator_config.signal.output.arguments.get("channel"), simulator)
            )
            for index, output_file in enumerate(signal_files):
                output_records.append(
                    {
                        "kind": "signal",
                        "path": _to_path_string(output_file, working_directory),
                        "channels": signal_channels[index : index + 1] if signal_channels else [],
                        "t0": _to_plain_number(batch_data.signal_segment.start_time),
                        "duration": _to_plain_number(batch_data.signal_segment.duration),
                        "sha256": compute_file_hash(output_file),
                        "content_sha256": compute_content_hash(output_file),
                    }
                )

        if batch_data.noise_result is not None:
            noise_output_config = batch_data.noise_result.config.output
            for detector, output_path in batch_data.noise_result.output_paths.items():
                if noise_output_config.channels and detector in noise_output_config.channels:
                    channel_id = noise_output_config.channels[detector]
                else:
                    channel_id = f"{detector}:{noise_output_config.channel}"
                output_records.append(
                    {
                        "kind": "noise",
                        "path": _to_path_string(output_path, working_directory),
                        "channels": [channel_id],
                        "t0": _to_plain_number(simulator.start_time),
                        "duration": _to_plain_number(simulator.duration),
                        "sha256": compute_file_hash(output_path),
                        "content_sha256": compute_content_hash(output_path),
                    }
                )
        return output_records

    if isinstance(batch_data, SimulationResult):
        channel_prefix = str(getattr(simulator, "_active_channel_prefix", "MOCK"))
        for detector, output_path in batch_data.output_paths.items():
            output_records.append(
                {
                    "kind": batch.simulator_name,
                    "path": _to_path_string(output_path, working_directory),
                    "channels": [f"{detector}:{channel_prefix}"],
                    "t0": _to_plain_number(getattr(simulator, "start_time", None)),
                    "duration": _to_plain_number(getattr(simulator, "duration", None)),
                    "sha256": compute_file_hash(output_path),
                    "content_sha256": compute_content_hash(output_path),
                }
            )
        return output_records

    expanded_arguments = expand_template_variables(batch.simulator_config.output.arguments or {}, simulator)
    channels = _flatten_to_strings(expanded_arguments.get("channel"))
    for index, output_file in enumerate(output_files):
        output_records.append(
            {
                "kind": batch.simulator_name,
                "path": _to_path_string(output_file, working_directory),
                "channels": channels[index : index + 1] if channels else [],
                "t0": _to_plain_number(getattr(batch_data, "start_time", getattr(simulator, "start_time", None))),
                "duration": _to_plain_number(getattr(batch_data, "duration", getattr(simulator, "duration", None))),
                "sha256": compute_file_hash(output_file),
                "content_sha256": compute_content_hash(output_file),
            }
        )
    return output_records


def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    state_restore_func: Any = None,
) -> Any:
    """Retry a function with exponential backoff and optional state restoration.

    Args:
        func: Callable to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        state_restore_func: Optional callable to restore state before each retry.
                           Called before each retry attempt (not before first attempt).

    Returns:
        Result of function call

    Raises:
        Exception: If all retries fail
    """
    delay = initial_delay
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:  # pylint: disable=broad-exception-caught
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.2fs...",
                    attempt + 1,
                    max_retries + 1,
                    str(e),
                    delay,
                    exc_info=e,
                )
                time.sleep(delay)
                delay *= backoff_factor

                # Restore state before retry if function provided
                if state_restore_func is not None:
                    try:
                        state_restore_func()
                        logger.debug("State restored before retry attempt %d", attempt + 2)
                    except Exception as restore_error:
                        logger.error("Failed to restore state before retry: %s", restore_error)
                        raise RuntimeError(f"Cannot retry: failed to restore state: {restore_error}") from restore_error
            else:
                logger.error("All %d attempts failed for batch: %s", max_retries + 1, str(e))

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected retry failure")


def update_metadata_index(
    metadata_directory: Path,
    output_files: list[Path],
    metadata_file_name: str,
    encoding: str = "utf-8",
) -> None:
    """Update the central metadata index file.

    The index maps data file names to their corresponding metadata files,
    enabling O(1) lookup to find metadata for a given data file.

    Args:
        metadata_directory: Directory where metadata files are stored
        output_files: List of output data file Paths
        metadata_file_name: Name of the metadata file (e.g., "signal-0.metadata.yaml")
        encoding: File encoding for reading/writing the index file
    """
    index_file = metadata_directory / "index.yaml"

    # Load existing index or create new
    if index_file.exists():
        try:
            with index_file.open(encoding=encoding) as f:
                index = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Failed to load metadata index: %s. Creating new index.", e)
            index = {}
    else:
        index = {}

    # Add entries for all output files
    for output_file in output_files:
        index[output_file.name] = metadata_file_name
        logger.debug("Index entry: %s -> %s", output_file.name, metadata_file_name)

    # Save updated index
    try:
        with index_file.open("w") as f:
            yaml.safe_dump(index, f, default_flow_style=False, sort_keys=True)
        logger.debug("Updated metadata index: %s", index_file)
    except (OSError, yaml.YAMLError) as e:
        logger.error("Failed to save metadata index: %s", e)
        raise


def _withdraw_batch(index: dict[str, Any], metadata_file_name: str) -> dict[str, Any]:
    """Return *index* with every contribution from *metadata_file_name* removed.

    Entries left with no contributions are dropped, so a re-run that injects nothing no longer
    leaves an id pointing at frames it did not write.

    Pre-1.5.0 entries carried a single ``metadata`` string and a flat ``frames`` list. They are
    migrated in passing rather than rejected: an index is a rebuildable cache, and refusing to read
    one written by an older gwmock would make an upgrade look like data loss.
    """
    migrated: dict[str, Any] = {}
    for event_id, entry in index.items():
        batches = entry.get("batches")
        if batches is None:
            batches = [{"metadata": entry.get("metadata"), "frames": entry.get("frames") or []}]
        kept = [batch for batch in batches if batch.get("metadata") != metadata_file_name]
        if not kept:
            continue
        migrated[event_id] = {"batches": kept, "coa_time": entry.get("coa_time")}
    return migrated


def update_signal_index(
    metadata_directory: Path,
    metadata: dict[str, Any],
    metadata_file_name: str,
    encoding: str = "utf-8",
) -> None:
    """Update the signal index mapping each injected event to its frame file(s).

    The index (``signal_index.yaml``) maps a signal's ``event_id`` to the signal
    frame file(s) that contain it plus the batch metadata file, enabling O(1)
    signal->frame lookup by id.

    **Not safe against concurrent writers.** This is an unlocked read-modify-write, so two
    processes updating the index at once can lose one side's contribution -- reproduced
    deterministically with a barrier. It predates the per-batch accumulation below and is not
    made worse by it, but the accumulation does change what is lost: a dropped update used to
    cost one assignment, and now costs one batch's frames for every event in it. gwmock writes
    batches sequentially within a run, so this bites only when two runs share a metadata
    directory. Tracked as ``gwmock/signal-index-concurrent-writers``. Parameter-based lookup reads the injections
    recorded in the batch metadata files (their source of truth); this index is
    only the id shortcut. A batch with no injected signals writes nothing.

    Args:
        metadata_directory: Directory where metadata and the index live.
        metadata: The batch metadata record just written.
        metadata_file_name: File name of that batch metadata record.
        encoding: File encoding for reading/writing the index file.
    """
    injections = (metadata.get("signal") or {}).get("injections") or []
    index_file = metadata_directory / "signal_index.yaml"
    if not injections and not index_file.exists():
        return

    if index_file.exists():
        try:
            with index_file.open(encoding=encoding) as f:
                index = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Failed to load signal index: %s. Creating new index.", e)
            index = {}
    else:
        index = {}

    # Withdraw this batch's previous contribution, per event, so a re-run or overwrite (which may
    # now inject different or no events) cannot leave stale id -> frame rows the fast path would
    # trust. This used to drop whole entries whose `metadata` matched, which was equivalent only
    # while an entry belonged to exactly one batch -- and that is the assumption being removed here.
    index = _withdraw_batch(index, metadata_file_name)

    signal_frames = [
        output["path"] for output in metadata.get("outputs", []) if output.get("kind") == "signal" and "path" in output
    ]
    for injection in injections:
        event_id = injection.get("event_id")
        if event_id is None:
            continue
        # Appended, not assigned. A signal reaches every frame its samples land in, and each of those
        # frames belongs to a different batch writing this index in turn, so the previous
        # `index[event_id] = ...` kept whichever batch happened to write last. For a continuous wave
        # that is one frame out of every frame in the run; for a 48 s inspiral across 32 s segments it
        # was one of three, and not the one holding the merger.
        entry = index.setdefault(
            str(event_id),
            {"batches": [], "coa_time": (injection.get("parameters") or {}).get("coa_time")},
        )
        entry["batches"].append({"metadata": metadata_file_name, "frames": signal_frames})

    try:
        with index_file.open("w") as f:
            yaml.safe_dump(index, f, default_flow_style=False, sort_keys=True)
    except (OSError, yaml.YAMLError) as e:
        logger.error("Failed to save signal index: %s", e)
        raise


def instantiate_simulator(
    simulator_config: SimulatorConfig | OrchestrationConfig,
    simulator_name: str | None = None,
    global_simulator_arguments: dict[str, Any] | None = None,
) -> Simulator:
    """Instantiate a simulator from configuration.

    Creates a single simulator instance that will be reused across multiple batches.
    The simulator maintains state (RNG, counters, etc.) across iterations.

    Global simulator arguments are merged with simulator-specific arguments,
    with simulator-specific arguments taking precedence.

    Args:
        simulator_config: Configuration for this simulator
        simulator_name: Name of the simulator (used for class path resolution)
        global_simulator_arguments: Global fallback arguments for the simulator

    Returns:
        Instantiated Simulator

    Raises:
        ImportError: If simulator class cannot be imported
        TypeError: If simulator instantiation fails
    """
    if isinstance(simulator_config, OrchestrationConfig):
        simulator = AdapterOrchestrator.from_config(
            orchestration_config=simulator_config,
            global_simulator_arguments=global_simulator_arguments,
        )
        logger.info("Instantiated adapter-backed orchestration path")
        return simulator

    class_spec = simulator_config.class_

    # Resolve short class names to full paths
    class_spec = resolve_class_path(class_spec, simulator_name)

    module_name, class_name = class_spec.rsplit(".", 1)
    simulator_module = import_module(module_name)
    simulator_cls = getattr(simulator_module, class_name)

    # Merge global and simulator-specific arguments
    # Simulator-specific arguments override global defaults
    if global_simulator_arguments:
        merged_arguments = {**global_simulator_arguments, **simulator_config.arguments}
    else:
        merged_arguments = simulator_config.arguments

    # Normalize keys: convert hyphens to underscores (YAML uses hyphens, Python uses underscores)
    normalized_arguments = {k.replace("-", "_"): v for k, v in merged_arguments.items()}

    simulator = simulator_cls(**normalized_arguments)

    logger.info("Instantiated simulator from class %s", class_spec)
    return simulator


def restore_batch_state(
    simulator: Simulator,
    batch: SimulationBatch,
    last_simulator_state: dict[str, Any] | None = None,
    last_simulator_spillover: Any = None,
) -> None:
    """Restore simulator state from batch metadata or checkpoint file if available.

    This is used when reproducing a specific batch. It restores the RNG state,
    filter memory, and other stateful components that existed before this batch
    was generated.

    Args:
        simulator: Simulator instance
        batch: SimulationBatch potentially containing state snapshot
        last_simulator_state (optional): State dict of the last simulator from the checkpoint file, or None if unavailable

    Raises:
        ValueError: If state restoration fails
    """
    if batch.has_state_snapshot() and batch.pre_batch_state is not None:
        logger.debug(
            "[RESTORE] Batch %d: Restoring state from snapshot - state_keys=%s",
            batch.batch_index,
            list(batch.pre_batch_state.keys()),
        )
        try:
            logger.debug(
                "[RESTORE] Batch %d: Setting state dict - counter=%s",
                batch.batch_index,
                batch.pre_batch_state.get("counter"),
            )
            simulator.state = batch.pre_batch_state
            logger.debug(
                "[RESTORE] Batch %d: State restored successfully - new_counter=%s",
                batch.batch_index,
                simulator.counter,
            )
        except Exception as e:
            logger.error("Failed to restore batch state: %s", e)
            raise ValueError(f"Failed to restore state for batch {batch.batch_index}") from e
    elif last_simulator_state is not None and batch.batch_index == last_simulator_state.get("counter"):
        logger.debug(
            "[RESTORE] Batch %d: Restoring state from checkpoint last state - state_keys=%s",
            batch.batch_index,
            list(last_simulator_state.keys()),
        )
        try:
            logger.debug(
                "[RESTORE] Batch %d: Setting state dict - counter=%s",
                batch.batch_index,
                last_simulator_state.get("counter"),
            )
            simulator.state = last_simulator_state
            # Restored only on this branch -- the one that resumes from the checkpoint's *last*
            # state. The branch above restores from a batch metadata record, which by design does
            # not carry samples, so there is no spillover to restore there and a run resumed that
            # way still loses the tail. Stated rather than silently half-handled.
            if last_simulator_spillover is not None:
                simulator.cached_data_chunks = last_simulator_spillover
            logger.debug(
                "[RESTORE] Batch %d: State restored successfully - new_counter=%s",
                batch.batch_index,
                simulator.counter,
            )
        except Exception as e:
            logger.error("Failed to restore batch state: %s", e)
            raise ValueError(f"Failed to restore state for batch {batch.batch_index}") from e
    else:
        logger.debug(
            "[RESTORE] Batch %d: No pre-batch state snapshot available (fresh generation)",
            batch.batch_index,
        )


def save_batch_metadata(
    simulator: Simulator,
    batch: SimulationBatch,
    metadata_directory: Path,
    batch_data: object,
    output_files: list[Path],
    pre_batch_state: dict[str, Any] | None = None,
) -> None:
    """Save batch metadata including pre-batch simulator state and all output files.

    The metadata file uses batch-indexed naming ({simulator_name}-{batch_index}.metadata.yaml)
    to provide a single source of truth for all outputs from that batch. This handles
    cases where a single batch generates multiple output files (e.g., one per detector).

    An index file is also maintained to enable quick lookup of metadata for a given data file.

    Args:
        simulator: Simulator instance
        batch: SimulationBatch
        metadata_directory: Directory to save metadata
        batch_data: Generated batch artifact used to derive output provenance
        output_files: List of Path objects for all output files generated by this batch
        pre_batch_state: State of simulator before batch generation (for reproducibility).
                        If None, uses current simulator state.
    """
    metadata_directory.mkdir(parents=True, exist_ok=True)

    # Use provided pre_batch_state or current simulator state
    state_to_save = pre_batch_state if pre_batch_state is not None else simulator.state

    seed = _resolve_seed(simulator, batch)
    config_payload = _build_config_payload(batch, simulator)
    resolved_config, replayable = _build_resolved_config(simulator, config_payload)
    metadata = create_batch_metadata(
        simulator_name=batch.simulator_name,
        batch_index=batch.batch_index,
        simulator_config=batch.simulator_config,
        globals_config=batch.globals_config,
        simulator_metadata=simulator.metadata,
        pre_batch_state=state_to_save,
        source=batch.source,
        author=batch.author,
        email=batch.email,
        config_payload=config_payload,
        resolved_config=resolved_config,
        replayable=replayable,
        config_sha256=batch.config_sha256,
        seed=seed,
        segment_seeds=_resolve_segment_seeds(simulator, batch, seed),
        population=_build_population_section(simulator, batch),
        signal=_build_signal_section(simulator, batch),
        noise=_build_noise_section(simulator, batch),
        outputs=_build_output_records(simulator, batch, batch_data, output_files),
        host=_get_host_metadata(),
        environment=capture_environment(),
    )

    # Add output files to metadata for easy discovery
    # Store just the file names, not full paths
    metadata["output_files"] = [f.name for f in output_files]

    # Compute and add file hashes for integrity checking. Two hashes are kept:
    #   * file_hashes    -- raw container bytes (exact-file integrity)
    #   * content_hashes -- decoded scientific content, stable across write-time
    #                       and frame-library version (reproducibility check)
    file_hashes = {}
    content_hashes = {}
    for output_file in output_files:
        try:
            file_hash = compute_file_hash(output_file)
            file_hashes[output_file.name] = file_hash
            logger.debug("Compute hash for %s: %s", output_file.name, file_hash)
        except OSError as e:
            logger.warning("Failed to compute hash for %s: %s", output_file.name, e)
            # Continue without failing - metadata is still useful
        content_hash = compute_content_hash(output_file)
        if content_hash is not None:
            content_hashes[output_file.name] = content_hash

    metadata["file_hashes"] = file_hashes
    metadata["content_hashes"] = content_hashes

    metadata_file_name = f"{batch.simulator_name}-{batch.batch_index}.metadata.json"
    metadata_file = metadata_directory / metadata_file_name
    logger.debug("Saving batch metadata to %s with %d output files", metadata_file, len(output_files))

    save_metadata_record(metadata=metadata, metadata_file=metadata_file)

    # Update the metadata index for quick lookup
    update_metadata_index(metadata_directory, output_files, metadata_file_name)

    # Update the signal index (event id -> containing frame file(s)) for signal->frame lookup
    update_signal_index(metadata_directory, metadata, metadata_file_name)


def process_batch(
    simulator: Simulator,
    batch_data: object,
    batch: SimulationBatch,
    output_directory: Path,
    overwrite: bool,
) -> list[Path]:
    """Process and save a single batch of generated data.

    A single batch may generate multiple output files (e.g., one per detector).
    This function handles both single and multiple output files.

    Args:
        simulator: Simulator instance
        batch_data: Generated batch data (may contain multiple outputs)
        batch: SimulationBatch metadata
        output_directory: Directory for output files
        overwrite: Whether to overwrite existing files

    Returns:
        List of Path objects for all generated output files
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    if isinstance(batch_data, AdapterOrchestrationResult):
        if not isinstance(batch.simulator_config, OrchestrationConfig):
            raise TypeError("Adapter orchestration results require an OrchestrationConfig batch.")

        signal_output_files: list[Path] = []
        if batch.simulator_config.signal is not None and batch_data.signal_segment is not None:
            signal_output = batch.simulator_config.signal.output
            logger.debug(
                "[PROCESS] Batch %s: Saving signal output - counter=%s, template=%s",
                batch.batch_index,
                simulator.counter,
                signal_output.file_name,
            )
            signal_output_files = _resolve_output_paths(
                file_name_template=signal_output.file_name,
                simulator=simulator,
                output_directory=cast(AdapterOrchestrator, simulator).signal_output_directory(),
            )
            simulator.save_data(
                data=batch_data.signal_segment,
                file_name=signal_output.file_name,
                output_directory=cast(AdapterOrchestrator, simulator).signal_output_directory(),
                overwrite=overwrite,
                **cast(AdapterOrchestrator, simulator).signal_output_arguments(),
            )

        noise_output_files: list[Path] = []
        if batch_data.noise_result is not None:
            noise_output_files = list(batch_data.noise_result.output_paths.values())
            missing_noise_outputs = [path for path in noise_output_files if not path.exists()]
            if missing_noise_outputs:
                raise FileNotFoundError(
                    "Noise adapter reported output files that do not exist: "
                    + ", ".join(str(path) for path in missing_noise_outputs)
                )

        logger.debug(
            "[PROCESS] Batch %s: adapter outputs - signal=%d files, noise=%d files",
            batch.batch_index,
            len(signal_output_files),
            len(noise_output_files),
        )
        return [*signal_output_files, *noise_output_files]

    if isinstance(batch_data, SimulationResult):
        output_files_list = list(batch_data.output_paths.values())
        missing_outputs = [path for path in output_files_list if not path.exists()]
        if missing_outputs:
            raise FileNotFoundError(
                "Noise adapter reported output files that do not exist: "
                + ", ".join(str(path) for path in missing_outputs)
            )
        logger.debug(
            "[PROCESS] Batch %s: Using upstream-written outputs - %s",
            batch.batch_index,
            [str(path.name) for path in output_files_list],
        )
        return output_files_list

    # Build output configuration
    output_config = batch.simulator_config.output
    logger.debug(
        "[PROCESS] Batch %s: Saving data - counter=%s, file_template=%s",
        batch.batch_index,
        simulator.counter,
        output_config.file_name,
    )
    file_name_template = output_config.file_name
    output_args = output_config.arguments.copy() if output_config.arguments else {}

    # Save data with output directory
    logger.debug(
        "Saving batch data for %s batch %d",
        batch.simulator_name,
        batch.batch_index,
    )

    # Resolve the output file names (may be multiple if template contains arrays)
    output_files = expand_template_variables(value=file_name_template, simulator_instance=simulator)

    # Normalize to list of Paths
    if isinstance(output_files, str):
        output_files_list = [output_directory / Path(output_files)]
    else:
        # If it's an array (multiple detectors), flatten it
        output_files_list = [output_directory / Path(str(f)) for f in np.array(output_files).flatten()]

    logger.debug(
        "[PROCESS] Batch %s: Resolved filenames - %s", batch.batch_index, [str(f.name) for f in output_files_list]
    )

    simulator.save_data(
        data=batch_data,
        file_name=file_name_template,
        output_directory=output_directory,
        overwrite=overwrite,
        **output_args,
    )

    logger.debug("[PROCESS] Batch %s: Data saved - counter=%s", batch.batch_index, simulator.counter)

    return output_files_list


def _resolve_output_paths(file_name_template: str, simulator: Simulator, output_directory: Path) -> list[Path]:
    """Resolve one or more concrete output paths for a template."""
    output_files = expand_template_variables(value=file_name_template, simulator_instance=simulator)
    if isinstance(output_files, str):
        return [output_directory / Path(output_files)]
    return [output_directory / Path(str(f)) for f in np.array(output_files).flatten()]


def setup_signal_handlers(checkpoint_dirs: list[Path]) -> None:
    """Set up signal handlers for graceful shutdown.

    Args:
        checkpoint_dirs: List of checkpoint directories to clean up
    """

    def cleanup_checkpoints():
        """Clean up temporary checkpoint files."""
        for checkpoint_dir in checkpoint_dirs:
            for backup_file in checkpoint_dir.glob("*.bak"):
                try:
                    backup_file.unlink()
                    logger.debug("Cleaned up backup file: %s", backup_file)
                except OSError as e:
                    logger.warning("Failed to clean up backup file %s: %s", backup_file, e)

    atexit.register(cleanup_checkpoints)
    signal.signal(signal.SIGINT, handle_signal(cleanup_checkpoints))
    signal.signal(signal.SIGTERM, handle_signal(cleanup_checkpoints))


def validate_plan(plan: SimulationPlan) -> None:
    """Validate simulation plan before execution.

    Args:
        plan: SimulationPlan to validate

    Raises:
        ValueError: If plan validation fails
    """
    logger.info("Validating simulation plan with %d batches", plan.total_batches)

    if plan.total_batches == 0:
        raise ValueError("Simulation plan contains no batches")

    # Validate each batch
    for batch in plan.batches:
        if not batch.simulator_name:
            raise ValueError("Batch has empty simulator name")
        if batch.batch_index < 0:
            raise ValueError(f"Batch {batch.batch_index} has invalid index")

        if isinstance(batch.simulator_config, OrchestrationConfig):
            if batch.simulator_config.signal is not None and not batch.simulator_config.signal.output.file_name:
                raise ValueError(f"Batch {batch.simulator_name}-{batch.batch_index} missing signal output file_name")
            if batch.simulator_config.noise is not None and not batch.simulator_config.noise.output.file_name:
                raise ValueError(f"Batch {batch.simulator_name}-{batch.batch_index} missing noise output file_name")
        else:
            output_config = batch.simulator_config.output
            if not output_config.file_name:
                raise ValueError(f"Batch {batch.simulator_name}-{batch.batch_index} missing file_name")

    logger.info("Simulation plan validation completed successfully")


def _resolve_recorded_output_paths(metadata: dict[str, Any], working_directory: str | None) -> list[Path]:
    """Resolve the output file paths recorded in a batch's metadata.

    Paths in ``outputs[].path`` are stored relative to the working directory (see
    ``_to_path_string``); resolve them back against it so existence can be checked.
    """
    base = Path(working_directory) if working_directory else None
    paths: list[Path] = []
    for record in metadata.get("outputs", []) or []:
        raw = record.get("path") if isinstance(record, dict) else None
        if not raw:
            continue
        path = Path(raw)
        if base is not None and not path.is_absolute():
            path = base / path
        paths.append(path)
    return paths


def _batch_outputs_present(batch: SimulationBatch, metadata_directory: Path) -> bool | None:
    """Return whether the outputs recorded for ``batch`` all exist on disk.

    Returns:
        ``True`` if the batch metadata exists and every recorded output is present,
        ``False`` if the metadata exists but one or more recorded outputs are missing,
        ``None`` if the batch metadata file itself is missing.
    """
    metadata_file = metadata_directory / f"{batch.simulator_name}-{batch.batch_index}.metadata.json"
    if not metadata_file.exists():
        return None
    try:
        with metadata_file.open("r") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError) as error:
        logger.warning(
            "Failed to read metadata for batch %d during checkpoint reconciliation: %s", batch.batch_index, error
        )
        return None
    working_directory = getattr(batch.globals_config, "working_directory", None)
    recorded = _resolve_recorded_output_paths(metadata, working_directory)
    if not recorded:
        # No output paths recorded to verify; trust the checkpoint for this batch.
        return True
    missing = [path for path in recorded if not path.exists()]
    if missing:
        logger.warning(
            "Checkpoint marks batch %d complete but output(s) are missing: %s",
            batch.batch_index,
            ", ".join(str(path) for path in missing),
        )
        return False
    return True


def reconcile_completed_batches(
    plan: SimulationPlan,
    metadata_directory: Path,
    completed_batch_indices: set[int],
) -> set[int]:
    """Prune checkpointed batches whose outputs are missing from disk.

    Resume restores simulator state from a single tail snapshot and assumes the
    completed batches form a contiguous prefix (the orchestration noise stream is
    sequential/stateful). So the completed set is validated in order and truncated at
    the FIRST batch whose recorded outputs (or metadata) are missing: that batch and
    every later batch are dropped so they get re-run.

    Returns the validated contiguous prefix of ``completed_batch_indices``.
    """
    if not completed_batch_indices:
        return set()
    batch_by_index = {batch.batch_index: batch for batch in plan.batches}
    valid: set[int] = set()
    for index in sorted(completed_batch_indices):
        batch = batch_by_index.get(index)
        if batch is None:
            logger.warning("Checkpoint references unknown batch %d; it and later batches will be re-run.", index)
            break
        present = _batch_outputs_present(batch, metadata_directory)
        if present is None:
            logger.warning(
                "Checkpoint marks batch %d complete but its metadata is missing; it and later batches will be re-run.",
                index,
            )
            break
        if not present:
            break
        valid.add(index)
    return valid


def execute_plan(  # noqa: PLR0915
    plan: SimulationPlan,
    output_directory: Path,
    metadata_directory: Path,
    overwrite: bool,
    ignore_checkpoint: bool = False,
    max_retries: int = 3,
) -> None:
    """Execute a complete simulation plan.

    The key insight: Simulators are stateful objects. Each simulator is instantiated
    once and then generates multiple batches by calling next() repeatedly. State
    (RNG, counters, filters) accumulates across batches.

    Checkpoint behavior:
    1. After each successfully completed batch, save checkpoint with updated state
    2. Checkpoint contains: completed batch indices, simulator state
    3. On next run, already-completed batches are skipped (resumption)
    4. On successful completion of all batches, checkpoint is cleaned up

    Workflow:
    1. Group batches by simulator name
    2. Load checkpoint to find already-completed batches
    3. For each simulator:
       a. Create ONE simulator instance
       b. For each batch of that simulator:
          - Skip if already completed (from checkpoint)
          - Restore state if reproducing from metadata
          - Call next(simulator) to generate batch (increments state)
          - Save batch output and metadata
          - Save checkpoint with updated state (for resumption)

    Args:
        plan: SimulationPlan to execute
        output_directory: Directory for output files
        metadata_directory: Directory for metadata files
        overwrite: Whether to overwrite existing files
        max_retries: Maximum retries per batch
    """
    logger.info("Executing simulation plan: %d batches", plan.total_batches)

    validate_plan(plan)
    setup_signal_handlers([plan.checkpoint_directory] if plan.checkpoint_directory else [])

    # Initialize checkpoint manager for resumption support
    checkpoint_manager = CheckpointManager(plan.checkpoint_directory)
    # One decode for the whole setup. The file now carries the spillover -- 131 MB of base64 for a
    # 1000 s tail -- and every `load_checkpoint` decodes all of it, so each convenience getter used
    # here would pay that again before the run started.
    # `--ignore-checkpoint` discards it here rather than deleting the file: the refusal below is a
    # dead end for anything that cannot answer a prompt -- an automated campaign would fail on a
    # stale file with no way forward but manual intervention -- and deleting on the user's behalf is
    # the one action that cannot be undone.
    checkpoint = {} if ignore_checkpoint else (checkpoint_manager.load_checkpoint() or {})
    if ignore_checkpoint:
        logger.warning("Ignoring any checkpoint in %s: --ignore-checkpoint was given.", plan.checkpoint_directory)
    # Checked before anything is read from it. A checkpoint another configuration wrote will
    # otherwise be believed: the batches it records as complete are skipped and their outputs never
    # produced, with no warning and exit code 0.
    # Not `batch.config_sha256` on its own: that hashes the config *file*, so the same file run with
    # a different `--output-dir` fingerprints identically and the guard waves it through -- measured
    # at 2 frames where a clean run writes 3. The identity has to include where the outputs go.
    plan_sha256 = run_fingerprint([batch.config_sha256 for batch in plan.batches], output_directory, metadata_directory)
    if checkpoint:
        require_matching_config(checkpoint.get("config_sha256"), plan_sha256, checkpoint_manager.checkpoint_file)
    # A set, matching what `get_completed_batch_indices` returned: it is compared against
    # `reconcile_completed_batches`'s output below, and a list never equals a set, which silently
    # sends every resume down the "outputs are missing" branch.
    loaded_batch_indices = set(checkpoint.get("completed_batch_indices") or [])
    resuming = bool(loaded_batch_indices)

    # Reconcile the checkpoint against the filesystem: a batch may be recorded as
    # completed while its output is missing (partial write at interrupt, an external
    # move/backup, fs hiccup). Skipping such a batch would silently drop a file.
    completed_batch_indices = reconcile_completed_batches(plan, metadata_directory, loaded_batch_indices)

    if not resuming:
        logger.debug("No checkpoint found or no batches completed yet")
        last_simulator_state = None
        last_simulator_spillover = None
        spillover_simulator_name = None
        spillover_batch_index = None
    elif completed_batch_indices == loaded_batch_indices:
        logger.info("Loaded checkpoint: %d batches already completed", len(completed_batch_indices))
        # From the single decode above. The per-batch scoping the getter would apply is done at the
        # restore call instead, from these values.
        last_simulator_state = checkpoint.get("last_simulator_state")
        last_simulator_state = last_simulator_state if isinstance(last_simulator_state, dict) else None
        last_simulator_spillover = checkpoint.get("last_simulator_spillover")
        spillover_simulator_name = checkpoint.get("last_simulator_name")
        spillover_batch_index = checkpoint.get("last_completed_batch_index")
    else:
        # One or more checkpointed batches are missing their outputs. The checkpoint
        # only holds the tail simulator state, so an interior batch cannot be
        # regenerated in isolation; discard the checkpoint and regenerate from the
        # first batch to keep the simulator/noise-stream/RNG state consistent.
        logger.warning(
            "Checkpoint listed %d completed batch(es) but only the first %d still have all outputs "
            "on disk; regenerating the simulation from the start to restore the missing output(s).",
            len(loaded_batch_indices),
            len(completed_batch_indices),
        )
        completed_batch_indices = set()
        last_simulator_state = None
        last_simulator_spillover = None
        spillover_simulator_name = None
        spillover_batch_index = None

    # Group batches by simulator name to execute sequentially per simulator
    simulator_batches: dict[str, list[SimulationBatch]] = {}
    for batch in plan.batches:
        if batch.simulator_name not in simulator_batches:
            simulator_batches[batch.simulator_name] = []
        simulator_batches[batch.simulator_name].append(batch)

    logger.info("Executing %d simulators", len(simulator_batches))

    with tqdm(total=plan.total_batches, desc="Executing simulation plan") as p_bar:
        for simulator_name, batches in simulator_batches.items():
            logger.info("Starting simulator: %s with %d batches", simulator_name, len(batches))

            # Create ONE simulator instance for all batches of this simulator
            # Extract global simulator arguments from the first batch's global config
            global_sim_args = batches[0].globals_config.simulator_arguments if batches else {}
            simulator = instantiate_simulator(batches[0].simulator_config, simulator_name, global_sim_args)

            # Process batches sequentially, maintaining state across them
            for batch_idx, batch in enumerate(batches):
                # Skip batches that were already completed AND whose outputs were verified
                # on disk during reconciliation (for resumption after interrupt).
                if batch.batch_index in completed_batch_indices:
                    logger.info(
                        "Skipping batch %d (already completed from checkpoint)",
                        batch.batch_index,
                    )
                    continue

                # On resume, any output present for a batch we are about to run is an
                # unverified leftover (orphan/partial) from the interrupted attempt, not
                # user data, so allow overwriting it even without --overwrite.
                batch_overwrite = overwrite or resuming

                try:
                    logger.debug(
                        "Executing batch %d/%d for simulator %s",
                        batch_idx + 1,
                        len(batches),
                        simulator_name,
                    )

                    # Capture pre-batch state first for potential retries
                    logger.debug(
                        "[EXECUTE] Batch %s: Before restore - counter=%s, has_state_snapshot=%s",
                        batch.batch_index,
                        simulator.counter,
                        batch.has_state_snapshot(),
                    )
                    # Scoped before it is handed over. A plan can execute several simulators and
                    # the checkpoint holds one tail, so an unscoped hand-off can put one simulator's
                    # spillover into another's segment -- real strain of the right shape, in the
                    # wrong place. It is also only valid for the batch immediately after the one
                    # that produced it.
                    spillover_for_batch = (
                        last_simulator_spillover
                        if spillover_applies(
                            spillover_simulator_name,
                            spillover_batch_index,
                            batch.simulator_name,
                            batch.batch_index,
                        )
                        else None
                    )
                    restore_batch_state(simulator, batch, last_simulator_state, spillover_for_batch)
                    logger.debug("[EXECUTE] Batch %s: After restore - counter=%s", batch.batch_index, simulator.counter)
                    pre_batch_state = copy.deepcopy(simulator.state)
                    # Spillover too, and separately, because it is not part of `state`. `simulate`
                    # consumes `cached_data_chunks` and replaces it with the new tail, so a retry
                    # after a failed write would re-run against consumed chunks and produce
                    # different data than the first attempt -- silently, since a retry that
                    # succeeds looks like a success.
                    pre_batch_spillover = copy.deepcopy(getattr(simulator, "cached_data_chunks", None))
                    logger.debug(
                        "[EXECUTE] Batch %s: Captured pre_batch_state - keys=%s",
                        batch.batch_index,
                        list(pre_batch_state.keys()),
                    )

                    def execute_batch(
                        _simulator=simulator,
                        _batch=batch,
                        _output_directory=output_directory,
                        _pre_batch_state=pre_batch_state,
                        _overwrite=batch_overwrite,
                    ):
                        """Execute a single batch with state management."""
                        set_batch_context = getattr(_simulator, "set_batch_context", None)
                        if callable(set_batch_context):
                            set_batch_context(
                                batch=_batch,
                                output_directory=_output_directory,
                                overwrite=_overwrite,
                            )

                        # Generate data by calling next() - this advances simulator state
                        logger.debug("[BATCH] %s: Before next() - counter=%s", _batch.batch_index, _simulator.counter)
                        batch_data = _simulator.simulate()
                        logger.debug("[BATCH] %s: After next() - counter=%s", _batch.batch_index, _simulator.counter)

                        # Save the generated data and get all output file paths
                        output_files = process_batch(
                            simulator=_simulator,
                            batch_data=batch_data,
                            batch=_batch,
                            output_directory=_output_directory,
                            overwrite=_overwrite,
                        )

                        # Only save metadata if data save succeeded
                        # This ensures metadata only exists for successfully saved data
                        save_batch_metadata(
                            _simulator,
                            _batch,
                            metadata_directory,
                            batch_data,
                            output_files,
                            pre_batch_state=_pre_batch_state,
                        )
                        # Update the state after successful save
                        _simulator.update_state()

                    def restore_state_for_retry(
                        _simulator=simulator,
                        _pre_batch_state=pre_batch_state,
                        _pre_batch_spillover=pre_batch_spillover,
                    ):
                        """Restore simulator state to pre-batch state before retry."""
                        _simulator.state = copy.deepcopy(_pre_batch_state)
                        if _pre_batch_spillover is not None:
                            _simulator.cached_data_chunks = copy.deepcopy(_pre_batch_spillover)

                    # Execute batch with retry mechanism that restores state on failure
                    retry_with_backoff(
                        execute_batch,
                        max_retries=max_retries,
                        state_restore_func=restore_state_for_retry,
                    )

                    # After successful completion, save checkpoint with updated state
                    # At this point, state has been incremented by next() -> update_state()
                    # Save checkpoint to enable resumption if interrupted before next batch
                    completed_batch_indices.add(batch.batch_index)
                    checkpoint_manager.save_checkpoint(
                        completed_batch_indices=sorted(completed_batch_indices),
                        last_simulator_name=simulator_name,
                        last_completed_batch_index=batch.batch_index,
                        last_simulator_state=copy.deepcopy(simulator.state),
                        # Beside the state, not inside it: `state` also goes into every batch
                        # metadata record, and spillover is raw samples. See `save_checkpoint`.
                        last_simulator_spillover=copy.deepcopy(getattr(simulator, "cached_data_chunks", None)),
                        config_sha256=plan_sha256,
                    )
                    logger.debug(
                        "Checkpoint saved after batch %d - state counter=%s",
                        batch.batch_index,
                        simulator.counter,
                    )
                    p_bar.update(1)

                except Exception as e:
                    logger.error(
                        "Failed to execute batch %d for simulator %s after %d retries: %s",
                        batch.batch_index,
                        simulator_name,
                        max_retries,
                        e,
                    )
                    raise

    # All batches completed successfully - clean up checkpoint files
    checkpoint_manager.cleanup()
    logger.info("All batches completed successfully. Checkpoint files cleaned up.")
