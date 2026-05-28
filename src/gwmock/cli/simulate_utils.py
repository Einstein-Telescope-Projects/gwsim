"""
Utilities for executing simulation plans via CLI.
"""

from __future__ import annotations

import atexit
import copy
import json
import logging
import platform
import signal
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
from gwmock.cli.utils.checkpoint import CheckpointManager
from gwmock.cli.utils.config import OrchestrationConfig, SimulatorConfig, resolve_class_path
from gwmock.cli.utils.hash import compute_file_hash
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
    git_sha = _get_distribution_git_sha()
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
        return {
            "backend": _backend_path_from_object(simulator.signal_adapter._backend),
            "waveform_model": simulator.waveform_model,
            "detector_network": list(simulator.detectors),
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
                }
            )

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
    simulator: Simulator, batch: SimulationBatch, last_simulator_state: dict[str, Any] | None = None
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
        config_payload=_build_config_payload(batch, simulator),
        config_sha256=batch.config_sha256,
        seed=seed,
        segment_seeds=_resolve_segment_seeds(simulator, batch, seed),
        population=_build_population_section(simulator, batch),
        signal=_build_signal_section(simulator, batch),
        noise=_build_noise_section(simulator, batch),
        outputs=_build_output_records(simulator, batch, batch_data, output_files),
        host=_get_host_metadata(),
    )

    # Add output files to metadata for easy discovery
    # Store just the file names, not full paths
    metadata["output_files"] = [f.name for f in output_files]

    # Compute and add file hashes for integrity checking
    file_hashes = {}
    for output_file in output_files:
        try:
            file_hash = compute_file_hash(output_file)
            file_hashes[output_file.name] = file_hash
            logger.debug("Compute hash for %s: %s", output_file.name, file_hash)
        except OSError as e:
            logger.warning("Failed to compute hash for %s: %s", output_file.name, e)
            # Continue without failing - metadata is still useful

    metadata["file_hashes"] = file_hashes

    metadata_file_name = f"{batch.simulator_name}-{batch.batch_index}.metadata.json"
    metadata_file = metadata_directory / metadata_file_name
    logger.debug("Saving batch metadata to %s with %d output files", metadata_file, len(output_files))

    save_metadata_record(metadata=metadata, metadata_file=metadata_file)

    # Update the metadata index for quick lookup
    update_metadata_index(metadata_directory, output_files, metadata_file_name)


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

        signal_output = batch.simulator_config.signal.output
        logger.debug(
            "[PROCESS] Batch %s: Saving adapter-backed outputs - counter=%s, signal_template=%s, noise_template=%s",
            batch.batch_index,
            simulator.counter,
            signal_output.file_name,
            batch.simulator_config.noise.output.file_name,
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
        noise_output_files = list(batch_data.noise_result.output_paths.values())
        missing_noise_outputs = [path for path in noise_output_files if not path.exists()]
        if missing_noise_outputs:
            raise FileNotFoundError(
                "Noise adapter reported output files that do not exist: "
                + ", ".join(str(path) for path in missing_noise_outputs)
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
            if not batch.simulator_config.signal.output.file_name:
                raise ValueError(f"Batch {batch.simulator_name}-{batch.batch_index} missing signal output file_name")
            if not batch.simulator_config.noise.output.file_name:
                raise ValueError(f"Batch {batch.simulator_name}-{batch.batch_index} missing noise output file_name")
        else:
            output_config = batch.simulator_config.output
            if not output_config.file_name:
                raise ValueError(f"Batch {batch.simulator_name}-{batch.batch_index} missing file_name")

    logger.info("Simulation plan validation completed successfully")


def execute_plan(  # noqa: PLR0915
    plan: SimulationPlan,
    output_directory: Path,
    metadata_directory: Path,
    overwrite: bool,
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
    completed_batch_indices = checkpoint_manager.get_completed_batch_indices()

    if completed_batch_indices:
        logger.info("Loaded checkpoint: %d batches already completed", len(completed_batch_indices))
        last_simulator_state = checkpoint_manager.get_last_simulator_state()
    else:
        logger.debug("No checkpoint found or no batches completed yet")
        last_simulator_state = None

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
                # Skip batches that were already completed (for resumption after interrupt)
                if checkpoint_manager.should_skip_batch(batch.batch_index):
                    logger.info(
                        "Skipping batch %d (already completed from checkpoint)",
                        batch.batch_index,
                    )
                    continue

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
                    restore_batch_state(simulator, batch, last_simulator_state)
                    logger.debug("[EXECUTE] Batch %s: After restore - counter=%s", batch.batch_index, simulator.counter)
                    pre_batch_state = copy.deepcopy(simulator.state)
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
                    ):
                        """Execute a single batch with state management."""
                        set_batch_context = getattr(_simulator, "set_batch_context", None)
                        if callable(set_batch_context):
                            set_batch_context(
                                batch=_batch,
                                output_directory=_output_directory,
                                overwrite=overwrite,
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
                            overwrite=overwrite,
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

                    def restore_state_for_retry(_simulator=simulator, _pre_batch_state=pre_batch_state):
                        """Restore simulator state to pre-batch state before retry."""
                        _simulator.state = copy.deepcopy(_pre_batch_state)

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
