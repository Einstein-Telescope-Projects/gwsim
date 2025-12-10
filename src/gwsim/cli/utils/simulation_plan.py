"""Utility functions for creating and managing simulation plans."""

from __future__ import annotations

import datetime
import getpass
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gwsim.cli.utils.config import Config, GlobalsConfig, SimulatorConfig
from gwsim.cli.utils.config_resolution import resolve_max_samples
from gwsim.cli.utils.metadata import load_metadata_with_external_state
from gwsim.utils.log import get_dependency_versions

logger = logging.getLogger("gwsim")


@dataclass
class SimulationBatch:
    """Data class representing a single simulation batch.

    A batch is a unit of work for a particular simulator. For example:
    - A noise simulator might generate multiple batches (segments) of noise data
    - A signal simulator might generate multiple batches of gravitational wave signals

    The batch_index is per-simulator, so batch 0 from noise simulator and batch 0 from
    signal simulator are different batches.

    Metadata can contain two types of information:
    - Configuration metadata: Full config + max_samples (for reproducibility with fresh state)
    - State metadata: Pre-batch state (RNG state, etc.) for exact reproduction of a specific batch
    """

    simulator_name: str
    """Name of the simulator (e.g., 'noise', 'signal', 'glitch')"""

    simulator_config: SimulatorConfig
    """Configuration for this simulator"""

    globals_config: GlobalsConfig
    """Global configuration (shared across all simulators)"""

    batch_index: int
    """Index of this batch within the simulator (0-based, per-simulator)"""

    # Optional: For metadata-based reproduction
    metadata_file: Path | None = None
    """If reproducing from metadata, path to the metadata file"""

    batch_metadata: dict[str, Any] | None = None
    """Parsed metadata content (if metadata_file is provided)"""

    # State snapshot for exact reproduction
    pre_batch_state: dict[str, Any] | None = None
    """State snapshot taken before this batch was generated.

    Contains simulator-specific state that cannot be known a priori:
    - RNG state (numpy.random.RandomState or similar)
    - Simulator internal state (e.g., filter memory for colored noise)
    - Other stateful components

    If present, use this for exact reproduction. Otherwise, reconstruct from config.
    """

    # For tracking
    source: str = "config"
    """Source of this batch: 'config' (fresh), 'metadata_config' (from saved config),
    or 'metadata_state' (from saved state snapshot)"""

    author: str | None = None
    """Author of this batch (from metadata)"""

    email: str | None = None
    """Email of the author (from metadata)"""

    def __post_init__(self):
        """Post-initialization checks.

        Raises:
            ValueError: If simulator_name is empty or batch_index is negative.
        """
        if not self.simulator_name:
            raise ValueError("simulator_name must not be empty")
        if self.batch_index < 0:
            raise ValueError("batch_index must be non-negative")

    def is_metadata_based(self) -> bool:
        """Check if this batch is based on saved metadata.

        Returns:
            True if the batch is based on metadata, False otherwise.
        """
        return self.source in ("metadata_config", "metadata_state")

    def has_state_snapshot(self) -> bool:
        """Check if this batch has a pre-batch state snapshot for exact reproduction.

        Returns:
            True if pre_batch_state is available, False otherwise.
        """
        return self.pre_batch_state is not None


@dataclass
class SimulationPlan:
    """Data class representing a simulation plan."""

    batches: list[SimulationBatch] = field(default_factory=list)
    """List of batches to simulate"""

    source_config: Config | None = None
    """Original Config object (if config-based)"""

    checkpoint_directory: Path = Path("checkpoints")
    """Directory for checkpoint files"""

    total_batches: int = 0
    """Total number of batches"""

    def add_batch(self, batch: SimulationBatch) -> None:
        """Add a batch to the plan.

        Args:
            batch: SimulationBatch to add
        """
        self.batches.append(batch)
        self.total_batches = len(self.batches)
        logger.debug(
            "Added batch %d: simulator=%s, source=%s",
            batch.batch_index,
            batch.simulator_name,
            batch.source,
        )

    def get_batches_for_simulator(self, simulator_name: str) -> list[SimulationBatch]:
        """Get all batches for a specific simulator.

        Args:
            simulator_name: Name of the simulator

        Returns:
            List of batches for that simulator, in order
        """
        return [b for b in self.batches if b.simulator_name == simulator_name]


def parse_batch_metadata(metadata_file: Path, metadata_dir: Path | None = None) -> dict[str, Any]:
    """Parse a batch metadata file.

    Args:
        metadata_file: Path to BATCH-*.metadata.yaml file
        metadata_dir: Directory containing external state files. If None, uses parent of metadata_file

    Returns:
        Parsed metadata dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML is invalid
    """
    metadata = load_metadata_with_external_state(metadata_file, metadata_dir)

    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")

    return metadata


def create_batch_metadata(
    simulator_name: str,
    batch_index: int,
    simulator_config: SimulatorConfig,
    globals_config: GlobalsConfig,
    pre_batch_state: dict[str, Any] | None = None,
    source: str = "config",
    author: str | None = None,
    email: str | None = None,
    timestamp: datetime.datetime | None = None,
) -> dict[str, Any]:
    """Create metadata for a simulation batch.

    This metadata can be used to reproduce a specific batch. It includes:
    1. Configuration: Simulator and global configs for reproducibility
    2. State snapshot: Pre-batch state (RNG, etc.) for exact reproduction
    3. Version information: gwsim and key dependency versions for traceability

    Args:
        simulator_name: Name of the simulator
        batch_index: Index of the batch within the simulator
        simulator_config: Configuration for this simulator
        globals_config: Global configuration
        pre_batch_state: Optional state snapshot taken before batch generation
        source: Source of this batch: 'config', 'metadata_config', or 'metadata_state'
        author: Optional author name
        email: Optional author email
        timestamp: Optional timestamp for when the batch was created

    Returns:
        Metadata dictionary suitable for YAML serialization
    """
    if author is None:
        author = getpass.getuser()

    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc)

    metadata: dict[str, Any] = {
        "simulator_name": simulator_name,
        "batch_index": batch_index,
        "simulator_config": simulator_config.model_dump(mode="python"),
        "globals_config": globals_config.model_dump(mode="python"),
        "author": author,
        "email": email,
        "timestamp": timestamp.isoformat(),
        "versions": get_dependency_versions(),
    }

    if pre_batch_state is not None:
        metadata["pre_batch_state"] = pre_batch_state

    metadata["source"] = source

    return metadata


def create_plan_from_config(
    config: Config, checkpoint_dir: Path, author: str | None = None, email: str | None = None
) -> SimulationPlan:
    """Create a simulation plan from a configuration file.

    This is the standard workflow: start fresh with a config.
    Each simulator can generate multiple batches (e.g., segments or samples).

    Note: State (like RNG state) will be captured during simulation and stored in
    metadata for exact reproduction of individual batches.

    Args:
        config: Parsed Config object
        checkpoint_dir: Directory for checkpoints
        author: Optional author name for metadata
        email: Optional author email for metadata

    Returns:
        SimulationPlan with all batches defined across all simulators

    Example:
        >>> from gwsim.cli.utils.config import load_config
        >>> cfg = load_config(Path("config.yaml"))
        >>> plan = create_plan_from_config(cfg, Path("checkpoints"))
        >>> print(f"Total batches: {plan.total_batches}")
        >>> # Get all batches from a specific simulator
        >>> noise_batches = plan.get_batches_for_simulator("noise")
    """
    plan = SimulationPlan(
        source_config=config,
        checkpoint_directory=checkpoint_dir,
    )

    # For each simulator, create batches (each simulator can generate multiple batches)
    global_batch_index = 0
    for simulator_name, simulator_config in config.simulators.items():
        # Determine number of batches for this simulator
        # This comes from simulator_arguments in globals_config (max_samples parameter)
        # First check simulator-specific arguments, then fall back to global simulator_arguments
        # Note: Keys in simulator_arguments may have hyphens (YAML style), so normalize them
        global_sim_args = {k.replace("-", "_"): v for k, v in config.globals.simulator_arguments.items()}
        local_sim_args = {k.replace("-", "_"): v for k, v in simulator_config.arguments.items()}

        max_samples = resolve_max_samples(simulator_args=local_sim_args, global_args=global_sim_args)

        for _ in range(max_samples):
            batch = SimulationBatch(
                simulator_name=simulator_name,
                simulator_config=simulator_config,
                globals_config=config.globals,
                batch_index=global_batch_index,
                source="config",
                author=author,
                email=email,
            )
            plan.add_batch(batch)
            global_batch_index += 1

    logger.info("Created simulation plan from config: %d batches", plan.total_batches)
    return plan


def create_plan_from_metadata_files(
    metadata_files: list[Path],
    checkpoint_dir: Path,
    author: str | None = None,
    email: str | None = None,
) -> SimulationPlan:
    """Create a simulation plan from individual metadata files.

    This allows exact reproduction of specific batches by restoring their pre-batch state.
    Metadata files should follow the naming pattern: SIMULATOR-BATCH_INDEX.metadata.yaml

    Args:
        metadata_files: List of paths to individual metadata YAML files
        checkpoint_dir: Directory for checkpoints
        author: Optional author name for metadata
        email: Optional author email for metadata

    Returns:
        SimulationPlan with batches reconstructed from metadata

    Raises:
        FileNotFoundError: If any metadata file doesn't exist
        ValueError: If metadata files are malformed

    Example:
        >>> files = [Path("signal-0.metadata.yaml"), Path("signal-1.metadata.yaml")]
        >>> plan = create_plan_from_metadata_files(files, Path("checkpoints"))
        >>> # Reproduces specific batches with exact state snapshots
    """
    plan = SimulationPlan(checkpoint_directory=checkpoint_dir)

    for metadata_file in sorted(metadata_files):
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        metadata = parse_batch_metadata(metadata_file)

        # Reconstruct configs from metadata
        try:
            globals_config = GlobalsConfig(**metadata["globals_config"])
            simulator_config = SimulatorConfig(**metadata["simulator_config"])
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid metadata in {metadata_file}: missing or malformed config: {e}") from e

        simulator_name = metadata.get("simulator_name")
        batch_index = metadata.get("batch_index")
        pre_batch_state = metadata.get("pre_batch_state")

        if not simulator_name or batch_index is None:
            raise ValueError(f"Invalid metadata in {metadata_file}: missing simulator_name or batch_index")

        batch = SimulationBatch(
            simulator_name=simulator_name,
            simulator_config=simulator_config,
            globals_config=globals_config,
            batch_index=batch_index,
            metadata_file=metadata_file,
            batch_metadata=metadata,
            pre_batch_state=pre_batch_state,
            source="metadata_state" if pre_batch_state else "metadata_config",
            author=author,
            email=email,
        )
        plan.add_batch(batch)

    logger.info(
        "Created simulation plan from %d metadata files",
        len(metadata_files),
    )
    return plan


def create_plan_from_metadata(
    metadata_dir: Path,
    checkpoint_dir: Path,
    author: str | None = None,
    email: str | None = None,
) -> SimulationPlan:
    """Create a simulation plan from a directory of metadata files.

    This allows exact reproduction of specific batches by restoring their pre-batch state.
    Metadata files should follow the naming pattern: SIMULATOR-BATCH_INDEX.metadata.yaml

    Args:
        metadata_dir: Directory containing metadata YAML files
        checkpoint_dir: Directory for checkpoints
        author: Optional author name for metadata
        email: Optional author email for metadata

    Returns:
        SimulationPlan with batches reconstructed from metadata

    Raises:
        FileNotFoundError: If metadata_dir doesn't exist
        ValueError: If metadata files are malformed

    Example:
        >>> plan = create_plan_from_metadata(Path("metadata"), Path("checkpoints"))
        >>> # Reproduces batches with exact state snapshots
    """
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    # Find all metadata files in directory
    metadata_files = list(metadata_dir.glob("*.metadata.yaml"))

    plan = create_plan_from_metadata_files(metadata_files, checkpoint_dir, author=author, email=email)
    logger.info(
        "Created simulation plan from metadata directory: %d batches from %s",
        plan.total_batches,
        metadata_dir,
    )
    return plan


def merge_plans(*plans: SimulationPlan) -> SimulationPlan:
    """Merge multiple simulation plans into one.

    Useful for combining config-based and metadata-based workflows.

    Args:
        *plans: SimulationPlan objects to merge

    Returns:
        Merged SimulationPlan

    Example:
        >>> plan_config = create_plan_from_config(cfg, Path("checkpoints"))
        >>> plan_metadata = create_plan_from_metadata_files([meta1, meta2], Path("checkpoints"))
        >>> combined_plan = merge_plans(plan_config, plan_metadata)
    """
    merged = SimulationPlan()
    batch_index = 0

    for plan in plans:
        for batch in plan.batches:
            # Reassign batch indices to maintain order
            batch.batch_index = batch_index
            merged.add_batch(batch)
            batch_index += 1

    merged.checkpoint_directory = plans[0].checkpoint_directory if plans else Path("checkpoints")
    logger.info("Merged %d plans into one: %d total batches", len(plans), merged.total_batches)
    return merged
