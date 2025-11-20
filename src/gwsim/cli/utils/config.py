"""
Utility functions to load and save configuration files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("gwsim")


class SimulatorOutputConfig(BaseModel):
    """Configuration for simulator output handling."""

    file_name: str = Field(..., description="Output file name template (supports {{ variable }} placeholders)")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Output-specific arguments (e.g., channel name)"
    )
    output_directory: str | None = Field(
        default=None, description="Optional directory override for this simulator's output"
    )
    metadata_directory: str | None = Field(
        default=None, description="Optional directory override for this simulator's metadata"
    )

    # Allow unknown fields
    model_config = ConfigDict(extra="allow")


class SimulatorConfig(BaseModel):
    """Configuration for a single simulator."""

    class_: str = Field(alias="class", description="Simulator class name or full import path")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to simulator constructor")
    output: SimulatorOutputConfig = Field(
        default_factory=lambda: SimulatorOutputConfig(file_name="output-{{counter}}.hdf5"),
        description="Output configuration for this simulator",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @field_validator("class_", mode="before")
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        """Validate class specification is non-empty."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("'class' must be a non-empty string")
        return v


class GlobalsConfig(BaseModel):
    """Global configuration applying to all simulators."""

    working_directory: str = Field(
        default=".", alias="working-directory", description="Base working directory for all output"
    )
    output_directory: str | None = Field(
        default=None, alias="output-directory", description="Default output directory (can be overridden per simulator)"
    )
    metadata_directory: str | None = Field(
        default=None,
        alias="metadata-directory",
        description="Default metadata directory (can be overridden per simulator)",
    )
    sampling_frequency: float | None = Field(
        default=None, alias="sampling-frequency", description="Default sampling frequency in Hz"
    )
    duration: float | None = Field(default=None, description="Default segment duration in seconds")
    max_samples: int | None = Field(
        default=None, alias="max-samples", description="Maximum number of segments to generate"
    )
    seed: int | None = Field(default=None, description="Base seed for random number generation")

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Config(BaseModel):
    """Top-level configuration model."""

    globals: GlobalsConfig = Field(default_factory=GlobalsConfig, description="Global configuration")
    simulators: dict[str, SimulatorConfig] = Field(..., description="Dictionary of simulators")

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @field_validator("simulators", mode="before")
    @classmethod
    def validate_simulators_not_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure simulators section is not empty."""
        if not v:
            raise ValueError("'simulators' section cannot be empty")
        return v


def load_config(file_name: Path, encoding: str = "utf-8") -> Config:
    """Load configuration file with validation.

    Args:
        file_name (Path): File name.
        encoding (str, optional): File encoding. Defaults to "utf-8".

    Returns:
        Config: Validated configuration dataclass.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration is invalid or cannot be parsed.
    """
    if not file_name.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_name}")
    try:
        with file_name.open(encoding=encoding) as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}") from e

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration must be a YAML dictionary")

    # Validate and convert to Config dataclass
    try:
        config = Config(**raw_config)
        logger.info("Configuration loaded and validated: %s simulators", len(config.simulators))
        return config
    except ValueError as e:
        raise ValueError(f"Configuration validation failed: {e}") from e


def save_config(
    file_name: Path, config: Config, overwrite: bool = False, encoding: str = "utf-8", backup: bool = True
) -> None:
    """Save configuration to YAML file safely.

    Args:
        file_name: Path to save configuration to
        config: Config dataclass instance
        overwrite: If True, overwrite existing file
        encoding: File encoding (default: utf-8)
        backup: If True and overwriting, create backup

    Raises:
        FileExistsError: If file exists and overwrite=False
    """
    if file_name.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {file_name}. Use overwrite=True to overwrite.")

    # Create backup if needed
    if file_name.exists() and overwrite and backup:
        backup_path = file_name.with_suffix(f"{file_name.suffix}.backup")
        logger.info("Creating backup: %s", backup_path)
        backup_path.write_text(file_name.read_text(encoding=encoding), encoding=encoding)

    # Atomic write
    temp_file = file_name.with_suffix(f"{file_name.suffix}.tmp")
    try:
        # Convert to dict, excluding internal fields
        config_dict = config.model_dump(by_alias=True, exclude_none=False)

        with temp_file.open("w", encoding=encoding) as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

        temp_file.replace(file_name)
        logger.info("Configuration saved to: %s", file_name)

    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise ValueError(f"Failed to save configuration: {e}") from e


def validate_config(config: dict) -> None:
    """Validate configuration structure and provide helpful error messages.

    Args:
        config (dict): Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid with detailed error message
    """
    # Check for required top-level structure
    if "simulators" not in config:
        raise ValueError("Invalid configuration: Must contain 'simulators' section with simulator definitions")

    simulators = config["simulators"]

    if not isinstance(simulators, dict):
        raise ValueError("'simulators' must be a dictionary")

    if not simulators:
        raise ValueError("'simulators' section cannot be empty")

    for name, sim_config in simulators.items():
        if not isinstance(sim_config, dict):
            raise ValueError(f"Simulator '{name}' configuration must be a dictionary")

        # Check required fields
        if "class" not in sim_config:
            raise ValueError(f"Simulator '{name}' missing required 'class' field")

        # Validate class specification
        class_spec = sim_config["class"]
        if not isinstance(class_spec, str) or not class_spec.strip():
            raise ValueError(f"Simulator '{name}' 'class' must be a non-empty string")

        # Validate arguments if present
        if "arguments" in sim_config and not isinstance(sim_config["arguments"], dict):
            raise ValueError(f"Simulator '{name}' 'arguments' must be a dictionary")

        # Validate output configuration if present
        if "output" in sim_config:
            output_config = sim_config["output"]
            if not isinstance(output_config, dict):
                raise ValueError(f"Simulator '{name}' 'output' must be a dictionary")

    # Validate globals section if present
    if "globals" in config:
        globals_config = config["globals"]
        if not isinstance(globals_config, dict):
            raise ValueError("'globals' must be a dictionary")

    logger.info("Configuration validation passed")


def resolve_class_path(class_spec: str, section_name: str) -> str:
    """Resolve class specification to full module path.

    Args:
        class_spec: Either 'ClassName' or 'third_party.module.ClassName'
        section_name: Section name (e.g., 'noise', 'signal', 'glitch')

    Returns:
        Full path like 'gwsim.noise.ClassName' or 'third_party.module.ClassName'

    Examples:
        resolve_class_path("WhiteNoise", "noise") -> "gwsim.noise.WhiteNoise"
        resolve_class_path("numpy.random.Generator", "noise") -> "numpy.random.Generator"
    """
    if "." not in class_spec:
        # Just a class name - use section_name as submodule, class imported in __init__.py
        return f"gwsim.{section_name}.{class_spec}"
    # Contains dots - assume it's a third-party package, use as-is
    return class_spec


def merge_parameters(globals_config: GlobalsConfig, simulator_args: dict[str, Any]) -> dict[str, Any]:
    """Merge global and simulator-specific parameters.

    Args:
        globals_config: GlobalsConfig dataclass instance
        simulator_args: Simulator-specific arguments dict

    Returns:
        Merged parameters with simulator args taking precedence

    Note:
        All non-private fields from globals_config are merged into the result.
    """
    # Convert globals to dict, excluding None values and private fields
    merged = {
        k: v for k, v in globals_config.model_dump(by_alias=True).items() if v is not None and not k.startswith("_")
    }

    # Override with simulator-specific arguments
    merged.update(simulator_args)

    return merged


def get_output_directories(
    globals_config: GlobalsConfig,
    simulator_config: SimulatorConfig,
    simulator_name: str,
    working_directory: Path | None = None,
) -> tuple[Path, Path]:
    """Get output and metadata directories for a simulator.

    Args:
        globals_config: Global configuration
        simulator_config: Simulator-specific configuration
        simulator_name: Name of the simulator
        working_directory: Override working directory (for testing)

    Returns:
        Tuple of (output_directory, metadata_directory)

    Priority (highest to lowest):
        1. Simulator output.output_directory / output.metadata_directory
        2. Global output-directory / metadata-directory
        3. working-directory / output / {simulator_name}

    Examples:
        >>> globals_cfg = GlobalsConfig(working_directory="/data")
        >>> sim_cfg = SimulatorConfig(class_="Noise")
        >>> get_output_directories(globals_cfg, sim_cfg, "noise")
        (Path("/data/output/noise"), Path("/data/output/noise"))
    """
    working_dir = working_directory or Path(globals_config.working_directory)

    # Simulator-specific overrides
    if simulator_config.output.output_directory:
        output_directory = Path(simulator_config.output.output_directory)
    elif globals_config.output_directory:
        output_directory = Path(globals_config.output_directory)
    else:
        output_directory = working_dir / "output" / simulator_name

    if simulator_config.output.metadata_directory:
        metadata_directory = Path(simulator_config.output.metadata_directory)
    elif globals_config.metadata_directory:
        metadata_directory = Path(globals_config.metadata_directory)
    else:
        metadata_directory = working_dir / "metadata" / simulator_name

    return output_directory, metadata_directory
