"""
Utility functions to load and save configuration files.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from ..utils.io import check_file_overwrite

logger = logging.getLogger("gwsim")


def validate_config(config: dict) -> None:
    """Validate configuration structure and provide helpful error messages.

    Args:
        config: Configuration dictionary to validate

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


def load_config(file_name: Path, encoding: str = "utf-8") -> dict:
    """Load configuration file with validation.

    Args:
        file_name (Path): File name.
        encoding (str, optional): File encoding. Defaults to "utf-8".

    Returns:
        dict: A dictionary of the configuration.

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If configuration file does not exist
    """
    if not file_name.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_name}")

    with open(file_name, encoding=encoding) as f:
        config = yaml.safe_load(f)

    # Validate the loaded configuration
    validate_config(config)

    return config


@check_file_overwrite()
def save_config(
    file_name: Path, config: dict, overwrite: bool = False, encoding: str = "utf-8", backup: bool = True
) -> None:
    """Save configuration file safely with optional backup.

    Args:
        file_name (Path): File name.
        config (dict): A dictionary of configuration.
        overwrite (bool, optional): If True, overwrite the existing file, or otherwise raise an error.
            Defaults to False.
        encoding (str, optional): File encoding. Defaults to "utf-8".
        backup (bool, optional): If True and overwriting, create a backup of the existing file.
            Defaults to True.

    Raises:
        FileExistsError: If file_name exists and overwrite is False, raise an error.
    """
    # Create backup if file exists and we're overwriting
    if file_name.exists() and overwrite and backup:
        backup_path = file_name.with_suffix(f"{file_name.suffix}.backup")
        logger.info("Creating backup: %s", backup_path)
        backup_path.write_text(file_name.read_text(encoding=encoding), encoding=encoding)

    # Atomic write using temporary file
    temp_file = file_name.with_suffix(f"{file_name.suffix}.tmp")
    try:
        with open(temp_file, "w", encoding=encoding) as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        # Atomic move (rename) - this is atomic on most filesystems
        temp_file.replace(file_name)
        logger.info("Configuration saved to: %s", file_name)

    except Exception:
        # Clean up temp file if something went wrong
        if temp_file.exists():
            temp_file.unlink()
        raise


def get_config_value(config: dict, key: str, default_value: Any | None = None) -> Any:
    """Get the argument

    Args:
        config (dict): A dictionary of configuration.
        key (str): Key of the entry.
        default_value (Any | None, optional): Default value if key is not present. Defaults to None.

    Returns:
        Any: Value of the corresponding key in config.
    """
    if key in config:
        return config[key]
    return default_value


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


def merge_parameters(globals_config: dict, simulator_config: dict) -> dict:
    """Merge global and simulator-specific parameters.

    Args:
        globals_config: Global configuration parameters
        simulator_config: Simulator-specific configuration

    Returns:
        Merged parameters with simulator config taking precedence

    Note:
        All global parameters are passed to simulators. Simulator-specific
        arguments override global parameters when the same key exists.
    """
    # Start with all global parameters
    merged = globals_config.copy() if globals_config else {}

    # Simulator-specific arguments take precedence over global parameters
    if "arguments" in simulator_config:
        merged.update(simulator_config["arguments"])

    return merged


def merge_output_parameters(globals_config: dict, simulator_config: dict) -> dict:
    """Merge global and simulator-specific output parameters.

    Args:
        globals_config: Global configuration parameters
        simulator_config: Simulator-specific configuration

    Returns:
        Merged output parameters with simulator config taking precedence

    Note:
        All global parameters are passed to output functions. Simulator-specific
        output arguments override global parameters when the same key exists.
        Output functions should ignore irrelevant parameters gracefully.
    """
    # Start with all global parameters
    merged = globals_config.copy() if globals_config else {}

    # Merge simulator-specific output arguments if they exist
    if "output" in simulator_config and "arguments" in simulator_config["output"]:
        merged.update(simulator_config["output"]["arguments"])

    return merged


def expand_templates(text: str, context: dict) -> str:
    """Expand template variables in text strings.

    Args:
        text: String that may contain template variables like {{ key }}
        context: Dictionary containing template variable values

    Returns:
        String with template variables expanded

    Examples:
        expand_templates("file-{{ duration }}.gwf", {"duration": 4}) -> "file-4.gwf"
    """

    def replace_var(match):
        var_name = match.group(1).strip()
        # Support nested access like globals.duration
        keys = var_name.split(".")
        value = context
        try:
            for key in keys:
                value = value[key]
            return str(value)
        except (KeyError, TypeError):
            logger.warning("Template variable '%s' not found in context", var_name)
            return match.group(0)  # Return original if not found

    # Match {{ variable }} or {{ nested.variable }}
    pattern = r"\{\{\s*([^}]+)\s*\}\}"
    return re.sub(pattern, replace_var, text)


def expand_detector_templates(config: Any, detectors: list[str] | None = None) -> Any:
    """Expand {detector} placeholders in configuration.

    Args:
        config: Configuration value (dict, list, str, or other)
        detectors: List of detector names to expand for

    Returns:
        Configuration with expanded detector templates

    Note:
        This function handles {detector} placeholders which are different from
        {{ variable }} template variables. The {detector} placeholder is used
        for multi-detector file naming patterns.
    """
    if detectors is None:
        detectors = []

    if isinstance(config, str):
        # Preserve all strings unchanged - {detector} placeholders are handled at runtime
        return config
    if isinstance(config, dict):
        return {key: expand_detector_templates(value, detectors) for key, value in config.items()}
    if isinstance(config, list):
        return [expand_detector_templates(item, detectors) for item in config]
    return config


def expand_config_templates(config: Any, context: dict) -> Any:
    """Recursively expand template variables in configuration.

    Args:
        config: Configuration value (dict, list, str, or other)
        context: Template variable context

    Returns:
        Configuration with expanded templates
    """
    if isinstance(config, str):
        return expand_templates(config, context)
    if isinstance(config, dict):
        return {key: expand_config_templates(value, context) for key, value in config.items()}
    if isinstance(config, list):
        return [expand_config_templates(item, context) for item in config]
    return config


def normalize_config(config: dict) -> dict:
    """Normalize configuration ensuring proper structure.

    Args:
        config: Raw configuration dictionary

    Returns:
        Normalized configuration with 'globals' and 'simulators' sections
    """
    # Ensure we have the required structure
    if "simulators" not in config:
        raise ValueError("Configuration must contain 'simulators' section")

    # Ensure we have a globals section (can be empty)
    if "globals" not in config:
        config["globals"] = {}

    return config


def process_config(config: dict) -> dict:
    """Process configuration with template expansion and parameter inheritance.

    Args:
        config: Raw configuration dictionary

    Returns:
        Processed configuration ready for use
    """
    # Normalize configuration structure
    normalized = normalize_config(config)

    # Extract globals
    globals_config = normalized.get("globals", {})

    # Create template context
    context = {"globals": globals_config, **globals_config}  # Also make globals available at root level

    # Process each simulator
    processed_simulators = {}
    for name, simulator_config in normalized["simulators"].items():
        # Merge parameters for simulator arguments
        merged_args = merge_parameters(globals_config, simulator_config)

        # Merge parameters for output arguments
        merged_output_args = merge_output_parameters(globals_config, simulator_config)

        # Add simulator name to context
        simulator_context = {**context, "simulator_name": name, **merged_args}  # Make arguments available for templates

        # Expand templates in the simulator config
        processed_config = expand_config_templates(simulator_config, simulator_context)

        # Update arguments with merged parameters
        processed_config["arguments"] = merged_args

        # Update output arguments with merged parameters
        if "output" not in processed_config:
            processed_config["output"] = {}
        processed_config["output"]["arguments"] = merged_output_args

        processed_simulators[name] = processed_config

    return {"globals": expand_config_templates(globals_config, context), "simulators": processed_simulators}
