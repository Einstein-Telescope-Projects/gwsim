# ruff: noqa PLC0415
"""
A sub-command to handle data generation using simulation plans.
"""

from __future__ import annotations

from typing import Annotated

import typer


def _simulate_impl(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    config_file_names: str | list[str],
    output_dir: str | None = None,
    metadata_dir: str | None = None,
    overwrite: bool = False,
    metadata: bool = True,
    author: str | None = None,
    email: str | None = None,
    dry_run: bool = False,
) -> None:
    """Internal implementation of simulate command that accepts both str and list[str].

    Unified approach: All simulation goes through the same plan execution system.
    The difference is only in how we create the initial plan:
    - Config file → SimulationPlan with pre_batch_state=None (fresh simulation)
    - Metadata files → SimulationPlan with pre_batch_state=<dict> (reproduce)

    Both cases end up calling execute_plan with batches that may or may not have
    state snapshots. The execute_plan function handles both transparently.

    Args:
        config_file_names: Path to YAML config file OR one or more metadata files
        output_dir: Output directory override
        metadata_dir: Metadata directory override (config mode only)
        overwrite: Whether to overwrite existing files
        metadata: Whether to save metadata files (config mode only)
        author: Author name for metadata
        email: Author email for metadata
        dry_run: If True, only validate the plan without executing.

    Returns:
        None
    """
    import logging  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    from gwsim.cli.simulate_utils import (  # pylint: disable=import-outside-toplevel
        execute_plan,
        validate_plan,
    )
    from gwsim.cli.utils.config import load_config  # pylint: disable=import-outside-toplevel
    from gwsim.cli.utils.simulation_plan import (  # pylint: disable=import-outside-toplevel
        create_plan_from_config,
        create_plan_from_metadata_files,
    )
    from gwsim.monitor.resource import ResourceMonitor  # pylint: disable=import-outside-toplevel

    logger = logging.getLogger("gwsim")
    logger.setLevel(logging.DEBUG)

    checkpoint_dir = Path(".gwsim_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ===== Normalize input: accept both string and list =====
        config_file_names_list = [config_file_names] if isinstance(config_file_names, str) else config_file_names

        # ===== Auto-detect mode: Config file vs Metadata files =====
        if len(config_file_names_list) == 1:
            # Single argument: could be YAML config or metadata file
            single_path = Path(config_file_names_list[0])
            is_metadata = (
                single_path.suffix == ".metadata.yaml"
                or (single_path.is_file() and "metadata" in single_path.name)
                or single_path.is_dir()  # Directory assumed to be metadata dir
            )
        else:
            # Multiple arguments: must be metadata files
            is_metadata = True

        # ===== Create plan (unified approach: both modes create same data structure) =====
        if is_metadata:
            logger.info("Reproduction mode: %d metadata file(s)", len(config_file_names_list))

            metadata_paths = [Path(f) for f in config_file_names_list]

            # If single directory, load all metadata files from it
            if len(metadata_paths) == 1 and metadata_paths[0].is_dir():
                metadata_dir_path = metadata_paths[0]
                metadata_files = list(metadata_dir_path.glob("*.metadata.yaml"))
                if not metadata_files:
                    raise ValueError(f"No metadata files found in directory: {metadata_dir_path}")
                logger.info("Found %d metadata files in directory: %s", len(metadata_files), metadata_dir_path)
                plan = create_plan_from_metadata_files(metadata_files, checkpoint_dir, author=author, email=email)
            else:
                # Individual metadata files
                plan = create_plan_from_metadata_files(metadata_paths, checkpoint_dir, author=author, email=email)

            logger.info(
                "Created reproduction plan from %d metadata file(s) with %d batches",
                len(config_file_names_list),
                plan.total_batches,
            )
        else:
            logger.info("Config mode: %s", config_file_names_list[0])
            config_path = Path(config_file_names_list[0])
            config = load_config(file_name=config_path)
            logger.debug("Configuration loaded successfully from %s", config_file_names_list[0])

            plan = create_plan_from_config(config, checkpoint_dir, author=author, email=email)
            logger.info("Created simulation plan with %d batches", plan.total_batches)

        # ===== Determine output directories (same logic for both modes) =====
        # Get reference config from first batch
        if not plan.batches:
            raise ValueError("No batches found in simulation plan")

        first_batch = plan.batches[0]
        globals_cfg = first_batch.globals_config
        working_dir = Path(globals_cfg.working_directory or ".")  # pylint: disable=no-member
        output_dir_config = globals_cfg.output_directory or "output"  # pylint: disable=no-member

        # Handle absolute vs relative paths
        config_output_dir = (
            Path(output_dir_config) if Path(output_dir_config).is_absolute() else working_dir / output_dir_config
        )
        final_output_dir = Path(output_dir) if output_dir else config_output_dir

        # Metadata directory only used in config mode (fresh simulation)
        final_metadata_dir = None
        if not is_metadata and metadata:
            metadata_dir_config = globals_cfg.metadata_directory or "metadata"  # pylint: disable=no-member
            config_metadata_dir = (
                Path(metadata_dir_config)
                if Path(metadata_dir_config).is_absolute()
                else working_dir / metadata_dir_config
            )
            final_metadata_dir = Path(metadata_dir) if metadata_dir else config_metadata_dir

        logger.debug("Output directory: %s", final_output_dir)
        if final_metadata_dir:
            logger.debug("Metadata directory: %s", final_metadata_dir)

        # ===== Validate and execute plan =====
        validate_plan(plan)
        logger.info("Simulation plan validation passed")

        if dry_run:
            logger.info("Dry run mode: Simulation plan validated but not executed")
            return

        resource_monitor = ResourceMonitor()

        with resource_monitor.measure():
            execute_plan(
                plan=plan,
                output_directory=final_output_dir,
                metadata_directory=final_metadata_dir or Path("metadata"),
                overwrite=overwrite,
                max_retries=3,
            )

        resource_monitor.log_summary(logger)

        # Write the resource usage summary to a json file.
        resource_file = working_dir / "resource_usage_summary.json"
        resource_monitor.save_metrics(resource_file, overwrite=True)

        logger.info("Simulation completed successfully. Output written to %s", final_output_dir)

    except Exception as e:
        logger.error("Simulation failed: %s", str(e), exc_info=True)
        raise typer.Exit(code=1) from e


def simulate_command(
    config_file_names: Annotated[
        list[str],
        typer.Argument(help="Configuration file (YAML) or metadata files (can specify multiple metadata files)."),
    ],
    output_dir: Annotated[
        str | None, typer.Option("--output-dir", help="Output directory (overrides config/metadata defaults).")
    ] = None,
    metadata_dir: Annotated[
        str | None,
        typer.Option(
            "--metadata-dir", help="Metadata directory (overrides config/metadata defaults, only for config mode)."
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing files.")] = False,
    metadata: Annotated[bool, typer.Option("--metadata", help="Generate metadata files (only in config mode).")] = True,
    author: Annotated[str | None, typer.Option("--author", help="Author name for the simulation.")] = None,
    email: Annotated[str | None, typer.Option("--email", help="Author email for the simulation.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate the plan without executing.")] = False,
) -> None:
    """Generate gravitational wave simulation data using specified simulators.

    This command can run in two modes, automatically detected from the input:

    1. **Config Mode**: Provide a single YAML configuration file to create a simulation plan.
       Executes all simulators with state tracking for reproducibility.
       Example:

         `gwsim simulate config.yaml`

    2. **Reproduction Mode**: Provide one or more metadata files to reproduce specific batches.
       Each metadata file contains the exact configuration and pre-batch state needed for
       exact reproducibility. Users can distribute individual metadata files, and anyone
       can reproduce those specific batches independently.
       Example:

         `gwsim simulate signal-0.metadata.yaml signal-1.metadata.yaml`

    **Path Overrides:**

    - Use `--output-dir` to specify where output files should be saved
    - Use `--metadata-dir` to specify where metadata should be saved (config mode only)
    - These override paths from the configuration or metadata files

    Args:
        config_file_names: Path to YAML config file OR one or more metadata files
        output_dir: Output directory override
        metadata_dir: Metadata directory override (config mode only)
        overwrite: Whether to overwrite existing files
        metadata: Whether to save metadata files (config mode only)

    Returns:
        None
    """
    _simulate_impl(
        config_file_names=config_file_names,
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        overwrite=overwrite,
        metadata=metadata,
        author=author,
        email=email,
        dry_run=dry_run,
    )
