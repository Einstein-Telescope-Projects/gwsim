# ruff: noqa PLC0415

"""
Command-line tool for creating and submitting batch jobs (Slurm) for gwsim simulations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

SchedulerType = Literal["slurm"]

# pylint: disable=no-member


def _batch_command_impl(
    config: Path | None,
    get: Path | None,
    scheduler: SchedulerType,
    job_name: str,
    account: str | None,
    cluster: str | None,
    time: str | None,
    extra_lines: list[str] | None,
    submit: bool,
    output: Path,
    overwrite: bool,
) -> None:
    """Internal implementation of the batch command."""

    import logging

    from gwsim.cli.utils.config import (
        get_examples_dir,
        load_config,
        save_config,
    )

    def copy_and_update_config_file(
        src_path: Path,
        dst_path: Path,
        scheduler: SchedulerType,
        job_name: str,
        account: str | None,
        cluster: str | None,
        time: str | None,
        extra_lines: list[str] | None,
        overwrite: bool,
    ) -> None:
        """Copy a configuration file from src_path to dst_path,
        adding the batch section with default resources and chosen scheduler.

        Args:
            src_path: Source file path.
            dst_path: Destination file path.
            scheduler: Name of the scheduler (only `slurm` currently supported).
            job_name: Name of the job.
            account: Cluster account to charge.
            cluster: Cluster or partition to run on.
            time: Wall time limit for the job (e.g. "04:00:00" or "2-12:30:00").
            extra_lines: One or more custom shell lines to insert into the submit script before the simulation command.
            overwrite: Whether to overwrite existing files.
        """
        from gwsim.cli.utils.config import Config  # pylint: disable=import-outside-toplevel

        # Load the configuration file
        config = load_config(src_path)

        # Convert to dict
        config_dict = config.model_dump(by_alias=True, exclude_none=True)

        # Add/override the batch section with default resources and chosen scheduler
        batch_section = {
            "scheduler": scheduler,
            "job-name": job_name,
            "resources": {
                "nodes": 1,
                "ntasks-per-node": 1,
                "cpus-per-task": 1,
                "mem": "16GB",
            },
        }

        submit_options = {}
        if account:
            submit_options["account"] = account
        if cluster:
            submit_options["cluster"] = cluster
        if time:
            submit_options["time"] = time

        if submit_options:
            batch_section["submit"] = submit_options

        if extra_lines:
            batch_section["extra_lines"] = extra_lines

        config_dict["batch"] = batch_section
        updated_config = Config(**config_dict)

        # Create parent directories if they don't exist
        dst_path.resolve().parent.mkdir(parents=True, exist_ok=True)

        # Save the configuration file to the destination
        try:
            save_config(file_name=dst_path, config=updated_config, overwrite=overwrite, backup=False)
        except FileExistsError as e:
            raise FileExistsError(f"File already exists: {dst_path}. Use --overwrite to overwrite.") from e

    logger = logging.getLogger("gwsim")

    examples_dir = get_examples_dir()

    # Get option
    if get is not None:
        src_path = examples_dir / get / "config.yaml"
        if not src_path.exists():
            logger.error("Example configuration '%s' does not exist in examples directory.", get)
            raise typer.Exit(1)
        if output.is_dir():
            dst_path = output / "config.yaml"
        else:
            dst_path = output
        copy_and_update_config_file(
            src_path=src_path,
            dst_path=dst_path,
            scheduler=scheduler,
            job_name=job_name,
            account=account,
            cluster=cluster,
            time=time,
            extra_lines=extra_lines,
            overwrite=overwrite,
        )
        logger.info("Copied and prepared batch-ready config from example '%s' to %s", get, dst_path)
        logger.info("Scheduler set to: %s", scheduler)
        return

    # Regular case: use provided config.yaml to generate submit file
    if not config.exists():
        logger.error("Configuration file not found: %s", config)
        raise typer.Exit(1)

    config_data = load_config(config)

    if not config_data.batch:
        logger.error("Configuration file is missing a 'batch' section required for batch submission.")
        logger.error("Use 'gwsim batch --get <example>' to create a batch-ready config first.")
        raise typer.Exit(1)

    batch_section = config_data.batch

    scheduler = batch_section.scheduler
    if scheduler not in ["slurm"]:
        logger.error("Invalid or missing scheduler in 'batch.scheduler'. Must be 'slurm'.")
        raise typer.Exit(1)

    job_name = batch_section.job_name

    working_dir = Path(config_data.globals.working_directory)
    job_dir = working_dir / scheduler
    job_dir.mkdir(parents=True, exist_ok=True)

    out_dir = job_dir / "output"
    err_dir = job_dir / "error"
    submit_dir = job_dir / "submit"
    out_dir.mkdir(parents=True, exist_ok=True)
    err_dir.mkdir(parents=True, exist_ok=True)
    submit_dir.mkdir(parents=True, exist_ok=True)

    submit_file = submit_dir / f"{job_name}.submit"

    # Generate submit file content
    if scheduler == "slurm":
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={out_dir}/{job_name}.output",
            f"#SBATCH --error={err_dir}/{job_name}.error",
            "",
        ]
        resources = batch_section.resources
        for key, val in resources.items():
            lines.append(f"#SBATCH --{key}={val}")

        if batch_section.submit:
            submit_options = batch_section.submit
            for key, val in submit_options.items():
                lines.append(f"#SBATCH --{key}={val}")

        lines.append("")

        # Add user-defined extra lines (e.g. conda setup)
        if batch_section.extra_lines:
            lines.extend(batch_section.extra_lines)
            lines.append("")

        lines.append(f"gwsim simulate {config.resolve()}")

    content = "\n".join(lines) + "\n"

    if submit_file.exists() and not overwrite:
        raise FileExistsError(f"Submit file already exists: {submit_file}. Use --overwrite to overwrite.")
    submit_file.write_text(content)
    logger.info("Generated SLURM submit file: %s", submit_file)

    # Optional detailed logging
    logger.info("Generated %s submit file: %s", scheduler.upper(), submit_file)
    logger.info("\tscheduler: %s", scheduler)
    logger.info("\tjob-name: %s", job_name)
    if batch_section.submit:
        for key, val in submit_options.items():
            logger.info("\t%s: %s", key, val)
    for key, val in resources.items():
        logger.info("\t%s: %s", key, val)
    if batch_section.extra_lines:
        for line in batch_section.extra_lines:
            logger.info("\textra-line: %s", line)

    # Submit file
    if submit:
        import subprocess  # pylint: disable=import-outside-toplevel    # nosec B404 # B404: subprocess import is required for job submission to Slurm

        result = subprocess.run(  # pylint: disable=line-too-long # nosec B603 B607     # B603/B607: sbatch is a trusted system command     # submit_file path is generated and controlled by this tool — no injection risk
            ["sbatch", str(submit_file)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            logger.info("Job submitted successfully.")
            logger.info(result.stdout.strip())
        else:
            logger.error("Failed to submit job.")
            logger.error(result.stderr.strip())
            raise typer.Exit(1)
    else:
        logger.info("Job prepared. Use --submit to send it to the cluster.")


def batch_command(
    config: Annotated[
        Path | None,
        typer.Argument(
            help="Path to the simulation configuration file (config.yaml). Required when not using --get.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    get: Annotated[
        Path | None,
        typer.Option("--get", help="Label of the example config to copy and make batch-ready"),
    ] = None,
    scheduler: Annotated[
        SchedulerType,
        typer.Option("--scheduler", "-s", help="Batch system (only 'slurm' supported)"),
    ] = "slurm",
    job_name: Annotated[
        str,
        typer.Option("--job-name", "-j", help="Job name to set in batch section (only used with --get)"),
    ] = "gwsim_job",
    account: Annotated[
        str | None,
        typer.Option("--account", help="Cluster account (only with --get)"),
    ] = None,
    cluster: Annotated[
        str | None,
        typer.Option("--cluster", help="Cluster partition (only with --get)"),
    ] = None,
    time: Annotated[
        str | None,
        typer.Option("--time", "-t", help="Wall time limit, e.g. 12:00:00 (only with --get)"),
    ] = None,
    extra_lines: Annotated[
        list[str] | None,
        typer.Option(
            "--extra-line",
            help="Add custom lines to submit script before simulation command (e.g. environment setup)."
            "Can be used multiple times. (only with --get)",
        ),
    ] = None,
    submit: Annotated[
        bool,
        typer.Option(
            "--submit",
            help="Submit the job immediately after creating the submit file.",
        ),
    ] = False,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output file or directory for generated/copied config files (default: current directory)",
        ),
    ] = Path("config.yaml"),
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing files")] = False,
) -> None:
    """
    Prepare and optionally submit gwsim simulations as Slurm batch jobs.

    Two mutually exclusive modes:

    1. Create a batch-ready config from an example:

        gwsim batch --get default_config \\
         --job-name my_simulation \\
         --account my_account_name \\
         --time 08:00:00 \\
         --extra-line "module load Python/3.11" \\
         --extra-line "conda activate gwsim" \\
         --output my_config.yaml

       This copies examples/<label>/config.yaml and adds a complete `batch` section including:

         - batch.scheduler: slurm
         - batch.job-name: <your name>
         - batch.resources: default values
         - batch.submit: account, cluster, time (if provided)
         - batch.extra_lines: custom lines (if provided)

    2. Generate and optionally submit a Slurm job from an existing config:

       gwsim batch config.yaml
       gwsim batch config.yaml --submit

       The config must contain a valid `batch` section with at least:

         - batch.scheduler: slurm
         - batch.job-name: <name>
         - globals.working-directory

       Optional fields:

         - batch.resources: resource requests (nodes, mem, etc.)
         - batch.submit: additional sbatch options (account, cluster, time,  etc.)
         - batch.extra_lines: custom shell lines (environment setup, modules, etc.)

    Args:
        config: Path to the input config.yaml (ignored when --get is used).
        get: Label of example config to copy (triggers config creation mode).
        scheduler: Name of the scheduler (only `slurm` currently supported). Only allowed with --get.
        job_name: Name of the job. Only allowed with --get.
        account: Cluster account to charge (passed via --account to sbatch and stored in batch.submit).
            Only allowed with --get.
        cluster: Cluster or partition to run on (passed via --cluster to sbatch and stored in batch.submit).
            Only allowed with --get.
        time: Wall time limit for the job (e.g. "04:00:00" or "2-12:30:00").
            Passed via --time to sbatch and stored in batch.submit. Only allowed with --get.
        extra_lines: One or more custom shell lines to insert into the submit script before the simulation command
            (e.g. module loads, conda activate). Specified via --extra-line (can be repeated).
            Stored in batch.extra_lines. Only allowed with --get.
        submit: If True, submit the job via sbatch after creating the submit file.
        output: Destination path for the new config file. Only allowed with --get.
        overwrite: Overwrite existing files if they exist.
    """

    if get is not None:
        if config is not None:
            typer.echo("Error: Cannot provide a config file when using --get.", err=True)
            raise typer.Exit(1)

    if get is None:
        if config is None:
            typer.echo("Error: You must either provide a config file or use --get.", err=True)
            raise typer.Exit(1)
        if any(
            [
                scheduler != "slurm",
                job_name != "gwsim_job",
                output != Path("config.yaml"),
                account is not None,
                cluster is not None,
                time is not None,
                extra_lines is not None,
            ]
        ):
            typer.echo(
                "Error: --scheduler, --job-name, --output, --account, --cluster, --time, --extra-line can only be used with --get.",  # pylint: disable=line-too-long
                err=True,
            )
            raise typer.Exit(1)

    _batch_command_impl(
        config=config,
        get=get,
        scheduler=scheduler,
        job_name=job_name,
        account=account,
        cluster=cluster,
        time=time,
        extra_lines=extra_lines,
        submit=submit,
        output=output,
        overwrite=overwrite,
    )
