"""Tests for the `gwmock batch` command, covering Slurm and HTCondor generation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from gwmock.cli.batch import (
    _batch_command_impl,
    _default_resources,
    _htcondor_submit_content,
    _htcondor_wrapper_content,
    _slurm_submit_content,
)

# --- pure content builders -------------------------------------------------


def test_default_resources_are_scheduler_native():
    """Slurm and HTCondor get resource keys in their own dialect."""
    slurm = _default_resources("slurm")
    condor = _default_resources("htcondor")
    assert "mem" in slurm
    assert "nodes" in slurm
    assert "request_cpus" in condor
    assert "request_memory" in condor
    assert "nodes" not in condor


def test_slurm_submit_content():
    """The Slurm script carries #SBATCH directives, options, extras and the command."""
    content = _slurm_submit_content(
        job_name="job1",
        out_dir=Path("/w/output"),
        err_dir=Path("/w/error"),
        resources={"mem": "16GB", "cpus-per-task": 2},
        submit_options={"account": "acct", "time": "01:00:00"},
        extra_lines=["module load Python/3.12"],
        config_path=Path("config.yaml"),
    )
    assert content.startswith("#!/bin/bash\n")
    assert "#SBATCH --job-name=job1" in content
    assert "#SBATCH --output=/w/output/job1.output" in content
    assert "#SBATCH --mem=16GB" in content
    assert "#SBATCH --account=acct" in content
    assert "module load Python/3.12" in content
    assert content.rstrip().endswith("gwmock simulate " + str(Path("config.yaml").resolve()))


def test_htcondor_wrapper_content():
    """The wrapper carries env setup then the simulate command."""
    content = _htcondor_wrapper_content(
        extra_lines=["conda activate gwmock"],
        config_path=Path("config.yaml"),
    )
    assert content.startswith("#!/bin/bash\n")
    assert "conda activate gwmock" in content
    assert content.rstrip().endswith("gwmock simulate " + str(Path("config.yaml").resolve()))
    # Wrapper must not contain HTCondor submit syntax.
    assert "universe" not in content


def test_htcondor_submit_content():
    """The submit description carries universe, executable, log, request_* and queue."""
    content = _htcondor_submit_content(
        job_name="job1",
        wrapper_path=Path("/w/submit/job1.sh"),
        out_dir=Path("/w/output"),
        err_dir=Path("/w/error"),
        log_dir=Path("/w/log"),
        resources={"request_cpus": 1, "request_memory": "16GB"},
        submit_options={"accounting_group": "ligo.dev"},
    )
    assert "universe = vanilla" in content
    assert "executable = /w/submit/job1.sh" in content
    assert "output = /w/output/job1.output" in content
    assert "log = /w/log/job1.log" in content
    assert "request_cpus = 1" in content
    assert "request_memory = 16GB" in content
    assert "accounting_group = ligo.dev" in content
    assert content.rstrip().endswith("queue")
    # Must not leak Slurm syntax.
    assert "#SBATCH" not in content


# --- end-to-end generation via --get then submit-file creation -------------


def _make_batch_config(tmp_path: Path, scheduler: str, work_dir: Path) -> Path:
    """Create a batch-ready config from the default example and point it at work_dir."""
    cfg_path = tmp_path / "config.yaml"
    _batch_command_impl(
        config=None,
        get=Path("default_config"),
        scheduler=scheduler,
        job_name="test_job",
        account=None,
        cluster=None,
        time=None,
        extra_lines=["module load Python/3.12"],
        submit=False,
        output=cfg_path,
        overwrite=True,
    )
    data = yaml.safe_load(cfg_path.read_text())
    data.setdefault("globals", {})["working-directory"] = str(work_dir)
    cfg_path.write_text(yaml.safe_dump(data))
    return cfg_path


def _generate(cfg_path: Path, *, submit: bool = False, overwrite: bool = True) -> None:
    _batch_command_impl(
        config=cfg_path,
        get=None,
        scheduler="slurm",
        job_name="gwmock_job",
        account=None,
        cluster=None,
        time=None,
        extra_lines=None,
        submit=submit,
        output=Path("config.yaml"),
        overwrite=overwrite,
    )


def test_htcondor_generation_writes_sub_and_executable_wrapper(tmp_path):
    """htcondor produces a .sub description plus an executable wrapper with the command."""
    work_dir = tmp_path / "work"
    cfg_path = _make_batch_config(tmp_path, "htcondor", work_dir)
    _generate(cfg_path)

    submit_dir = work_dir / "htcondor" / "submit"
    sub_file = submit_dir / "test_job.sub"
    wrapper = submit_dir / "test_job.sh"

    assert sub_file.exists()
    sub = sub_file.read_text()
    assert "universe = vanilla" in sub
    assert f"executable = {wrapper}" in sub
    assert "request_cpus = 1" in sub
    assert sub.rstrip().endswith("queue")

    assert wrapper.exists()
    assert os.access(wrapper, os.X_OK)
    wrapper_text = wrapper.read_text()
    assert "module load Python/3.12" in wrapper_text
    assert "gwmock simulate" in wrapper_text


def test_slurm_generation_still_writes_sbatch_script(tmp_path):
    """Regression: slurm generation is unchanged (.submit with #SBATCH, no wrapper)."""
    work_dir = tmp_path / "work"
    cfg_path = _make_batch_config(tmp_path, "slurm", work_dir)
    _generate(cfg_path)

    submit_dir = work_dir / "slurm" / "submit"
    submit_file = submit_dir / "test_job.submit"
    assert submit_file.exists()
    content = submit_file.read_text()
    assert "#SBATCH --job-name=test_job" in content
    assert "gwmock simulate" in content
    assert not (submit_dir / "test_job.sh").exists()


def test_htcondor_submit_uses_condor_submit(tmp_path):
    """--submit for htcondor invokes condor_submit on the .sub file."""
    work_dir = tmp_path / "work"
    cfg_path = _make_batch_config(tmp_path, "htcondor", work_dir)

    class _Result:
        returncode = 0
        stdout = "submitted"
        stderr = ""

    with patch("subprocess.run", return_value=_Result()) as mock_run:
        _generate(cfg_path, submit=True)

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "condor_submit"
    assert cmd[1].endswith("test_job.sub")


def test_slurm_submit_uses_sbatch(tmp_path):
    """--submit for slurm invokes sbatch on the .submit file."""
    work_dir = tmp_path / "work"
    cfg_path = _make_batch_config(tmp_path, "slurm", work_dir)

    class _Result:
        returncode = 0
        stdout = "submitted"
        stderr = ""

    with patch("subprocess.run", return_value=_Result()) as mock_run:
        _generate(cfg_path, submit=True)

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "sbatch"
    assert cmd[1].endswith("test_job.submit")


def test_existing_submit_file_without_overwrite_raises(tmp_path):
    """A pre-existing .sub is not clobbered without --overwrite."""
    work_dir = tmp_path / "work"
    cfg_path = _make_batch_config(tmp_path, "htcondor", work_dir)
    submit_dir = work_dir / "htcondor" / "submit"
    submit_dir.mkdir(parents=True)
    (submit_dir / "test_job.sub").write_text("old")
    with pytest.raises(FileExistsError, match="Submit file already exists"):
        _generate(cfg_path, overwrite=False)


def test_existing_wrapper_without_overwrite_raises(tmp_path):
    """A pre-existing HTCondor wrapper .sh is not clobbered without --overwrite."""
    work_dir = tmp_path / "work"
    cfg_path = _make_batch_config(tmp_path, "htcondor", work_dir)
    submit_dir = work_dir / "htcondor" / "submit"
    submit_dir.mkdir(parents=True)
    # Only the wrapper exists, so the .sub guard passes and the wrapper guard fires.
    (submit_dir / "test_job.sh").write_text("old")
    with pytest.raises(FileExistsError, match="Wrapper script already exists"):
        _generate(cfg_path, overwrite=False)


def test_invalid_scheduler_exits(tmp_path):
    """An unknown scheduler in the config is rejected."""
    import typer

    work_dir = tmp_path / "work"
    cfg_path = _make_batch_config(tmp_path, "slurm", work_dir)
    data = yaml.safe_load(cfg_path.read_text())
    data["batch"]["scheduler"] = "pbs"
    cfg_path.write_text(yaml.safe_dump(data))

    with pytest.raises(typer.Exit):
        _generate(cfg_path)
