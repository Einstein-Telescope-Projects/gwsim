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


# --- the exact text a scheduler is handed ----------------------------------


class TestTheDefaultResourceRequest:
    """The defaults are what a user gets when the config names no resources.

    Both dialects are pinned key by key and value by value: an HTCondor submit file with a
    misspelled ``request_memory`` is accepted by the scheduler and silently gets the pool default
    instead, so a wrong key here is not a syntax error anywhere -- it is a job that runs with the
    wrong memory.
    """

    def test_the_htcondor_request_block(self):
        assert _default_resources("htcondor") == {
            "request_cpus": 1,
            "request_memory": "16GB",
            "request_disk": "4GB",
        }

    def test_the_slurm_request_block(self):
        assert _default_resources("slurm") == {
            "nodes": 1,
            "ntasks-per-node": 1,
            "cpus-per-task": 1,
            "mem": "16GB",
        }

    def test_anything_that_is_not_htcondor_gets_the_slurm_dialect(self):
        """The check is on ``htcondor`` alone, so slurm is the fallback rather than a second case."""
        assert _default_resources("slurm") == _default_resources("anything-else")


class TestTheGeneratedSlurmScript:
    def test_the_whole_script_line_by_line(self):
        """Whitespace and order are part of the file: ``#SBATCH`` directives are only read from the
        header block, before the first command, so a blank line inserted or dropped in the wrong
        place silently stops the rest of them being applied."""
        content = _slurm_submit_content(
            job_name="job1",
            out_dir=Path("/w/output"),
            err_dir=Path("/w/error"),
            resources={"mem": "16GB"},
            submit_options=None,
            extra_lines=None,
            config_path=Path("config.yaml"),
        )
        assert content.splitlines() == [
            "#!/bin/bash",
            "#SBATCH --job-name=job1",
            "#SBATCH --output=/w/output/job1.output",
            "#SBATCH --error=/w/error/job1.error",
            "",
            "#SBATCH --mem=16GB",
            "",
            f"gwmock simulate {Path('config.yaml').resolve()}",
        ]

    def test_the_file_ends_with_a_newline(self):
        """A shell script whose last line is unterminated is a portability trap."""
        content = _slurm_submit_content(
            job_name="j",
            out_dir=Path("/w"),
            err_dir=Path("/w"),
            resources={},
            submit_options=None,
            extra_lines=None,
            config_path=Path("config.yaml"),
        )
        assert content.endswith("\n")

    def test_extra_lines_sit_between_the_directives_and_the_command(self):
        """Environment setup has to run before the simulation and after the header."""
        content = _slurm_submit_content(
            job_name="j",
            out_dir=Path("/w"),
            err_dir=Path("/w"),
            resources={},
            submit_options=None,
            extra_lines=["module load Python/3.12", "source venv/bin/activate"],
            config_path=Path("config.yaml"),
        )
        lines = content.splitlines()
        assert lines[-4:] == [
            "module load Python/3.12",
            "source venv/bin/activate",
            "",
            f"gwmock simulate {Path('config.yaml').resolve()}",
        ]

    def test_submit_options_follow_the_resources(self):
        content = _slurm_submit_content(
            job_name="j",
            out_dir=Path("/w"),
            err_dir=Path("/w"),
            resources={"mem": "16GB"},
            submit_options={"account": "acct", "time": "01:00:00"},
            extra_lines=None,
            config_path=Path("config.yaml"),
        )
        lines = content.splitlines()
        assert lines[5:8] == ["#SBATCH --mem=16GB", "#SBATCH --account=acct", "#SBATCH --time=01:00:00"]

    def test_the_config_path_is_absolute(self):
        """The job runs from whatever directory the scheduler chooses, so a relative path would
        resolve against the wrong root on the execute node."""
        content = _slurm_submit_content(
            job_name="j",
            out_dir=Path("/w"),
            err_dir=Path("/w"),
            resources={},
            submit_options=None,
            extra_lines=None,
            config_path=Path("config.yaml"),
        )
        assert content.splitlines()[-1] == f"gwmock simulate {Path('config.yaml').resolve()}"


class TestTheGeneratedHtcondorFiles:
    def test_the_whole_submit_description_line_by_line(self):
        content = _htcondor_submit_content(
            job_name="job1",
            wrapper_path=Path("/w/submit/job1.sh"),
            out_dir=Path("/w/output"),
            err_dir=Path("/w/error"),
            log_dir=Path("/w/log"),
            resources={"request_cpus": 1},
            submit_options=None,
        )
        assert content.splitlines() == [
            "universe = vanilla",
            "executable = /w/submit/job1.sh",
            "output = /w/output/job1.output",
            "error = /w/error/job1.error",
            "log = /w/log/job1.log",
            "request_cpus = 1",
            "queue",
        ]

    def test_submit_options_are_emitted_after_the_resources_and_before_queue(self):
        """``queue`` ends the description: anything after it is not part of the job."""
        content = _htcondor_submit_content(
            job_name="j",
            wrapper_path=Path("/w/j.sh"),
            out_dir=Path("/w"),
            err_dir=Path("/w"),
            log_dir=Path("/w"),
            resources={"request_cpus": 1},
            submit_options={"accounting_group": "ligo.dev"},
        )
        assert content.splitlines()[-3:] == ["request_cpus = 1", "accounting_group = ligo.dev", "queue"]

    def test_the_description_ends_with_a_newline(self):
        content = _htcondor_submit_content(
            job_name="j",
            wrapper_path=Path("/w/j.sh"),
            out_dir=Path("/w"),
            err_dir=Path("/w"),
            log_dir=Path("/w"),
            resources={},
            submit_options=None,
        )
        assert content.endswith("queue\n")

    def test_the_wrapper_is_a_shell_script_ending_in_the_simulation(self):
        content = _htcondor_wrapper_content(extra_lines=["conda activate gwmock"], config_path=Path("config.yaml"))
        assert content.splitlines() == [
            "#!/bin/bash",
            "conda activate gwmock",
            f"gwmock simulate {Path('config.yaml').resolve()}",
        ]

    def test_a_wrapper_without_setup_is_just_the_command(self):
        content = _htcondor_wrapper_content(extra_lines=None, config_path=Path("config.yaml"))
        assert content.splitlines() == ["#!/bin/bash", f"gwmock simulate {Path('config.yaml').resolve()}"]


class TestPreparingAConfigFromAnExample:
    def test_the_batch_section_names_the_scheduler_the_job_and_the_resources(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        _batch_command_impl(
            config=None,
            get=Path("default_config"),
            scheduler="htcondor",
            job_name="my_job",
            account=None,
            cluster=None,
            time=None,
            extra_lines=None,
            submit=False,
            output=cfg_path,
            overwrite=True,
        )
        batch = yaml.safe_load(cfg_path.read_text())["batch"]
        assert batch["scheduler"] == "htcondor"
        assert batch["job-name"] == "my_job"
        assert batch["resources"] == _default_resources("htcondor")

    def test_nothing_optional_is_invented(self, tmp_path):
        """No account, cluster or time means no ``submit`` block at all, rather than an empty one
        that would emit ``#SBATCH --account=None``."""
        cfg_path = tmp_path / "config.yaml"
        _batch_command_impl(
            config=None,
            get=Path("default_config"),
            scheduler="slurm",
            job_name="j",
            account=None,
            cluster=None,
            time=None,
            extra_lines=None,
            submit=False,
            output=cfg_path,
            overwrite=True,
        )
        batch = yaml.safe_load(cfg_path.read_text())["batch"]
        assert "submit" not in batch
        assert "extra_lines" not in batch

    def test_the_submit_block_carries_what_was_given(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        _batch_command_impl(
            config=None,
            get=Path("default_config"),
            scheduler="slurm",
            job_name="j",
            account="acct",
            cluster="part",
            time="04:00:00",
            extra_lines=["module load Python/3.12"],
            submit=False,
            output=cfg_path,
            overwrite=True,
        )
        batch = yaml.safe_load(cfg_path.read_text())["batch"]
        assert batch["submit"] == {"account": "acct", "cluster": "part", "time": "04:00:00"}
        assert batch["extra_lines"] == ["module load Python/3.12"]

    def test_the_copied_config_is_written_with_the_aliases_the_loader_reads(self, tmp_path):
        """The example is dumped by alias, so the file it writes is a file it can read back."""
        cfg_path = tmp_path / "config.yaml"
        _batch_command_impl(
            config=None,
            get=Path("default_config"),
            scheduler="slurm",
            job_name="j",
            account=None,
            cluster=None,
            time=None,
            extra_lines=None,
            submit=False,
            output=cfg_path,
            overwrite=True,
        )
        from gwmock.cli.utils.config import load_config

        assert "globals" in yaml.safe_load(cfg_path.read_text())
        assert load_config(cfg_path).batch.job_name == "j"

    def test_a_directory_output_gets_a_config_yaml_inside_it(self, tmp_path):
        out_dir = tmp_path / "prepared"
        out_dir.mkdir()
        _batch_command_impl(
            config=None,
            get=Path("default_config"),
            scheduler="slurm",
            job_name="j",
            account=None,
            cluster=None,
            time=None,
            extra_lines=None,
            submit=False,
            output=out_dir,
            overwrite=True,
        )
        assert (out_dir / "config.yaml").exists()

    def test_an_unknown_example_exits_non_zero(self, tmp_path):
        import typer

        with pytest.raises(typer.Exit) as exit_info:
            _batch_command_impl(
                config=None,
                get=Path("no_such_example"),
                scheduler="slurm",
                job_name="j",
                account=None,
                cluster=None,
                time=None,
                extra_lines=None,
                submit=False,
                output=tmp_path / "config.yaml",
                overwrite=True,
            )
        assert exit_info.value.exit_code == 1

    def test_an_existing_config_is_not_overwritten_silently(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("existing: true\n")
        with pytest.raises(FileExistsError, match="Use --overwrite"):
            _batch_command_impl(
                config=None,
                get=Path("default_config"),
                scheduler="slurm",
                job_name="j",
                account=None,
                cluster=None,
                time=None,
                extra_lines=None,
                submit=False,
                output=cfg_path,
                overwrite=False,
            )
        assert cfg_path.read_text() == "existing: true\n"


class TestSubmitting:
    @staticmethod
    def _result(returncode: int = 0):
        class _Result:
            pass

        result = _Result()
        result.returncode = returncode
        result.stdout = "Submitted batch job 1"
        result.stderr = "rejected"
        return result

    def test_the_submission_is_bounded_in_time_and_its_status_is_read(self, tmp_path):
        """``check=False`` plus an explicit look at the return code is what turns a rejected
        submission into this command's own error; a timeout stops a hung scheduler client from
        holding the terminal forever."""
        work_dir = tmp_path / "work"
        cfg_path = _make_batch_config(tmp_path, "slurm", work_dir)

        with patch("subprocess.run", return_value=self._result()) as run:
            _generate(cfg_path, submit=True)

        assert run.call_args.kwargs["timeout"] == 60
        assert run.call_args.kwargs["check"] is False
        assert run.call_args.kwargs["capture_output"] is True

    def test_a_rejected_submission_exits_non_zero(self, tmp_path):
        import typer

        work_dir = tmp_path / "work"
        cfg_path = _make_batch_config(tmp_path, "slurm", work_dir)

        with patch("subprocess.run", return_value=self._result(returncode=1)), pytest.raises(typer.Exit) as exit_info:
            _generate(cfg_path, submit=True)
        assert exit_info.value.exit_code == 1

    def test_a_submission_that_hangs_exits_non_zero(self, tmp_path):
        import subprocess

        import typer

        work_dir = tmp_path / "work"
        cfg_path = _make_batch_config(tmp_path, "slurm", work_dir)

        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sbatch", 60)),
            pytest.raises(typer.Exit) as exit_info,
        ):
            _generate(cfg_path, submit=True)
        assert exit_info.value.exit_code == 1

    def test_nothing_is_submitted_without_the_flag(self, tmp_path):
        work_dir = tmp_path / "work"
        cfg_path = _make_batch_config(tmp_path, "slurm", work_dir)
        with patch("subprocess.run") as run:
            _generate(cfg_path, submit=False)
        run.assert_not_called()
