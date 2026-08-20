"""The Zenodo repository commands: what reaches the client, and what the exit status is.

The whole ``gwmock repository`` subtree had no test at all. These commands are the only way a user
publishes a dataset, and every one of them is a thin layer over ``ZenodoClient`` whose job is to
pick the right host and token, decide what to send, and turn a failure into a non-zero exit status.
The rendered console output is deliberately not asserted line by line -- only the parts a script
or a user acts on: the exit code, whether the call happened at all, and the arguments it carried.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typer
import yaml
from typer.testing import CliRunner

from gwmock.cli.repository.main import repository_app
from gwmock.cli.repository.utils import get_zenodo_client

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture
def client(mocker):
    """Replace the Zenodo client factory, and hand back the client the commands will get."""
    fake = MagicMock()
    fake.create_deposition.return_value = {"id": 123}
    fake.publish_deposition.return_value = {"doi": "10.5281/zenodo.123"}
    fake.list_depositions.return_value = []
    mocker.patch("gwmock.cli.repository.utils.get_zenodo_client", return_value=fake)
    return fake


@pytest.fixture(autouse=True)
def _no_inherited_tokens(monkeypatch):
    """The commands read the token from the environment, which the test host may well have set."""
    monkeypatch.delenv("ZENODO_API_TOKEN", raising=False)
    monkeypatch.delenv("ZENODO_SANDBOX_API_TOKEN", raising=False)


class TestTheClientFactory:
    def test_an_explicit_token_is_used_as_is(self):
        built = get_zenodo_client(token="explicit")  # noqa: S106
        assert built.access_token == "explicit"  # noqa: S105
        assert built.sandbox is False

    def test_production_reads_the_production_variable(self, monkeypatch):
        monkeypatch.setenv("ZENODO_API_TOKEN", "prod")
        assert get_zenodo_client().access_token == "prod"  # noqa: S105

    def test_the_sandbox_reads_its_own_variable(self, monkeypatch):
        """Two hosts, two tokens: a production token sent to the sandbox is simply invalid, so the
        variables must not be interchangeable."""
        monkeypatch.setenv("ZENODO_API_TOKEN", "prod")
        monkeypatch.setenv("ZENODO_SANDBOX_API_TOKEN", "sand")
        assert get_zenodo_client(sandbox=True).access_token == "sand"  # noqa: S105
        assert get_zenodo_client(sandbox=True).base_url == "https://sandbox.zenodo.org/api/"

    def test_a_production_token_does_not_stand_in_for_a_sandbox_one(self, monkeypatch):
        monkeypatch.setenv("ZENODO_API_TOKEN", "prod")
        with pytest.raises(typer.Exit):
            get_zenodo_client(sandbox=True)

    def test_a_sandbox_token_does_not_stand_in_for_a_production_one(self, monkeypatch):
        monkeypatch.setenv("ZENODO_SANDBOX_API_TOKEN", "sand")
        with pytest.raises(typer.Exit):
            get_zenodo_client()

    def test_no_token_at_all_exits_non_zero(self):
        with pytest.raises(typer.Exit) as exit_info:
            get_zenodo_client()
        assert exit_info.value.exit_code == 1

    def test_an_empty_token_is_treated_as_no_token(self, monkeypatch):
        """An unset variable and one set to the empty string are the same mistake."""
        monkeypatch.setenv("ZENODO_API_TOKEN", "")
        with pytest.raises(typer.Exit):
            get_zenodo_client()

    def test_the_explicit_token_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("ZENODO_API_TOKEN", "env")
        assert get_zenodo_client(token="explicit").access_token == "explicit"  # noqa: S106,S105


class TestTheCommandsAreRegistered:
    """``register_commands`` runs at import; if a name is dropped the CLI silently loses it."""

    @pytest.mark.parametrize(
        "name", ["create", "upload", "download", "list", "delete", "verify", "publish", "metadata"]
    )
    def test_the_command_is_reachable(self, name: str) -> None:
        result = runner.invoke(repository_app, [name, "--help"])
        assert result.exit_code == 0, result.output

    def test_the_metadata_subcommand_is_reachable(self) -> None:
        result = runner.invoke(repository_app, ["metadata", "update", "--help"])
        assert result.exit_code == 0, result.output

    def test_an_unknown_command_is_refused(self) -> None:
        assert runner.invoke(repository_app, ["frobnicate"]).exit_code != 0


class TestCreate:
    def test_the_title_and_description_are_sent(self, client) -> None:
        result = runner.invoke(repository_app, ["create", "--title", "T", "--description", "D"])
        assert result.exit_code == 0, result.output
        assert client.create_deposition.call_args.kwargs["metadata"] == {"title": "T", "description": "D"}

    def test_a_missing_title_is_prompted_for(self, client) -> None:
        result = runner.invoke(repository_app, ["create", "--description", "D"], input="Prompted\n")
        assert result.exit_code == 0, result.output
        assert client.create_deposition.call_args.kwargs["metadata"]["title"] == "Prompted"

    def test_a_metadata_file_adds_its_keys(self, client, tmp_path: Path) -> None:
        path = tmp_path / "metadata.yaml"
        path.write_text(yaml.safe_dump({"creators": [{"name": "A"}], "upload_type": "dataset"}))

        result = runner.invoke(
            repository_app, ["create", "--title", "T", "--description", "D", "--metadata-file", str(path)]
        )

        assert result.exit_code == 0, result.output
        sent = client.create_deposition.call_args.kwargs["metadata"]
        assert sent["upload_type"] == "dataset"
        assert sent["title"] == "T"

    def test_a_metadata_file_overrides_the_options(self, client, tmp_path: Path) -> None:
        """The file is merged last on purpose: it is the fuller description of the deposition."""
        path = tmp_path / "metadata.yaml"
        path.write_text(yaml.safe_dump({"title": "From the file"}))

        runner.invoke(repository_app, ["create", "--title", "T", "--description", "D", "--metadata-file", str(path)])

        assert client.create_deposition.call_args.kwargs["metadata"]["title"] == "From the file"

    def test_an_empty_metadata_file_changes_nothing(self, client, tmp_path: Path) -> None:
        path = tmp_path / "metadata.yaml"
        path.write_text("")
        result = runner.invoke(
            repository_app, ["create", "--title", "T", "--description", "D", "--metadata-file", str(path)]
        )
        assert result.exit_code == 0, result.output
        assert client.create_deposition.call_args.kwargs["metadata"] == {"title": "T", "description": "D"}

    def test_a_missing_metadata_file_exits_non_zero_without_creating_anything(self, client, tmp_path: Path) -> None:
        result = runner.invoke(
            repository_app,
            ["create", "--title", "T", "--description", "D", "--metadata-file", str(tmp_path / "nope.yaml")],
        )
        assert result.exit_code == 1
        client.create_deposition.assert_not_called()

    def test_a_failure_from_zenodo_exits_non_zero(self, client) -> None:
        client.create_deposition.side_effect = RuntimeError("no")
        result = runner.invoke(repository_app, ["create", "--title", "T", "--description", "D"])
        assert result.exit_code == 1

    def test_the_new_id_is_reported(self, client) -> None:
        """The id is what the next command needs, so it has to reach stdout."""
        result = runner.invoke(repository_app, ["create", "--title", "T", "--description", "D"])
        assert "123" in result.output


class TestUpload:
    def test_each_file_is_uploaded_to_the_deposition(self, client, tmp_path: Path) -> None:
        first, second = tmp_path / "a.gwf", tmp_path / "b.gwf"
        first.write_bytes(b"a")
        second.write_bytes(b"b")

        result = runner.invoke(repository_app, ["upload", "123", "--file", str(first), "--file", str(second)])

        assert result.exit_code == 0, result.output
        assert [call.args[0] for call in client.upload_file.call_args_list] == ["123", "123"]
        assert [Path(call.args[1]).name for call in client.upload_file.call_args_list] == ["a.gwf", "b.gwf"]

    def test_the_size_derived_timeout_is_asked_for(self, client, tmp_path: Path) -> None:
        """A deposited frame file is large; without this the client's 300 s default cuts it off."""
        path = tmp_path / "a.gwf"
        path.write_bytes(b"a")
        runner.invoke(repository_app, ["upload", "123", "--file", str(path)])
        assert client.upload_file.call_args.kwargs["auto_timeout"] is True

    def test_no_files_is_an_error_rather_than_a_no_op(self, client) -> None:
        result = runner.invoke(repository_app, ["upload", "123"])
        assert result.exit_code == 1
        client.upload_file.assert_not_called()

    def test_a_missing_file_exits_non_zero(self, client, tmp_path: Path) -> None:
        result = runner.invoke(repository_app, ["upload", "123", "--file", str(tmp_path / "nope.gwf")])
        assert result.exit_code == 1
        client.upload_file.assert_not_called()

    def test_the_files_that_do_exist_are_still_uploaded(self, client, tmp_path: Path) -> None:
        """One bad path must not abandon the rest of the batch -- but the run still fails, so a
        script does not treat a partial upload as a complete one."""
        good = tmp_path / "a.gwf"
        good.write_bytes(b"a")

        result = runner.invoke(
            repository_app, ["upload", "123", "--file", str(tmp_path / "nope.gwf"), "--file", str(good)]
        )

        assert client.upload_file.call_count == 1
        assert result.exit_code == 1

    def test_a_failed_upload_exits_non_zero(self, client, tmp_path: Path) -> None:
        path = tmp_path / "a.gwf"
        path.write_bytes(b"a")
        client.upload_file.side_effect = RuntimeError("rejected")
        assert runner.invoke(repository_app, ["upload", "123", "--file", str(path)]).exit_code == 1


class TestDownload:
    def test_an_explicit_output_path_is_used(self, client, tmp_path: Path) -> None:
        """The regression: with the branches the wrong way round, passing ``--output`` -- the form
        the command's own help shows -- refused with "Output path must be specified"."""
        output = tmp_path / "saved.gwf"
        result = runner.invoke(repository_app, ["download", "123", "--file", "data.gwf", "--output", str(output)])
        assert result.exit_code == 0, result.output
        assert Path(client.download_file.call_args.args[2]) == output

    def test_without_an_output_the_file_name_is_used(self, client) -> None:
        result = runner.invoke(repository_app, ["download", "123", "--file", "data.gwf"])
        assert result.exit_code == 0, result.output
        assert Path(client.download_file.call_args.args[2]) == Path("data.gwf")

    def test_the_deposition_and_file_are_passed_through(self, client, tmp_path: Path) -> None:
        runner.invoke(repository_app, ["download", "123", "--file", "data.gwf", "--output", str(tmp_path / "o.gwf")])
        assert client.download_file.call_args.args[0] == "123"
        assert client.download_file.call_args.args[1] == "data.gwf"

    def test_a_declared_size_is_passed_through_for_the_timeout(self, client, tmp_path: Path) -> None:
        runner.invoke(
            repository_app,
            ["download", "123", "--file", "d.gwf", "--output", str(tmp_path / "o"), "--file-size-mb", "50"],
        )
        assert client.download_file.call_args.kwargs["file_size_in_mb"] == 50

    def test_the_deposition_is_prompted_for_when_omitted(self, client) -> None:
        result = runner.invoke(repository_app, ["download", "--file", "data.gwf"], input="9876\n")
        assert result.exit_code == 0, result.output
        assert client.download_file.call_args.args[0] == "9876"

    def test_a_failed_download_exits_non_zero(self, client, tmp_path: Path) -> None:
        client.download_file.side_effect = RuntimeError("404")
        result = runner.invoke(
            repository_app, ["download", "123", "--file", "d.gwf", "--output", str(tmp_path / "o.gwf")]
        )
        assert result.exit_code == 1


class TestDelete:
    def test_a_refused_confirmation_deletes_nothing_and_succeeds(self, client) -> None:
        """Declining is not an error: the exit status must stay zero so a script can offer the
        prompt without treating "no" as a failure."""
        result = runner.invoke(repository_app, ["delete", "123"], input="n\n")
        assert result.exit_code == 0, result.output
        client.delete_deposition.assert_not_called()

    def test_a_confirmed_delete_goes_through(self, client) -> None:
        result = runner.invoke(repository_app, ["delete", "123"], input="y\n")
        assert result.exit_code == 0, result.output
        client.delete_deposition.assert_called_once_with("123")

    def test_the_default_answer_is_no(self, client) -> None:
        """An empty answer at the prompt must not delete anything."""
        runner.invoke(repository_app, ["delete", "123"], input="\n")
        client.delete_deposition.assert_not_called()

    def test_force_skips_the_prompt(self, client) -> None:
        result = runner.invoke(repository_app, ["delete", "123", "--force"])
        assert result.exit_code == 0, result.output
        client.delete_deposition.assert_called_once_with("123")

    def test_a_failed_delete_exits_non_zero(self, client) -> None:
        client.delete_deposition.side_effect = RuntimeError("published")
        assert runner.invoke(repository_app, ["delete", "123", "--force"]).exit_code == 1


class TestPublish:
    def test_a_refused_confirmation_publishes_nothing_and_succeeds(self, client) -> None:
        result = runner.invoke(repository_app, ["publish", "123"], input="n\n")
        assert result.exit_code == 0, result.output
        client.publish_deposition.assert_not_called()

    def test_a_confirmed_publish_goes_through_and_reports_the_doi(self, client) -> None:
        """Publishing is irreversible and mints the DOI, which is the one thing the user needs."""
        result = runner.invoke(repository_app, ["publish", "123"], input="y\n")
        assert result.exit_code == 0, result.output
        client.publish_deposition.assert_called_once_with("123")
        assert "10.5281/zenodo.123" in result.output

    def test_a_failed_publish_exits_non_zero(self, client) -> None:
        client.publish_deposition.side_effect = RuntimeError("incomplete")
        assert runner.invoke(repository_app, ["publish", "123"], input="y\n").exit_code == 1


class TestVerify:
    def test_a_working_token_reports_the_draft_count(self, client) -> None:
        client.list_depositions.return_value = [{"id": 1}, {"id": 2}]
        result = runner.invoke(repository_app, ["verify"])
        assert result.exit_code == 0, result.output
        assert client.list_depositions.call_args.kwargs["status"] == "draft"
        assert "2" in result.output

    def test_a_rejected_token_exits_non_zero(self, client) -> None:
        client.list_depositions.side_effect = RuntimeError("401")
        assert runner.invoke(repository_app, ["verify"]).exit_code == 1

    def test_a_missing_token_exits_non_zero(self, mocker) -> None:
        """The factory's own ``typer.Exit`` must not be swallowed into a success."""
        assert runner.invoke(repository_app, ["verify"]).exit_code == 1


class TestListing:
    def test_the_status_filter_is_passed_through(self, client) -> None:
        result = runner.invoke(repository_app, ["list", "--status", "draft"])
        assert result.exit_code == 0, result.output
        assert client.list_depositions.call_args.kwargs["status"] == "draft"

    def test_it_lists_published_depositions_by_default(self, client) -> None:
        runner.invoke(repository_app, ["list"])
        assert client.list_depositions.call_args.kwargs["status"] == "published"

    def test_an_empty_list_is_reported_rather_than_an_empty_table(self, client) -> None:
        client.list_depositions.return_value = []
        result = runner.invoke(repository_app, ["list"])
        assert result.exit_code == 0
        assert "No published depositions found" in result.output

    def test_a_deposition_is_shown_with_its_id_and_doi(self, client) -> None:
        client.list_depositions.return_value = [
            {"id": 555, "doi": "10.5281/zenodo.555", "created": "2026-01-02T03:04:05", "metadata": {"title": "Run A"}}
        ]
        result = runner.invoke(repository_app, ["list"])
        assert "555" in result.output
        assert "Run A" in result.output

    def test_a_deposition_missing_its_fields_does_not_crash_the_listing(self, client) -> None:
        client.list_depositions.return_value = [{"created": "2026-01-02T03:04:05"}]
        assert runner.invoke(repository_app, ["list"]).exit_code == 0

    def test_a_long_title_is_shortened_to_fit_the_column(self, client) -> None:
        client.list_depositions.return_value = [
            {"id": 1, "doi": "d", "created": "2026-01-02T03:04:05", "metadata": {"title": "T" * 60}}
        ]
        result = runner.invoke(repository_app, ["list"])
        assert "..." in result.output

    def test_a_failure_exits_non_zero(self, client) -> None:
        client.list_depositions.side_effect = RuntimeError("500")
        assert runner.invoke(repository_app, ["list"]).exit_code == 1


class TestMetadataUpdate:
    def test_the_parsed_metadata_is_sent(self, client, tmp_path: Path) -> None:
        path = tmp_path / "metadata.yaml"
        path.write_text(yaml.safe_dump({"title": "T", "upload_type": "dataset"}))

        result = runner.invoke(repository_app, ["metadata", "update", "123", "--metadata-file", str(path)])

        assert result.exit_code == 0, result.output
        client.update_metadata.assert_called_once_with("123", {"title": "T", "upload_type": "dataset"})

    def test_a_missing_file_exits_non_zero_without_sending_anything(self, client, tmp_path: Path) -> None:
        result = runner.invoke(
            repository_app, ["metadata", "update", "123", "--metadata-file", str(tmp_path / "nope.yaml")]
        )
        assert result.exit_code == 1
        client.update_metadata.assert_not_called()

    def test_an_empty_file_is_refused_rather_than_wiping_the_metadata(self, client, tmp_path: Path) -> None:
        """``update_metadata`` with an empty block would replace the deposition's metadata."""
        path = tmp_path / "metadata.yaml"
        path.write_text("")
        result = runner.invoke(repository_app, ["metadata", "update", "123", "--metadata-file", str(path)])
        assert result.exit_code == 1
        client.update_metadata.assert_not_called()

    def test_the_file_is_prompted_for_when_omitted(self, client, tmp_path: Path) -> None:
        path = tmp_path / "metadata.yaml"
        path.write_text(yaml.safe_dump({"title": "T"}))
        result = runner.invoke(repository_app, ["metadata", "update", "123"], input=f"{path}\n")
        assert result.exit_code == 0, result.output
        client.update_metadata.assert_called_once_with("123", {"title": "T"})

    def test_a_failed_update_exits_non_zero(self, client, tmp_path: Path) -> None:
        path = tmp_path / "metadata.yaml"
        path.write_text(yaml.safe_dump({"title": "T"}))
        client.update_metadata.side_effect = RuntimeError("400")
        result = runner.invoke(repository_app, ["metadata", "update", "123", "--metadata-file", str(path)])
        assert result.exit_code == 1
