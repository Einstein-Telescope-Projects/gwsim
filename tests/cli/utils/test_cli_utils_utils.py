"""The CLI helper module: template substitution, signal handling, and safe saves.

``handle_signal`` is the live one -- ``execute_plan`` installs it on SIGINT and SIGTERM so an
interrupted run still removes its checkpoints. The template helpers and ``save_file_safely`` are
part of the published surface of the package and had no test at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gwmock.cli.utils.utils import (
    get_file_name_from_template,
    get_file_name_from_template_with_dict,
    handle_signal,
    save_file_safely,
)

pytestmark = pytest.mark.unit


class TestTemplateSubstitutionFromADictionary:
    def test_every_placeholder_is_substituted(self) -> None:
        assert get_file_name_from_template_with_dict("{{ x }}-{{ y }}.txt", {"x": "a", "y": 2}) == "a-2.txt"

    def test_whitespace_inside_the_braces_is_ignored(self) -> None:
        values = {"detector": "H1"}
        assert get_file_name_from_template_with_dict("{{detector}}.gwf", values) == "H1.gwf"
        assert get_file_name_from_template_with_dict("{{   detector   }}.gwf", values) == "H1.gwf"

    def test_a_value_is_rendered_with_str(self) -> None:
        """Not formatted, not repr'd: whatever ``str`` gives, so a float keeps its own spelling."""
        assert get_file_name_from_template_with_dict("{{ n }}", {"n": 1.5}) == "1.5"

    def test_text_outside_the_placeholders_is_untouched(self) -> None:
        assert get_file_name_from_template_with_dict("out/{{ a }}/file.txt", {"a": "x"}) == "out/x/file.txt"

    def test_a_template_without_placeholders_is_returned_as_is(self) -> None:
        assert get_file_name_from_template_with_dict("plain.txt", {"a": "x"}) == "plain.txt"

    def test_a_repeated_placeholder_is_substituted_everywhere(self) -> None:
        assert get_file_name_from_template_with_dict("{{ a }}-{{ a }}", {"a": "z"}) == "z-z"

    def test_an_excluded_placeholder_is_left_for_a_later_pass(self) -> None:
        """Exclusion has to leave the *placeholder*, braces and all, not an empty string: the
        batch layer substitutes the rest of them later."""
        result = get_file_name_from_template_with_dict("{{ a }}-{{ b }}", {"a": "1", "b": "2"}, exclude={"b"})
        assert result == "1-{{ b }}"

    def test_an_excluded_placeholder_keeps_its_original_spacing(self) -> None:
        result = get_file_name_from_template_with_dict("{{b}}", {"b": "2"}, exclude={"b"})
        assert result == "{{b}}"

    def test_a_missing_key_names_the_key_it_could_not_find(self) -> None:
        with pytest.raises(ValueError, match="Key 'missing' not found"):
            get_file_name_from_template_with_dict("{{ missing }}", {"a": "1"})

    def test_a_single_brace_is_not_a_placeholder(self) -> None:
        assert get_file_name_from_template_with_dict("{ a }", {"a": "1"}) == "{ a }"


class TestTemplateSubstitutionFromAnInstance:
    class _Batch:
        detector = "L1"
        index = 3

    def test_attributes_are_substituted(self) -> None:
        assert get_file_name_from_template("{{ detector }}-{{ index }}.gwf", self._Batch()) == "L1-3.gwf"

    def test_an_excluded_attribute_is_left_alone(self) -> None:
        result = get_file_name_from_template("{{ detector }}-{{ index }}", self._Batch(), exclude={"index"})
        assert result == "L1-{{ index }}"

    def test_a_missing_attribute_names_the_attribute_and_the_type(self) -> None:
        with pytest.raises(ValueError, match="Attribute 'nope' not found in instance of type _Batch"):
            get_file_name_from_template("{{ nope }}", self._Batch())


class TestTheSignalHandler:
    def test_the_cleanup_runs_and_the_process_exits_non_zero(self) -> None:
        """What ``execute_plan`` relies on: an interrupted run cleans up, then exits with a
        failure status so a wrapper script does not treat it as a completed run."""
        calls: list[str] = []
        handler = handle_signal(lambda: calls.append("cleaned"))

        with pytest.raises(SystemExit) as exit_info:
            handler(2, None)

        assert calls == ["cleaned"]
        assert exit_info.value.code == 1

    def test_the_cleanup_runs_before_the_exit(self) -> None:
        """Ordering is the whole point: exiting first would leave the checkpoints behind."""
        order: list[str] = []

        def cleanup() -> None:
            order.append("cleanup")

        try:
            handle_signal(cleanup)(15, None)
        except SystemExit:
            order.append("exit")

        assert order == ["cleanup", "exit"]

    def test_the_signal_number_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR, logger="gwmock"), pytest.raises(SystemExit):
            handle_signal(lambda: None)(15, None)
        assert "15" in caplog.text

    def test_a_failing_cleanup_is_not_swallowed(self) -> None:
        """A cleanup that itself fails must not be reported to the shell as a clean interrupt."""

        def cleanup() -> None:
            raise RuntimeError("cleanup failed")

        with pytest.raises(RuntimeError, match="cleanup failed"):
            handle_signal(cleanup)(2, None)


class TestSavingSafely:
    def test_a_first_save_writes_the_file_and_leaves_no_backup(self, tmp_path: Path) -> None:
        target, backup = tmp_path / "checkpoint.json", tmp_path / "checkpoint.json.bak"
        save_file_safely(target, backup, lambda file_name: Path(file_name).write_text("new"))
        assert target.read_text() == "new"
        assert not backup.exists()

    def test_an_existing_file_is_moved_aside_and_then_removed(self, tmp_path: Path) -> None:
        target, backup = tmp_path / "checkpoint.json", tmp_path / "checkpoint.json.bak"
        target.write_text("old")

        seen: list[str] = []

        def save(file_name):
            # The backup exists while the save is in flight -- that is the window it protects.
            seen.append(backup.read_text())
            Path(file_name).write_text("new")

        save_file_safely(target, backup, save)

        assert seen == ["old"]
        assert target.read_text() == "new"
        assert not backup.exists()

    def test_extra_arguments_reach_the_save_function(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        save_file_safely(
            target,
            tmp_path / "out.txt.bak",
            lambda file_name, payload: Path(file_name).write_text(payload),
            payload="body",
        )
        assert target.read_text() == "body"

    def test_a_failed_save_restores_the_previous_file_and_raises(self, tmp_path: Path) -> None:
        """The reason the function exists: a half-written checkpoint must not replace a good one."""
        target, backup = tmp_path / "checkpoint.json", tmp_path / "checkpoint.json.bak"
        target.write_text("old")

        def save(file_name):
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            save_file_safely(target, backup, save)

        assert target.read_text() == "old"
        assert not backup.exists()

    def test_a_failed_first_save_leaves_nothing_behind(self, tmp_path: Path) -> None:
        target, backup = tmp_path / "checkpoint.json", tmp_path / "checkpoint.json.bak"

        def save(file_name):
            raise ValueError("bad record")

        with pytest.raises(ValueError, match="bad record"):
            save_file_safely(target, backup, save)

        assert not target.exists()
        assert not backup.exists()

    def test_an_unexpected_failure_is_not_treated_as_a_save_failure(self, tmp_path: Path) -> None:
        """Only OSError/PermissionError/ValueError are handled. Anything else propagates with the
        backup still on disk rather than being quietly restored over."""
        target, backup = tmp_path / "checkpoint.json", tmp_path / "checkpoint.json.bak"
        target.write_text("old")

        def save(file_name):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            save_file_safely(target, backup, save)

        assert backup.read_text() == "old"

    def test_a_directory_in_place_of_the_target_is_not_backed_up(self, tmp_path: Path) -> None:
        """``is_file`` rather than ``exists``: renaming a directory aside would be destructive."""
        target = tmp_path / "checkpoint.json"
        target.mkdir()
        backup = tmp_path / "checkpoint.json.bak"

        with pytest.raises(IsADirectoryError):
            save_file_safely(target, backup, lambda file_name: Path(file_name).write_text("new"))

        assert target.is_dir()
        assert not backup.exists()
