"""Tests for the interactive config editor."""

from __future__ import annotations

import pytest

from gwmock.cli.utils import config_editor
from gwmock.cli.utils.config_editor import ConfigEditorApp
from gwmock.cli.utils.config_state import SECTION_EXTRA, SECTION_KEYS


@pytest.fixture
def app():
    return ConfigEditorApp()


def _submit(pilot, app, command: str):
    """Helper to submit a command to the editor."""
    from textual.widgets import Input

    input_widget = app.query_one("#command-input")
    input_widget.value = command
    input_widget.post_message(Input.Submitted(input_widget, command))


class TestConfigEditorCommands:
    @pytest.mark.asyncio
    async def test_app_starts(self, app):
        async with app.run_test():
            assert app.query_one("#config-panel") is not None
            assert app.query_one("#output-panel") is not None
            assert app.query_one("#command-input") is not None

    @pytest.mark.asyncio
    async def test_set_noise_psd(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/noise psd ET_10_full_cryo_psd")
            await pilot.pause()
            assert app._state.get("noise", "psd") == "ET_10_full_cryo_psd"

    @pytest.mark.asyncio
    async def test_set_signal_detectors(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/signal detectors E0 E1 E2")
            await pilot.pause()
            assert app._state.get("signal", "detectors") == ["E0", "E1", "E2"]

    @pytest.mark.asyncio
    async def test_set_globals_sampling_frequency(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/globals sampling-frequency 4096")
            await pilot.pause()
            assert app._state.get("globals", "sampling-frequency") == 4096

    @pytest.mark.asyncio
    async def test_set_population_path(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/population path /data/catalog.h5")
            await pilot.pause()
            assert app._state.get("population", "path") == "/data/catalog.h5"

    @pytest.mark.asyncio
    async def test_unknown_command_does_not_crash(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/nonexistent")
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_command_without_slash_does_not_crash(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "noise psd test")
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_empty_command_ignored(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "   ")
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_reset_noise_section(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/noise psd test")
            await pilot.pause()
            assert app._state.get("noise", "psd") == "test"

            _submit(pilot, app, "/reset noise")
            await pilot.pause()
            assert app._state.get("noise", "psd") is None

    @pytest.mark.asyncio
    async def test_reset_all(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/noise psd test")
            await pilot.pause()
            _submit(pilot, app, "/signal waveform-model IMRPhenomXPHM")
            await pilot.pause()

            _submit(pilot, app, "/reset all")
            await pilot.pause()
            assert app._state.is_empty

    @pytest.mark.asyncio
    async def test_add_glitch(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/noise glitch add gengli_blip")
            await pilot.pause()
            d = app._state.to_dict()
            glitches = d["orchestration"]["noise"]["arguments"]["glitches"]
            assert len(glitches) == 1
            assert glitches[0]["kind"] == "gengli_blip"

    @pytest.mark.asyncio
    async def test_remove_glitch(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/noise glitch add gengli_blip")
            await pilot.pause()
            _submit(pilot, app, "/noise glitch add blip")
            await pilot.pause()

            _submit(pilot, app, "/noise glitch remove 0")
            await pilot.pause()
            d = app._state.to_dict()
            glitches = d["orchestration"]["noise"]["arguments"]["glitches"]
            assert len(glitches) == 1
            assert glitches[0]["kind"] == "blip"

    @pytest.mark.asyncio
    async def test_batch_resources(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/batch resources nodes 4")
            await pilot.pause()
            d = app._state.to_dict()
            assert d["batch"]["resources"]["nodes"] == "4"

    @pytest.mark.asyncio
    async def test_batch_submit_option(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/batch submit account myaccount")
            await pilot.pause()
            d = app._state.to_dict()
            assert d["batch"]["submit"]["account"] == "myaccount"

    @pytest.mark.asyncio
    async def test_batch_simple_key(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/batch scheduler slurm")
            await pilot.pause()
            assert app._state.get("batch", "scheduler") == "slurm"

    @pytest.mark.asyncio
    async def test_save_valid_config(self, app, tmp_path):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/noise psd test_psd")
            await pilot.pause()
            _submit(pilot, app, "/noise seed 42")
            await pilot.pause()

            output_path = tmp_path / "config.yaml"
            _submit(pilot, app, f"/save {output_path}")
            await pilot.pause()

            assert output_path.exists()
            content = output_path.read_text()
            assert "test_psd" in content

    @pytest.mark.asyncio
    async def test_load_config(self, app, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("orchestration:\n  noise:\n    arguments:\n      psd_file: loaded_psd\n      seed: 99\n")

        async with app.run_test() as pilot:
            _submit(pilot, app, f"/load {config_file}")
            await pilot.pause()

            assert app._state.get("noise", "psd") == "loaded_psd"
            assert app._state.get("noise", "seed") == 99

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/load /nonexistent/path.yaml")
            await pilot.pause()
            # Should not crash, just show error

    @pytest.mark.asyncio
    async def test_multiple_commands_sequence(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/noise psd ET_10_full_cryo_psd")
            await pilot.pause()
            _submit(pilot, app, "/noise seed 42")
            await pilot.pause()
            _submit(pilot, app, "/noise detectors E0 E1 E2")
            await pilot.pause()
            _submit(pilot, app, "/signal waveform-model IMRPhenomXPHM")
            await pilot.pause()
            _submit(pilot, app, "/globals sampling-frequency 4096")
            await pilot.pause()

            d = app._state.to_dict()
            assert d["orchestration"]["noise"]["arguments"]["psd_file"] == "ET_10_full_cryo_psd"
            assert d["orchestration"]["noise"]["arguments"]["seed"] == 42
            assert d["orchestration"]["noise"]["arguments"]["detectors"] == ["E0", "E1", "E2"]
            assert d["orchestration"]["signal"]["waveform-model"] == "IMRPhenomXPHM"
            assert d["globals"]["simulator-arguments"]["sampling-frequency"] == 4096

    @pytest.mark.asyncio
    async def test_help_does_not_crash(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/help")
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_config_does_not_crash(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/config")
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_discovery_commands_do_not_crash(self, app):
        async with app.run_test() as pilot:
            for cmd in ["/geometries", "/psds", "/source-types", "/glitches", "/presets"]:
                _submit(pilot, app, cmd)
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_section_help_shown_without_args(self, app):
        async with app.run_test() as pilot:
            _submit(pilot, app, "/noise")
            await pilot.pause()
            # Should not crash and should not modify state
            assert app._state.is_empty


class TestConfigEditorWithLoad:
    @pytest.mark.asyncio
    async def test_load_on_startup(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("orchestration:\n  noise:\n    arguments:\n      psd_file: startup_psd\n")
        app = ConfigEditorApp(load_path=config_file)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._state.get("noise", "psd") == "startup_psd"


class TestSuggestions:
    """The completion engine behind the command input.

    ``_get_suggestions`` decides what the editor offers for a half-typed command.
    It is pure, so it is exercised directly rather than through the TUI: a wrong
    branch here shows up as a menu that is merely unhelpful, never as an error.
    """

    @pytest.fixture(autouse=True)
    def _fixed_discovery(self, monkeypatch):
        """Pin the discovery helpers so a suggestion list is exactly checkable."""
        for name, values in {
            "discover_psds": ["psd-a", "psd-b"],
            "discover_geometries": ["ET", "CE"],
            "discover_glitch_models": ["blip", "koi_fish"],
            "discover_source_types": ["bbh", "bns"],
            "discover_waveform_models": ["IMRPhenomD", "TaylorF2"],
        }.items():
            monkeypatch.setattr(config_editor, name, lambda values=values: list(values))

    @pytest.mark.parametrize("text", ["", "   ", "\t"])
    def test_nothing_typed_suggests_nothing(self, app, text: str):
        """An empty input has no prefix to complete."""
        assert app._get_suggestions(text) == []

    def test_a_bare_word_suggests_commands_case_insensitively(self, app):
        """Typing without the leading slash still completes command names."""
        assert app._get_suggestions("NOI") == ["noise"]

    def test_a_lone_slash_suggests_every_command(self, app):
        """The slash alone offers the full command list, sorted."""
        suggestions = app._get_suggestions("/")

        assert suggestions == sorted(app._handlers.keys())

    def test_a_command_prefix_is_filtered(self, app):
        """A partial command name narrows the list to that prefix."""
        assert app._get_suggestions("/noi") == ["noise"]

    def test_a_completed_command_suggests_its_keys(self, app):
        """The trailing space marks the command as finished, so keys come next."""
        suggestions = app._get_suggestions("/noise ")

        assert suggestions == list(SECTION_KEYS["noise"].keys()) + [e[0] for e in SECTION_EXTRA.get("noise", [])]

    def test_a_key_prefix_is_filtered_against_keys_and_extras(self, app):
        """Without a trailing space the second word completes a key name."""
        assert app._get_suggestions("/noise ps") == ["psd"]

    def test_an_unknown_section_suggests_nothing(self, app):
        """A command that owns no keys has nothing to offer for its arguments."""
        assert app._get_suggestions("/help x") == []

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Without the trailing space the command name itself is still being typed.
            ("/template", ["template"]),
            ("/template ", ["noise", "signal", "glitch"]),
            ("/template s", ["signal"]),
            ("/template G", ["glitch"]),
            ("/generate-script", ["generate-script"]),
            ("/generate-script ", ["slurm", "local"]),
            ("/generate-script s", ["slurm"]),
            # The script kind is the last argument, so nothing follows it.
            ("/generate-script slurm ", []),
        ],
    )
    def test_commands_with_a_fixed_argument_list(self, app, text: str, expected: list[str]):
        """``/template`` and ``/generate-script`` complete from their own fixed lists."""
        assert app._get_suggestions(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("/noise psd ", ["psd-a", "psd-b"]),
            ("/noise detectors ", ["ET", "CE"]),
            ("/signal detectors ", ["ET", "CE"]),
            ("/noise glitch add", ["blip", "koi_fish"]),
            ("/signal source-type ", ["bbh", "bns"]),
            ("/signal waveform-model ", ["IMRPhenomD", "TaylorF2"]),
            ("/population source-type ", ["bbh", "bns"]),
            ("/population backend ", ["file", "cbc_prior", "bbh", "bns_prior", "nsbh_prior"]),
            ("/globals total-duration ", ["1 day", "6 hours", "1 hour", "3600"]),
            # A key with no value suggestions offers nothing rather than the key list again.
            ("/noise seed ", []),
        ],
    )
    def test_value_suggestions_per_key(self, app, text: str, expected: list[str]):
        """Each key that has known values completes from its own source."""
        assert app._get_suggestions(text) == expected

    def test_a_glitch_value_is_only_suggested_after_add(self, app):
        """``glitch remove`` takes an index, not a model name."""
        assert app._get_suggestions("/noise glitch remove") == []

    def test_psd_values_are_not_offered_for_another_section(self, app):
        """``psd`` is a noise key; the same word elsewhere is not a PSD."""
        assert app._get_suggestions("/signal psd ") == []
