"""Tests for the interactive config editor."""

from __future__ import annotations

import pytest

from gwmock.cli.utils.config_editor import ConfigEditorApp


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
