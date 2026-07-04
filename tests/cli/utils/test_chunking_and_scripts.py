"""Tests for chunking configuration and script generation."""

from __future__ import annotations

import pytest

from gwmock.cli.utils.config_editor import ConfigEditorApp


@pytest.fixture
def app():
    return ConfigEditorApp()


class TestChunkingConfiguration:
    @pytest.mark.asyncio
    async def test_chunks_enabled(self, app):
        """Test setting chunks-enabled."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            input_widget.value = "/batch chunks-enabled true"
            from textual.widgets import Input

            input_widget.post_message(Input.Submitted(input_widget, "/batch chunks-enabled true"))
            await pilot.pause()
            assert app._state.get("batch", "chunks-enabled") is True

    @pytest.mark.asyncio
    async def test_chunks_n_chunks(self, app):
        """Test setting chunks-n-chunks."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            input_widget.value = "/batch chunks-n-chunks 4"
            from textual.widgets import Input

            input_widget.post_message(Input.Submitted(input_widget, "/batch chunks-n-chunks 4"))
            await pilot.pause()
            assert app._state.get("batch", "chunks-n-chunks") == 4

    @pytest.mark.asyncio
    async def test_chunks_parallel(self, app):
        """Test setting chunks-parallel."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            input_widget.value = "/batch chunks-parallel true"
            from textual.widgets import Input

            input_widget.post_message(Input.Submitted(input_widget, "/batch chunks-parallel true"))
            await pilot.pause()
            assert app._state.get("batch", "chunks-parallel") is True

    @pytest.mark.asyncio
    async def test_chunks_n_chunks_validation(self, app):
        """Test that chunks-n-chunks must be >= 1 during validation."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            input_widget.value = "/batch chunks-n-chunks 0"
            from textual.widgets import Input

            input_widget.post_message(Input.Submitted(input_widget, "/batch chunks-n-chunks 0"))
            await pilot.pause()
            # Value is set in state, but validation will fail when saving
            assert app._state.get("batch", "chunks-n-chunks") == 0

            # Try to save - should fail validation
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmpdir:
                config_file = Path(tmpdir) / "config.yaml"
                input_widget.value = f"/save {config_file}"
                input_widget.post_message(Input.Submitted(input_widget, f"/save {config_file}"))
                await pilot.pause()
                # File should not be created due to validation error
                assert not config_file.exists()


class TestTemplateCommands:
    @pytest.mark.asyncio
    async def test_template_noise(self, app):
        """Test loading noise template."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            input_widget.value = "/template noise"
            from textual.widgets import Input

            input_widget.post_message(Input.Submitted(input_widget, "/template noise"))
            await pilot.pause()

            assert app._state.get("noise", "psd") == "ET_10_full_cryo_psd"
            assert app._state.get("noise", "seed") == 42
            assert app._state.get("noise", "detectors") == ["ET-Triangle-EMR"]

    @pytest.mark.asyncio
    async def test_template_signal(self, app):
        """Test loading signal template."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            input_widget.value = "/template signal"
            from textual.widgets import Input

            input_widget.post_message(Input.Submitted(input_widget, "/template signal"))
            await pilot.pause()

            assert app._state.get("noise", "psd") == "ET_10_full_cryo_psd"
            assert app._state.get("signal", "source-type") == "bbh"
            assert app._state.get("signal", "detectors") == ["ET-Triangle-EMR"]
            assert app._state.get("population", "backend") == "file"

    @pytest.mark.asyncio
    async def test_template_glitch(self, app):
        """Test loading glitch template."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            input_widget.value = "/template glitch"
            from textual.widgets import Input

            input_widget.post_message(Input.Submitted(input_widget, "/template glitch"))
            await pilot.pause()

            assert app._state.get("noise", "psd") == "ET_10_full_cryo_psd"
            # Check glitches in the actual data structure
            glitches = app._state._data.get("orchestration", {}).get("noise", {}).get("arguments", {}).get("glitches")
            assert glitches is not None
            assert len(glitches) == 1
            assert glitches[0]["kind"] == "gengli_blip"

    @pytest.mark.asyncio
    async def test_template_invalid(self, app):
        """Test loading invalid template."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            input_widget.value = "/template invalid"
            from textual.widgets import Input

            input_widget.post_message(Input.Submitted(input_widget, "/template invalid"))
            await pilot.pause()

            # Should not crash, just show error message


class TestScriptGeneration:
    @pytest.mark.asyncio
    async def test_generate_script_no_batch(self, app, tmp_path):
        """Test script generation without batch config."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            output_file = tmp_path / "submit.sh"
            input_widget.value = f"/generate-script slurm {output_file}"
            from textual.widgets import Input

            input_widget.post_message(Input.Submitted(input_widget, f"/generate-script slurm {output_file}"))
            await pilot.pause()

            # Should show error message, file should not be created
            assert not output_file.exists()

    @pytest.mark.asyncio
    async def test_generate_script_slurm_basic(self, app, tmp_path):
        """Test basic SLURM script generation."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            from textual.widgets import Input

            # Configure batch
            input_widget.value = "/batch job-name test_job"
            input_widget.post_message(Input.Submitted(input_widget, "/batch job-name test_job"))
            await pilot.pause()

            # Save config
            config_file = tmp_path / "config.yaml"
            input_widget.value = f"/save {config_file}"
            input_widget.post_message(Input.Submitted(input_widget, f"/save {config_file}"))
            await pilot.pause()

            # Generate script
            output_file = tmp_path / "submit.sh"
            input_widget.value = f"/generate-script slurm {output_file}"
            input_widget.post_message(Input.Submitted(input_widget, f"/generate-script slurm {output_file}"))
            await pilot.pause()

            assert output_file.exists()
            content = output_file.read_text()
            assert "#!/bin/bash" in content
            assert "#SBATCH --job-name=test_job" in content
            assert "gwmock simulate" in content

    @pytest.mark.asyncio
    async def test_generate_script_slurm_with_chunks(self, app, tmp_path):
        """Test SLURM script generation with chunking."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            from textual.widgets import Input

            # Configure batch with chunking
            input_widget.value = "/batch job-name test_job"
            input_widget.post_message(Input.Submitted(input_widget, "/batch job-name test_job"))
            await pilot.pause()

            input_widget.value = "/batch chunks-enabled true"
            input_widget.post_message(Input.Submitted(input_widget, "/batch chunks-enabled true"))
            await pilot.pause()

            input_widget.value = "/batch chunks-n-chunks 4"
            input_widget.post_message(Input.Submitted(input_widget, "/batch chunks-n-chunks 4"))
            await pilot.pause()

            # Save config
            config_file = tmp_path / "config.yaml"
            input_widget.value = f"/save {config_file}"
            input_widget.post_message(Input.Submitted(input_widget, f"/save {config_file}"))
            await pilot.pause()

            # Generate script
            output_file = tmp_path / "submit.sh"
            input_widget.value = f"/generate-script slurm {output_file}"
            input_widget.post_message(Input.Submitted(input_widget, f"/generate-script slurm {output_file}"))
            await pilot.pause()

            assert output_file.exists()
            content = output_file.read_text()
            assert "#SBATCH --array=0-3" in content
            assert "SLURM_ARRAY_TASK_ID" in content

    @pytest.mark.asyncio
    async def test_generate_script_local_basic(self, app, tmp_path):
        """Test basic local script generation."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            from textual.widgets import Input

            # Configure batch
            input_widget.value = "/batch job-name test_job"
            input_widget.post_message(Input.Submitted(input_widget, "/batch job-name test_job"))
            await pilot.pause()

            # Save config
            config_file = tmp_path / "config.yaml"
            input_widget.value = f"/save {config_file}"
            input_widget.post_message(Input.Submitted(input_widget, f"/save {config_file}"))
            await pilot.pause()

            # Generate script
            output_file = tmp_path / "run.sh"
            input_widget.value = f"/generate-script local {output_file}"
            input_widget.post_message(Input.Submitted(input_widget, f"/generate-script local {output_file}"))
            await pilot.pause()

            assert output_file.exists()
            content = output_file.read_text()
            assert "#!/bin/bash" in content
            assert "gwmock simulate" in content

    @pytest.mark.asyncio
    async def test_generate_script_local_parallel(self, app, tmp_path):
        """Test local script generation with parallel execution."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            from textual.widgets import Input

            # Configure batch with chunking
            input_widget.value = "/batch job-name test_job"
            input_widget.post_message(Input.Submitted(input_widget, "/batch job-name test_job"))
            await pilot.pause()

            input_widget.value = "/batch chunks-enabled true"
            input_widget.post_message(Input.Submitted(input_widget, "/batch chunks-enabled true"))
            await pilot.pause()

            input_widget.value = "/batch chunks-n-chunks 3"
            input_widget.post_message(Input.Submitted(input_widget, "/batch chunks-n-chunks 3"))
            await pilot.pause()

            input_widget.value = "/batch chunks-parallel true"
            input_widget.post_message(Input.Submitted(input_widget, "/batch chunks-parallel true"))
            await pilot.pause()

            # Save config
            config_file = tmp_path / "config.yaml"
            input_widget.value = f"/save {config_file}"
            input_widget.post_message(Input.Submitted(input_widget, f"/save {config_file}"))
            await pilot.pause()

            # Generate script
            output_file = tmp_path / "run.sh"
            input_widget.value = f"/generate-script local {output_file}"
            input_widget.post_message(Input.Submitted(input_widget, f"/generate-script local {output_file}"))
            await pilot.pause()

            assert output_file.exists()
            content = output_file.read_text()
            assert content.count("&") == 3  # Three background processes
            assert "wait" in content


class TestAutocomplete:
    @pytest.mark.asyncio
    async def test_template_autocomplete(self, app):
        """Test autocomplete for /template command."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            input_widget.value = "/template "
            from textual.widgets import Input

            input_widget.post_message(Input.Changed(input_widget, "/template "))
            await pilot.pause()

            panel = app.query_one("#suggestion-panel")
            assert panel.option_count == 3
            options = [panel.get_option_at_index(i).prompt for i in range(3)]
            assert "noise" in options
            assert "signal" in options
            assert "glitch" in options

    @pytest.mark.asyncio
    async def test_generate_script_autocomplete(self, app):
        """Test autocomplete for /generate-script command."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            input_widget.value = "/generate-script "
            from textual.widgets import Input

            input_widget.post_message(Input.Changed(input_widget, "/generate-script "))
            await pilot.pause()

            panel = app.query_one("#suggestion-panel")
            assert panel.option_count == 2
            options = [panel.get_option_at_index(i).prompt for i in range(2)]
            assert "slurm" in options
            assert "local" in options

    @pytest.mark.asyncio
    async def test_batch_chunks_autocomplete(self, app):
        """Test autocomplete for batch chunks-* keys."""
        async with app.run_test() as pilot:
            input_widget = app.query_one("#command-input")
            input_widget.value = "/batch chunks"
            from textual.widgets import Input

            input_widget.post_message(Input.Changed(input_widget, "/batch chunks"))
            await pilot.pause()

            panel = app.query_one("#suggestion-panel")
            options = [panel.get_option_at_index(i).prompt for i in range(panel.option_count)]
            assert "chunks-enabled" in options
            assert "chunks-n-chunks" in options
            assert "chunks-parallel" in options
