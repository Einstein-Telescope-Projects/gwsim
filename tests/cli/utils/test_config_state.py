"""Unit tests for ConfigState."""

from __future__ import annotations

import pytest

from gwmock.cli.utils.config_state import ConfigState


class TestConfigStateSet:
    def test_set_noise_psd(self):
        state = ConfigState()
        state.set("noise", "psd", "ET_10_full_cryo_psd")
        assert state.get("noise", "psd") == "ET_10_full_cryo_psd"

    def test_set_noise_seed_coerces_to_int(self):
        state = ConfigState()
        state.set("noise", "seed", "42")
        assert state.get("noise", "seed") == 42
        assert isinstance(state.get("noise", "seed"), int)

    def test_set_noise_detectors_splits_to_list(self):
        state = ConfigState()
        state.set("noise", "detectors", "E0 E1 E2")
        assert state.get("noise", "detectors") == ["E0", "E1", "E2"]

    def test_set_noise_minimum_frequency_coerces_to_float(self):
        state = ConfigState()
        state.set("noise", "minimum-frequency", "5.0")
        assert state.get("noise", "minimum-frequency") == 5.0

    def test_set_signal_waveform_model(self):
        state = ConfigState()
        state.set("signal", "waveform-model", "IMRPhenomXPHM")
        assert state.get("signal", "waveform-model") == "IMRPhenomXPHM"

    def test_set_signal_earth_rotation_bool(self):
        state = ConfigState()
        state.set("signal", "earth-rotation", "true")
        assert state.get("signal", "earth-rotation") is True
        state.set("signal", "earth-rotation", "false")
        assert state.get("signal", "earth-rotation") is False

    def test_set_globals_sampling_frequency(self):
        state = ConfigState()
        state.set("globals", "sampling-frequency", "4096")
        assert state.get("globals", "sampling-frequency") == 4096

    def test_set_globals_total_duration_keeps_string(self):
        state = ConfigState()
        state.set("globals", "total-duration", "1 day")
        assert state.get("globals", "total-duration") == "1 day"

    def test_set_population_n_samples(self):
        state = ConfigState()
        state.set("population", "n-samples", "10")
        assert state.get("population", "n-samples") == 10

    def test_set_unknown_section_raises(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="Unknown section"):
            state.set("unknown", "key", "value")

    def test_set_unknown_key_raises(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="Unknown key"):
            state.set("noise", "nonexistent", "value")

    def test_set_invalid_bool_raises(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="boolean"):
            state.set("signal", "earth-rotation", "maybe")

    def test_set_invalid_int_raises(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="invalid literal"):
            state.set("noise", "seed", "not_a_number")


class TestConfigStateGet:
    def test_get_unset_returns_none(self):
        state = ConfigState()
        assert state.get("noise", "psd") is None

    def test_get_unknown_key_returns_none(self):
        state = ConfigState()
        assert state.get("noise", "nonexistent") is None

    def test_get_section_returns_none_when_empty(self):
        state = ConfigState()
        assert state.get_section("noise") is None

    def test_get_section_returns_dict_when_set(self):
        state = ConfigState()
        state.set("noise", "psd", "ET_10_full_cryo_psd")
        section = state.get_section("noise")
        assert section is not None
        assert "arguments" in section


class TestConfigStateToDict:
    def test_empty_state_produces_empty_dict(self):
        state = ConfigState()
        assert state.to_dict() == {}

    def test_single_setting(self):
        state = ConfigState()
        state.set("noise", "psd", "ET_10_full_cryo_psd")
        d = state.to_dict()
        assert d["orchestration"]["noise"]["arguments"]["psd_file"] == "ET_10_full_cryo_psd"

    def test_multiple_sections(self):
        state = ConfigState()
        state.set("noise", "psd", "ET_10_full_cryo_psd")
        state.set("signal", "waveform-model", "IMRPhenomXPHM")
        state.set("globals", "sampling-frequency", "4096")
        d = state.to_dict()
        assert "orchestration" in d
        assert "globals" in d
        assert d["orchestration"]["signal"]["waveform-model"] == "IMRPhenomXPHM"
        assert d["globals"]["simulator-arguments"]["sampling-frequency"] == 4096

    def test_empty_sections_are_excluded(self):
        state = ConfigState()
        state.set("noise", "psd", "test")
        state.reset("noise")
        d = state.to_dict()
        assert "orchestration" not in d or not d.get("orchestration")


class TestConfigStateReset:
    def test_reset_section(self):
        state = ConfigState()
        state.set("noise", "psd", "test")
        state.set("noise", "seed", "42")
        state.reset("noise")
        assert state.get_section("noise") is None
        # Other sections unaffected
        state.set("signal", "waveform-model", "IMRPhenomXPHM")
        state.reset("noise")
        assert state.get("signal", "waveform-model") == "IMRPhenomXPHM"

    def test_reset_all(self):
        state = ConfigState()
        state.set("noise", "psd", "test")
        state.set("signal", "waveform-model", "test")
        state.reset()
        assert state.to_dict() == {}
        assert state.is_empty

    def test_reset_unknown_section_raises(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="Unknown section"):
            state.reset("nonexistent")


class TestConfigStateGlitches:
    def test_add_glitch(self):
        state = ConfigState()
        idx = state.add_glitch("gengli_blip")
        assert idx == 0
        d = state.to_dict()
        glitches = d["orchestration"]["noise"]["arguments"]["glitches"]
        assert len(glitches) == 1
        assert glitches[0]["kind"] == "gengli_blip"

    def test_add_multiple_glitches(self):
        state = ConfigState()
        state.add_glitch("gengli_blip")
        state.add_glitch("blip")
        d = state.to_dict()
        assert len(d["orchestration"]["noise"]["arguments"]["glitches"]) == 2

    def test_remove_glitch(self):
        state = ConfigState()
        state.add_glitch("gengli_blip")
        state.add_glitch("blip")
        removed = state.remove_glitch(0)
        assert removed["kind"] == "gengli_blip"
        d = state.to_dict()
        glitches = d["orchestration"]["noise"]["arguments"]["glitches"]
        assert len(glitches) == 1
        assert glitches[0]["kind"] == "blip"

    def test_remove_last_glitch_cleans_key(self):
        state = ConfigState()
        state.add_glitch("gengli_blip")
        state.remove_glitch(0)
        d = state.to_dict()
        noise_args = d.get("orchestration", {}).get("noise", {}).get("arguments", {})
        assert "glitches" not in noise_args

    def test_remove_glitch_out_of_range_raises(self):
        state = ConfigState()
        with pytest.raises(IndexError):
            state.remove_glitch(0)


class TestConfigStateBatch:
    def test_set_batch_resource(self):
        state = ConfigState()
        state.set_batch_resource("nodes", "4")
        d = state.to_dict()
        assert d["batch"]["resources"]["nodes"] == "4"

    def test_set_batch_submit(self):
        state = ConfigState()
        state.set_batch_submit("account", "myaccount")
        d = state.to_dict()
        assert d["batch"]["submit"]["account"] == "myaccount"


class TestConfigStateLoad:
    def test_load_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "orchestration:\n  noise:\n    arguments:\n      psd_file: ET_10_full_cryo_psd\n      seed: 42\n"
        )
        state = ConfigState()
        state.load(config_file)
        assert state.get("noise", "psd") == "ET_10_full_cryo_psd"
        assert state.get("noise", "seed") == 42

    def test_load_invalid_yaml_raises(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- just\n- a\n- list\n")
        state = ConfigState()
        with pytest.raises(ValueError, match="YAML mapping"):
            state.load(config_file)


class TestConfigStateValidate:
    def test_empty_config_is_invalid(self):
        state = ConfigState()
        valid, error = state.validate()
        assert not valid
        assert error

    def test_noise_only_is_valid(self):
        state = ConfigState()
        state.set("noise", "psd", "test_psd")
        state.set("noise", "seed", "42")
        valid, _ = state.validate()
        assert valid
