"""Unit tests for template variable expansion utilities."""

from __future__ import annotations

from typing import ClassVar

import pytest
from astropy.units import Quantity

from gwsim.cli.utils.template import (
    _expand_string_template,
    _value_to_string_list,
    expand_template_variables,
)


class TestValueToStringList:
    """Tests for _value_to_string_list function."""

    def test_convert_string(self):
        """Test converting a string returns single-item list."""
        assert _value_to_string_list("H1") == ["H1"]

    def test_convert_integer(self):
        """Test converting an integer to single-item list."""
        assert _value_to_string_list(42) == ["42"]

    def test_convert_float(self):
        """Test converting a float to single-item list."""
        assert _value_to_string_list(3.14) == ["3.14"]

    def test_convert_quantity_with_integer_value(self):
        """Test converting astropy Quantity with integer value."""
        qty = Quantity(4096, "Hz")
        assert _value_to_string_list(qty) == ["4096"]

    def test_convert_quantity_with_float_value(self):
        """Test converting astropy Quantity with float value."""
        qty = Quantity(4096.5, "Hz")
        assert _value_to_string_list(qty) == ["4096.5"]

    def test_convert_list(self):
        """Test converting a list to list of strings."""
        result = _value_to_string_list(["H1", "L1"])
        assert result == ["H1", "L1"]

    def test_convert_tuple(self):
        """Test converting a tuple to list of strings."""
        result = _value_to_string_list(("H1", "L1"))
        assert result == ["H1", "L1"]

    def test_convert_empty_list(self):
        """Test converting an empty list."""
        result = _value_to_string_list([])
        assert result == []

    def test_convert_mixed_list(self):
        """Test converting a list with mixed types."""
        result = _value_to_string_list([1, "H1", 3.14])
        assert result == ["1", "H1", "3.14"]


class TestExpandStringTemplate:
    """Tests for _expand_string_template function."""

    def test_expand_single_variable(self):
        """Test expanding a single template variable."""

        class MockSimulator:
            detector = "H1"

        sim = MockSimulator()
        result = _expand_string_template("{{ detector }}-STRAIN", sim)
        assert result == "H1-STRAIN"

    def test_expand_multiple_variables(self):
        """Test expanding multiple template variables."""

        class MockSimulator:
            detector = "H1"
            counter = 5

        sim = MockSimulator()
        result = _expand_string_template("{{ detector }}-{{ counter }}", sim)
        assert result == "H1-5"

    def test_expand_with_spaces_in_template(self):
        """Test expanding template with extra spaces."""

        class MockSimulator:
            detector = "H1"

        sim = MockSimulator()
        result = _expand_string_template("{{  detector  }}-STRAIN", sim)
        assert result == "H1-STRAIN"

    def test_expand_from_simulator_attribute(self):
        """Test expanding template from simulator attribute."""

        class MockSimulator:
            sampling_frequency = 4096

        sim = MockSimulator()
        result = _expand_string_template("fs={{ sampling_frequency }}", sim)
        assert result == "fs=4096"

    def test_expand_quantity_attribute(self):
        """Test expanding template with Quantity attribute."""

        class MockSimulator:
            sampling_frequency = Quantity(4096, "Hz")

        sim = MockSimulator()
        result = _expand_string_template("fs={{ sampling_frequency }}", sim)
        assert result == "fs=4096"

    def test_expand_list_attribute(self):
        """Test expanding template with list attribute creates Cartesian product."""

        class MockSimulator:
            detectors: ClassVar[list[str]] = ["H1", "L1"]

        sim = MockSimulator()
        result = _expand_string_template("{{ detectors }}:STRAIN", sim)
        assert result == ["H1:STRAIN", "L1:STRAIN"]

    def test_expand_missing_variable_raises_error(self):
        """Test that missing variable raises AttributeError."""

        class MockSimulator:
            pass

        sim = MockSimulator()
        with pytest.raises(AttributeError, match="Template variable 'missing' not found"):
            _expand_string_template("{{ missing }}", sim)

    def test_expand_no_template_variables(self):
        """Test expanding string with no template variables."""

        class MockSimulator:
            pass

        sim = MockSimulator()
        result = _expand_string_template("static-string.txt", sim)
        assert result == "static-string.txt"

    def test_expand_repeated_variable(self):
        """Test expanding the same variable multiple times."""

        class MockSimulator:
            detector = "H1"

        sim = MockSimulator()
        result = _expand_string_template("{{ detector }}-{{ detector }}.gwf", sim)
        assert result == "H1-H1.gwf"

    def test_expand_multiple_list_attributes(self):
        """Test Cartesian product with multiple list attributes."""

        class MockSimulator:
            detectors: ClassVar[list[str]] = ["H1", "L1"]
            channels: ClassVar[list[str]] = ["STRAIN", "AUX"]

        sim = MockSimulator()
        result = _expand_string_template("{{ detectors }}:{{ channels }}", sim)
        # Should return 2D nested array: detectors (2) x channels (2)
        assert result == [["H1:STRAIN", "H1:AUX"], ["L1:STRAIN", "L1:AUX"]]

    def test_expand_mixed_scalar_and_list(self):
        """Test Cartesian product with scalar and list attributes."""

        class MockSimulator:
            detector = "H1"
            channels: ClassVar[list[str]] = ["STRAIN", "AUX"]

        sim = MockSimulator()
        result = _expand_string_template("{{ detector }}:{{ channels }}", sim)
        # Should return 1D list since only channels is an array (scalar x 2-list = 2)
        assert result == ["H1:STRAIN", "H1:AUX"]


class TestExpandTemplateVariables:
    """Tests for expand_template_variables recursive function."""

    def test_expand_string_value(self):
        """Test expanding a string value."""

        class MockSimulator:
            detector = "H1"

        sim = MockSimulator()
        result = expand_template_variables("{{ detector }}-STRAIN", sim)
        assert result == "H1-STRAIN"

    def test_expand_dict_values(self):
        """Test recursively expanding dict values."""

        class MockSimulator:
            detector = "H1"
            channel_type = "STRAIN"

        sim = MockSimulator()
        template_dict = {
            "channel": "{{ detector }}:{{ channel_type }}",
            "static": "value",
        }
        result = expand_template_variables(template_dict, sim)
        assert result == {
            "channel": "H1:STRAIN",
            "static": "value",
        }

    def test_expand_nested_dict_values(self):
        """Test recursively expanding nested dict values."""

        class MockSimulator:
            detector = "H1"

        sim = MockSimulator()
        template_dict = {
            "outer": {
                "inner": "{{ detector }}-STRAIN",
                "static": "value",
            }
        }
        result = expand_template_variables(template_dict, sim)
        assert result == {
            "outer": {
                "inner": "H1-STRAIN",
                "static": "value",
            }
        }

    def test_expand_list_items(self):
        """Test recursively expanding list items."""

        class MockSimulator:
            detector = "H1"

        sim = MockSimulator()
        template_list = ["{{ detector }}-STRAIN", "static", "{{ detector }}-AUX"]
        result = expand_template_variables(template_list, sim)
        assert result == ["H1-STRAIN", "static", "H1-AUX"]

    def test_expand_nested_list_items(self):
        """Test recursively expanding nested list items."""

        class MockSimulator:
            detector = "H1"

        sim = MockSimulator()
        template_list = [
            ["{{ detector }}-STRAIN", "static"],
            ["{{ detector }}-AUX"],
        ]
        result = expand_template_variables(template_list, sim)
        assert result == [
            ["H1-STRAIN", "static"],
            ["H1-AUX"],
        ]

    def test_expand_dict_with_list_items(self):
        """Test recursively expanding dict containing lists."""

        class MockSimulator:
            detector = "H1"

        sim = MockSimulator()
        template = {
            "channels": ["{{ detector }}:STRAIN", "{{ detector }}:AUX"],
            "name": "{{ detector }}",
        }
        result = expand_template_variables(template, sim)
        assert result == {
            "channels": ["H1:STRAIN", "H1:AUX"],
            "name": "H1",
        }

    def test_expand_non_string_values_unchanged(self):
        """Test that non-string values in non-template contexts are unchanged."""

        class MockSimulator:
            pass

        sim = MockSimulator()
        value = 42
        result = expand_template_variables(value, sim)
        assert result == value

    def test_expand_multiple_attributes(self):
        """Test expansion using multiple simulator attributes."""

        class MockSimulator:
            detector = "H1"
            sampling_frequency = 4096

        sim = MockSimulator()
        result = expand_template_variables(
            {
                "channel": "{{ detector }}:STRAIN",
                "fs": "{{ sampling_frequency }}",
            },
            sim,
        )
        assert result == {
            "channel": "H1:STRAIN",
            "fs": "4096",
        }

    def test_expand_complex_structure(self):
        """Test expansion with complex nested structure."""

        class MockSimulator:
            detector = "H1"
            counter = 5

        sim = MockSimulator()
        template = {
            "file": "{{ detector }}-STRAIN-{{ counter }}.gwf",
            "metadata": {
                "detector": "{{ detector }}",
                "channels": ["{{ detector }}:STRAIN", "{{ detector }}:AUX"],
            },
        }
        result = expand_template_variables(template, sim)
        assert result == {
            "file": "H1-STRAIN-5.gwf",
            "metadata": {
                "detector": "H1",
                "channels": ["H1:STRAIN", "H1:AUX"],
            },
        }

    def test_expand_empty_dict(self):
        """Test expanding an empty dict."""

        class MockSimulator:
            pass

        sim = MockSimulator()
        result = expand_template_variables({}, sim)
        assert result == {}

    def test_expand_empty_list(self):
        """Test expanding an empty list."""

        class MockSimulator:
            pass

        sim = MockSimulator()
        result = expand_template_variables([], sim)
        assert result == []


class TestTemplateExpansionIntegration:
    """Integration tests for template expansion with realistic scenarios."""

    def test_output_config_expansion(self):
        """Test expanding output configuration similar to real use case."""

        class MockSimulator:
            detectors: ClassVar[list[str]] = ["H1", "L1"]
            start_time = 1696291200
            duration = 4096

        sim = MockSimulator()
        output_args = {
            "channel": "{{ detectors }}:STRAIN",
            "metadata_key": "{{ start_time }}-{{ duration }}",
        }
        result = expand_template_variables(output_args, sim)
        assert result == {
            "channel": ["H1:STRAIN", "L1:STRAIN"],
            "metadata_key": "1696291200-4096",
        }

    def test_detector_specific_channels(self):
        """Test expanding channel names for detector-specific output."""

        class MockSimulator:
            detector = "H1"

        sim = MockSimulator()
        channel_template = "{{ detector }}:STRAIN"
        result = expand_template_variables(channel_template, sim)
        assert result == "H1:STRAIN"

    def test_quantity_in_waveform_parameters(self):
        """Test expanding waveform parameters with Quantity objects."""

        class MockSimulator:
            minimum_frequency = Quantity(20, "Hz")
            maximum_frequency = Quantity(2048, "Hz")

        sim = MockSimulator()
        params = {
            "f_min": "{{ minimum_frequency }}",
            "f_max": "{{ maximum_frequency }}",
        }
        result = expand_template_variables(params, sim)
        assert result == {
            "f_min": "20",
            "f_max": "2048",
        }

    def test_changing_simulator_state(self):
        """Test that expansion captures current simulator state."""

        class MockSimulator:
            counter = 0

        sim = MockSimulator()

        # First expansion
        result1 = _expand_string_template("batch-{{ counter }}.gwf", sim)
        assert result1 == "batch-0.gwf"

        # Simulator state changes
        sim.counter = 5

        # Second expansion captures new state
        result2 = _expand_string_template("batch-{{ counter }}.gwf", sim)
        assert result2 == "batch-5.gwf"
