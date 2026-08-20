"""The YAML representers that let a resolved config carry units and arrays.

Importing ``gwmock.cli.utils`` registers these on ``yaml.SafeDumper``/``SafeLoader``, so every
``safe_dump`` of a resolved configuration or a saved plan goes through them. They are the reason
a ``Quantity`` survives a round trip through the config on disk instead of becoming a string,
and the reason an array survives as bytes instead of a lossy list of floats.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from astropy.units import Quantity

import gwmock.cli.utils  # noqa: F401  (imported for the side effect of registering the tags)

pytestmark = pytest.mark.unit


class TestQuantities:
    def test_a_quantity_round_trips(self) -> None:
        loaded = yaml.safe_load(yaml.safe_dump({"duration": Quantity(4096.0, "s")}))["duration"]
        assert isinstance(loaded, Quantity)
        assert loaded.value == 4096.0
        assert str(loaded.unit) == "s"

    def test_it_is_written_under_its_own_tag(self) -> None:
        """The tag is the contract with anything that reads these files without astropy."""
        assert "!Quantity" in yaml.safe_dump({"duration": Quantity(1.0, "s")})

    def test_the_value_and_unit_are_written_as_separate_keys(self) -> None:
        """A mapping of ``value``/``unit``, not a single string: the loader reads them by name."""
        text = yaml.safe_dump({"d": Quantity(2048.0, "Hz")})
        assert "value:" in text
        assert "unit:" in text

    def test_an_integer_quantity_is_written_as_a_float(self) -> None:
        """``float(obj.value)`` is load-bearing: a numpy scalar is not a YAML scalar type, and
        SafeDumper refuses it outright."""
        text = yaml.safe_dump({"n": Quantity(4, "s")})
        assert "4.0" in text
        assert yaml.safe_load(text)["n"].value == 4.0

    def test_a_unit_that_is_not_seconds_survives(self) -> None:
        loaded = yaml.safe_load(yaml.safe_dump({"f": Quantity(16384.0, "Hz")}))["f"]
        assert str(loaded.unit) == "Hz"
        assert loaded.value == 16384.0

    def test_a_derived_unit_survives(self) -> None:
        loaded = yaml.safe_load(yaml.safe_dump({"rate": Quantity(3.5, "m / s")}))["rate"]
        assert loaded == Quantity(3.5, "m / s")


class TestArrays:
    def test_a_float_array_round_trips_exactly(self) -> None:
        """Base64 of the raw bytes, so the values come back bit for bit rather than rounded."""
        array = np.array([1.0, 1 / 3, np.pi, -2.5e-24])
        loaded = yaml.safe_load(yaml.safe_dump({"a": array}))["a"]
        assert np.array_equal(loaded, array)

    def test_the_dtype_is_preserved(self) -> None:
        loaded = yaml.safe_load(yaml.safe_dump({"a": np.array([1, 2, 3], dtype=np.int32)}))["a"]
        assert loaded.dtype == np.int32

    def test_a_float32_array_does_not_come_back_widened(self) -> None:
        loaded = yaml.safe_load(yaml.safe_dump({"a": np.arange(4, dtype=np.float32)}))["a"]
        assert loaded.dtype == np.float32

    def test_the_shape_is_preserved(self) -> None:
        array = np.arange(6, dtype=np.float64).reshape(2, 3)
        loaded = yaml.safe_load(yaml.safe_dump({"a": array}))["a"]
        assert loaded.shape == (2, 3)
        assert np.array_equal(loaded, array)

    def test_an_empty_array_round_trips(self) -> None:
        loaded = yaml.safe_load(yaml.safe_dump({"a": np.array([], dtype=np.float64)}))["a"]
        assert loaded.shape == (0,)

    def test_it_is_written_under_its_own_tag_with_the_encoding_named(self) -> None:
        text = yaml.safe_dump({"a": np.arange(3)})
        assert "!ndarray" in text
        assert "base64" in text

    def test_the_payload_is_not_the_repr_of_the_array(self) -> None:
        """A representer that fell back to a string would still round trip as *something*; the
        point of this one is that the bytes are the payload."""
        assert "1.0" not in yaml.safe_dump({"a": np.array([1.0, 2.0])})

    def test_an_unknown_encoding_is_refused(self) -> None:
        """The encoding key is checked rather than assumed, so a hand-edited or future-format
        document fails loudly instead of decoding garbage into an array."""
        document = "a: !ndarray\n  data: AAAA\n  dtype: float64\n  shape: [1]\n  encoding: hex\n"
        with pytest.raises(ValueError, match="base64"):
            yaml.safe_load(document)

    def test_a_missing_encoding_is_refused(self) -> None:
        document = "a: !ndarray\n  data: AAAA\n  dtype: float64\n  shape: [1]\n"
        with pytest.raises(ValueError, match="base64"):
            yaml.safe_load(document)


def test_a_quantity_and_an_array_in_one_document(tmp_path) -> None:
    """Both tags in one file, which is what a resolved config actually looks like."""
    path = tmp_path / "resolved.yaml"
    payload = {"sampling_frequency": Quantity(4096.0, "Hz"), "psd": np.array([1e-46, 2e-46])}
    path.write_text(yaml.safe_dump(payload))
    loaded = yaml.safe_load(path.read_text())
    assert loaded["sampling_frequency"] == Quantity(4096.0, "Hz")
    assert np.array_equal(loaded["psd"], payload["psd"])
