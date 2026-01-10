"""Unit tests for the Encoder class."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gwsim.data.serialize.encoder import Encoder


class MockSerializable:
    """Mock class for testing Encoder."""

    def __init__(self, value: int, name: str):
        self.value = value
        self.name = name

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON dict without __type__."""
        return {
            "value": self.value,
            "name": self.name,
        }


class MockSerializableWithType:
    """Mock class with __type__ already in to_json_dict."""

    def __init__(self, data: str):
        self.data = data

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON dict with __type__."""
        return {
            "__type__": "MockSerializableWithType",
            "data": self.data,
        }


class NonSerializable:
    """Mock class without to_json_dict."""

    def __init__(self, item: str):
        self.item = item


class TestEncoder:
    """Test the Encoder class."""

    def test_encodes_serializable_object(self):
        """Test encoding an object with to_json_dict."""
        value = 42
        obj = MockSerializable(value=value, name="test")
        json_str = json.dumps(obj, cls=Encoder)
        data = json.loads(json_str)
        assert data["__type__"] == "MockSerializable"
        assert data["value"] == value
        assert data["name"] == "test"

    def test_adds_type_if_missing(self):
        """Test that __type__ is added if not present in to_json_dict."""
        obj = MockSerializable(value=1, name="add_type")
        json_str = json.dumps(obj, cls=Encoder)
        data = json.loads(json_str)
        assert "__type__" in data
        assert data["__type__"] == "MockSerializable"

    def test_preserves_existing_type(self):
        """Test that existing __type__ is preserved."""
        obj = MockSerializableWithType(data="preserve")
        json_str = json.dumps(obj, cls=Encoder)
        data = json.loads(json_str)
        assert data["__type__"] == "MockSerializableWithType"
        assert data["data"] == "preserve"

    def test_fallback_for_non_serializable(self):
        """Test fallback to default encoding for objects without to_json_dict."""
        obj = NonSerializable(item="fallback")
        # This should raise TypeError since NonSerializable is not JSON serializable
        with pytest.raises(TypeError):
            json.dumps(obj, cls=Encoder)

    def test_encodes_nested_structures(self):
        """Test encoding nested structures with serializable objects."""
        value = 100
        serializable = MockSerializable(value=value, name="nested")
        nested_data = {
            "metadata": {"version": "1.0"},
            "object": serializable,
            "list": [1, 2, serializable],
        }
        json_str = json.dumps(nested_data, cls=Encoder)
        data = json.loads(json_str)
        assert data["metadata"]["version"] == "1.0"
        assert data["object"]["__type__"] == "MockSerializable"
        assert data["object"]["value"] == value
        assert data["list"][0] == 1
        assert data["list"][2]["name"] == "nested"

    def test_handles_non_serializable_in_nested(self):
        """Test that non-serializable objects in nested structures raise errors."""
        non_ser = NonSerializable(item="bad")
        nested_data = {"good": MockSerializable(1, "ok"), "bad": non_ser}
        with pytest.raises(TypeError):
            json.dumps(nested_data, cls=Encoder)
