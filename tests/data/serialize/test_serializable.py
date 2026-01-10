"""Unit tests for JSONSerializable protocol using mock classes."""

from __future__ import annotations

import pytest

from gwsim.data.serialize.serializable import JSONSerializable


class MockSerializable(JSONSerializable):
    """Mock class implementing JSONSerializable for testing."""

    def __init__(self, value: int, name: str):
        self.value = value
        self.name = name

    def to_json_dict(self) -> dict[str, int | str]:
        """Convert to JSON dict."""
        return {
            "__type__": "MockSerializable",
            "value": self.value,
            "name": self.name,
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict[str, int | str]) -> MockSerializable:
        """Create from JSON dict."""
        return cls(
            value=int(json_dict["value"]),
            name=str(json_dict["name"]),
        )


class MockNestedSerializable(JSONSerializable):
    """Mock class with nested serializable objects."""

    def __init__(self, mock_obj: MockSerializable, count: int):
        self.mock_obj = mock_obj
        self.count = count

    def to_json_dict(self) -> dict[str, int | dict]:
        """Convert to JSON dict."""
        return {
            "__type__": "MockNestedSerializable",
            "mock_obj": self.mock_obj.to_json_dict(),
            "count": self.count,
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict[str, int | dict]) -> MockNestedSerializable:
        """Create from JSON dict."""
        mock_obj_dict = json_dict["mock_obj"]
        if isinstance(mock_obj_dict, dict):
            mock_obj = MockSerializable.from_json_dict(mock_obj_dict)
        else:
            raise ValueError("Invalid mock_obj in JSON")
        return cls(
            mock_obj=mock_obj,
            count=int(json_dict["count"]),
        )


@pytest.fixture
def mock_obj() -> MockSerializable:
    """Fixture for MockSerializable."""
    return MockSerializable(value=42, name="test")


@pytest.fixture
def mock_nested_obj(mock_obj: MockSerializable) -> MockNestedSerializable:
    """Fixture for MockNestedSerializable."""
    return MockNestedSerializable(mock_obj=mock_obj, count=10)


class TestJSONSerializableProtocol:
    """Test the JSONSerializable protocol with mock classes."""

    def test_to_json_dict_returns_dict_with_type(self, mock_obj: MockSerializable):
        """Test that to_json_dict returns a dict with __type__."""
        result = mock_obj.to_json_dict()
        assert isinstance(result, dict)
        assert "__type__" in result
        assert result["__type__"] == "MockSerializable"
        expected_value = 42
        assert result["value"] == expected_value
        assert result["name"] == "test"

    def test_from_json_dict_reconstructs_object(self, mock_obj: MockSerializable):
        """Test that from_json_dict reconstructs the object."""
        json_dict = mock_obj.to_json_dict()
        reconstructed = MockSerializable.from_json_dict(json_dict)
        assert isinstance(reconstructed, MockSerializable)
        assert reconstructed.value == mock_obj.value
        assert reconstructed.name == mock_obj.name

    def test_round_trip_serialization(self, mock_obj: MockSerializable):
        """Test round-trip serialization preserves state."""
        json_dict = mock_obj.to_json_dict()
        reconstructed = MockSerializable.from_json_dict(json_dict)
        assert reconstructed.value == mock_obj.value
        assert reconstructed.name == mock_obj.name

    def test_nested_serialization(self, mock_nested_obj: MockNestedSerializable):
        """Test serialization of nested serializable objects."""
        json_dict = mock_nested_obj.to_json_dict()
        assert json_dict["__type__"] == "MockNestedSerializable"
        assert isinstance(json_dict["mock_obj"], dict)
        assert json_dict["mock_obj"]["__type__"] == "MockSerializable"

    def test_nested_deserialization(self, mock_nested_obj: MockNestedSerializable):
        """Test deserialization of nested objects."""
        json_dict = mock_nested_obj.to_json_dict()
        reconstructed = MockNestedSerializable.from_json_dict(json_dict)
        assert isinstance(reconstructed, MockNestedSerializable)
        assert reconstructed.count == mock_nested_obj.count
        assert reconstructed.mock_obj.value == mock_nested_obj.mock_obj.value
        assert reconstructed.mock_obj.name == mock_nested_obj.mock_obj.name

    def test_invalid_json_dict_raises_error(self):
        """Test that invalid JSON dict raises ValueError."""
        invalid_dict = {"__type__": "MockSerializable", "value": "not_int"}
        with pytest.raises((ValueError, KeyError)):
            MockSerializable.from_json_dict(invalid_dict)
