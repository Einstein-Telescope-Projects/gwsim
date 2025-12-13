"""Unit tests for io utility functions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gwsim.utils.io import atomic_writer, get_file_name_from_template


class MockInstance:
    """Mock instance for testing template expansion."""

    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


class TestGetFileNameFromTemplate:
    """Test suite for get_file_name_from_template function."""

    def test_no_placeholders(self):
        """Test template with no placeholders returns as-is."""
        instance = MockInstance()
        template = "static_filename.txt"
        result = get_file_name_from_template(template, instance)
        assert result == Path("static_filename.txt")

    def test_single_placeholder_non_array(self):
        """Test single non-array placeholder substitution."""
        instance = MockInstance(name="test")
        template = "{{ name }}.txt"
        result = get_file_name_from_template(template, instance)
        assert result == Path("test.txt")

    def test_single_placeholder_array(self):
        """Test single array placeholder returns list."""
        instance = MockInstance(items=[1, 2, 3])
        template = "file_{{ items }}.txt"
        result = get_file_name_from_template(template, instance)

        assert np.array_equal(result, np.array([Path("file_1.txt"), Path("file_2.txt"), Path("file_3.txt")]))

    def test_multiple_placeholders_no_arrays(self):
        """Test multiple non-array placeholders."""
        instance = MockInstance(prefix="data", suffix="end")
        template = "{{ prefix }}_{{ suffix }}.txt"
        result = get_file_name_from_template(template, instance)
        assert result == Path("data_end.txt")

    def test_multiple_placeholders_with_arrays(self):
        """Test multiple placeholders including arrays."""
        instance = MockInstance(prefix="data", numbers=[1, 2], letters=["a", "b"])
        template = "{{ prefix }}_{{ numbers }}_{{ letters }}.txt"
        result = get_file_name_from_template(template, instance)
        expected = [[Path("data_1_a.txt"), Path("data_1_b.txt")], [Path("data_2_a.txt"), Path("data_2_b.txt")]]
        assert np.array_equal(result, np.array(expected))

    def test_excluded_placeholders(self):
        """Test excluded placeholders are not substituted."""
        instance = MockInstance(name="test", excluded="ignore")
        template = "{{ name }}_{{ excluded }}.txt"
        result = get_file_name_from_template(template, instance, exclude={"excluded"})
        assert result == Path("test_{{ excluded }}.txt")

    def test_excluded_with_arrays(self):
        """Test excluded array placeholders are not expanded."""
        instance = MockInstance(name="test", excluded=[1, 2])
        template = "{{ name }}_{{ excluded }}.txt"
        result = get_file_name_from_template(template, instance, exclude={"excluded"})
        assert result == Path("test_{{ excluded }}.txt")

    def test_missing_attribute(self):
        """Test missing attribute raises ValueError."""
        instance = MockInstance(name="test")
        template = "{{ name }}_{{ missing }}.txt"
        with pytest.raises(ValueError, match="Attribute 'missing' not found"):
            get_file_name_from_template(template, instance)

    def test_tuple_as_array(self):
        """Test tuple is treated as array-like."""
        instance = MockInstance(items=(1, 2))
        template = "file_{{ items }}.txt"
        result = get_file_name_from_template(template, instance)
        assert np.array_equal(result, np.array([Path("file_1.txt"), Path("file_2.txt")]))

    def test_iterable_non_string(self):
        """Test iterable (non-string) is treated as array-like."""
        instance = MockInstance(items=range(3))
        template = "file_{{ items }}.txt"
        result = get_file_name_from_template(template, instance)
        assert np.array_equal(result, np.array([Path("file_0.txt"), Path("file_1.txt"), Path("file_2.txt")]))

    def test_string_not_treated_as_array(self):
        """Test string is not treated as array-like."""
        instance = MockInstance(name="hello")
        template = "{{ name }}.txt"
        result = get_file_name_from_template(template, instance)
        assert result == Path("hello.txt")

    def test_empty_template(self):
        """Test empty template."""
        instance = MockInstance()
        template = ""
        result = get_file_name_from_template(template, instance)
        assert result == Path("")

    def test_duplicate_placeholders(self):
        """Test duplicate placeholders are handled correctly."""
        instance = MockInstance(name="test")
        template = "{{ name }}_{{ name }}.txt"
        result = get_file_name_from_template(template, instance)
        assert result == Path("test_test.txt")

    def test_mixed_excluded_and_included(self):
        """Test mix of excluded and included placeholders with arrays."""
        instance = MockInstance(included=[1, 2], excluded=["a", "b"])
        template = "{{ included }}_{{ excluded }}.txt"
        result = get_file_name_from_template(template, instance, exclude={"excluded"})
        expected = [Path("1_{{ excluded }}.txt"), Path("2_{{ excluded }}.txt")]
        assert np.array_equal(result, np.array(expected))

    def test_output_directory_single_file(self):
        """Test output_directory with single file name."""
        instance = MockInstance(name="test")
        template = "{{ name }}.txt"
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_file_name_from_template(template, instance, output_directory=tmpdir)
            expected = Path(tmpdir) / "test.txt"
            assert result == expected

    def test_output_directory_with_arrays(self):
        """Test output_directory with array placeholders."""
        instance = MockInstance(items=[1, 2, 3])
        template = "file_{{ items }}.txt"
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_file_name_from_template(template, instance, output_directory=tmpdir)
            expected = np.array(
                [
                    Path(tmpdir) / "file_1.txt",
                    Path(tmpdir) / "file_2.txt",
                    Path(tmpdir) / "file_3.txt",
                ]
            )
            assert np.array_equal(result, expected)

    def test_output_directory_with_multiple_arrays(self):
        """Test output_directory with multiple array placeholders."""
        instance = MockInstance(prefix="data", numbers=[1, 2], letters=["a", "b"])
        template = "{{ prefix }}_{{ numbers }}_{{ letters }}.txt"
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_file_name_from_template(template, instance, output_directory=tmpdir)
            expected = np.array(
                [
                    [Path(tmpdir) / "data_1_a.txt", Path(tmpdir) / "data_1_b.txt"],
                    [Path(tmpdir) / "data_2_a.txt", Path(tmpdir) / "data_2_b.txt"],
                ]
            )
            assert np.array_equal(result, expected)

    def test_output_directory_pathlib_input(self):
        """Test output_directory accepts Path object."""
        instance = MockInstance(name="test")
        template = "{{ name }}.txt"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            result = get_file_name_from_template(template, instance, output_directory=tmpdir_path)
            expected = tmpdir_path / "test.txt"
            assert result == expected

    def test_output_directory_none(self):
        """Test output_directory=None does not prepend path."""
        instance = MockInstance(name="test")
        template = "{{ name }}.txt"
        result = get_file_name_from_template(template, instance, output_directory=None)
        assert result == Path("test.txt")


class TestAtomicWriter:
    """Test suite for atomic_writer context manager."""

    def test_successful_write_with_open(self, tmp_path):
        """Test successful atomic write using built-in open."""
        file_path = tmp_path / "test.txt"
        test_content = "Hello, World!"

        with atomic_writer(file_path, mode="w") as f:
            f.write(test_content)

        # Check final file exists and has content
        assert file_path.exists()
        assert file_path.read_text() == test_content

        # Check temp file is cleaned up
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        assert not temp_path.exists()

    def test_failure_during_write(self, tmp_path):
        """Test that temp file is cleaned up on failure."""
        file_path = tmp_path / "test.txt"

        def _write_and_fail():
            with atomic_writer(file_path, mode="w") as f:
                f.write("partial content")
                raise ValueError("Simulated failure")

        with pytest.raises(ValueError, match="Simulated failure"):
            _write_and_fail()

        # Check final file does not exist
        assert not file_path.exists()

        # Check temp file is cleaned up
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        assert not temp_path.exists()

    def test_custom_open_func(self, tmp_path, monkeypatch):
        """Test atomic_writer with a custom open function (e.g., h5py)."""
        file_path = tmp_path / "test.h5"

        # Mock a file-like object
        mock_file = MagicMock()
        mock_open_func = MagicMock(return_value=mock_file)

        # Mock shutil.move to avoid actual file operations
        mock_move = MagicMock()
        monkeypatch.setattr("shutil.move", mock_move)

        with atomic_writer(file_path, mock_open_func, mode="w") as f:
            f.write("data")

        # Verify the mock was called with temp path
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        mock_open_func.assert_called_once_with(str(temp_path), mode="w")

        # Verify file was closed and move was attempted
        mock_file.close.assert_called_once()
        mock_move.assert_called_once_with(str(temp_path), str(file_path))

    def test_file_already_exists_overwrite(self, tmp_path):
        """Test atomic write when final file already exists."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("existing content")

        new_content = "new content"
        with atomic_writer(file_path, mode="w") as f:
            f.write(new_content)

        # Check content was overwritten
        assert file_path.read_text() == new_content

    def test_temp_file_cleanup_on_exception_in_yield(self, tmp_path):
        """Test temp file cleanup when exception occurs inside the context."""
        file_path = tmp_path / "test.txt"

        def _write_and_fail():
            with atomic_writer(file_path, mode="w") as f:
                f.write("start")
                raise RuntimeError("Error inside context")

        with pytest.raises(RuntimeError):
            _write_and_fail()

        # Final file should not exist
        assert not file_path.exists()

        # Temp file should be cleaned up
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        assert not temp_path.exists()
