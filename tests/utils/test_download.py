"""Unit tests for download utility functions."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import filelock
import pytest
import requests

from gwsim.utils.download import determine_dest_path, download_file, download_file_with_lock, handle_existing_file


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_response():
    """Mock requests.Response object."""
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.headers = {}
    # Return an iterator that yields data chunks
    response.iter_content = MagicMock(return_value=iter([b"test data"]))
    response.raise_for_status.return_value = None
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=None)
    return response


def test_download_file_basic(temp_dir, mocker, mock_response):
    """Test basic file download."""
    url = "https://example.com/file.txt"
    mock_response.headers = {"Content-Type": "application/octet-stream"}
    mocker.patch("gwsim.utils.download.requests.get", return_value=mock_response)

    dest_path = download_file(url, outdir=temp_dir)

    assert dest_path.exists()
    assert dest_path.name == "file.txt"
    with dest_path.open("rb") as f:
        assert f.read() == b"test data"


def test_download_file_with_dest_path(temp_dir, mocker, mock_response):
    """Test download with specified dest_path."""
    url = "https://example.com/file.txt"
    mocker.patch("gwsim.utils.download.requests.get", return_value=mock_response)

    dest_path = download_file(url, dest_path="custom.txt", outdir=temp_dir)

    assert dest_path.exists()
    assert dest_path.name == "custom.txt"


def test_download_file_hashed_url(temp_dir, mocker, mock_response):
    """Test download with hashed URL filename."""
    url = "https://example.com/file.txt"
    mocker.patch("gwsim.utils.download.requests.get", return_value=mock_response)

    dest_path = download_file(url, dest_path_from_hashed_url=True, outdir=temp_dir)

    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    assert dest_path.name == f"{url_hash}.txt"


def test_download_file_allow_existing(temp_dir, mocker, mock_response):
    """Test skipping download if file exists and allow_existing=True."""
    url = "https://example.com/file.txt"
    existing_file = temp_dir / "file.txt"
    existing_file.write_text("existing")

    dest_path = download_file(url, outdir=temp_dir, allow_existing=True)

    assert dest_path == existing_file
    # Ensure requests.get was not called
    mocker.patch("requests.get", return_value=mock_response)
    # Since file exists, no download should happen


def test_download_file_overwrite_false_existing(temp_dir, mocker, mock_response):
    """Test raising error if file exists and overwrite=False, allow_existing=False."""
    url = "https://example.com/file.txt"
    existing_file = temp_dir / "file.txt"
    existing_file.write_text("existing")

    with pytest.raises(FileExistsError):
        download_file(url, outdir=temp_dir, overwrite=False, allow_existing=False)


def test_download_file_infer_extension(temp_dir, mocker):
    """Test inferring file extension from Content-Type."""
    url = "https://example.com/file"
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/plain"}
    mock_response.iter_content = MagicMock(return_value=iter([b"test data"]))
    mock_response.raise_for_status.return_value = None
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    mocker.patch("gwsim.utils.download.requests.get", return_value=mock_response)

    dest_path = download_file(url, outdir=temp_dir)

    assert dest_path.suffix == ".txt"


def test_download_file_request_exception(temp_dir, mocker):
    """Test handling of request exceptions."""
    url = "https://example.com/file.txt"
    mocker.patch("gwsim.utils.download.requests.get", side_effect=requests.RequestException("Network error"))

    with pytest.raises(ValueError, match="Failed to download file"):
        download_file(url, outdir=temp_dir)


def test_download_file_lock_timeout(temp_dir, mocker, mock_response):
    """Test handling of lock timeout."""
    url = "https://example.com/file.txt"
    mocker.patch("gwsim.utils.download.requests.get", return_value=mock_response)
    mocker.patch("gwsim.utils.download.filelock.FileLock", side_effect=filelock.Timeout("Lock timeout"))

    with pytest.raises(ValueError, match="Timeout waiting for download lock"):
        download_file(url, outdir=temp_dir, timeout=1)


def test_download_file_no_extension_fallback(temp_dir, mocker):
    """Test fallback to .bin when no extension can be inferred."""
    url = "https://example.com/file"
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/octet-stream"}
    mock_response.iter_content = MagicMock(return_value=iter([b"test data"]))
    mock_response.raise_for_status.return_value = None
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    mocker.patch("gwsim.utils.download.requests.get", return_value=mock_response)

    dest_path = download_file(url, outdir=temp_dir)

    assert dest_path.suffix == ".bin"


def test_determine_dest_path(temp_dir):
    """Test determining destination path from URL."""
    url = "https://example.com/file.txt"
    dest_path = determine_dest_path(url, outdir=temp_dir)
    assert dest_path == temp_dir / "file.txt"


def test_determine_dest_path_hashed(temp_dir):
    """Test determining destination path with hashed URL."""
    url = "https://example.com/file.txt"
    dest_path = determine_dest_path(url, outdir=temp_dir, dest_path_from_hashed_url=True)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    assert dest_path == temp_dir / f"{url_hash}.txt"


def test_determine_dest_path_with_dest_path(temp_dir):
    """Test determining destination path with provided dest_path."""
    url = "https://example.com/file.txt"
    dest_path = determine_dest_path(url, dest_path="custom.txt", outdir=temp_dir)
    assert dest_path == temp_dir / "custom.txt"


def test_handle_existing_file_allow_existing(temp_dir):
    """Test handle_existing_file with allow_existing=True."""
    dest_path = temp_dir / "file.txt"
    dest_path.write_text("existing")
    result = handle_existing_file(dest_path, overwrite=False, allow_existing=True)
    assert result == dest_path


def test_handle_existing_file_overwrite_false(temp_dir):
    """Test handle_existing_file with overwrite=False, allow_existing=False."""
    dest_path = temp_dir / "file.txt"
    dest_path.write_text("existing")
    with pytest.raises(FileExistsError):
        handle_existing_file(dest_path, overwrite=False, allow_existing=False)


def test_handle_existing_file_no_file(temp_dir):
    """Test handle_existing_file when file does not exist."""
    dest_path = temp_dir / "file.txt"
    result = handle_existing_file(dest_path, overwrite=False, allow_existing=True)
    assert result is None


def test_download_file_with_lock(temp_dir, mocker, mock_response):
    """Test download_file_with_lock function."""
    url = "https://example.com/file.txt"
    dest_path = temp_dir / "file.txt"
    lock_path = dest_path.with_suffix(dest_path.suffix + ".lock")
    mock_response.headers = {"Content-Type": "application/octet-stream"}
    mocker.patch("gwsim.utils.download.requests.get", return_value=mock_response)

    result = download_file_with_lock(url, dest_path, lock_path, timeout=300)

    assert result == dest_path
    assert dest_path.exists()
    with dest_path.open("rb") as f:
        assert f.read() == b"test data"
