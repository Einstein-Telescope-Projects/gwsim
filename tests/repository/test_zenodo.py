"""Unit tests for the ZenodoClient."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from gwsim.repository.zenodo import ZenodoClient


@pytest.fixture
def zenodo_client():
    """Fixture to create a ZenodoClient instance."""
    return ZenodoClient(access_token="fake_token", sandbox=True)  # nosec B106


@pytest.fixture
def mock_response():
    """Fixture to create a mock response."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"id": "123", "links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
    return response


class TestZenodoClient:
    """Test suite for ZenodoClient."""

    def test_init(self):
        """Test client initialization."""
        client = ZenodoClient("token", sandbox=True)
        assert client.access_token == "token"
        assert client.sandbox is True
        assert client.base_url == "https://sandbox.zenodo.org/api/"
        assert client.headers == {"Authorization": "Bearer token"}

        client_prod = ZenodoClient("token", sandbox=False)
        assert client_prod.base_url == "https://zenodo.org/api/"

    @patch("gwsim.repository.zenodo.requests.request")
    def test_request_success(self, mock_request, zenodo_client, mock_response):
        """Test successful _request call."""
        mock_request.return_value = mock_response
        result = zenodo_client._request("GET", "test_url", headers={}, timeout=60).json()
        assert result == {"id": "123", "links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        mock_request.assert_called_once()

    @patch("gwsim.repository.zenodo.requests.request")
    def test_request_failure(self, mock_request, zenodo_client):
        """Test _request with HTTP error."""
        mock_request.side_effect = requests.HTTPError("404")
        with pytest.raises(requests.HTTPError):
            zenodo_client._request("GET", "test_url", headers={}, timeout=60)

    @patch.object(ZenodoClient, "_request")
    def test_create_deposition(self, mock_request_method, zenodo_client, mock_response):
        """Test create_deposition."""
        mock_request_method.return_value = mock_response
        result = zenodo_client.create_deposition(metadata={"title": "Test"})
        assert result == {"id": "123", "links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        mock_request_method.assert_called_with(
            "POST",
            "https://sandbox.zenodo.org/api/deposit/depositions",
            headers={"Content-Type": "application/json", "Authorization": "Bearer fake_token"},
            timeout=60,
            json={"metadata": {"title": "Test"}},
        )

    @patch.object(ZenodoClient, "_request")
    @patch.object(ZenodoClient, "get_deposition")
    def test_upload_file(self, mock_get_deposition, mock_request_method, zenodo_client, tmp_path, mock_response):
        """Test upload_file."""
        mock_get_deposition.return_value = {"links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        mock_request_method.return_value = mock_response

        file_path = tmp_path / "test.txt"
        file_path.write_text("test content")

        result = zenodo_client.upload_file("dep123", file_path)
        assert result == {"id": "123", "links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        mock_request_method.assert_called_once()

    @patch.object(ZenodoClient, "_request")
    def test_update_metadata(self, mock_request_method, zenodo_client, mock_response):
        """Test update_metadata."""
        mock_request_method.return_value = mock_response
        result = zenodo_client.update_metadata("dep123", {"title": "Updated"})
        assert result == {"id": "123", "links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        mock_request_method.assert_called_with(
            "PUT",
            "https://sandbox.zenodo.org/api/deposit/depositions/dep123",
            headers={"Content-Type": "application/json", "Authorization": "Bearer fake_token"},
            timeout=60,
            data=json.dumps({"metadata": {"title": "Updated"}}),
        )

    @patch.object(ZenodoClient, "_request")
    def test_publish_deposition(self, mock_request_method, zenodo_client, mock_response):
        """Test publish_deposition."""
        mock_response.json.return_value = {"doi": "10.5281/zenodo.123"}
        mock_request_method.return_value = mock_response
        result = zenodo_client.publish_deposition("dep123")
        assert result == {"doi": "10.5281/zenodo.123"}
        mock_request_method.assert_called_with(
            "POST",
            "https://sandbox.zenodo.org/api/deposit/depositions/dep123/actions/publish",
            headers={"Authorization": "Bearer fake_token"},
            timeout=300,
        )

    @patch.object(ZenodoClient, "_request")
    def test_get_deposition(self, mock_request_method, zenodo_client, mock_response):
        """Test get_deposition."""
        mock_response.json.return_value = {"id": "123"}
        mock_request_method.return_value = mock_response
        result = zenodo_client.get_deposition("dep123")
        assert result == {"id": "123"}
        mock_request_method.assert_called_with(
            "GET",
            "https://sandbox.zenodo.org/api/deposit/depositions/dep123",
            headers={"Content-Type": "application/json", "Authorization": "Bearer fake_token"},
            timeout=60,
        )

    @patch.object(ZenodoClient, "_request")
    def test_download_file(self, mock_request_method, zenodo_client, tmp_path):
        """Test download_file."""
        # Mock response object with iter_content for streaming
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_request_method.return_value = mock_response

        output_path = tmp_path / "downloaded.txt"
        _result = zenodo_client.download_file("dep123", "file.txt", output_path)

        # Verify the file was created with correct content
        assert output_path.exists()
        assert output_path.read_bytes() == b"chunk1chunk2"
        # Verify _request was called
        mock_request_method.assert_called_once()

    @patch.object(ZenodoClient, "_request")
    def test_list_depositions(self, mock_request_method, zenodo_client):
        """Test list_depositions."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"id": "123"}]
        mock_request_method.return_value = mock_response
        result = zenodo_client.list_depositions(status="draft")
        assert result == [{"id": "123"}]
        mock_request_method.assert_called_with(
            "GET",
            "https://sandbox.zenodo.org/api/deposit/depositions",
            headers={"Content-Type": "application/json", "Authorization": "Bearer fake_token"},
            timeout=60,
            params={"status": "draft"},
        )

    @patch.object(ZenodoClient, "_request")
    def test_delete_deposition(self, mock_request_method, zenodo_client):
        """Test delete_deposition."""
        mock_request_method.return_value = {"message": "deleted"}
        result = zenodo_client.delete_deposition("dep123")
        assert result == {"message": "deleted"}
        mock_request_method.assert_called_with(
            "DELETE",
            "https://sandbox.zenodo.org/api/deposit/depositions/dep123",
            headers={"Authorization": "Bearer fake_token"},
            timeout=60,
        )
