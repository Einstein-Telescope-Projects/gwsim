"""Zenodo publishing client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from requests import Response

from gwsim.utils.retry import retry_on_failure


def get_deposition_id_from_doi(doi: str) -> tuple[str, bool]:
    """Extract deposition ID from a Zenodo DOI.

    Args:
        doi: The DOI string.

    returns:
        A tuple containing the deposition ID and a boolean indicating if it's from sandbox.
    """
    parts = doi.split(".")
    deposition_id = parts[1]
    if parts[0] == "10.5072/zenodo":
        sandbox = True
    elif parts[0] == "10.5281/zenodo":
        sandbox = False
    else:
        raise ValueError(f"Invalid Zenodo DOI: {doi}")

    return deposition_id, sandbox


class ZenodoClient:
    """Client for interacting with Zenodo API (production and sandbox)."""

    def __init__(self, access_token: str, sandbox: bool = False):
        """Initialize the Zenodo client.

        Args:
            access_token: Zenodo API access token.
            sandbox: Whether to use the sandbox environment. Default is False.
        """
        self.access_token = access_token
        self.sandbox = sandbox
        self.base_url = "https://sandbox.zenodo.org/api/" if sandbox else "https://zenodo.org/api/"

        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
        }

    @retry_on_failure()
    def _request(self, method: str, url: str, headers: dict, timeout: int = 60, **kwargs) -> Response:
        """Make a request to the Zenodo API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            url: URL to make the request to.
            headers: Headers to include in the request.
            timeout: Timeout for the request in seconds. Default is 60.
            **kwargs: Additional arguments to pass to requests.

        Returns:
            Response object.
        """
        response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response

    def create_deposition(self, metadata: dict[str, Any] | None = None, timeout=60) -> dict[str, Any]:
        """Create a new deposition.

        Args:
            metadata: Optional metadata dictionary for the deposition.
            timeout: Timeout for the request in seconds. Default is 60.

        Returns:
            Response JSON as a dictionary.
        """
        data = {"metadata": metadata} if metadata else {}

        response: Response = self._request(
            "POST",
            f"{self.base_url}deposit/depositions",
            headers={"Content-Type": "application/json", **self.headers},
            timeout=timeout,
            json=data,
        )
        return response.json()

    def upload_file(
        self, deposition_id: str, file_path: Path, timeout=300, auto_timeout: bool = True
    ) -> dict[str, Any]:
        """Upload a file to a deposition.

        Args:
            deposition_id: ID of the deposition to upload the file to.
            file_path: Path to the file to upload.
            timeout: Timeout for the request in seconds. Default is 300.
            auto_timeout: Whether to automatically adjust timeout based on file size. Default is True.
                If True, the timeout is set to max(timeout, file_size_in_MB * 10).

        Returns:
            Response JSON as a dictionary.
        """
        deposition = self.get_deposition(deposition_id)

        bucket = deposition["links"]["bucket"]

        if auto_timeout:
            # Get the size of the file in MB
            file_size = file_path.stat().st_size / (1024 * 1024)

            timeout = max(timeout, int(file_size * 10))  # 10 seconds per MB, minimum 300 seconds

        with file_path.open("rb") as f:
            response: Response = self._request(
                "PUT",
                f"{bucket}/{file_path.name}",
                headers=self.headers,
                timeout=timeout,
                data=f,
            )

        return response.json()

    def update_metadata(self, deposition_id: str, metadata: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
        """Update metadata for a deposition.

        Args:
            deposition_id: ID of the deposition to update.
            metadata: Metadata dictionary to update.
            timeout: Timeout for the request in seconds. Default is 60.

        Returns:
            Response JSON as a dictionary.
        """
        data = {"metadata": metadata}
        response: Response = self._request(
            "PUT",
            f"{self.base_url}deposit/depositions/{deposition_id}",
            headers={"Content-Type": "application/json", **self.headers},
            timeout=timeout,
            data=json.dumps(data),
        )

        return response.json()

    def publish_deposition(self, deposition_id: str, timeout: int = 300) -> dict[str, Any]:
        """Publish a deposition.

        Args:
            deposition_id: ID of the deposition to publish.
            timeout: Timeout for the request in seconds. Default is 300.

        Returns:
            Response JSON as a dictionary.
        """
        response: Response = self._request(
            "POST",
            f"{self.base_url}deposit/depositions/{deposition_id}/actions/publish",
            headers=self.headers,
            timeout=timeout,
        )
        return response.json()

    def get_deposition(self, deposition_id: str, timeout: int = 60) -> dict[str, Any]:
        """Retrieve a deposition's details.

        Args:
            deposition_id: ID of the deposition to retrieve.
            timeout: Timeout for the request in seconds. Default is 60.

        Returns:
            Response JSON as a dictionary.
        """
        response: Response = self._request(
            "GET",
            f"{self.base_url}deposit/depositions/{deposition_id}",
            headers={"Content-Type": "application/json", **self.headers},
            timeout=timeout,
        )
        return response.json()

    def download_file(
        self,
        deposition_id: str,
        filename: str,
        output_path: Path,
        is_draft: bool = False,
        timeout: int = 300,
        file_size_in_mb: int | None = None,
    ) -> dict[str, Any]:
        """Download a file from Zenodo.

        Args:
            deposition_id: ID of the deposition.
            filename: Name of the file to download.
            output_path: Path to save the downloaded file.
            is_draft: Whether the file is in a draft deposition. Default is False.
            timeout: Timeout for the request in seconds. Default is 300.
            file_size_in_mb: Optional size of the file in MB to adjust timeout. If provided

        Returns:
            Response JSON as a dictionary.
        """
        if is_draft:
            deposition_id += "/draft"

        file_url = f"{self.base_url}records/{deposition_id}/files/{filename}"

        if file_size_in_mb is not None:
            timeout = max(timeout, int(file_size_in_mb * 10))  # 10 seconds per MB

        response: Response = self._request(
            "GET",
            file_url,
            headers={"Content-Type": "application/json", **self.headers},
            timeout=timeout,
            stream=True,
        )

        # Atomic write to avoid incomplete files
        output_path_tmp = output_path.with_suffix(".tmp")
        with output_path_tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        output_path_tmp.rename(output_path)
        return response.json()

    def list_depositions(self, status: str = "published", timeout: int = 60) -> list[dict[str, Any]]:
        """List all depositions for the authenticated user.

        Args:
            status: Filter by deposition status ('draft', 'unsubmitted', 'published'). Default is 'published'.
            timeout: Timeout for the request in seconds. Default is 60.

        Returns:
            List of deposition dictionaries.
        """
        params = {"status": status}
        response = self._request(
            "GET",
            f"{self.base_url}deposit/depositions",
            headers={"Content-Type": "application/json", **self.headers},
            timeout=timeout,
            params=params,
        )
        return response.json()

    def delete_deposition(self, deposition_id: str, timeout: int = 60) -> dict[str, Any]:
        """Delete a deposition.

        Args:
            deposition_id: ID of the deposition to delete.
            timeout: Timeout for the request in seconds. Default is 60.

        Returns:
            Response JSON as a dictionary.
        """
        response = self._request(
            "DELETE",
            f"{self.base_url}deposit/depositions/{deposition_id}",
            headers=self.headers,
            timeout=timeout,
        )
        return response
