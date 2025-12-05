"""Zenodo publishing client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


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

    def create_deposition(self, metadata: dict[str, Any] | None = None, timeout=60) -> dict[str, Any]:
        """Create a new deposition.

        Args:
            metadata: Optional metadata dictionary for the deposition.
            timeout: Timeout for the request in seconds. Default is 60.

        Returns:
            Response JSON as a dictionary.
        """
        data = {"metadata": metadata} if metadata else {}

        response = requests.post(
            f"{self.base_url}deposit/depositions",
            headers={"Content-Type": "application/json", **self.headers},
            json=data,
            timeout=timeout,
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
            response = requests.put(
                f"{bucket}/{file_path.name}",
                data=f,
                headers=self.headers,
                timeout=timeout,
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
        response = requests.put(
            f"{self.base_url}deposit/depositions/{deposition_id}",
            data=data,
            headers={"Content-Type": "application/json", **self.headers},
            timeout=timeout,
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
        response = requests.post(
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
        response = requests.get(
            f"{self.base_url}deposit/depositions/{deposition_id}",
            headers={"Content-Type": "application/json", **self.headers},
            timeout=timeout,
        )
        return response.json()

    def download_file(
        self,
        doi: str,
        filename: str,
        output_path: Path,
        is_draft: bool = False,
        timeout: int = 300,
        file_size_in_mb: int | None = None,
    ) -> None:
        """Download a file from Zenodo.

        Args:
            file_url: URL of the file to download.
            destination: Path to save the downloaded file.
        """
        record_id = doi.split(".")[-1]
        if is_draft:
            record_id += "/draft"

        file_url = f"{self.base_url}records/{record_id}/files/{filename}"

        if file_size_in_mb is not None:
            timeout = max(timeout, int(file_size_in_mb * 10))  # 10 seconds per MB

        response = requests.get(
            file_url, headers={"Content-Type": "application/json", **self.headers}, timeout=timeout, stream=True
        )

        # Atomic write to avoid incomplete files
        output_path_tmp = output_path.with_suffix(".tmp")
        with output_path_tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        output_path_tmp.rename(output_path)

    def list_depositions(self, status: str = "published", timeout: int = 60) -> list[dict[str, Any]]:
        """List all depositions for the authenticated user.

        Args:
            status: Filter by deposition status ('draft', 'unsubmitted', 'published'). Default is 'published'.
            timeout: Timeout for the request in seconds. Default is 60.

        Returns:
            List of deposition dictionaries.
        """
        params = {"status": status}
        response = requests.get(
            f"{self.base_url}deposit/depositions",
            headers={"Content-Type": "application/json", **self.headers},
            params=params,
            timeout=timeout,
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
        response = requests.delete(
            f"{self.base_url}deposit/depositions/{deposition_id}",
            headers={"Content-Type": "application/json", **self.headers},
            timeout=timeout,
        )
        return response.json()
