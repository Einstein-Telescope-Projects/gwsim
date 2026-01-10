"""Utility functions for downloading files with concurrency control."""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse

import filelock
import requests

logger = logging.getLogger("gwsim")


def determine_dest_path(
    url: str,
    dest_path: Path | str | None = None,
    outdir: Path | str | None = None,
    dest_path_from_hashed_url: bool = False,
) -> Path:
    """Determine the destination path for the downloaded file.

    Args:
        url: The URL to download the file from.
        dest_path: The destination file path or name. If None, the file name is derived from the URL.
        outdir: The output directory to save the file. Defaults to the current working directory.
        dest_path_from_hashed_url: If True, derive the file name from a hash of the URL. Default is False.

    Returns:
        The determined destination file path.
    """
    outdir = Path.cwd() if outdir is None else Path(outdir)

    if dest_path is None:
        parsed_url = urlparse(url)
        if dest_path_from_hashed_url:
            _, url_ext = os.path.splitext(parsed_url.path)
            url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
            return outdir / f"{url_hash}{url_ext}"
        return outdir / os.path.basename(parsed_url.path)
    return outdir / Path(dest_path)


def handle_existing_file(dest_path: Path, overwrite: bool, allow_existing: bool = True) -> Path | None:
    """Handle existing file based on overwrite and allow_existing flags.

    Args:
        dest_path: The destination file path.
        overwrite: Whether to overwrite the file if it already exists.
        allow_existing: If True and the file exists, skip downloading.

    Returns:
        The dest_path if skipping download, None otherwise.
    """
    if not overwrite and dest_path.exists():
        if allow_existing:
            logger.info("File %s already exists. Skipping download.", dest_path)
            return dest_path
        raise FileExistsError(f"File {dest_path} already exists and overwrite is set to False.")
    return None


def download_file_with_lock(url: str, dest_path: Path | str, lock_path: Path | str, timeout: int) -> Path:
    """Download a file with a file lock to prevent concurrent downloads.

    Args:
        url: The URL to download the file from.
        dest_path: The destination file path.
        lock_path: The path for the lock file.
        timeout: Timeout in seconds for acquiring the lock.

    Returns:
        The path to the downloaded file.
    """
    dest_path = Path(dest_path)
    lock_path = Path(lock_path)

    with filelock.FileLock(lock_path, timeout=timeout):
        if Path(dest_path).exists():
            logger.info("File was downloaded by another process: %s", dest_path)
            return dest_path

        with requests.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()

            if not dest_path.suffix:
                content_type = response.headers.get("Content-Type", "")
                guessed_ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".bin"
                dest_path = dest_path.with_suffix(guessed_ext)

            Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(dest_path).open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        logger.info("File downloaded successfully to: %s", dest_path)
    return dest_path


def download_file(
    url: str,
    dest_path: Path | str | None = None,
    outdir: Path | str | None = None,
    overwrite: bool = False,
    allow_existing: bool = True,
    dest_path_from_hashed_url: bool = False,
    timeout: int = 300,
) -> Path:
    """Download a file from a URL with concurrency control.

    Args:
        url: The URL to download the file from.
        dest_path: The destination file path or name. If None, the file name is derived from the URL.
        outdir: The output directory to save the file. Defaults to the current working directory.
        overwrite: Whether to overwrite the file if it already exists. Default is False.
        allow_existing: If True and the file exists, skip downloading. Default is True.
        dest_path_from_hashed_url: If True, derive the file name from a hash of the URL. Default is False.
        timeout: Timeout in seconds for the download operation. Default is 300 seconds.

    Returns:
        The path to the downloaded file.
    """
    dest_path = determine_dest_path(
        url=url,
        dest_path=dest_path,
        outdir=outdir,
        dest_path_from_hashed_url=dest_path_from_hashed_url,
    )

    result = handle_existing_file(dest_path=dest_path, overwrite=overwrite, allow_existing=allow_existing)
    if result is not None:
        return result

    logger.info("Downloading file from: %s", url)

    lock_path = dest_path.with_suffix(dest_path.suffix + ".lock")
    try:
        dest_path = download_file_with_lock(url=url, dest_path=dest_path, lock_path=lock_path, timeout=timeout)
    except filelock.Timeout as e:
        raise ValueError(f"Timeout waiting for download lock on {url}: {e}") from e
    except requests.RequestException as e:
        raise ValueError(f"Failed to download file from {url}: {e}") from e
    return dest_path
