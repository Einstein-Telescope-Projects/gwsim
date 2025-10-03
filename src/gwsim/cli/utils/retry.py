"""Utility functions for retrying operations with exponential backoff."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("gwsim")


class RetryManager:
    """Manages retry logic with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        retryable_exceptions: tuple[type[Exception], ...] = (
            OSError,
            PermissionError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ),
    ):
        """Initialize the RetryManager.

        Args:
            max_retries: Maximum number of retries.
            base_delay: Base delay in seconds for exponential backoff.
            retryable_exceptions: Tuple of exception types that are considered retryable.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retryable_exception = retryable_exceptions

    def retry_with_backoff(self, operation, *args, **kwargs) -> Any | None:
        """Retry operation with exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except self.retryable_exception as e:
                if attempt == self.max_retries:
                    logger.error("Operation failed after %s retries: %s", self.max_retries, e)
                    raise

                delay = self.base_delay * (2**attempt)
                logger.warning("Attempt %s failed: %s. Retrying in %ss...", attempt + 1, e, delay)
                time.sleep(delay)
        return None

    def is_retryable_exception(self, exception: Exception) -> bool:
        """Check if an exception is retryable.

        Args:
            exception: The exception to check

        Returns:
            bool: True if the exception is retryable, False otherwise
        """
        return isinstance(exception, self.retryable_exception)
