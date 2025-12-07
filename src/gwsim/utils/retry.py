"""Retry decorator for handling transient failures."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("gwsim")


def retry_on_failure(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to retry a function on failure with exponential backoff.

    Retries the decorated function up to `max_retries` times on specified exceptions,
    with exponential backoff between attempts. Useful for handling transient I/O errors.

    Args:
        max_retries: Maximum number of retry attempts (default: 3).
        initial_delay: Initial delay in seconds before the first retry (default: 1.0).
        backoff_factor: Factor by which the delay increases after each retry (default: 2.0).
        exceptions: Tuple of exception types to catch and retry on (default: all Exceptions).

    Returns:
        The decorated function.

    Example:
        @retry_on_failure(max_retries=5, initial_delay=0.5)
        def unstable_io_operation():
            # Code that might fail transiently
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. Retrying in %.2f seconds...",
                            attempt + 1,
                            max_retries + 1,
                            func.__name__,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_retries + 1,
                            func.__name__,
                            e,
                        )
                        raise last_exception from e
            return None

        return wrapper

    return decorator
