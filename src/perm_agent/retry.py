from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from .exceptions import RetryExhaustedError

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    retry_on: tuple[type[Exception], ...] = field(default=(Exception,))
    backoff_factor: float = 1.0
    max_backoff: float = 30.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed)."""
        delay = self.backoff_factor * (2**attempt)
        return min(delay, self.max_backoff)


def with_retry(
    fn: Callable[..., T],
    config: RetryConfig,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a callable with retry logic.

    Retries on exceptions matching config.retry_on, with exponential backoff.
    Raises RetryExhaustedError if all attempts fail.
    """
    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except config.retry_on as e:
            last_error = e
            if attempt < config.max_retries:
                delay = config.delay_for_attempt(attempt)
                time.sleep(delay)

    raise RetryExhaustedError(
        attempts=config.max_retries + 1,
        last_error=last_error,  # type: ignore[arg-type]
    ) from last_error
