"""Async retry helper shared by tools, agents and workflow nodes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.schemas.common import RetryPolicy

T = TypeVar("T")


async def retry_async(
    op: Callable[[int], Awaitable[T]],
    policy: RetryPolicy,
    *,
    retry_on: tuple[type[BaseException], ...],
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Run `op(attempt)` until it succeeds or the policy is exhausted. Only `retry_on` errors are retried."""
    attempt = 1
    while True:
        try:
            return await op(attempt)
        except retry_on as exc:
            if attempt >= policy.max_attempts:
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            attempt += 1
            delay = policy.delay_before(attempt)
            if delay:
                await asyncio.sleep(delay)
