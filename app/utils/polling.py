"""Generic bounded polling for asynchronous work (provider jobs, long renders).

`poll_until` calls `fetch` until `done` says the value is final, or until the policy's timeout or attempt limit,
or until `cancelled` returns True. It never waits forever: the outcome says why it stopped, and the caller decides
what an unfinished result means (the workflow puts the task in a resumable WAITING state). Errors that `retryable`
accepts are counted and polled through; any other error is raised at once, and too many retryable errors in a row
are raised too.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

T = TypeVar("T")
PollStop = Literal["done", "timeout", "exhausted", "cancelled"]


@dataclass(frozen=True)
class PollPolicy:
    interval_seconds: float = 5.0  # wait between polls (the first poll does not wait)
    timeout_seconds: float = 600.0  # total time spent polling in one call
    max_attempts: int = 120  # polls in one call
    backoff_multiplier: float = 1.0  # >1 stretches the interval after every poll
    max_interval_seconds: float | None = None
    max_consecutive_errors: int = 5  # retryable errors in a row before the last one is raised

    def __post_init__(self) -> None:
        if self.interval_seconds < 0 or self.timeout_seconds <= 0 or self.max_attempts < 1:
            raise ValueError("a poll policy needs interval >= 0, timeout > 0 and at least one attempt")
        if self.backoff_multiplier < 1 or self.max_consecutive_errors < 1:
            raise ValueError("backoff_multiplier must be >= 1 and max_consecutive_errors >= 1")

    def interval(self, attempt: int) -> float:
        """Delay before poll `attempt` (1-based)."""
        if attempt <= 1:
            return 0.0
        delay = self.interval_seconds * self.backoff_multiplier ** (attempt - 2)
        return min(delay, self.max_interval_seconds) if self.max_interval_seconds is not None else delay


@dataclass
class PollOutcome(Generic[T]):
    stopped: PollStop
    value: T | None  # the last value fetched (None when every poll failed or none ran)
    attempts: int
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)  # retryable errors polled through

    @property
    def done(self) -> bool:
        return self.stopped == "done"


async def poll_until(
    fetch: Callable[[int], Awaitable[T]],
    done: Callable[[T], bool],
    policy: PollPolicy,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    retryable: Callable[[BaseException], bool] = lambda exc: False,
    on_poll: Callable[[int, T], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> PollOutcome[T]:
    started = clock()
    value: T | None = None
    errors: list[str] = []
    consecutive = 0
    attempt = 0
    while True:
        if cancelled():
            return PollOutcome("cancelled", value, attempt, clock() - started, errors)
        if attempt >= policy.max_attempts:
            return PollOutcome("exhausted", value, attempt, clock() - started, errors)
        delay = policy.interval(attempt + 1)
        if delay and clock() - started + delay > policy.timeout_seconds:
            return PollOutcome("timeout", value, attempt, clock() - started, errors)
        if delay:
            await sleep(delay)
            if cancelled():
                return PollOutcome("cancelled", value, attempt, clock() - started, errors)
        attempt += 1
        try:
            value = await fetch(attempt)
        except Exception as exc:  # noqa: BLE001 - classified by `retryable`
            if not retryable(exc):
                raise
            consecutive += 1
            errors.append(f"{type(exc).__name__}: {exc}"[:300])
            if consecutive >= policy.max_consecutive_errors:
                raise
            continue
        consecutive = 0
        if on_poll is not None:
            on_poll(attempt, value)
        if done(value):
            return PollOutcome("done", value, attempt, clock() - started, errors)
        if clock() - started >= policy.timeout_seconds:
            return PollOutcome("timeout", value, attempt, clock() - started, errors)
