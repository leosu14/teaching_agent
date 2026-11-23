"""Rate limiting per (capability, provider): a requests-per-minute sliding window and a concurrency limit.

Two implementations behind one interface (`slot()`, `limited`):

- `RateLimiter`: in this process's memory. Development and SQLite (one process), and unit tests. Several processes
  each apply their own limits, so the startup check refuses it where several processes share a PostgreSQL database
  (`Settings.shared_state_problems`).
- `SharedRateLimiter`: counted in the database shared by every API process and worker (issue #66 PR 2,
  app/storage/shared_state.py). The window holds the starts admitted in the last 60 s of database time; each
  concurrency slot is a lease that its holder renews while the request runs and deletes when it ends, so a killed
  process's slot frees itself at its expiry. Waiting is a poll with backoff.

Fail closed (D3): when the shared limiter cannot decide (the database fails or times out), `slot()` raises
`RateLimiterUnavailable` before anything is sent, and the invoker reserves no spend for that attempt."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from app.providers.core.errors import RateLimiterUnavailable

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0

OnWait = Callable[[str, float | None], None]  # (reason, seconds to wait or None when unknown)


class RateLimiter:
    def __init__(self, *, requests_per_minute: int | None = None, max_concurrency: int | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.requests_per_minute = requests_per_minute
        self.max_concurrency = max_concurrency
        self._clock = clock
        self._sleep = sleep
        self._starts: deque[float] = deque()
        self._semaphores: dict[int, asyncio.Semaphore] = {}  # one per event loop: a semaphore binds to its loop

    @property
    def limited(self) -> bool:
        return self.requests_per_minute is not None or self.max_concurrency is not None

    def _semaphore(self) -> asyncio.Semaphore | None:
        if self.max_concurrency is None:
            return None
        loop = id(asyncio.get_running_loop())
        return self._semaphores.setdefault(loop, asyncio.Semaphore(self.max_concurrency))

    async def _admit(self, on_wait: OnWait | None) -> None:
        if self.requests_per_minute is None:
            return
        while True:
            now = self._clock()
            while self._starts and now - self._starts[0] >= WINDOW_SECONDS:
                self._starts.popleft()
            if len(self._starts) < self.requests_per_minute:
                self._starts.append(now)
                return
            wait = WINDOW_SECONDS - (now - self._starts[0])
            if on_wait is not None:
                on_wait("requests_per_minute", round(wait, 3))
            await self._sleep(wait)

    @asynccontextmanager
    async def slot(self, on_wait: OnWait | None = None) -> AsyncIterator[None]:
        semaphore = self._semaphore()
        if semaphore is not None:
            if semaphore.locked() and on_wait is not None:
                on_wait("concurrency", None)
            await semaphore.acquire()
        try:
            await self._admit(on_wait)
            yield
        finally:
            if semaphore is not None:
                semaphore.release()


# --- shared across processes ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Admission:
    admitted: bool
    slot_id: str | None = None
    reason: str | None = None
    wait_seconds: float | None = None


class RateLimitStore(Protocol):
    """What the shared limiter needs of its store (app/storage/shared_state.py, `SqlRateLimits`)."""

    def admit(self, key: str, *, requests_per_minute: int | None, max_concurrency: int | None,
              slot_ttl: timedelta) -> Admission: ...

    def renew(self, slot_id: str, ttl: timedelta) -> bool: ...

    def release(self, slot_id: str) -> None: ...


UNAVAILABLE = "provider rate limiter unavailable: no rate-limit decision could be made; the request is not sent"


class SharedRateLimiter:
    """The limits of one (capability, provider) key, counted in the shared store.

    `slot()` polls the store until it admits the request (backoff from `poll_seconds` doubling to `max_poll_seconds`;
    for a full window, the time until its oldest start leaves it), then holds the concurrency slot, renewing it every
    third of `slot_ttl` until the request ends, and deletes it. A release that fails is left to the slot's expiry."""

    def __init__(self, store: RateLimitStore, key: str, *, requests_per_minute: int | None = None,
                 max_concurrency: int | None = None, slot_ttl: timedelta = timedelta(seconds=30),
                 poll_seconds: float = 0.05, max_poll_seconds: float = 1.0,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, provider: str | None = None) -> None:
        self.requests_per_minute = requests_per_minute
        self.max_concurrency = max_concurrency
        self.key = key
        self._store = store
        self._slot_ttl = slot_ttl
        self._poll = poll_seconds
        self._max_poll = max_poll_seconds
        self._sleep = sleep
        self._provider = provider

    @property
    def limited(self) -> bool:
        return self.requests_per_minute is not None or self.max_concurrency is not None

    def _admit_once(self) -> Admission:
        try:
            return self._store.admit(self.key, requests_per_minute=self.requests_per_minute,
                                     max_concurrency=self.max_concurrency, slot_ttl=self._slot_ttl)
        except Exception as exc:  # noqa: BLE001 - any failure to decide fails closed (D3)
            # A fixed message: the key and the error type only (a driver's message could carry a connection string).
            logger.warning("%s (key=%s, error=%s)", UNAVAILABLE, self.key, type(exc).__name__)
            raise RateLimiterUnavailable(provider=self._provider) from None

    @asynccontextmanager
    async def slot(self, on_wait: OnWait | None = None) -> AsyncIterator[None]:
        if not self.limited:
            yield
            return
        delay = self._poll
        reported: str | None = None
        while True:
            admission = self._admit_once()
            if admission.admitted:
                break
            if admission.reason == "requests_per_minute" and admission.wait_seconds is not None:
                wait = admission.wait_seconds
            else:
                wait = delay
                delay = min(delay * 2, self._max_poll)
            if on_wait is not None and (admission.reason != reported or admission.reason == "requests_per_minute"):
                on_wait(admission.reason or "concurrency", admission.wait_seconds)
            reported = admission.reason
            await self._sleep(wait)
        slot_id = admission.slot_id
        renewer = asyncio.create_task(self._renew(slot_id)) if slot_id is not None else None
        try:
            yield
        finally:
            if renewer is not None:
                renewer.cancel()
            if slot_id is not None:
                try:
                    self._store.release(slot_id)
                except Exception as exc:  # noqa: BLE001 - the slot expires on its own
                    logger.warning("provider rate limiter: a concurrency slot could not be released; it frees "
                                   "itself at its expiry (key=%s, error=%s)", self.key, type(exc).__name__)

    async def _renew(self, slot_id: str) -> None:
        every = max(self._slot_ttl.total_seconds() / 3, 0.01)
        while True:
            await asyncio.sleep(every)
            try:
                held = self._store.renew(slot_id, self._slot_ttl)
            except Exception as exc:  # noqa: BLE001 - tried again on the next beat
                logger.warning("provider rate limiter: a concurrency slot could not be renewed; trying again "
                               "(key=%s, error=%s)", self.key, type(exc).__name__)
                continue
            if not held:
                logger.warning("provider rate limiter: a concurrency slot expired while its request ran (the "
                               "renewals failed for longer than its lease); key=%s", self.key)
                return
