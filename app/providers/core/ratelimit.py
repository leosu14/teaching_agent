"""Local (in-process) rate limiting per provider: a requests-per-minute sliding window and a concurrency limit.
Not distributed: several processes each apply their own limits."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

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
