"""Research cache. Entries hold complete search results (sources, retrieval times, provider), so a cached answer is
exactly as traceable as a fresh one.

Per learner (issue #66, decision A2): an entry is stored and found under the learner whose lesson searched, so one
learner's query, which can come from their own lesson request, is never served to another learner. Entries live for
`CACHE_TTL` (7 days, D4), and a learner's entries are deleted with their data (the TTL does not replace that).

The key (`research_cache_key`) is a SHA-256 of everything that shapes the results: the normalised query (text folded
to NFKC, case and whitespace; language, subject, domain, the number of results) and the configuration (the search
providers in order, the operator's include and exclude domain lists), with `CACHE_FORMAT`, raised whenever the stored
results' shape or the key's own rules change, so an entry written under another configuration or version is never
read.

Two implementations: `InMemoryResearchCache` (one process: development, SQLite, tests) and `SharedResearchCache`, on
the database every process shares (app/storage/shared_state.py, `SqlResearchCacheStore`)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Protocol

from pydantic import TypeAdapter

from app.schemas.common import utcnow
from app.schemas.research import SearchQuery, SearchResult

logger = logging.getLogger(__name__)

CACHE_TTL = timedelta(days=7)
CACHE_FORMAT = 2  # 1: the process-local cache keyed by SearchQuery.cache_key(); 2: per learner, with configuration
_RESULTS = TypeAdapter(list[SearchResult])


def _fold(value: str | None) -> str | None:
    if value is None:
        return None
    folded = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()
    return folded or None


def research_cache_key(query: SearchQuery, *, providers: Sequence[str], include_domains: Sequence[str] = (),
                       exclude_domains: Sequence[str] = ()) -> str:
    """Equal for searches that would return the same results under the same configuration, whatever the query's id
    or purpose; different when any input or setting that shapes the results differs."""
    payload = {
        "format": CACHE_FORMAT,
        "text": _fold(query.text), "language": _fold(query.language), "subject": _fold(query.subject),
        "domain": _fold(query.domain), "max_results": query.max_results,
        "providers": list(providers),
        "include_domains": sorted({d.strip().lower() for d in include_domains}),
        "exclude_domains": sorted({d.strip().lower() for d in exclude_domains}),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResearchCache(ABC):
    @abstractmethod
    def get(self, learner_id: str, key: str) -> list[SearchResult] | None: ...

    @abstractmethod
    def set(self, learner_id: str, key: str, results: list[SearchResult], *, task_id: str) -> None:
        """Store an entry for the learner, written for their lesson task `task_id` (refused once it is deleted)."""

    @abstractmethod
    def invalidate(self, learner_id: str, key: str) -> None: ...

    def forget(self, learner_id: str) -> None:
        """Drop the learner's entries (their data deletion). The shared cache's rows are deleted by the deletion's own
        transaction instead (app/storage/repositories.py, ERASED_TABLES)."""


class InMemoryResearchCache(ResearchCache):
    """Process-local. Stores deep copies so callers cannot mutate it. Expiry by the injected clock."""

    def __init__(self, *, ttl: timedelta = CACHE_TTL, clock: Callable[[], datetime] = utcnow) -> None:
        self._ttl = ttl
        self._clock = clock
        self._entries: dict[tuple[str, str], tuple[datetime, list[SearchResult]]] = {}

    def get(self, learner_id: str, key: str) -> list[SearchResult] | None:
        hit = self._entries.get((learner_id, key))
        if hit is None:
            return None
        expires_at, results = hit
        if self._clock() >= expires_at:
            self._entries.pop((learner_id, key), None)
            return None
        return [r.model_copy(deep=True) for r in results]

    def set(self, learner_id: str, key: str, results: list[SearchResult], *, task_id: str) -> None:
        self._entries[(learner_id, key)] = (self._clock() + self._ttl, [r.model_copy(deep=True) for r in results])

    def invalidate(self, learner_id: str, key: str) -> None:
        self._entries.pop((learner_id, key), None)

    def forget(self, learner_id: str) -> None:
        for entry in [k for k in self._entries if k[0] == learner_id]:
            del self._entries[entry]

    def __len__(self) -> int:
        return len(self._entries)


class ResearchCacheStore(Protocol):
    """What the shared cache needs of its store (app/storage/shared_state.py, `SqlResearchCacheStore`)."""

    def get(self, learner_id: str, key: str) -> str | None: ...

    def put(self, learner_id: str, key: str, body: str, *, ttl: timedelta, task_id: str) -> None: ...

    def delete(self, learner_id: str, key: str) -> None: ...


class SharedResearchCache(ResearchCache):
    """On the database every process shares. A cache never fails a search: an entry that cannot be read is a miss, and
    one that cannot be written (the store failed, or the learner's data was deleted meanwhile) is not cached."""

    def __init__(self, store: ResearchCacheStore, *, ttl: timedelta = CACHE_TTL) -> None:
        self._store = store
        self._ttl = ttl

    def get(self, learner_id: str, key: str) -> list[SearchResult] | None:
        try:
            body = self._store.get(learner_id, key)
            return None if body is None else _RESULTS.validate_json(body)
        except Exception as exc:  # noqa: BLE001 - a miss: the search runs (and is rate-limited and budgeted)
            logger.warning("research cache: an entry could not be read (%s); searching instead", type(exc).__name__)
            return None

    def set(self, learner_id: str, key: str, results: list[SearchResult], *, task_id: str) -> None:
        try:
            self._store.put(learner_id, key, _RESULTS.dump_json(results).decode("utf-8"), ttl=self._ttl,
                            task_id=task_id)
        except Exception as exc:  # noqa: BLE001 - not cached; LearnerDataGone included (nothing may be stored)
            logger.info("research cache: an entry was not stored (%s)", type(exc).__name__)

    def invalidate(self, learner_id: str, key: str) -> None:
        self._store.delete(learner_id, key)
