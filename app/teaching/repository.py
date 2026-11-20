"""Teaching session store. `app/storage/repositories.py` implements it on SQL; the in-memory version serves unit tests.

Every change is stored atomically and only against the session version it was computed from (optimistic lock): a
concurrent or stale change raises SessionConflict and stores nothing. Turns, evidence and outbox items are append-only
and keyed by deterministic ids, so a repeated transition can never store a second copy of anything.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Protocol

from app.schemas.common import new_id, utcnow
from app.schemas.teaching import (
    InteractionEvidence,
    OutboxItem,
    SessionChange,
    SessionConflict,
    TeachingRequestRecord,
    TeachingSession,
    TeachingTurn,
    TurnLease,
    TurnLeaseLost,
)


class TeachingRepository(Protocol):
    def create(self, change: SessionChange) -> bool:
        """Store a new session; False when the session id exists (nothing is stored)."""
        ...

    def apply(self, change: SessionChange, expected_version: int, request: TeachingRequestRecord | None = None, *,
              turn_lease: TurnLease | None = None) -> None:
        """Store the change if the stored session is still at `expected_version`; SessionConflict otherwise (also
        for a duplicate turn sequence or client_turn_id). With `turn_lease` (teacher turns), the same statement also
        requires that lease to be the stored one and unexpired, and gives it back: TurnLeaseLost otherwise."""
        ...

    # The teaching-turn lease: single-flight production of the teacher turns a session owes. Expiry is the
    # database's clock on PostgreSQL, read in each statement's own transaction; the injected clock elsewhere.

    def claim_turns(self, session_id: str, ttl: timedelta) -> TurnLease | None:
        """Take the lease when no unexpired one is held; None while another request's lease runs."""
        ...

    def renew_turns(self, lease: TurnLease, ttl: timedelta) -> TurnLease | None:
        """Extend a lease that is still current and unexpired; None once it ran out or was taken over."""
        ...

    def release_turns(self, lease: TurnLease) -> None:
        """Give the lease back (nothing when it was taken over)."""
        ...

    def turn_lease(self, session_id: str) -> datetime | None:
        """When the session's live lease ends; None when nobody holds one."""
        ...

    def get(self, session_id: str) -> TeachingSession | None: ...

    def for_learner(self, learner_id: str) -> list[TeachingSession]: ...

    def turns(self, session_id: str) -> list[TeachingTurn]:
        """In sequence order."""
        ...

    def evidence(self, session_id: str) -> list[InteractionEvidence]: ...

    def request(self, session_id: str, client_turn_id: str) -> TeachingRequestRecord | None: ...

    def pending_outbox(self, session_id: str) -> list[OutboxItem]:
        """Unpublished items in the order they were stored."""
        ...

    def mark_published(self, item_id: str) -> None: ...


class InMemoryTeachingRepository:
    def __init__(self, clock: Callable[[], datetime] = utcnow) -> None:
        self._clock = clock
        self._leases: dict[str, TurnLease] = {}
        self._lock = threading.Lock()
        self._sessions: dict[str, TeachingSession] = {}
        self._turns: dict[str, list[TeachingTurn]] = {}
        self._evidence: dict[str, list[InteractionEvidence]] = {}
        self._requests: dict[tuple[str, str], TeachingRequestRecord] = {}
        self._outbox: list[OutboxItem] = []

    def create(self, change: SessionChange) -> bool:
        with self._lock:
            if change.session.session_id in self._sessions:
                return False
            self._sessions[change.session.session_id] = change.session.model_copy(deep=True)
            self._store(change)
            return True

    def apply(self, change: SessionChange, expected_version: int, request: TeachingRequestRecord | None = None, *,
              turn_lease: TurnLease | None = None) -> None:
        sid = change.session.session_id
        with self._lock:
            current = self._sessions.get(sid)
            if current is None or current.version != expected_version:
                raise SessionConflict(f"session {sid} changed concurrently (expected version {expected_version})")
            if turn_lease is not None and not self._live(turn_lease):
                raise TurnLeaseLost(f"session {sid}: the teaching-turn lease expired or was taken over")
            sequences = {t.sequence for t in self._turns.get(sid, [])}
            if any(t.sequence in sequences for t in change.turns):
                raise SessionConflict(f"session {sid}: turn sequence already stored")
            if request is not None and (sid, request.client_turn_id) in self._requests:
                raise SessionConflict(f"client turn {request.client_turn_id} was already applied")
            self._sessions[sid] = change.session.model_copy(deep=True)
            self._store(change)
            if request is not None:
                self._requests[(sid, request.client_turn_id)] = request
            if turn_lease is not None:
                del self._leases[sid]

    def _live(self, lease: TurnLease) -> bool:
        held = self._leases.get(lease.session_id)
        return held is not None and held.token == lease.token and held.until > self._clock()

    def claim_turns(self, session_id: str, ttl: timedelta) -> TurnLease | None:
        with self._lock:
            now = self._clock()
            held = self._leases.get(session_id)
            if held is not None and held.until > now:
                return None
            lease = TurnLease(session_id=session_id, token=new_id("turnlease"), until=now + ttl)
            self._leases[session_id] = lease
            return lease.model_copy()

    def renew_turns(self, lease: TurnLease, ttl: timedelta) -> TurnLease | None:
        with self._lock:
            if not self._live(lease):
                return None
            renewed = lease.model_copy(update={"until": self._clock() + ttl})
            self._leases[lease.session_id] = renewed
            return renewed.model_copy()

    def release_turns(self, lease: TurnLease) -> None:
        with self._lock:
            held = self._leases.get(lease.session_id)
            if held is not None and held.token == lease.token:
                del self._leases[lease.session_id]

    def turn_lease(self, session_id: str) -> datetime | None:
        with self._lock:
            held = self._leases.get(session_id)
            return held.until if held is not None and held.until > self._clock() else None

    def _store(self, change: SessionChange) -> None:
        sid = change.session.session_id
        self._turns.setdefault(sid, []).extend(t.model_copy(deep=True) for t in change.turns)
        self._evidence.setdefault(sid, []).extend(e.model_copy(deep=True) for e in change.evidence)
        self._outbox.extend(o.model_copy(deep=True) for o in change.outbox)

    def get(self, session_id: str) -> TeachingSession | None:
        found = self._sessions.get(session_id)
        return found.model_copy(deep=True) if found else None

    def for_learner(self, learner_id: str) -> list[TeachingSession]:
        return sorted((s.model_copy(deep=True) for s in self._sessions.values() if s.learner_id == learner_id),
                      key=lambda s: (s.started_at, s.session_id))

    def turns(self, session_id: str) -> list[TeachingTurn]:
        return sorted((t.model_copy(deep=True) for t in self._turns.get(session_id, [])), key=lambda t: t.sequence)

    def evidence(self, session_id: str) -> list[InteractionEvidence]:
        return [e.model_copy(deep=True) for e in self._evidence.get(session_id, [])]

    def request(self, session_id: str, client_turn_id: str) -> TeachingRequestRecord | None:
        found = self._requests.get((session_id, client_turn_id))
        return found.model_copy(deep=True) if found else None

    def pending_outbox(self, session_id: str) -> list[OutboxItem]:
        return [o.model_copy(deep=True) for o in self._outbox if o.session_id == session_id and not o.published]

    def mark_published(self, item_id: str) -> None:
        with self._lock:
            for o in self._outbox:
                if o.item_id == item_id:
                    o.published = True
