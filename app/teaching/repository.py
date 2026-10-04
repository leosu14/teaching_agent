"""Teaching session store. `app/storage/repositories.py` implements it on SQL; the in-memory version serves unit tests.

Every change is stored atomically and only against the session version it was computed from (optimistic lock): a
concurrent or stale change raises SessionConflict and stores nothing. Turns, evidence and outbox items are append-only
and keyed by deterministic ids, so a repeated transition can never store a second copy of anything.
"""

from __future__ import annotations

import threading
from typing import Protocol

from app.schemas.teaching import (
    InteractionEvidence,
    OutboxItem,
    SessionChange,
    SessionConflict,
    TeachingRequestRecord,
    TeachingSession,
    TeachingTurn,
)


class TeachingRepository(Protocol):
    def create(self, change: SessionChange) -> bool:
        """Store a new session; False when the session id exists (nothing is stored)."""
        ...

    def apply(self, change: SessionChange, expected_version: int, request: TeachingRequestRecord | None = None) -> None:
        """Store the change if the stored session is still at `expected_version`; SessionConflict otherwise (also
        for a duplicate turn sequence or client_turn_id)."""
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
    def __init__(self) -> None:
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

    def apply(self, change: SessionChange, expected_version: int, request: TeachingRequestRecord | None = None) -> None:
        sid = change.session.session_id
        with self._lock:
            current = self._sessions.get(sid)
            if current is None or current.version != expected_version:
                raise SessionConflict(f"session {sid} changed concurrently (expected version {expected_version})")
            sequences = {t.sequence for t in self._turns.get(sid, [])}
            if any(t.sequence in sequences for t in change.turns):
                raise SessionConflict(f"session {sid}: turn sequence already stored")
            if request is not None and (sid, request.client_turn_id) in self._requests:
                raise SessionConflict(f"client turn {request.client_turn_id} was already applied")
            self._sessions[sid] = change.session.model_copy(deep=True)
            self._store(change)
            if request is not None:
                self._requests[(sid, request.client_turn_id)] = request

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
