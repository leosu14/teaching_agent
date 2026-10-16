"""In-process event bus. Subscribers (persistence, logs, API streams) observe every event."""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.schemas.common import new_id
from app.schemas.events import Event
from app.schemas.lease import Lease, LeaseLost

logger = logging.getLogger(__name__)

Subscriber = Callable[[Event], None]
FencedSubscriber = Callable[[Event, Lease | None], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[tuple[Callable, bool]] = []

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append((subscriber, False))

    def subscribe_fenced(self, subscriber: FencedSubscriber) -> None:
        """A subscriber that also receives the emitting worker's lease (the event store: it refuses an event whose
        worker lost its lease)."""
        self._subscribers.append((subscriber, True))

    def emit(
        self,
        type: str,
        *,
        task_id: str | None = None,
        node_id: str | None = None,
        agent_id: str | None = None,
        tool: str | None = None,
        event_id: str | None = None,
        fence: Lease | None = None,
        **data: object,
    ) -> Event:
        """`event_id`: a stable id for an event that may be published again (e.g. from an outbox after a restart);
        the event store keeps one copy per id. Default: a new id. `fence`: the lease of the task worker emitting it;
        the event is not stored once that lease is lost."""
        event = Event(
            event_id=event_id or new_id("evt"),
            type=type,
            task_id=task_id,
            node_id=node_id,
            agent_id=agent_id,
            tool=tool,
            data=data,
        )
        for subscriber, fenced in self._subscribers:
            try:
                if fenced:
                    subscriber(event, fence)
                else:
                    subscriber(event)
            except LeaseLost:  # a stale worker's event: refused; its next state write stops the worker
                logger.warning("event %s of task %s not stored: lease fence %s lost", type, task_id,
                               fence.fence if fence else None)
            except Exception:  # an observer must never break execution
                logger.exception("event subscriber failed for %s", type)
        return event
