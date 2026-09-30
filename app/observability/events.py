"""In-process event bus. Subscribers (persistence, logs, API streams) observe every event."""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.schemas.common import new_id
from app.schemas.events import Event

logger = logging.getLogger(__name__)

Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    def emit(
        self,
        type: str,
        *,
        task_id: str | None = None,
        node_id: str | None = None,
        agent_id: str | None = None,
        tool: str | None = None,
        **data: object,
    ) -> Event:
        event = Event(
            event_id=new_id("evt"),
            type=type,
            task_id=task_id,
            node_id=node_id,
            agent_id=agent_id,
            tool=tool,
            data=data,
        )
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception:  # an observer must never break execution
                logger.exception("event subscriber failed for %s", type)
        return event
