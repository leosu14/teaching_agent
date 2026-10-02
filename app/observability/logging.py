"""Structured JSON logging. Every event is logged so a task can be reconstructed from logs alone."""

from __future__ import annotations

import json
import logging
import sys

from app.observability.redaction import redact, redact_text
from app.schemas.events import Event

EVENT_LOGGER = "teaching_agent.events"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage()),
        }
        extra = getattr(record, "event", None)
        if extra is not None:
            payload["event"] = extra
        if record.exc_info:
            payload["exc"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root.handlers = [handler]


def log_event(event: Event) -> None:
    logging.getLogger(EVENT_LOGGER).info(event.type, extra={"event": redact(event.model_dump(mode="json"))})
