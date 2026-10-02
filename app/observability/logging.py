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


# --- Production logs -------------------------------------------------------------------------------------------
# One compact JSON line per event: where it happened (task, node, agent, tool), which provider request, how long and
# how it ended. Prompts, model output, binary payloads and credentials are never part of a line.

PRODUCTION_FIELDS = ("provider", "capability", "operation", "model", "request_id", "attempt", "artifact_id",
                     "artifact_type", "error_type", "category", "will_retry")
ERROR_CHARS = 300


def _status(event_type: str, data: dict) -> str | None:
    if event_type.endswith(("failed", "validation_failed")) or data.get("ok") is False:
        return "failed"
    if event_type.endswith(("started", "_started")):
        return "started"
    if event_type.endswith("retry") or event_type.endswith("rate_limited") or event_type.endswith("fallback"):
        return event_type.rsplit(".", 1)[-1]
    if event_type.endswith(("completed", "finished", "created", "validated")):
        return "ok"
    return None


def production_record(event: Event) -> dict:
    data = event.data
    record: dict = {"ts": event.at.isoformat(), "event": event.type, "task_id": event.task_id,
                    "node": event.node_id, "agent": event.agent_id, "tool": event.tool}
    record.update({k: data[k] for k in PRODUCTION_FIELDS if data.get(k) is not None})
    duration = data.get("latency_ms", data.get("duration_ms"))
    if duration is not None:
        record["duration_ms"] = duration
    record["status"] = _status(event.type, data)
    if data.get("error"):
        record["error"] = str(data["error"])[:ERROR_CHARS]
    return redact({k: v for k, v in record.items() if v is not None})


class ProductionLogWriter:
    """Event subscriber writing production log lines (JSON, one per event) to a text stream."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def __call__(self, event: Event) -> None:
        self._stream.write(json.dumps(production_record(event), ensure_ascii=False, default=str) + "\n")
        self._stream.flush()
