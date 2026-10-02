"""Secret redaction for events, logs and error messages.

Three layers, so a secret is caught even when it ends up somewhere unexpected:
- keys that name a credential (api_key, authorization, token, ...) have their values replaced;
- values registered at startup (the configured API keys) are replaced wherever they appear in text;
- strings shaped like common credentials (bearer tokens, sk-... keys) are replaced.
"""

from __future__ import annotations

import re
import threading

REDACTED = "***"
SECRET_KEY = re.compile(r"(api[_-]?key|authorization|auth[_-]?token|access[_-]?token|secret|password|"
                        r"x-api-key|xi-api-key|credential|cookie)", re.IGNORECASE)
SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\btvly-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|x-api-key|token)=([^&\s\"']{6,})"),
)
MIN_SECRET_LENGTH = 6  # shorter values are not registered: replacing them would mangle ordinary text

_lock = threading.Lock()
_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Remember a secret value so it is redacted from any text that contains it."""
    if value and len(value) >= MIN_SECRET_LENGTH:
        with _lock:
            _secrets.add(value)


def redact_text(text: str) -> str:
    with _lock:
        known = sorted(_secrets, key=len, reverse=True)
    for secret in known:
        if secret in text:
            text = text.replace(secret, REDACTED)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda m: REDACTED if m.lastindex is None else f"{m.group(1)}={REDACTED}", text)
    return text


def redact(value: object) -> object:
    """A copy of `value` with secrets removed: dict values under credential-like keys, registered secrets and
    credential-shaped strings. Non-string scalars are returned unchanged."""
    if isinstance(value, dict):
        return {k: REDACTED if isinstance(k, str) and SECRET_KEY.search(k) and v not in (None, "")
                else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact(v) for v in value)
    if isinstance(value, str):
        return redact_text(value)
    return value
