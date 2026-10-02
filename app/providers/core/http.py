"""The HTTP client every network adapter uses. It is the only place that talks to vendor APIs over the network.

- httpx is an optional dependency (`pip install -e ".[providers]"`), imported only when a request is made.
- Offline mode (TEACHING_AGENT_OFFLINE=true, checked on every request) rejects the request before any I/O.
- HTTPS only, except to a loopback host (a local OpenAI-compatible server); redirects are never followed.
- Every request has a timeout; request and response bodies are size-limited.
- The request id of the current invocation is sent as a header; the vendor's own request id is returned.
- Vendor failures become typed provider errors whose messages are redacted and never include headers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from urllib.parse import urlsplit

from app.config.routing import ConfigError
from app.observability.redaction import redact_text, register_secret
from app.providers.core.base import current_request_id
from app.providers.core.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderInvalidRequest,
    ProviderOfflineError,
    ProviderRateLimit,
    ProviderResponseError,
    ProviderTimeout,
    ProviderUnavailable,
)

LOOPBACK = {"localhost", "127.0.0.1", "::1"}
TRANSIENT_STATUS = {408, 409, 425, 500, 502, 503, 504, 529}
VENDOR_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-amzn-requestid")
ERROR_EXCERPT = 300


def offline_env() -> bool:
    return os.environ.get("TEACHING_AGENT_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def validate_base_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in {"https", "http"} or not parts.hostname:
        raise ConfigError(f"provider base URL must be an absolute http(s) URL: {url!r}")
    if parts.scheme == "http" and parts.hostname not in LOOPBACK:
        raise ConfigError(f"provider base URL must use https (plain http is allowed only for loopback): {url!r}")
    if parts.username or parts.password:
        raise ConfigError("provider base URL must not contain credentials")
    return url.rstrip("/")


def _httpx():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise ProviderUnavailable("httpx is not installed; install the provider extras: "
                                  "pip install -e \".[providers]\"", transient=False) from exc
    return httpx


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


@dataclass(frozen=True)
class HttpResponse:
    status: int
    content: bytes
    content_type: str
    vendor_request_id: str | None

    def json(self) -> dict:
        try:
            data = json.loads(self.content)
        except ValueError as exc:
            raise ProviderResponseError(f"response is not JSON ({self.content_type or 'no content type'})") from exc
        if not isinstance(data, dict):
            raise ProviderResponseError("response JSON is not an object")
        return data


class HttpClient:
    def __init__(self, *, provider: str, base_url: str, headers: Mapping[str, str] | None = None,
                 secrets: tuple[str, ...] = (), timeout_seconds: float = 60.0, offline: bool = False,
                 max_request_bytes: int = 4_000_000, max_response_bytes: int = 32_000_000,
                 request_id_header: str = "X-Request-Id", transport=None) -> None:
        self.provider = provider
        self.base_url = validate_base_url(base_url)
        self._headers = dict(headers or {})
        for secret in secrets:
            register_secret(secret)
        self.timeout_seconds = timeout_seconds
        self.offline = offline
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.request_id_header = request_id_header
        self._transport = transport  # tests pass an httpx.MockTransport

    def __repr__(self) -> str:  # never show headers: they carry credentials
        return f"HttpClient(provider={self.provider!r}, base_url={self.base_url!r})"

    def _error(self, cls: type[ProviderError], message: str, **kwargs) -> ProviderError:
        return cls(redact_text(f"{self.provider}: {message}"), provider=self.provider, **kwargs)

    async def request(self, method: str, path: str, *, json_body: dict | None = None,
                      params: Mapping[str, str] | None = None) -> HttpResponse:
        if self.offline or offline_env():
            raise self._error(ProviderOfflineError, "network request refused: offline mode is on "
                                                    "(TEACHING_AGENT_OFFLINE=true)")
        body = None
        if json_body is not None:
            body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            if len(body) > self.max_request_bytes:
                raise self._error(ProviderInvalidRequest, f"request body is {len(body)} bytes, over the "
                                                          f"{self.max_request_bytes}-byte limit")
        httpx = _httpx()
        headers = {**self._headers, "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request_id = current_request_id()
        if request_id and self.request_id_header:
            headers[self.request_id_header] = request_id
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds), follow_redirects=False,
                                         transport=self._transport) as client:
                async with client.stream(method, url, content=body, headers=headers, params=params) as response:
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.max_response_bytes:
                            raise self._error(ProviderResponseError, f"response over the {self.max_response_bytes}"
                                                                     "-byte limit")
                        chunks.append(chunk)
                    content = b"".join(chunks)
                    status, resp_headers = response.status_code, response.headers
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise self._error(ProviderTimeout, f"timed out after {self.timeout_seconds}s "
                                               f"({type(exc).__name__})") from None
        except (httpx.TransportError, OSError) as exc:
            raise self._error(ProviderUnavailable, f"connection failed ({type(exc).__name__})") from None
        vendor_id = next((resp_headers[h] for h in VENDOR_REQUEST_ID_HEADERS if h in resp_headers), None)
        result = HttpResponse(status=status, content=content, content_type=resp_headers.get("content-type", ""),
                              vendor_request_id=vendor_id)
        if status >= 400 or 300 <= status < 400:
            raise self._status_error(result, resp_headers.get("retry-after"))
        return result

    def _status_error(self, response: HttpResponse, retry_after: str | None) -> ProviderError:
        status = response.status
        detail = self._detail(response)
        suffix = f" (vendor request id {response.vendor_request_id})" if response.vendor_request_id else ""
        message = f"HTTP {status}: {detail}{suffix}" if detail else f"HTTP {status}{suffix}"
        if status in (401, 403):
            return self._error(ProviderAuthenticationError, message, status=status)
        if status == 429:
            return self._error(ProviderRateLimit, message, status=status, retry_after=_retry_after(retry_after))
        if status in TRANSIENT_STATUS or status >= 500:
            return self._error(ProviderUnavailable, message, status=status)
        if 300 <= status < 400:
            return self._error(ProviderResponseError, f"{message}: redirects are not followed", status=status)
        return self._error(ProviderInvalidRequest, message, status=status)

    @staticmethod
    def _detail(response: HttpResponse) -> str:
        """A short excerpt of the vendor's error message (bodies can echo input; never the whole body)."""
        try:
            data = json.loads(response.content)
        except ValueError:
            return response.content[:ERROR_EXCERPT].decode("utf-8", "replace").strip()
        if isinstance(data, dict):
            err = data.get("error", data.get("detail", data.get("message", "")))
            if isinstance(err, dict):
                err = err.get("message") or err.get("type") or ""
            return str(err)[:ERROR_EXCERPT]
        return ""
