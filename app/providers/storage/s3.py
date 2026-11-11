"""S3-compatible object store (AWS S3, MinIO, Cloudflare R2, ...), behind the artifact service's `ObjectStore` protocol.

Objects are addressed by store-relative keys and recorded as `object:<key>` URIs (app/utils/object_uri.py); the
optional prefix (`TA_S3_PREFIX`) is applied here only, so a recorded URI never names a bucket, a prefix or a host.

- Plain HTTP with AWS Signature Version 4 (`sign_v4`), through httpx: no vendor SDK. Credentials are an access key
  pair (and an optional session token) from the configuration; they are registered for redaction and never logged,
  put in errors or returned.
- `put_if_absent` and `put_file_if_absent` are conditional writes (`If-None-Match: *`): of two concurrent writers of
  one key exactly one creates it and an existing object is never replaced, as with the filesystem store's hard links.
  S3 has supported this since 2024; MinIO does too.
- HTTPS only, unless `allow_http` is set for a store on a private network (a MinIO container next to the app).
- Not a vendor API: it does not go through the provider HTTP client, so offline mode (mock providers) still stores
  artifacts in a bucket.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import uuid
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote, unquote, urlsplit

from app.config.routing import ConfigError
from app.config.storage import S3Settings as S3Config
from app.observability.redaction import redact_text, register_secret
from app.utils.object_uri import is_object_uri, key_of, object_uri, validate_key

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
UNSIGNED = "UNSIGNED-PAYLOAD"
CHUNK = 1 << 20


class ObjectStoreError(OSError):
    """The object store refused or failed a request. The message has the operation, status and S3 error code only."""


def _httpx():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise ConfigError('TA_OBJECT_STORE=s3 needs httpx: pip install -e ".[providers]"') from exc
    return httpx


# --- AWS Signature Version 4 ---------------------------------------------------------------------------------------

def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _encode(value: str, safe: str = "") -> str:
    return quote(value, safe="-_.~" + safe)


def canonical_query(query: str) -> str:
    pairs = []
    for part in filter(None, query.split("&")):
        name, _, value = part.partition("=")
        pairs.append((_encode(unquote(name)), _encode(unquote(value))))
    return "&".join(f"{k}={v}" for k, v in sorted(pairs))


def sign_v4(method: str, url: str, headers: Mapping[str, str], payload_hash: str, *, access_key: str, secret_key: str,
            region: str, service: str = "s3", now: datetime | None = None,
            session_token: str | None = None) -> dict[str, str]:
    """The request's headers with `x-amz-date`, the session token (if any) and `Authorization` added. Every header
    passed in is signed, plus `host`. `url`'s path must already be percent-encoded as it will be sent."""
    when = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    amz_date, day = when.strftime("%Y%m%dT%H%M%SZ"), when.strftime("%Y%m%d")
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port and (parts.scheme, parts.port) not in {("https", 443), ("http", 80)}:
        host = f"{host}:{parts.port}"
    signed = {k.lower(): " ".join(str(v).split()) for k, v in headers.items()}
    signed["host"] = host
    signed["x-amz-date"] = amz_date
    if session_token:
        signed["x-amz-security-token"] = session_token
    names = sorted(signed)
    canonical = "\n".join([method.upper(), parts.path or "/", canonical_query(parts.query),
                           "".join(f"{n}:{signed[n]}\n" for n in names), ";".join(names), payload_hash])
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    key = _hmac(_hmac(_hmac(_hmac(("AWS4" + secret_key).encode(), day), region), service), "aws4_request")
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    out = {k: v for k, v in headers.items()}
    out["x-amz-date"] = amz_date
    if session_token:
        out["x-amz-security-token"] = session_token
    out["Authorization"] = (f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={';'.join(names)}, "
                            f"Signature={signature}")
    return out


# --- the store -----------------------------------------------------------------------------------------------------

class _Reader(io.RawIOBase):
    """A streaming GET body as a binary file object (closing it closes the response)."""

    def __init__(self, response) -> None:
        self._response = response
        self._chunks: Iterator[bytes] = response.iter_bytes(CHUNK)
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self._buffer:
            try:
                self._buffer = next(self._chunks)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buffer))
        b[:n], self._buffer = self._buffer[:n], self._buffer[n:]
        return n

    def close(self) -> None:
        if not self.closed:
            self._response.close()
        super().close()


class S3ObjectStore:
    scheme = "s3"

    def __init__(self, config: S3Config, *, transport=None, clock=None) -> None:
        self._config = config
        register_secret(config.secret_access_key)
        if config.session_token:
            register_secret(config.session_token)
        httpx = _httpx()
        self._client = httpx.Client(timeout=config.timeout_seconds, follow_redirects=False, transport=transport)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # ObjectStore protocol

    def put(self, key: str, data: bytes) -> str:
        self._request("PUT", key, content=data, payload=data, expect={200})
        return object_uri(key)

    def put_if_absent(self, key: str, data: bytes) -> tuple[str, bool]:
        status = self._request("PUT", key, content=data, payload=data, headers={"If-None-Match": "*"},
                               expect={200, 409, 412})
        return object_uri(key), status == 200  # 412: it exists; 409: another conditional write of it is in flight

    def put_file_if_absent(self, key: str, source: Path) -> tuple[str, bool]:
        size = source.stat().st_size

        def body() -> Iterator[bytes]:
            with source.open("rb") as f:
                while chunk := f.read(CHUNK):
                    yield chunk

        status = self._request("PUT", key, content=body(), payload=None, expect={200, 409, 412},
                               headers={"If-None-Match": "*", "Content-Length": str(size)})
        return object_uri(key), status == 200

    def get(self, uri: str) -> bytes:
        with self.open(uri) as reader:
            return reader.read()

    def get_key(self, key: str) -> bytes | None:
        try:
            with self._open_key(key) as reader:
                return reader.read()
        except FileNotFoundError:
            return None

    def open(self, uri: str) -> BinaryIO:
        return self._open_key(self._key(uri))

    def exists(self, uri: str) -> bool:
        return self._request("HEAD", self._key(uri), expect={200, 404}) == 200

    def delete(self, uri: str) -> None:
        self._request("DELETE", self._key(uri), expect={200, 204, 404})

    def check(self) -> None:
        """Readiness: write, read back and delete a probe object (raises ObjectStoreError otherwise)."""
        key = f".probe/{uuid.uuid4().hex}"
        self._request("PUT", key, content=b"ok", payload=b"ok", expect={200})
        try:
            with self._open_key(key) as reader:
                if reader.read() != b"ok":
                    raise ObjectStoreError("object store check: the probe object read back differently")
        finally:
            self._request("DELETE", key, expect={200, 204, 404})

    def close(self) -> None:
        self._client.close()

    # internals

    @staticmethod
    def _key(uri: str) -> str:
        if not is_object_uri(uri):
            raise ValueError("unsupported object uri")
        return key_of(uri)

    def _url(self, key: str) -> str:
        c = self._config
        full = "/".join(p for p in (c.prefix.strip("/"), validate_key(key)) if p)
        path = _encode(full, safe="/")
        if c.uses_path_style:
            return f"{c.base}/{_encode(c.bucket)}/{path}"
        parts = urlsplit(c.base)
        return f"{parts.scheme}://{c.bucket}.{parts.netloc}/{path}"

    def _signed(self, method: str, url: str, payload: bytes | None, headers: dict[str, str] | None) -> dict[str, str]:
        c = self._config
        digest = EMPTY_SHA256 if payload is None and method in {"GET", "HEAD", "DELETE"} else (
            UNSIGNED if payload is None else hashlib.sha256(payload).hexdigest())
        base = {"x-amz-content-sha256": digest}
        unsigned = {}
        for name, value in (headers or {}).items():
            (unsigned if name.lower() == "content-length" else base)[name] = value
        signed = sign_v4(method, url, base, digest, access_key=c.access_key_id, secret_key=c.secret_access_key,
                         region=c.region, now=self._clock(), session_token=c.session_token)
        return {**signed, **unsigned}

    def _request(self, method: str, key: str, *, content=None, payload: bytes | None = None,
                 headers: dict[str, str] | None = None, expect: set[int]) -> int:
        url = self._url(key)
        httpx = _httpx()
        try:
            response = self._client.request(method, url, content=content,
                                            headers=self._signed(method, url, payload, headers))
        except httpx.HTTPError as exc:
            raise ObjectStoreError(f"object store {method} failed: {type(exc).__name__}") from None
        try:
            if response.status_code not in expect:
                raise self._error(method, response)
            return response.status_code
        finally:
            response.close()

    def _open_key(self, key: str) -> BinaryIO:
        url = self._url(key)
        httpx = _httpx()
        request = self._client.build_request("GET", url, headers=self._signed("GET", url, None, None))
        try:
            response = self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise ObjectStoreError(f"object store GET failed: {type(exc).__name__}") from None
        if response.status_code == 404:
            response.close()
            raise FileNotFoundError("object not found")
        if response.status_code != 200:
            response.read()
            response.close()
            raise self._error("GET", response)
        return io.BufferedReader(_Reader(response), CHUNK)

    @staticmethod
    def _error(method: str, response) -> OSError:
        if response.status_code == 404:
            return FileNotFoundError("object not found")
        code = ""
        try:
            body = response.read()[:2000].decode("utf-8", "replace")
            if "<Code>" in body:
                code = body.split("<Code>", 1)[1].split("</Code>", 1)[0][:64]
        except Exception:  # noqa: BLE001 - the status alone is enough
            pass
        return ObjectStoreError(redact_text(f"object store {method} answered {response.status_code}"
                                            + (f" ({code})" if code else "")))
