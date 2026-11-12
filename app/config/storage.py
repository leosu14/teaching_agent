"""The S3 object store's settings, as plain data (the store itself is app/providers/storage/s3.py)."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from app.config.routing import ConfigError


@dataclass(frozen=True)
class S3Settings:
    bucket: str
    access_key_id: str
    secret_access_key: str = ""
    region: str = "us-east-1"
    endpoint_url: str | None = None  # unset: AWS (https://s3.<region>.amazonaws.com)
    prefix: str = ""
    path_style: bool | None = None  # unset: path style for a custom endpoint, virtual-hosted for AWS
    allow_http: bool = False
    session_token: str | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        problems = []
        if not self.bucket or "/" in self.bucket:
            problems.append("TA_S3_BUCKET must name a bucket")
        if not self.access_key_id or not self.secret_access_key:
            problems.append("TA_S3_ACCESS_KEY_ID and TA_S3_SECRET_ACCESS_KEY are required")
        if self.endpoint_url:
            parts = urlsplit(self.endpoint_url)
            if parts.scheme not in {"http", "https"} or not parts.hostname or parts.path not in {"", "/"}:
                problems.append("TA_S3_ENDPOINT_URL must be an http(s) URL with no path")
            elif parts.username or parts.password:
                problems.append("TA_S3_ENDPOINT_URL must not contain credentials")
            elif parts.scheme == "http" and not self.allow_http:
                problems.append("TA_S3_ENDPOINT_URL uses plain http: use https, or set TA_S3_ALLOW_HTTP=true for a "
                                "store on a private network")
        if self.prefix and (self.prefix.startswith("/") or ".." in self.prefix.split("/")):
            problems.append("TA_S3_PREFIX must be a relative key prefix")
        if problems:
            raise ConfigError("; ".join(problems))

    @property
    def base(self) -> str:
        return (self.endpoint_url or f"https://s3.{self.region}.amazonaws.com").rstrip("/")

    @property
    def uses_path_style(self) -> bool:
        return self.endpoint_url is not None if self.path_style is None else self.path_style


