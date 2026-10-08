"""Authentication at the API boundary: who is calling, and which learners they may act for.

The rest of the application never sees how a caller authenticated. Every protected route receives a `Principal`
(user id, learner scope, roles), and an `Authenticator` turns a request into one. Two authenticators exist:

- `GatewayAuthenticator` (production): a trusted auth gateway in front of the API (OIDC, SSO, session cookies,
  whatever the deployment uses) authenticates the user, maps them to their learner scope, and forwards that identity
  in headers signed with a shared secret. The API verifies the signature and a timestamp; it never talks to an
  identity provider itself. `gateway_headers` builds the headers and is the reference for the gateway's side.
- `StaticTokenAuthenticator` (development and tests): fixed bearer tokens mapped to principals, loaded from a local
  file (never committed) or injected by tests.

There is no anonymous mode: a request without valid credentials is rejected, and a misconfigured authenticator
refuses to start rather than letting requests through.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from fastapi import Request

from app.config.routing import ConfigError
from app.config.settings import Settings
from app.observability.redaction import register_secret

AUTHOR = "author"  # may register assessment items (with their answer keys and rubrics) on lessons in scope
ROLES = frozenset({AUTHOR})


@dataclass(frozen=True)
class Principal:
    """An authenticated caller. `learner_ids` is the learner scope: the learners whose data this caller may read and
    change (a learner has their own id; a teacher or parent may have several). Nothing outside the scope exists for
    them."""

    user_id: str
    learner_ids: frozenset[str] = field(default_factory=frozenset)
    roles: frozenset[str] = field(default_factory=frozenset)
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("a principal needs a user id")
        if unknown := self.roles - ROLES:
            raise ValueError(f"unknown roles: {sorted(unknown)}")

    def may_access(self, learner_id: str | None) -> bool:
        return learner_id is not None and learner_id in self.learner_ids

    def has_role(self, role: str) -> bool:
        return role in self.roles


class AuthenticationFailed(Exception):
    """The request carries no credentials, or credentials that do not verify (401)."""


class Authenticator(Protocol):
    def authenticate(self, request: Request) -> Principal:
        """The request's principal, or AuthenticationFailed. Never returns an anonymous principal."""
        ...


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationFailed("missing bearer token")
    return token.strip()


class StaticTokenAuthenticator:
    """Bearer tokens mapped to principals. For development and tests only: tokens are fixed and never expire."""

    MIN_TOKEN_LENGTH = 16

    def __init__(self, tokens: dict[str, Principal] | None = None) -> None:
        self._tokens: dict[str, Principal] = {}
        for token, principal in (tokens or {}).items():
            self.add(token, principal)

    def add(self, token: str, principal: Principal) -> None:
        if len(token) < self.MIN_TOKEN_LENGTH:
            raise ConfigError(f"static auth tokens must be at least {self.MIN_TOKEN_LENGTH} characters")
        register_secret(token)
        self._tokens[token] = principal

    def authenticate(self, request: Request) -> Principal:
        presented = _bearer(request).encode()
        for token, principal in self._tokens.items():
            if hmac.compare_digest(presented, token.encode()):
                return principal
        raise AuthenticationFailed("unknown bearer token")

    @classmethod
    def from_file(cls, path: Path) -> StaticTokenAuthenticator:
        """`{"<token>": {"user_id": "...", "learner_ids": ["..."], "roles": ["author"]}, ...}`"""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls({token: _principal(spec) for token, spec in raw.items()})
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ConfigError(f"cannot load static auth tokens from {path}: {exc}") from exc


def _principal(spec: dict) -> Principal:
    return Principal(user_id=spec["user_id"], learner_ids=frozenset(spec.get("learner_ids", [])),
                     roles=frozenset(spec.get("roles", [])), tenant_id=spec.get("tenant_id"))


# The gateway's identity headers. Lists are comma-separated; the signature covers every one of them plus the method,
# the path and the timestamp, so a captured set of headers cannot be replayed against another route or after the
# allowed clock skew.
SUBJECT, LEARNERS, ROLES_HEADER, TENANT = "x-auth-subject", "x-auth-learners", "x-auth-roles", "x-auth-tenant"
TIMESTAMP, SIGNATURE = "x-auth-timestamp", "x-auth-signature"


def _canonical(method: str, path: str, timestamp: str, subject: str, tenant: str, learners: str, roles: str) -> bytes:
    return "\n".join(["v1", method.upper(), path, timestamp, subject, tenant, learners, roles]).encode()


def gateway_headers(secret: str, *, method: str, path: str, user_id: str, learner_ids: list[str] | tuple[str, ...],
                    roles: list[str] | tuple[str, ...] = (), tenant_id: str | None = None,
                    timestamp: int | None = None) -> dict[str, str]:
    """The headers an auth gateway sends with a request it has authenticated (the gateway's side of the contract)."""
    ts = str(int(time.time()) if timestamp is None else timestamp)
    learners, role_list, tenant = ",".join(learner_ids), ",".join(roles), tenant_id or ""
    digest = hmac.new(secret.encode(), _canonical(method, path, ts, user_id, tenant, learners, role_list),
                      hashlib.sha256).hexdigest()
    headers = {SUBJECT: user_id, LEARNERS: learners, ROLES_HEADER: role_list, TIMESTAMP: ts, SIGNATURE: digest}
    if tenant_id:
        headers[TENANT] = tenant_id
    return headers


class GatewayAuthenticator:
    """Identity asserted by a trusted auth gateway in HMAC-SHA256-signed headers (see `gateway_headers`)."""

    def __init__(self, secret: str, *, max_skew_seconds: int = 300, clock=time.time) -> None:
        if len(secret) < 32:
            raise ConfigError("the auth gateway secret must be at least 32 characters")
        register_secret(secret)
        self._secret = secret.encode()
        self._max_skew = max_skew_seconds
        self._clock = clock

    def authenticate(self, request: Request) -> Principal:
        h = request.headers
        subject, signature, ts = h.get(SUBJECT, ""), h.get(SIGNATURE, ""), h.get(TIMESTAMP, "")
        if not subject or not signature or not ts:
            raise AuthenticationFailed("missing gateway identity")
        try:
            skew = abs(self._clock() - int(ts))
        except ValueError:
            raise AuthenticationFailed("invalid gateway timestamp") from None
        if skew > self._max_skew:
            raise AuthenticationFailed("gateway identity expired")
        learners, roles, tenant = h.get(LEARNERS, ""), h.get(ROLES_HEADER, ""), h.get(TENANT, "")
        expected = hmac.new(self._secret, _canonical(request.method, request.url.path, ts, subject, tenant, learners,
                                                     roles), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise AuthenticationFailed("invalid gateway signature")
        try:
            return Principal(user_id=subject, learner_ids=_split(learners), roles=_split(roles),
                             tenant_id=tenant or None)
        except ValueError as exc:
            raise AuthenticationFailed(str(exc)) from None


def _split(value: str) -> frozenset[str]:
    return frozenset(x.strip() for x in value.split(",") if x.strip())


def authenticator_from_settings(settings: Settings) -> Authenticator:
    """The configured authenticator. Fails (ConfigError) instead of falling back to anything weaker."""
    if settings.auth_mode == "gateway":
        if settings.auth_gateway_secret is None:
            raise ConfigError("TA_AUTH_MODE=gateway needs TA_AUTH_GATEWAY_SECRET; the API does not run unauthenticated")
        return GatewayAuthenticator(settings.auth_gateway_secret.get_secret_value(),
                                    max_skew_seconds=settings.auth_gateway_max_skew_seconds)
    if settings.auth_mode == "static":
        if settings.auth_static_tokens_file is None:
            raise ConfigError("TA_AUTH_MODE=static needs TA_AUTH_STATIC_TOKENS_FILE")
        return StaticTokenAuthenticator.from_file(settings.auth_static_tokens_file)
    raise ConfigError(f"unknown TA_AUTH_MODE {settings.auth_mode!r}")
