"""Verifying the identity provider's tokens, and turning their claims into the API's principal.

A token is accepted only when its signature verifies with one of the issuer's published public keys, with an
allowed asymmetric algorithm (never `none` or a shared-secret HMAC: the header's `alg` is checked against the
configured list before anything else), and its `iss`, `aud`, `exp` (required), `nbf` and `iat` hold. The key set is
cached and fetched again, at most every `MIN_REFETCH_SECONDS`, when a token names a key id it does not have (key
rotation).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import jwt
from jwt import PyJWKSet

from app.config.routing import ConfigError

MIN_REFETCH_SECONDS = 30
MAX_TOKEN_BYTES = 16 * 1024
# The identifiers that go into the signed headers: no separators (comma, newline), no whitespace.
ID = re.compile(r"^[A-Za-z0-9._:@|+=-]{1,128}$")


class InvalidToken(Exception):
    """The token is missing, malformed, unsigned, signed by an unknown key, expired or for someone else (401)."""


@dataclass(frozen=True)
class Identity:
    user_id: str
    learner_ids: tuple[str, ...]
    roles: tuple[str, ...]
    tenant_id: str | None = None


class KeySet:
    """The issuer's public signing keys: from a URL (fetched with `fetch`, cached) or a file."""

    def __init__(self, *, url: str | None = None, file: Path | None = None, fetch=None, cache_seconds: int = 300,
                 clock=time.monotonic) -> None:
        if not url and not file:
            raise ConfigError("the identity gateway needs a JWKS URL or file")
        self._url, self._file, self._fetch = url, file, fetch
        self._ttl, self._clock = cache_seconds, clock
        self._keys: dict[str, object] = {}
        self._fetched_at: float | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        raw = self._file.read_text(encoding="utf-8") if self._file else self._fetch(self._url)
        try:
            keyset = PyJWKSet.from_dict(json.loads(raw) if isinstance(raw, str) else raw)
        except (ValueError, jwt.PyJWKSetError) as exc:
            raise InvalidToken(f"the identity provider's key set is unusable: {type(exc).__name__}") from None
        self._keys = {k.key_id or "": k for k in keyset.keys}
        self._fetched_at = self._clock()

    def get(self, kid: str | None):
        with self._lock:
            now = self._clock()
            if self._fetched_at is None or now - self._fetched_at > self._ttl:
                self._load()
            key = self._keys.get(kid or "")
            if key is None and kid is None and len(self._keys) == 1:
                key = next(iter(self._keys.values()))
            if key is None and now - (self._fetched_at or 0) > MIN_REFETCH_SECONDS:
                self._load()  # rotated keys
                key = self._keys.get(kid or "")
            if key is None:
                raise InvalidToken("the token is signed by an unknown key")
            return key


class TokenVerifier:
    def __init__(self, keys: KeySet, *, issuer: str, audience: str, algorithms: list[str], leeway: int = 30,
                 user_claim: str = "sub", learners_claim: str = "ta_learners", self_learner: bool = True,
                 roles_claim: str = "ta_roles", role_map: dict[str, str] | None = None,
                 tenant_claim: str | None = None) -> None:
        self._keys, self._issuer, self._audience = keys, issuer, audience
        self._algorithms, self._leeway = algorithms, leeway
        self._user_claim, self._learners_claim, self._self_learner = user_claim, learners_claim, self_learner
        self._roles_claim, self._role_map, self._tenant_claim = roles_claim, role_map or {}, tenant_claim

    def verify(self, token: str) -> Identity:
        if not token or len(token) > MAX_TOKEN_BYTES:
            raise InvalidToken("missing or oversized token")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise InvalidToken("malformed token") from None
        if header.get("alg") not in self._algorithms:
            raise InvalidToken("token algorithm not allowed")
        key = self._keys.get(header.get("kid"))
        try:
            claims = jwt.decode(token, key=key, algorithms=self._algorithms, audience=self._audience,
                                issuer=self._issuer, leeway=self._leeway,
                                options={"require": ["exp", "iss", "aud"], "verify_signature": True})
        except jwt.PyJWTError as exc:
            raise InvalidToken(f"token rejected: {type(exc).__name__}") from None
        return self.identity(claims)

    def identity(self, claims: dict) -> Identity:
        user = claims.get(self._user_claim)
        if not isinstance(user, str) or not ID.match(user):
            raise InvalidToken("the token has no usable user id")
        learners = _strings(claims.get(self._learners_claim))
        if not learners and self._self_learner:
            learners = [user]
        roles = sorted({self._role_map[v] for v in _strings(claims.get(self._roles_claim)) if v in self._role_map})
        tenant = claims.get(self._tenant_claim) if self._tenant_claim else None
        for value in [*learners, *([tenant] if tenant is not None else [])]:
            if not isinstance(value, str) or not ID.match(value):
                raise InvalidToken("the token has an unusable learner or tenant id")
        return Identity(user_id=user, learner_ids=tuple(sorted(set(learners))), roles=tuple(roles),
                        tenant_id=tenant)


def _strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list):
        if not all(isinstance(v, str) for v in value):
            raise InvalidToken("a list claim holds something other than strings")
        return [v.strip() for v in value if v.strip()]
    raise InvalidToken("a list claim has an unusable type")
