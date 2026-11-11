"""The identity gateway's settings (TA_GATEWAY_*). The shared signing secret is the API's TA_AUTH_GATEWAY_SECRET."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.routing import ConfigError

ALLOWED_ALGORITHMS = frozenset({"RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512",
                                "EdDSA"})  # asymmetric only: the gateway never holds a key that can mint tokens


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TA_GATEWAY_", env_file=".env", extra="ignore")

    upstream_url: str = "http://127.0.0.1:8000"  # the API, on the private network
    # The identity provider (OIDC). Tokens are verified against its signing keys: from `jwks_url`, or from the
    # issuer's discovery document (<issuer>/.well-known/openid-configuration) when unset, or from `jwks_file` (a
    # static key set, for an IdP whose keys are distributed as a file, and for tests).
    oidc_issuer: str = ""
    oidc_audience: str = ""
    jwks_url: str | None = None
    jwks_file: Path | None = None
    algorithms: str = "RS256,ES256"
    leeway_seconds: int = Field(default=30, ge=0, le=300)
    jwks_cache_seconds: int = Field(default=300, ge=10, le=86400)
    # Where the token comes from: the Authorization bearer header (API clients), or a header a login proxy in front
    # sets after the browser signs in (oauth2-proxy's X-Forwarded-Access-Token, for example). The proxy must strip
    # that header from what the browser sends; the token is verified here either way.
    token_headers: str = "authorization,x-forwarded-access-token"
    # Claims -> principal. The user id is `user_claim`. The learner scope is `learners_claim` (a list or a
    # comma-separated string); when the token has none and `self_learner` is on, the user is their own learner.
    # Roles come from `roles_claim` (a list, e.g. groups) through `role_map` (claim value=role, comma-separated);
    # values not in the map are ignored. The tenant is `tenant_claim`, if set.
    user_claim: str = "sub"
    learners_claim: str = "ta_learners"
    self_learner: bool = True
    roles_claim: str = "ta_roles"
    role_map: str = "teacher=teacher,author=author"
    tenant_claim: str | None = None
    timeout_seconds: float = Field(default=300.0, gt=0, le=3600)  # a lesson's upload or a video download

    auth_gateway_secret: SecretStr | None = Field(default=None, validation_alias=AliasChoices(
        "TA_AUTH_GATEWAY_SECRET", "auth_gateway_secret"))

    def algorithm_list(self) -> list[str]:
        return _csv(self.algorithms)

    def header_list(self) -> list[str]:
        return [h.lower() for h in _csv(self.token_headers)]

    def roles(self) -> dict[str, str]:
        mapping = {}
        for entry in _csv(self.role_map):
            source, sep, role = entry.partition("=")
            if not sep or not source.strip() or not role.strip():
                raise ConfigError(f"TA_GATEWAY_ROLE_MAP: {entry!r} is not claim-value=role")
            mapping[source.strip()] = role.strip()
        return mapping

    def problems(self) -> list[str]:
        from app.api.auth import ROLES

        problems = []
        secret = self.auth_gateway_secret.get_secret_value() if self.auth_gateway_secret else ""
        if len(secret) < 32:
            problems.append("TA_AUTH_GATEWAY_SECRET (or TA_AUTH_GATEWAY_SECRET_FILE) must be set, at least 32 "
                            "characters: the same secret as the API's")
        if not self.oidc_issuer:
            problems.append("TA_GATEWAY_OIDC_ISSUER must name the identity provider's issuer")
        if not self.oidc_audience:
            problems.append("TA_GATEWAY_OIDC_AUDIENCE must be set (the client id or API audience tokens are for)")
        if not self.jwks_url and not self.jwks_file and not self.oidc_issuer.startswith("https://"):
            problems.append("set TA_GATEWAY_JWKS_URL or TA_GATEWAY_JWKS_FILE, or an https issuer to discover keys")
        bad = [a for a in self.algorithm_list() if a not in ALLOWED_ALGORITHMS]
        if bad or not self.algorithm_list():
            problems.append(f"TA_GATEWAY_ALGORITHMS must be asymmetric signature algorithms (one of "
                            f"{sorted(ALLOWED_ALGORITHMS)}), not {bad or 'none'}")
        try:
            unknown = sorted(set(self.roles().values()) - ROLES)
            if unknown:
                problems.append(f"TA_GATEWAY_ROLE_MAP maps to unknown roles {unknown} (known: {sorted(ROLES)})")
        except ConfigError as exc:
            problems.append(str(exc))
        if not self.upstream_url.startswith(("http://", "https://")):
            problems.append("TA_GATEWAY_UPSTREAM_URL must be the API's http(s) URL")
        return problems

    def require_valid(self) -> None:
        if problems := self.problems():
            raise ConfigError("identity gateway: " + "; ".join(problems))
