"""An identity provider for the gateway's tests: RSA (and EC) keys, their JWKS, and tokens signed with them."""

from __future__ import annotations

import base64
import json
import time

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from app.api.app import create_app
from app.api.auth import GatewayAuthenticator
from app.gateway.app import create_gateway
from app.gateway.config import GatewaySettings

ISSUER, AUDIENCE = "https://idp.example.test/", "teaching-agent"
GATEWAY_SECRET = "gateway-shared-secret-" + "s" * 32


def _b64(n: int) -> str:
    return base64.urlsafe_b64encode(n.to_bytes((n.bit_length() + 7) // 8, "big")).rstrip(b"=").decode()


class IdP:
    def __init__(self, kid: str = "k1") -> None:
        self.kid = kid
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.ec_key = ec.generate_private_key(ec.SECP256R1())

    def jwk(self, key=None, kid: str | None = None) -> dict:
        key = key or self.key
        if isinstance(key, ec.EllipticCurvePrivateKey):
            nums = key.public_key().public_numbers()
            return {"kty": "EC", "crv": "P-256", "kid": kid or "ec1", "alg": "ES256", "use": "sig",
                    "x": _b64(nums.x).rjust(43, "A"), "y": _b64(nums.y).rjust(43, "A")}
        nums = key.public_key().public_numbers()
        return {"kty": "RSA", "kid": kid or self.kid, "alg": "RS256", "use": "sig", "n": _b64(nums.n),
                "e": _b64(nums.e)}

    def jwks(self) -> dict:
        return {"keys": [self.jwk(), self.jwk(self.ec_key)]}

    def token(self, sub: str = "learner-1", *, key=None, kid: str | None = None, alg: str = "RS256",
              lifetime: int = 600, **claims) -> str:
        now = int(time.time())
        body = {"iss": ISSUER, "aud": AUDIENCE, "sub": sub, "iat": now, "exp": now + lifetime} | claims
        body = {k: v for k, v in body.items() if v is not None}
        signing = key or (self.ec_key if alg.startswith("ES") else self.key)
        return jwt.encode(body, signing, algorithm=alg, headers={"kid": kid or (self.kid if alg.startswith("RS")
                                                                              else "ec1")})

    def write_jwks(self, path) -> None:
        path.write_text(json.dumps(self.jwks()), encoding="utf-8")


def gateway_settings(jwks_file, **overrides) -> GatewaySettings:
    values = dict(upstream_url="http://api.internal", oidc_issuer=ISSUER, oidc_audience=AUDIENCE,
                  jwks_file=jwks_file, auth_gateway_secret=GATEWAY_SECRET)
    return GatewaySettings(**(values | overrides))


def gateway_for(container, jwks_file, **overrides):
    """The gateway in front of the real API (in process), which authenticates with the gateway's signed headers."""
    api = create_app(container, authenticator=GatewayAuthenticator(GATEWAY_SECRET))
    api.state.container = container
    return create_gateway(gateway_settings(jwks_file, **overrides), upstream_transport=httpx.ASGITransport(app=api))
