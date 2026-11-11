"""The identity gateway's token verification (MVP item 8): only the issuer's asymmetric signatures, for this
audience, unexpired; claims become the API's principal and nothing else."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import jwt
import pytest

from app.config.routing import ConfigError
from app.gateway.config import GatewaySettings
from app.gateway.tokens import InvalidToken, KeySet, TokenVerifier
from tests.gateway_fixtures import AUDIENCE, ISSUER, IdP, gateway_settings


@pytest.fixture
def idp():
    return IdP()


def verifier(idp: IdP, **kw) -> TokenVerifier:
    keys = KeySet(url="https://idp.example.test/jwks", fetch=lambda _url: idp.jwks())
    return TokenVerifier(keys, issuer=ISSUER, audience=AUDIENCE, algorithms=["RS256", "ES256"],
                         role_map={"teachers": "teacher", "teacher": "teacher"}, **kw)


def test_a_valid_token_becomes_an_identity(idp) -> None:
    identity = verifier(idp).verify(idp.token("learner-1"))
    assert (identity.user_id, identity.learner_ids, identity.roles) == ("learner-1", ("learner-1",), ())
    es = verifier(idp).verify(idp.token("learner-2", alg="ES256"))
    assert es.learner_ids == ("learner-2",)


def test_claims_map_to_learners_and_roles(idp) -> None:
    v = verifier(idp)
    guardian = v.verify(idp.token("parent-1", ta_learners=["kid-1", "kid-2"]))
    assert guardian.learner_ids == ("kid-1", "kid-2")
    teacher = v.verify(idp.token("t-1", ta_roles=["teachers", "admins", "author"]))
    assert teacher.roles == ("teacher",)  # only mapped values; "author" is not in this map, "admins" means nothing
    assert v.verify(idp.token("auth0|abc", ta_learners="a,b")).learner_ids == ("a", "b")
    assert verifier(idp, self_learner=False).verify(idp.token("t-2")).learner_ids == ()


@pytest.mark.parametrize("claims", [{"sub": "a,b"}, {"sub": "x\ny"}, {"sub": None}, {"ta_learners": ["ok", "a b"]},
                                    {"ta_learners": [1, 2]}, {"ta_roles": {"teacher": True}}])
def test_ids_that_would_break_the_signed_headers_are_refused(idp, claims) -> None:
    with pytest.raises(InvalidToken):
        verifier(idp).verify(idp.token(**({"sub": "learner-1"} | claims)))


@pytest.mark.parametrize("change", ["expired", "audience", "issuer", "no_exp", "other_key", "tampered"])
def test_tokens_that_do_not_verify_are_refused(idp, change) -> None:
    other = IdP()
    token = {"expired": lambda: idp.token(lifetime=-120),
             "audience": lambda: idp.token(aud="another-app"),
             "issuer": lambda: idp.token(iss="https://evil.example/"),
             "no_exp": lambda: idp.token(exp=None),
             "other_key": lambda: idp.token(key=other.key),
             "tampered": lambda: _tamper(idp.token("learner-1"), sub="learner-2")}[change]()
    with pytest.raises(InvalidToken):
        verifier(idp).verify(token)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _tamper(token: str, **claims) -> str:
    head, body, sig = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=="))
    raw = base64.urlsafe_b64encode(json.dumps(payload | claims).encode()).rstrip(b"=").decode()
    return f"{head}.{raw}.{sig}"


def test_unsigned_and_shared_secret_tokens_are_refused(idp) -> None:
    unsigned = jwt.encode({"iss": ISSUER, "aud": AUDIENCE, "sub": "x", "exp": 9999999999}, None, algorithm="none")
    with pytest.raises(InvalidToken, match="algorithm"):
        verifier(idp).verify(unsigned)
    # Algorithm confusion: an HMAC token keyed with the public key's text must not verify.
    public = json.dumps(idp.jwk()).encode()
    head = _b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": idp.kid}).encode())
    body = _b64(json.dumps({"iss": ISSUER, "aud": AUDIENCE, "sub": "x", "exp": 9999999999}).encode())
    sig = _b64(hmac.new(public, f"{head}.{body}".encode(), hashlib.sha256).digest())
    hs = f"{head}.{body}.{sig}"
    with pytest.raises(InvalidToken, match="algorithm"):
        verifier(idp).verify(hs)
    for bad in ("", "x" * 20000, "not.a.jwt"):
        with pytest.raises(InvalidToken):
            verifier(idp).verify(bad)


def test_keys_are_fetched_again_on_rotation_but_not_on_every_unknown_kid(idp) -> None:
    fetches = []
    now = [0.0]
    current = {"jwks": idp.jwks()}

    def fetch(_url):
        fetches.append(1)
        return current["jwks"]

    keys = KeySet(url="https://idp/jwks", fetch=fetch, clock=lambda: now[0])
    v = TokenVerifier(keys, issuer=ISSUER, audience=AUDIENCE, algorithms=["RS256"])
    v.verify(idp.token())
    rotated = IdP(kid="k2")
    with pytest.raises(InvalidToken, match="unknown key"):  # just fetched: no refetch storm
        v.verify(rotated.token())
    assert len(fetches) == 1
    current["jwks"] = rotated.jwks()
    now[0] = 31
    assert v.verify(rotated.token("new")).user_id == "new"
    assert len(fetches) == 2


def test_gateway_settings_refuse_weak_configurations(tmp_path) -> None:
    jwks = tmp_path / "jwks.json"
    assert gateway_settings(jwks).problems() == []
    cases = {"auth_gateway_secret": "short", "oidc_audience": "", "algorithms": "HS256,RS256",
             "role_map": "admins=superuser", "upstream_url": "api:8000"}
    for field, value in cases.items():
        problems = gateway_settings(jwks, **{field: value}).problems()
        assert len(problems) == 1, (field, problems)
    with pytest.raises(ConfigError, match="identity gateway"):
        GatewaySettings(oidc_issuer="", auth_gateway_secret=None).require_valid()
