"""The identity gateway: an ASGI reverse proxy in front of the API.

For every request but the public ones (`/health` and the web app's static files under `/app`, which the API serves
without identity), the gateway:

1. takes the identity provider's token from the configured headers (a bearer token, or the header a login proxy sets
   once the browser has signed in) and verifies it (app/gateway/tokens.py); without a valid one it answers 401 and
   the API is never called;
2. removes everything identity-shaped the client sent: `Authorization`, cookies, every `x-auth-*` header and the
   token headers, so a client can neither assert an identity nor see its token reach the API;
3. signs the verified identity with the shared secret (`gateway_headers`, the API's own reference for the contract)
   over the method and the path the API will see, and forwards the request, streaming both ways.

The API keeps every authorization rule (learner scope, the teacher's rosters, answer isolation); the gateway only
says who is calling. It never follows redirects from the API and never forwards hop-by-hop headers.
"""

from __future__ import annotations

import contextlib
import logging
import time

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from app.api.auth import gateway_headers
from app.gateway.config import GatewaySettings
from app.gateway.tokens import InvalidToken, KeySet, TokenVerifier

logger = logging.getLogger(__name__)

PUBLIC_EXACT = {"/health", "/app"}
PUBLIC_PREFIXES = ("/app/",)
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
              "transfer-encoding", "upgrade", "host", "content-length"}
STRIPPED_INBOUND = {"authorization", "cookie", "forwarded", "x-forwarded-for", "x-forwarded-host",
                    "x-forwarded-proto", "x-real-ip"}
METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
UNAUTHORIZED = {"error": "Unauthorized", "detail": "sign in to continue"}


def is_public(path: str) -> bool:
    return path in PUBLIC_EXACT or path.startswith(PUBLIC_PREFIXES)


def bearer(value: str) -> str:
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else value.strip()


def create_gateway(settings: GatewaySettings | None = None, *, verifier: TokenVerifier | None = None,
                   upstream_transport: httpx.AsyncBaseTransport | None = None,
                   jwks_fetch=None, clock=time.time) -> Starlette:
    settings = settings or GatewaySettings()
    settings.require_valid()
    secret = settings.auth_gateway_secret.get_secret_value()
    if verifier is None:
        keys = KeySet(url=settings.jwks_url or (None if settings.jwks_file else _discover(settings, jwks_fetch)),
                      file=settings.jwks_file, fetch=jwks_fetch or _fetch_json,
                      cache_seconds=settings.jwks_cache_seconds)
        verifier = TokenVerifier(keys, issuer=settings.oidc_issuer, audience=settings.oidc_audience,
                                 algorithms=settings.algorithm_list(), leeway=settings.leeway_seconds,
                                 user_claim=settings.user_claim, learners_claim=settings.learners_claim,
                                 self_learner=settings.self_learner, roles_claim=settings.roles_claim,
                                 role_map=settings.roles(), tenant_claim=settings.tenant_claim)
    token_headers = settings.header_list()
    upstream = httpx.AsyncClient(base_url=settings.upstream_url.rstrip("/"), transport=upstream_transport,
                                 timeout=settings.timeout_seconds, follow_redirects=False)

    async def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok", "service": "identity-gateway"})

    async def proxy(request: Request) -> Response:
        path = request.url.path
        identity_headers: dict[str, str] = {}
        if not is_public(path):
            token = next((bearer(request.headers[h]) for h in token_headers if request.headers.get(h)), "")
            try:
                identity = verifier.verify(token)
            except InvalidToken as exc:
                logger.info("gateway: rejected a request to %s %s: %s", request.method, path, exc)
                return JSONResponse(UNAUTHORIZED, status_code=401, headers={"WWW-Authenticate": "Bearer"})
            identity_headers = gateway_headers(secret, method=request.method, path=path, user_id=identity.user_id,
                                               learner_ids=identity.learner_ids, roles=identity.roles,
                                               tenant_id=identity.tenant_id, timestamp=int(clock()))
        headers = [(k, v) for k, v in request.headers.items()
                   if k not in HOP_BY_HOP and k not in STRIPPED_INBOUND and k not in token_headers
                   and not k.startswith("x-auth-")]
        headers += list(identity_headers.items())
        raw_path = request.scope.get("raw_path") or path.encode()
        target = raw_path.decode("latin-1") + (f"?{request.url.query}" if request.url.query else "")
        outbound = upstream.build_request(request.method, target, headers=headers,
                                          content=request.stream() if request.method not in {"GET", "HEAD"} else None)
        try:
            response = await upstream.send(outbound, stream=True)
        except httpx.HTTPError as exc:
            logger.warning("gateway: the API is unreachable: %s", type(exc).__name__)
            return JSONResponse({"error": "Unavailable", "detail": "the service is unavailable"}, status_code=502)
        out_headers = {k: v for k, v in response.headers.items() if k.lower() not in HOP_BY_HOP}
        return StreamingResponse(response.aiter_raw(), status_code=response.status_code, headers=out_headers,
                                 background=BackgroundTask(response.aclose))

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        await upstream.aclose()

    return Starlette(routes=[Route("/gateway/health", health, methods=["GET"]),
                             Route("/{path:path}", proxy, methods=METHODS)], lifespan=lifespan)


def _fetch_json(url: str) -> dict:
    with httpx.Client(timeout=10, follow_redirects=False) as client:
        response = client.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        return response.json()


def _discover(settings: GatewaySettings, fetch=None) -> str:
    """The JWKS URL from the issuer's OpenID discovery document."""
    url = settings.oidc_issuer.rstrip("/") + "/.well-known/openid-configuration"
    document = (fetch or _fetch_json)(url)
    jwks = document.get("jwks_uri") if isinstance(document, dict) else None
    if not isinstance(jwks, str) or not jwks.startswith("https://"):
        raise RuntimeError("the identity provider's discovery document has no https jwks_uri")
    return jwks
