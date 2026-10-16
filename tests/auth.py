"""The explicit test authentication mechanism: deterministic bearer tokens, one per principal, registered with a
`StaticTokenAuthenticator` that the test app is built with. Nothing here is a credential outside the test process."""

from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.auth import Principal, StaticTokenAuthenticator
from app.services.container import Container


class TestAuth:
    __test__ = False  # not a test class

    def __init__(self) -> None:
        self.authenticator = StaticTokenAuthenticator()

    def principal(self, *learner_ids: str, user_id: str | None = None, roles: tuple[str, ...] = ()) -> Principal:
        return Principal(user_id=user_id or f"user:{','.join(learner_ids) or '-'}", learner_ids=frozenset(learner_ids),
                         roles=frozenset(roles))

    def headers(self, *learner_ids: str, user_id: str | None = None, roles: tuple[str, ...] = ()) -> dict[str, str]:
        """Authorization headers for a principal scoped to `learner_ids` (the same principal, the same token)."""
        p = self.principal(*learner_ids, user_id=user_id, roles=roles)
        spec = json.dumps([p.user_id, sorted(p.learner_ids), sorted(p.roles)])
        token = "test-" + hashlib.sha256(spec.encode()).hexdigest()
        self.authenticator.add(token, p)
        return {"Authorization": f"Bearer {token}"}


def client_for(container: Container, *learner_ids: str, user_id: str | None = None, roles: tuple[str, ...] = (),
               auth: TestAuth | None = None) -> TestClient:
    """A TestClient that authenticates every request as a principal scoped to `learner_ids` (use as a context
    manager). `client.test_auth` registers further principals on the same app."""
    auth = auth or TestAuth()
    client = TestClient(create_app(container, authenticator=auth.authenticator),
                        headers=auth.headers(*learner_ids, user_id=user_id, roles=roles))
    client.test_auth = auth
    return client
