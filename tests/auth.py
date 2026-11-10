"""The explicit test authentication mechanism: deterministic bearer tokens, one per principal, registered with a
`StaticTokenAuthenticator` that the test app is built with. Nothing here is a credential outside the test process."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading

import httpx

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.auth import Principal, StaticTokenAuthenticator
from app.services.container import Container


class TestAuth:
    __test__ = False  # not a test class

    def __init__(self) -> None:
        self.authenticator = StaticTokenAuthenticator()

    def principal(self, *learner_ids: str, user_id: str | None = None, roles: tuple[str, ...] = (),
                  tenant_id: str | None = None) -> Principal:
        return Principal(user_id=user_id or f"user:{','.join(learner_ids) or '-'}", learner_ids=frozenset(learner_ids),
                         roles=frozenset(roles), tenant_id=tenant_id)

    def headers(self, *learner_ids: str, user_id: str | None = None, roles: tuple[str, ...] = (),
                tenant_id: str | None = None) -> dict[str, str]:
        """Authorization headers for a principal scoped to `learner_ids` (the same principal, the same token)."""
        p = self.principal(*learner_ids, user_id=user_id, roles=roles, tenant_id=tenant_id)
        spec = json.dumps([p.user_id, sorted(p.learner_ids), sorted(p.roles)] + ([p.tenant_id] if tenant_id else []))
        token = "test-" + hashlib.sha256(spec.encode()).hexdigest()
        self.authenticator.add(token, p)
        return {"Authorization": f"Bearer {token}"}


def client_for(container: Container, *learner_ids: str, user_id: str | None = None, roles: tuple[str, ...] = (),
               auth: TestAuth | None = None, tenant_id: str | None = None) -> TestClient:
    """A TestClient that authenticates every request as a principal scoped to `learner_ids` (use as a context
    manager). `client.test_auth` registers further principals on the same app."""
    auth = auth or TestAuth()
    client = TestClient(create_app(container, authenticator=auth.authenticator),
                        headers=auth.headers(*learner_ids, user_id=user_id, roles=roles, tenant_id=tenant_id))
    client.test_auth = auth
    client.container = container
    return client


def drain(container: Container) -> list:
    """Run every queued task job now, as the worker process would (from a sync test, even inside a running loop)."""
    out: dict = {}

    def target() -> None:
        try:
            out["jobs"] = asyncio.run(container.task_worker.drain())
        except BaseException as exc:  # re-raised in the test's thread
            out["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in out:
        raise out["error"]
    return out["jobs"]


def settle(client: TestClient, response: httpx.Response) -> dict:
    """For a 202 from a task-executing route: let the worker run the queued job, then read the task back."""
    assert response.status_code == 202, response.text
    task_id = response.json()["task_id"]
    drain(client.container)
    fetched = client.get(f"/tasks/{task_id}")
    assert fetched.status_code == 200, fetched.text
    return fetched.json()
