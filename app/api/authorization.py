"""The authorization boundary: every protected route depends on one of these before it calls a service.

- `principal`: the request's authenticated principal (401 without one).
- `access`: what the principal may do with a learner's data in this request (`Access.check`):
  - a learner in the principal's own scope (the gateway's `X-Auth-Learners`): read and change;
  - a learner on the roster of a class the principal teaches (the `teacher` role, from stored rosters): read only.
    A read is a GET or HEAD request; anything else is a change, so a new non-GET route is closed to teachers by
    default. A teacher's change is 403 (they can already see the resource, so nothing is disclosed);
  - anything else, and a resource that does not exist: the same 404, so ids cannot be probed for existence.
- `require_learner`, `owned_*`: the resource's owning learner, read from stored data (`OwnershipService`), checked
  with `Access.check`.
- `require_role`: a capability beyond learner scope (403), e.g. registering assessment items or creating classes.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request

from app.api.auth import TEACHER, AuthenticationFailed, Authenticator, Principal
from app.api.deps import container
from app.services.classes import ClassNotFound, ClassService
from app.services.container import Container

READ_METHODS = frozenset({"GET", "HEAD"})


class ResourceNotFound(Exception):
    """Not found, or not visible to this principal: the two are deliberately indistinguishable (404)."""

    def __init__(self) -> None:
        super().__init__("not found")


class Forbidden(Exception):
    """Authenticated, but missing a role the action needs, or a teacher trying to change a learner's data (403)."""


def principal(request: Request) -> Principal:
    authenticator: Authenticator | None = getattr(request.app.state, "authenticator", None)
    if authenticator is None:  # never anonymous: an app without an authenticator rejects everything
        raise AuthenticationFailed("no authenticator configured")
    found = authenticator.authenticate(request)
    request.state.principal = found
    return found


@dataclass(frozen=True)
class Access:
    """The principal and whether this request only reads. Rosters are consulted only for a learner outside the
    principal's own scope, and only when the principal has the teacher role."""

    principal: Principal
    reading: bool
    classes: ClassService

    def teaches(self, learner_id: str) -> bool:
        p = self.principal
        return p.has_role(TEACHER) and self.classes.teaches(teacher_id=p.user_id, tenant_id=p.tenant_id,
                                                            learner_id=learner_id)

    def check(self, owner: str | None) -> str:
        """The owning learner id when this request may go ahead for it; ResourceNotFound or Forbidden otherwise."""
        if owner is None:
            raise ResourceNotFound()
        if self.principal.may_access(owner):
            return owner
        if self.teaches(owner):
            if self.reading:
                return owner
            raise Forbidden("a teacher may read the learners of their classes but not act for them")
        raise ResourceNotFound()


def access(request: Request, p: Principal = Depends(principal), c: Container = Depends(container)) -> Access:
    return Access(principal=p, reading=request.method.upper() in READ_METHODS, classes=c.classes)


def authorize(p: Principal, owner: str | None) -> str:
    """The owning learner id when `p` may act for it (its own scope only); ResourceNotFound otherwise."""
    if owner is None or not p.may_access(owner):
        raise ResourceNotFound()
    return owner


def require_role(p: Principal, role: str) -> None:
    if not p.has_role(role):
        raise Forbidden(f"this action needs the {role!r} role")


def require_learner(learner_id: str, a: Access = Depends(access)) -> str:
    return a.check(learner_id)


def owned_task(task_id: str, a: Access = Depends(access), c: Container = Depends(container)) -> str:
    return a.check(c.ownership.task(task_id))


def owned_goal(goal_id: str, a: Access = Depends(access), c: Container = Depends(container)) -> str:
    return a.check(c.ownership.goal(goal_id))


def owned_lesson(lesson_id: str, a: Access = Depends(access), c: Container = Depends(container)) -> str:
    return a.check(c.ownership.lesson(lesson_id))


def owned_session(session_id: str, a: Access = Depends(access), c: Container = Depends(container)) -> str:
    return a.check(c.ownership.teaching_session(session_id))


def owned_item(item_id: str, a: Access = Depends(access), c: Container = Depends(container)) -> str:
    return a.check(c.ownership.assessment_item(item_id))


def owned_attempt(attempt_id: str, a: Access = Depends(access), c: Container = Depends(container)) -> str:
    return a.check(c.ownership.assessment_attempt(attempt_id))


def owned_cycle(cycle_id: str, a: Access = Depends(access), c: Container = Depends(container)) -> str:
    return a.check(c.ownership.learning_cycle(cycle_id))


def owned_artifact(artifact_id: str, a: Access = Depends(access), c: Container = Depends(container)) -> str:
    return a.check(c.ownership.artifact(artifact_id))


def taught_class(class_id: str, p: Principal = Depends(principal), c: Container = Depends(container)):
    """A class the principal teaches (teacher role, same user and tenant). Any other class is the uniform 404."""
    if not p.has_role(TEACHER):
        raise ResourceNotFound()
    try:
        return c.classes.owned(class_id, teacher_id=p.user_id, tenant_id=p.tenant_id)
    except ClassNotFound:
        raise ResourceNotFound() from None
