"""MVP item 4: the access rule a request is checked with (`Access.check`), the teacher role and join codes."""

from __future__ import annotations

import pytest

from app.api.auth import AUTHOR, TEACHER, GatewayAuthenticator, Principal, gateway_headers
from app.api.authorization import READ_METHODS, Access, Forbidden, ResourceNotFound
from app.services.classes import hash_join_code, new_join_code


class FakeClasses:
    """Records which roster lookups were made; `rosters` maps (teacher, tenant) to the learners they teach."""

    def __init__(self, rosters: dict[tuple[str, str | None], set[str]] | None = None) -> None:
        self.rosters = rosters or {}
        self.asked: list[tuple[str, str | None, str]] = []

    def teaches(self, *, teacher_id: str, tenant_id: str | None, learner_id: str) -> bool:
        self.asked.append((teacher_id, tenant_id, learner_id))
        return learner_id in self.rosters.get((teacher_id, tenant_id), set())


def access(p: Principal, *, reading: bool, classes: FakeClasses | None = None) -> Access:
    return Access(principal=p, reading=reading, classes=classes or FakeClasses())  # type: ignore[arg-type]


TEACHER_P = Principal(user_id="t1", roles=frozenset({TEACHER}))


def test_the_teacher_role_is_known() -> None:
    assert Principal(user_id="t", roles=frozenset({TEACHER, AUTHOR})).has_role(TEACHER)
    with pytest.raises(ValueError):
        Principal(user_id="t", roles=frozenset({"admin"}))


def test_only_get_and_head_are_reads() -> None:
    assert READ_METHODS == {"GET", "HEAD"}


@pytest.mark.parametrize("reading", [True, False])
def test_the_principals_own_scope_reads_and_changes_without_a_roster_lookup(reading) -> None:
    classes = FakeClasses()
    p = Principal(user_id="u", learner_ids=frozenset({"a"}), roles=frozenset({TEACHER}))
    assert access(p, reading=reading, classes=classes).check("a") == "a"
    assert classes.asked == []


def test_a_teacher_reads_a_learner_on_their_roster() -> None:
    classes = FakeClasses({("t1", None): {"a"}})
    assert access(TEACHER_P, reading=True, classes=classes).check("a") == "a"
    assert classes.asked == [("t1", None, "a")]


def test_a_teacher_cannot_change_a_learner_on_their_roster() -> None:
    classes = FakeClasses({("t1", None): {"a"}})
    with pytest.raises(Forbidden):
        access(TEACHER_P, reading=False, classes=classes).check("a")


@pytest.mark.parametrize("reading", [True, False])
def test_a_learner_on_no_roster_of_theirs_is_not_found(reading) -> None:
    classes = FakeClasses({("t1", None): {"a"}, ("t2", None): {"b"}})
    with pytest.raises(ResourceNotFound):
        access(TEACHER_P, reading=reading, classes=classes).check("b")


def test_rosters_are_not_consulted_without_the_teacher_role() -> None:
    classes = FakeClasses({("t1", None): {"a"}})
    p = Principal(user_id="t1", roles=frozenset({AUTHOR}))
    with pytest.raises(ResourceNotFound):
        access(p, reading=True, classes=classes).check("a")
    assert classes.asked == []


def test_the_roster_lookup_is_scoped_to_the_principals_tenant() -> None:
    classes = FakeClasses({("t1", "school-a"): {"a"}})
    in_a = Principal(user_id="t1", roles=frozenset({TEACHER}), tenant_id="school-a")
    in_b = Principal(user_id="t1", roles=frozenset({TEACHER}), tenant_id="school-b")
    assert access(in_a, reading=True, classes=classes).check("a") == "a"
    with pytest.raises(ResourceNotFound):
        access(in_b, reading=True, classes=classes).check("a")
    with pytest.raises(ResourceNotFound):
        access(TEACHER_P, reading=True, classes=classes).check("a")  # no tenant is not any tenant


@pytest.mark.parametrize("reading", [True, False])
def test_a_resource_without_an_owner_is_not_found(reading) -> None:
    classes = FakeClasses()
    with pytest.raises(ResourceNotFound):
        access(TEACHER_P, reading=reading, classes=classes).check(None)
    assert classes.asked == []


def test_the_gateway_carries_the_teacher_role() -> None:
    from starlette.requests import Request

    secret = "s" * 40
    headers = gateway_headers(secret, method="GET", path="/classes", user_id="t1", learner_ids=[],
                              roles=[TEACHER], tenant_id="school-a")
    request = Request({"type": "http", "method": "GET", "path": "/classes", "query_string": b"",
                       "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]})
    p = GatewayAuthenticator(secret).authenticate(request)
    assert p.roles == {TEACHER} and p.learner_ids == frozenset() and p.tenant_id == "school-a"


def test_join_codes_are_random_and_hashed_stably() -> None:
    codes = {new_join_code() for _ in range(200)}
    assert len(codes) == 200 and all(len(c) == 16 for c in codes)
    code = new_join_code()
    assert hash_join_code(code) == hash_join_code(f"  {code} ")  # pasted with spaces: the same code
    assert hash_join_code(code) != hash_join_code(code.upper() if code.upper() != code else code + "x")
    assert code not in hash_join_code(code) and len(hash_join_code(code)) == 64
