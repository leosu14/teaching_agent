"""The teaching-turn lease (issue #66 PR 1, decision A4): producing a session's teacher turns is single-flight.

Deterministic: the teacher call is held at a gate (no timing guesses), and a lease runs out only when the test lets
it (a lease of a few milliseconds whose holder does not renew it). The same guarantees across API processes on one
PostgreSQL database are in tests/postgres/test_multi_replica_postgres.py.

- The same answer sent again while its reply is produced makes no teacher call: it waits and gets the stored result.
- A different input while a reply is produced is refused at once (TeachingTurnBusy, 409 with Retry-After), before
  any grading.
- A repeat that waits longer than the bound gets TeachingTurnBusy; once the reply is stored it gets the result.
- Concurrent resumes of a session that owes a reply make one teacher call per owed turn.
- A killed holder's lease runs out and the next request produces the reply.
- A holder whose lease ran out and was taken over stores nothing: the turns are stored once.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from app.schemas.teaching import LearnerInput, SessionChange, TeachingSessionStatus, TurnLeaseLost
from app.teaching.errors import TeacherUnavailable, TeachingTurnBusy
from tests.auth import client_for
from tests.integration.test_teaching_flow import AGENT, start
from tests.teaching_fixtures import KEY, Env, answer_for

Status = TeachingSessionStatus


class Gate:
    """Wraps the teacher call: counts every call and holds the first `hold` of them until `open()`."""

    def __init__(self, env: Env, hold: int = 1) -> None:
        self.real = env.service._runtime.teacher_turn
        self.calls = 0
        self.hold = hold
        self.entered = asyncio.Event()
        self._open = asyncio.Event()
        env.service._runtime.teacher_turn = self

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.hold:
            self.entered.set()
            await self._open.wait()
        return await self.real(*args, **kwargs)

    def open(self) -> None:
        self._open.set()


async def answer(env: Env, sid: str, cid: str, outcome: str = "incorrect"):
    view = await env.service.view(sid)
    return LearnerInput(answer=answer_for(view.state.waiting_question, KEY, outcome), client_turn_id=cid)


def teacher_turns(env: Env, sid: str) -> list[str]:
    return [t.turn_id for t in env.service.turns(sid) if t.speaker.value != "learner"]


async def owing_a_reply(env: Env) -> tuple[str, LearnerInput]:
    """A session whose answer is stored and whose reply is still owed (ACTIVE): its teacher call failed."""
    sid = (await start(env)).session_id
    data = await answer(env, sid, "c1")
    env.llm.inject(AGENT, *[RuntimeError("provider down")] * 20)
    with pytest.raises(TeacherUnavailable):
        await env.service.submit(sid, data)
    env.llm._faults[AGENT].clear()
    assert (await env.service.get(sid)).status == Status.ACTIVE
    return sid, data


async def test_the_same_answer_sent_again_while_its_reply_is_produced_makes_no_teacher_call(teaching_env) -> None:
    env = teaching_env
    sid = (await start(env)).session_id
    data = await answer(env, sid, "c1")
    gate = Gate(env)
    first = asyncio.create_task(env.service.submit(sid, data))
    await gate.entered.wait()  # the first request holds the lease and is inside the teacher call
    second = asyncio.create_task(env.service.submit(sid, data))
    await asyncio.sleep(0.2)  # the repeat polls: it is waiting, and has made no call
    assert not second.done() and gate.calls == 1
    gate.open()
    a, b = await asyncio.gather(first, second)
    assert sorted([a.replayed, b.replayed]) == [False, True]
    assert a.learner_turn.turn_id == b.learner_turn.turn_id
    assert [t.turn_id for t in a.teacher_turns] == [t.turn_id for t in b.teacher_turns]
    assert gate.calls == len(a.teacher_turns)  # one teacher call per owed turn, none for the repeat
    assert env.service.repository.turn_lease(sid) is None  # given back with the turns, in the same write


async def test_a_different_input_while_a_reply_is_produced_is_refused_at_once(teaching_env, monkeypatch) -> None:
    env = teaching_env
    sid = (await start(env)).session_id
    data = await answer(env, sid, "c1")
    gate = Gate(env)
    first = asyncio.create_task(env.service.submit(sid, data))
    await gate.entered.wait()
    graded = []
    real_assess = env.service._assessment.assess

    async def assess(*args, **kwargs):
        graded.append(1)
        return await real_assess(*args, **kwargs)

    monkeypatch.setattr(env.service._assessment, "assess", assess)
    for other in (LearnerInput(answer="otra respuesta", client_turn_id="c2"),
                  LearnerInput(answer="¿por qué?", kind="question", client_turn_id="c3"),
                  LearnerInput(answer="otra respuesta")):
        with pytest.raises(TeachingTurnBusy) as busy:
            await asyncio.wait_for(env.service.submit(sid, other), timeout=0.05)  # at once: it does not wait
        assert busy.value.retry_after >= 1 and str(busy.value) == TeachingTurnBusy.MESSAGE
    assert graded == [] and gate.calls == 1  # no grading call and no teacher call for them
    gate.open()
    done = await first
    assert done.status == Status.WAITING_FOR_LEARNER
    assert [t.turn_type.value for t in env.service.turns(sid)].count("LEARNER_ANSWER") == 1


async def test_a_repeat_that_waits_too_long_gets_busy_then_the_stored_result(teaching_env) -> None:
    env = teaching_env
    sid = (await start(env)).session_id
    data = await answer(env, sid, "c1")
    env.service._turn_wait = 0.1
    gate = Gate(env)
    first = asyncio.create_task(env.service.submit(sid, data))
    await gate.entered.wait()
    with pytest.raises(TeachingTurnBusy):
        await env.service.submit(sid, data)
    assert gate.calls == 1
    gate.open()
    stored = await first
    again = await env.service.submit(sid, data)
    assert again.replayed and [t.turn_id for t in again.teacher_turns] == [t.turn_id for t in stored.teacher_turns]
    assert gate.calls == len(stored.teacher_turns)


async def test_concurrent_resumes_make_one_teacher_call_per_owed_turn(teaching_env) -> None:
    env = teaching_env
    sid, data = await owing_a_reply(env)
    gate = Gate(env)
    resumes = [asyncio.create_task(env.service.resume(sid)) for _ in range(3)]
    await gate.entered.wait()
    replay = asyncio.create_task(env.service.submit(sid, data))
    await asyncio.sleep(0.1)
    gate.open()
    views = await asyncio.gather(*resumes)
    result = await replay
    assert {v.status for v in views} == {Status.WAITING_FOR_LEARNER}
    produced = teacher_turns(env, sid)
    assert gate.calls == len(result.teacher_turns) and result.replayed
    assert len(produced) == len(set(produced))


async def test_a_killed_holders_lease_runs_out_and_the_next_request_produces_the_reply(teaching_env) -> None:
    env = teaching_env
    sid, data = await owing_a_reply(env)
    # A process claimed the lease and died before storing anything: its lease is never renewed or given back.
    assert env.service.repository.claim_turns(sid, timedelta(milliseconds=300)) is not None
    env.service._turn_wait = 0.05
    gate = Gate(env, hold=0)
    with pytest.raises(TeachingTurnBusy):  # still live: nobody else may call the teacher
        await env.service.resume(sid)
    assert gate.calls == 0
    env.service._turn_wait = 5.0
    result = await env.service.submit(sid, data)  # waits for the dead lease to run out, then produces the reply
    assert result.replayed and result.status == Status.WAITING_FOR_LEARNER and result.teacher_turns
    assert gate.calls == len(result.teacher_turns)


async def test_a_holder_whose_lease_was_taken_over_stores_nothing(teaching_env, monkeypatch) -> None:
    env = teaching_env
    sid, data = await owing_a_reply(env)
    env.service._turn_lease = timedelta(milliseconds=200)

    async def hung(_lease) -> None:  # the holder is stuck (a long pause): it never renews its lease
        await asyncio.Event().wait()

    monkeypatch.setattr(env.service, "_renew_turns", hung)
    gate = Gate(env)
    stale = asyncio.create_task(env.service.resume(sid))
    await gate.entered.wait()
    while env.service.repository.turn_lease(sid) is not None:  # until the stale holder's lease runs out
        await asyncio.sleep(0.01)
    taken = await env.service.submit(sid, data)  # takes over: its own teacher call, its turns stored
    assert taken.replayed and taken.status == Status.WAITING_FOR_LEARNER
    produced = teacher_turns(env, sid)
    gate.open()
    view = await stale  # the stale holder's turns are refused; it returns what the new holder stored
    assert view.status == Status.WAITING_FOR_LEARNER
    assert teacher_turns(env, sid) == produced and len(produced) == len(set(produced))


async def test_a_turn_write_under_an_expired_lease_is_refused_and_stores_nothing(teaching_env) -> None:
    env = teaching_env
    sid, _ = await owing_a_reply(env)
    repo = env.service.repository
    lease = repo.claim_turns(sid, timedelta(milliseconds=50))
    session = repo.get(sid)
    while repo.turn_lease(sid) is not None:
        await asyncio.sleep(0.01)
    work = session.model_copy(deep=True)
    work.version += 1
    with pytest.raises(TurnLeaseLost):
        repo.apply(SessionChange(session=work), expected_version=session.version, turn_lease=lease)
    assert repo.get(sid).version == session.version
    assert repo.renew_turns(lease, timedelta(seconds=60)) is None  # an expired lease is never renewed
    newer = repo.claim_turns(sid, timedelta(seconds=60))
    assert newer is not None and newer.token != lease.token
    repo.release_turns(lease)  # the stale holder's release leaves the new lease alone
    assert repo.turn_lease(sid) is not None
    repo.release_turns(newer)
    assert repo.turn_lease(sid) is None


def test_the_api_answers_409_with_retry_after_while_a_reply_is_produced(teaching_env) -> None:
    env = teaching_env
    sid = asyncio.run(start(env)).session_id
    lease = env.service.repository.claim_turns(sid, timedelta(seconds=60))
    session = env.service.repository.get(sid)
    session.status = Status.ACTIVE  # a reply is owed, and another request produces it
    env.service.repository.apply(SessionChange(session=session.model_copy(update={"version": session.version + 1})),
                                 expected_version=session.version)
    with client_for(env.container, session.learner_id) as client:
        response = client.post(f"/teaching-sessions/{sid}/answers", json={"answer": "x", "client_turn_id": "z"})
    assert response.status_code == 409 and response.headers["Retry-After"].isdigit()
    assert response.json() == {"error": "TeachingTurnBusy", "detail": TeachingTurnBusy.MESSAGE}
    env.service.repository.release_turns(lease)
