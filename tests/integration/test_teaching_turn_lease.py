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


async def test_a_turn_write_is_fenced_by_the_lease_token(teaching_env) -> None:
    """The token fences the turns' write: a lease that ran out but that nobody took over still stores its holder's
    turns (they are the only ones produced); once another request claimed it, the old holder's write is refused."""
    env = teaching_env
    sid, _ = await owing_a_reply(env)
    repo = env.service.repository

    def change():
        session = repo.get(sid)
        work = session.model_copy(deep=True)
        work.version += 1
        return SessionChange(session=work), session.version

    lease = repo.claim_turns(sid, timedelta(milliseconds=50))
    while repo.turn_lease(sid) is not None:  # ran out, nobody claimed it since
        await asyncio.sleep(0.01)
    assert repo.renew_turns(lease, timedelta(seconds=60)) is None  # an expired lease is never renewed
    work, version = change()
    repo.apply(work, expected_version=version, turn_lease=lease)  # stored, and the lease given back
    assert repo.get(sid).version == version + 1 and repo.turn_lease(sid) is None

    stale = repo.claim_turns(sid, timedelta(milliseconds=50))
    while repo.turn_lease(sid) is not None:
        await asyncio.sleep(0.01)
    newer = repo.claim_turns(sid, timedelta(seconds=60))  # taken over
    assert newer is not None and newer.token != stale.token
    work, version = change()
    with pytest.raises(TurnLeaseLost):
        repo.apply(work, expected_version=version, turn_lease=stale)
    assert repo.get(sid).version == version  # nothing stored
    repo.release_turns(stale)  # the stale holder's release leaves the new lease alone
    assert repo.turn_lease(sid) is not None
    repo.release_turns(newer)
    assert repo.turn_lease(sid) is None


async def test_a_repeat_of_an_answered_input_never_waits_on_a_later_reply(teaching_env) -> None:
    """Review finding 3 of PR #67: the reply of `c1` is stored; replaying `c1` while `c2`'s reply is produced returns
    `c1`'s stored result at once, without waiting for the holder or calling the teacher."""
    env = teaching_env
    sid = (await start(env)).session_id
    data = await answer(env, sid, "c1")
    first = await env.service.submit(sid, data)
    assert first.teacher_turns
    gate = Gate(env)
    later = asyncio.create_task(env.service.submit(sid, await answer(env, sid, "c2")))
    await gate.entered.wait()
    again = await asyncio.wait_for(env.service.submit(sid, data), timeout=0.5)
    assert again.replayed and [t.turn_id for t in again.teacher_turns] == [t.turn_id for t in first.teacher_turns]
    assert gate.calls == 1
    gate.open()
    await later


async def test_a_turn_lease_is_not_claimed_for_a_deleted_learner(teaching_env) -> None:
    """The claim is a write for the learner: refused once their data is deleted (it takes the deletion lock first on
    PostgreSQL, see tests/postgres/test_multi_replica_postgres.py)."""
    from app.schemas.lease import LearnerDataGone, learner_writes

    env = teaching_env
    sid, _ = await owing_a_reply(env)
    session = env.service.repository.get(sid)
    with learner_writes("someone-deleted", session.task_id):
        with pytest.raises(LearnerDataGone):
            env.service.repository.claim_turns(sid, timedelta(seconds=60))
    assert env.service.repository.turn_lease(sid) is None


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
