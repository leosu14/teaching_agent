"""Crash recovery of the spend ledger (PR19), across two processes on one database.

A worker reserves a model call and never settles it (its process is stuck or gone). Nothing runs in the background:
the next reservation anyone makes, after the abandoned one's `reserved_until`, settles it as `unknown` and charges its
estimate. If the original call does come back later, its reported usage replaces the estimate, exactly once, even
though that worker has lost its lease by then.
"""

from __future__ import annotations

import asyncio

import pytest

from app.schemas.lease import LeaseLost
from app.schemas.spend import SpendState
from app.schemas.task import TaskStatus
from tests.conftest import LEARNER, add_demo_learner, answers_for
from tests.integration.test_task_concurrency import AHEAD, Clock, Gate, finish, open_container


async def test_an_abandoned_reservation_is_charged_by_the_next_worker_and_corrected_by_a_late_answer(
        tmp_path) -> None:
    data = tmp_path / "data"
    a, a_llm = open_container(data)
    b, _ = open_container(data, clock=Clock(AHEAD))
    try:
        lid = add_demo_learner(a)
        task = await a.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
        gate = Gate(a_llm)
        gate.arm()
        stuck = asyncio.create_task(a.task_service.submit_assessment(task.task_id, answers_for(task)))
        await gate.entered.wait()  # A reserved its call and is parked in it: as good as crashed
        [inflight] = [e for e in a.spend.entries(task_id=task.task_id) if e.state == SpendState.RESERVED]

        # B, later: its first reservation sweeps A's abandoned one, then it runs the task to the end.
        newer = await b.task_service.resume(task.task_id)
        final = await finish(b, newer.task_id) if newer.status == TaskStatus.WAITING else newer
        assert final.status == TaskStatus.COMPLETED
        swept = next(e for e in b.spend.entries(task_id=task.task_id) if e.entry_id == inflight.entry_id)
        assert (swept.state, swept.error_type, swept.usage_source) == (SpendState.UNKNOWN, "Abandoned", "estimate")
        assert swept.charged == swept.estimate  # conservative: the whole estimate, labelled as one
        assert swept.final_cost_microusd is None or swept.cost_source == "estimate"
        abandoned = [e for e in b.task_service.events(task.task_id) if e.type == "spend.abandoned"]
        assert [e.data["entry_id"] for e in abandoned] == [inflight.entry_id]
        before = b.spend.totals([f"task:{task.task_id}"])[f"task:{task.task_id}"]["llm_tokens"].settled

        # A's call returns after all: the real usage replaces the estimate; A's own writes are refused.
        gate.open()
        with pytest.raises(LeaseLost):
            await stuck
        late = next(e for e in a.spend.entries(task_id=task.task_id) if e.entry_id == inflight.entry_id)
        assert late.state == SpendState.SETTLED and late.usage_source == "reported"
        after = b.spend.totals([f"task:{task.task_id}"])[f"task:{task.task_id}"]["llm_tokens"].settled
        assert after == before - swept.charged["llm_tokens"] + late.charged["llm_tokens"]
        assert all(e.state != SpendState.RESERVED for e in b.spend.entries(task_id=task.task_id))
    finally:
        a.close()
        b.close()
