"""The learning sweeper (issue #66 PR 3): bounded, round-robin batches; `full` walks everything; one id's failure is
counted and logged and never stops a pass; ids deleted meanwhile are skipped; the pass logs one summary line."""

from __future__ import annotations

import logging
import math

import pytest

from app.curriculum.cycle import CycleNotFound
from app.schemas.lease import LearnerDataGone
from app.services.sweeper import LearningSweeper
from app.storage.repositories import NotFound
from app.teaching.errors import TeachingSessionNotFound


def ids_of(universe: list[str]):
    def ids(after: str | None, limit: int) -> list[str]:
        return [i for i in sorted(universe) if after is None or i > after][:limit]
    return ids


def sweeper(learners: list[str], *, batch: int, refresh=None, cycles=(), outboxes=()):
    seen: dict[str, list[str]] = {"curricula": [], "cycles": [], "outboxes": []}

    def recorder(kind, act=None):
        async def run(i: str) -> None:
            seen[kind].append(i)
            if act is not None:
                await act(i)
        return run

    s = LearningSweeper.of(learners=ids_of(learners), refresh=recorder("curricula", refresh),
                           cycles=ids_of(list(cycles)), reconcile=recorder("cycles"),
                           outboxes=ids_of(list(outboxes)), publish=recorder("outboxes"), batch=batch)
    return s, seen


async def test_each_pass_is_bounded_and_every_learner_is_reached_within_ceil_n_over_batch_passes() -> None:
    learners = [f"l{i:02d}" for i in range(23)]
    s, seen = sweeper(learners, batch=5)
    passes = math.ceil(len(learners) / 5)
    for n in range(passes):
        before = len(seen["curricula"])
        report = await s.sweep()
        assert len(seen["curricula"]) - before <= 5  # bounded
        assert report.wrapped["curricula"] == (n == passes - 1)
    assert sorted(seen["curricula"]) == learners  # each one exactly once in ceil(N / batch) passes
    await s.sweep()  # and round robin: the next pass starts over
    assert seen["curricula"][-5:] == learners[:5]


async def test_an_exact_multiple_wraps_with_an_empty_batch() -> None:
    s, seen = sweeper(["a", "b", "c", "d"], batch=2)
    for _ in range(3):
        await s.sweep()
    assert seen["curricula"] == ["a", "b", "c", "d"]  # the third pass found nothing after "d" and wrapped
    await s.sweep()
    assert seen["curricula"][-2:] == ["a", "b"]


async def test_full_walks_every_id_batch_by_batch_from_the_first() -> None:
    learners = [f"l{i}" for i in range(7)]
    s, seen = sweeper(learners, batch=3, cycles=["c1", "c2"], outboxes=["s1"])
    await s.sweep()  # leaves a cursor in the middle
    for kind in seen.values():
        kind.clear()
    report = await s.sweep(full=True)
    assert seen["curricula"] == learners and report.wrapped["curricula"]
    assert report.done == {"curricula": 7, "cycles": 2, "outboxes": 1}
    assert seen["cycles"] == ["c1", "c2"] and seen["outboxes"] == ["s1"]


@pytest.mark.parametrize("gone", [NotFound("x"), LearnerDataGone(), CycleNotFound("x"),
                                  TeachingSessionNotFound("x")])
async def test_an_id_deleted_meanwhile_is_skipped_without_an_error(gone) -> None:
    async def refresh(i: str) -> None:
        if i == "b":
            raise gone

    s, seen = sweeper(["a", "b", "c"], batch=10, refresh=refresh)
    report = await s.sweep()
    assert seen["curricula"] == ["a", "b", "c"]
    assert report.errors == 0 and report.done["curricula"] == 2


async def test_a_failure_is_counted_logged_without_the_learner_and_does_not_stop_the_pass(caplog) -> None:
    async def refresh(i: str) -> None:
        if i == "secret-learner":
            raise RuntimeError("database down")

    s, seen = sweeper(["a", "secret-learner", "z"], batch=10, refresh=refresh, cycles=["c1"])
    with caplog.at_level(logging.INFO, logger="app.services.sweeper"):
        report = await s.sweep()
    assert seen["curricula"] == ["a", "secret-learner", "z"] and seen["cycles"] == ["c1"]
    assert report.errors == 1 and report.failed == {"curricula": 1, "cycles": 0, "outboxes": 0}
    assert "secret-learner" not in caplog.text  # a log line never names a learner
    [summary] = [r for r in caplog.records if "learning pass" in r.getMessage()]
    assert summary.levelno == logging.WARNING
    assert "curricula: 2 done, 1 failed" in summary.getMessage()


async def test_a_clean_pass_logs_its_summary_at_info(caplog) -> None:
    s, _ = sweeper(["a"], batch=10)
    with caplog.at_level(logging.INFO, logger="app.services.sweeper"):
        report = await s.sweep()
    [summary] = [r for r in caplog.records if "learning pass" in r.getMessage()]
    assert summary.levelno == logging.INFO and report.summary() in summary.getMessage()


def test_the_batch_is_at_least_one() -> None:
    with pytest.raises(ValueError):
        LearningSweeper([], batch=0)
