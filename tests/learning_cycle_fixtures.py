"""Learning-cycle tests start from the learning-cycle demo's learner. Generating a lesson runs the whole lesson
workflow, so the cycle's stages are built once per test run and each test works on its own copy of a stage's data
directory:

- `base`: the learner and the active goal, no curriculum, no cycle;
- `diagnostic`: cycle `c1` started (LEARN es.preterite): the curriculum is built, the lesson waits on its diagnostic;
- `session`: the diagnostic answered: the lesson completed, the session waits on its first question;
- `done`: the session answered to completion: the cycle is COMPLETED with its outcome.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.events import Event
from app.schemas.learning_cycle import CycleStatus, LearningCycleView, StartLearningCycle
from app.services.container import Container, build_container
from scripts.run_learning_cycle_demo import ScriptedLearner, prepare, settings
from tests.teaching_fixtures import _copy

LEARNER = "curriculum-learner"
CONCEPT = "es.preterite"
STAGES = ("base", "diagnostic", "session", "done")
__all__ = ["CONCEPT", "CycleEnv", "LEARNER", "ScriptedLearner", "build_stages", "open_cycle_env"]


@dataclass
class CycleEnv:
    container: Container
    llm: MockLLMProvider
    data_dir: Path
    events: list[Event] = field(default_factory=list)
    opened: list[Container] = field(default_factory=list)

    @property
    def cycles(self):
        return self.container.learning_cycle_service

    def reopen(self, **overrides) -> CycleEnv:
        """A process restart: a new container on the same data directory."""
        self.container.close()
        return open_cycle_env(self.data_dir, copy_from=None, opened=self.opened, **overrides)

    def close(self) -> None:
        for c in self.opened:
            c.close()


def open_cycle_env(data_dir: Path, *, copy_from: Path | None, opened: list[Container] | None = None,
                   **overrides) -> CycleEnv:
    if copy_from is not None:
        _copy(copy_from, data_dir)
    llm = MockLLMProvider(default_responders())
    container = build_container(settings(data_dir, log_json=False, **overrides), llm_providers={"mock": llm})
    env = CycleEnv(container=container, llm=llm, data_dir=data_dir, opened=opened if opened is not None else [])
    env.opened.append(container)
    container.events.subscribe(env.events.append)
    return env


async def answer(env: CycleEnv, view: LearningCycleView, learner: ScriptedLearner, *,
                 until: str | None = None, limit: int = 40) -> LearningCycleView:
    """Answer prompts until the cycle stops waiting (or the prompt kind changes to `until`)."""
    for _ in range(limit):
        if view.status != CycleStatus.WAITING or view.prompt is None or view.prompt.kind == until:
            return view
        view = await env.cycles.respond(view.cycle_id, learner.respond(view.prompt))
    raise AssertionError("the cycle did not finish")


def build_stages(root: Path) -> dict[str, Path]:
    """Build every stage once; returns stage -> data directory to copy."""
    dirs = {name: root / name for name in STAGES}

    async def run() -> None:
        env = open_cycle_env(dirs["base"], copy_from=None)
        try:
            await prepare(env.container)
        finally:
            env.close()
        _copy(dirs["base"], dirs["diagnostic"])
        env = open_cycle_env(dirs["diagnostic"], copy_from=None)
        try:
            view = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))
            assert view.prompt is not None and view.prompt.kind == "DIAGNOSTIC_QUESTIONS", view
        finally:
            env.close()
        _copy(dirs["diagnostic"], dirs["session"])
        env = open_cycle_env(dirs["session"], copy_from=None)
        try:
            view = await env.cycles.get(view.cycle_id)
            view = await answer(env, view, ScriptedLearner(prefix="d"), until="SESSION_QUESTION")
            assert view.prompt is not None and view.prompt.kind == "SESSION_QUESTION", view
        finally:
            env.close()
        _copy(dirs["session"], dirs["done"])
        env = open_cycle_env(dirs["done"], copy_from=None)
        try:
            view = await answer(env, await env.cycles.get(view.cycle_id), ScriptedLearner(prefix="s"))
            assert view.status == CycleStatus.COMPLETED, view
        finally:
            env.close()

    asyncio.run(run())
    return dirs
