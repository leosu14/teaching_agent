"""Interactive teaching tests start from a generated curriculum lesson. Generating it runs the whole lesson workflow,
so it is generated once per test run (the interactive demo's `prepare`) and each test works on its own copy of that
data directory."""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.events import Event
from app.services.container import Container, build_container
from scripts.run_curriculum_demo import FIXTURES, Checks
from scripts.run_interactive_demo import answer_for, prepare, scenario

KEY = scenario()["answer_key"]
__all__ = ["KEY", "Env", "answer_for", "build_lesson", "open_env", "settings_for"]


def settings_for(data_dir: Path, **overrides) -> Settings:
    # The interactive demo's short session: two correct answers raise the difficulty, one miss lowers it.
    return Settings(data_dir=data_dir, corpus_dir=FIXTURES, log_json=False,
                    **{"teaching_increase_after": 2, "teaching_decrease_after": 1, **overrides})


def build_lesson(data_dir: Path) -> str:
    """Generate the curriculum lesson into `data_dir`; returns the lesson task id."""
    container = build_container(settings_for(data_dir), llm_providers={"mock": MockLLMProvider(default_responders())})
    try:
        check = Checks(lambda _line: None)
        task_id = asyncio.run(prepare(container, lambda _line: None, check))
        assert not check.failed, check.failed
    finally:
        container.close()
    return task_id


def _copy(source: Path, target: Path) -> None:
    """Copy a data directory. Stored objects are referenced by absolute file URIs, so they are re-pointed."""
    shutil.copytree(source, target)
    db = sqlite3.connect(target / "teaching_agent.db")
    try:
        old, new = f"file://{source.resolve()}/", f"file://{target.resolve()}/"
        for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall():
            for column in db.execute(f'PRAGMA table_info("{table}")').fetchall():
                if column[2].upper() in ("TEXT", "VARCHAR", "JSON") or column[2].upper().startswith("VARCHAR"):
                    db.execute(f'UPDATE "{table}" SET "{column[1]}" = replace("{column[1]}", ?, ?) '
                               f'WHERE "{column[1]}" LIKE ?', (old, new, f"%{old}%"))
        db.commit()
    finally:
        db.close()


@dataclass
class Env:
    container: Container
    llm: MockLLMProvider
    data_dir: Path
    lesson_task: str
    events: list[Event] = field(default_factory=list)
    opened: list[Container] = field(default_factory=list)  # every container of this test, closed at teardown

    @property
    def service(self):
        return self.container.teaching_service

    def reopen(self, **overrides) -> Env:
        """A process restart: close everything and build a new container on the same data directory."""
        self.container.close()
        return open_env(self.data_dir, self.lesson_task, copy_from=None, opened=self.opened, **overrides)

    def close(self) -> None:
        for container in self.opened:
            container.close()


def open_env(data_dir: Path, lesson_task: str, *, copy_from: Path | None, responders=None,
             opened: list[Container] | None = None, **overrides) -> Env:
    if copy_from is not None:
        _copy(copy_from, data_dir)
    llm = MockLLMProvider(responders or default_responders())
    container = build_container(settings_for(data_dir, **overrides), llm_providers={"mock": llm})
    env = Env(container=container, llm=llm, data_dir=data_dir, lesson_task=lesson_task,
              opened=opened if opened is not None else [])
    env.opened.append(container)
    container.events.subscribe(env.events.append)
    return env
