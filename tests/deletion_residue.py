"""Find any trace of a learner left in the database or the object store after their data was deleted (MVP item 9).

`learner_ids` collects, before the deletion, every id that leads back to the learner, following ids through every
erased table until nothing new turns up (the learner's tasks, their artifacts, sessions, goals, curricula, items,
attempts, grades...). `db_residue` then scans every string column of every erased table for any of them, so a row the
deletion does not know about is found too, not only the ones it was written to delete. The spend ledger and the
teachers' and authors' tables are kept on purpose (KEPT_TABLES) and not scanned."""

from __future__ import annotations

from sqlalchemy import String, Text, select

from app.storage.orm import Base
from app.storage.repositories import KEPT_TABLES

# The ids that name one record of the learner's, and so lead to more of them. Not followed: ids shared with other
# learners or naming someone else (concept, node and agent ids, the class they joined, an author's rubric, users,
# tenants, client request ids), which on a shared database would lead into everyone's data.
_FOLLOWED = {"artifact_id", "attempt_id", "curriculum_id", "cycle_id", "event_id", "evidence_id", "goal_id",
             "grade_id", "item_id", "job_id", "learner_id", "lesson_id", "session_id", "task_id", "turn_id",
             "version_id"}


def _columns(table):
    return [col for col in table.columns if isinstance(col.type, (String, Text))]


def learner_ids(sessions, learner_id: str) -> set[str]:
    found = {learner_id}
    with sessions() as s:
        changed = True
        while changed:
            changed = False
            for table in Base.metadata.sorted_tables:
                if table.name in KEPT_TABLES:
                    continue
                columns = _columns(table)
                id_columns = [col for col in columns if col.name in _FOLLOWED]
                for row in s.execute(select(*columns)):
                    values = [str(v) for v in row._mapping.values() if v]
                    if not any(i == v or f'"{i}"' in v for v in values for i in found):
                        continue
                    for col in id_columns:
                        value = row._mapping[col.name]
                        if value and value not in found and len(value) > 4:
                            found.add(value)
                            changed = True
    return found


def db_residue(sessions, ids: set[str]) -> list[tuple[str, str, str]]:
    """(table, column, id) for every value in an erased table that holds one of `ids`."""
    left = []
    with sessions() as s:
        for table in Base.metadata.sorted_tables:
            if table.name in KEPT_TABLES:
                continue
            for row in s.execute(select(*_columns(table))):
                for column, value in row._mapping.items():
                    hit = next((i for i in ids if value and i in str(value)), None)
                    if hit is not None:
                        left.append((table.name, column, hit))
    return left


def stored_files(c) -> dict[str, bytes]:
    """Every object in a filesystem object store, by key."""
    root = c.artifacts._store._root
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
