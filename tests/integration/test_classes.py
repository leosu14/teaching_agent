"""MVP item 4: classes and rosters through ClassService and the SQL repository (SQLite)."""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import text

from app.learner.memory import UnknownLearner
from app.schemas.learner import LearnerProfileInput
from app.services.classes import ClassNotFound


@pytest.fixture
def classes(container):
    for learner in ("a", "b", "c"):
        container.learner_service.upsert(learner, LearnerProfileInput(display_name=learner.upper()))
    return container.classes


def test_a_learner_joins_with_the_code_and_the_teacher_then_teaches_them(classes) -> None:
    made = classes.create(teacher_id="t1", tenant_id=None, name="  Spanish B1 ")
    assert made.name == "Spanish B1"
    assert not classes.teaches(teacher_id="t1", tenant_id=None, learner_id="a")
    enrollment, created = classes.join("a", made.join_code, tenant_id=None)
    assert created and (enrollment.class_id, enrollment.learner_id) == (made.class_id, "a")
    assert classes.teaches(teacher_id="t1", tenant_id=None, learner_id="a")
    assert not classes.teaches(teacher_id="t2", tenant_id=None, learner_id="a")
    assert not classes.teaches(teacher_id="t1", tenant_id=None, learner_id="b")
    again, created = classes.join("a", made.join_code, tenant_id=None)
    assert not created and again == enrollment  # idempotent: the first join's time is kept
    [view] = classes.for_teacher(teacher_id="t1", tenant_id=None)
    assert [m.learner_id for m in view.learners] == ["a"]


def test_wrong_retired_and_foreign_tenant_codes_are_not_found(classes) -> None:
    made = classes.create(teacher_id="t1", tenant_id=None, name="Mine")
    with pytest.raises(ClassNotFound):
        classes.join("a", "not-a-code", tenant_id=None)
    rotated = classes.rotate_join_code(classes.owned(made.class_id, teacher_id="t1", tenant_id=None))
    with pytest.raises(ClassNotFound):
        classes.join("a", made.join_code, tenant_id=None)
    classes.join("a", rotated.join_code, tenant_id=None)
    elsewhere = classes.create(teacher_id="t1", tenant_id="school-b", name="Theirs")
    with pytest.raises(ClassNotFound):
        classes.join("b", elsewhere.join_code, tenant_id=None)
    with pytest.raises(ClassNotFound):
        classes.join("b", elsewhere.join_code, tenant_id="school-c")
    classes.join("b", elsewhere.join_code, tenant_id="school-b")
    assert classes.teaches(teacher_id="t1", tenant_id="school-b", learner_id="b")
    assert not classes.teaches(teacher_id="t1", tenant_id=None, learner_id="b")
    assert classes.enrollments("b", tenant_id=None) == []


def test_an_unknown_learner_cannot_join(classes) -> None:
    made = classes.create(teacher_id="t1", tenant_id=None, name="Mine")
    with pytest.raises(UnknownLearner):
        classes.join("nobody", made.join_code, tenant_id=None)


def test_only_the_creating_teacher_in_their_tenant_owns_a_class(classes) -> None:
    made = classes.create(teacher_id="t1", tenant_id="school-a", name="Mine")
    assert classes.owned(made.class_id, teacher_id="t1", tenant_id="school-a").class_id == made.class_id
    for teacher, tenant in (("t2", "school-a"), ("t1", None), ("t1", "school-b")):
        with pytest.raises(ClassNotFound):
            classes.owned(made.class_id, teacher_id=teacher, tenant_id=tenant)
    with pytest.raises(ClassNotFound):
        classes.owned("cls_missing", teacher_id="t1", tenant_id="school-a")


def test_removal_ends_teaching_and_a_learner_in_two_classes_stays_taught_by_both(classes) -> None:
    first = classes.create(teacher_id="t1", tenant_id=None, name="First")
    second = classes.create(teacher_id="t1", tenant_id=None, name="Second")
    other = classes.create(teacher_id="t2", tenant_id=None, name="Other")
    for made in (first, second, other):
        classes.join("a", made.join_code, tenant_id=None)
    assert {e.name for e in classes.enrollments("a", tenant_id=None)} == {"First", "Second", "Other"}
    classes.remove(first.class_id, "a")
    assert classes.teaches(teacher_id="t1", tenant_id=None, learner_id="a")  # still on Second
    classes.remove(second.class_id, "a")
    assert not classes.teaches(teacher_id="t1", tenant_id=None, learner_id="a")
    assert classes.teaches(teacher_id="t2", tenant_id=None, learner_id="a")
    with pytest.raises(ClassNotFound):
        classes.remove(second.class_id, "a")


def test_join_codes_are_stored_only_as_hashes(container, classes) -> None:
    made = classes.create(teacher_id="t1", tenant_id=None, name="Mine")
    rotated = classes.rotate_join_code(classes.owned(made.class_id, teacher_id="t1", tenant_id=None))
    with container._sessions() as s:  # noqa: SLF001 - a test reads the raw table
        stored = " ".join(str(v) for row in s.execute(text("SELECT * FROM classes")) for v in row)
    assert made.join_code not in stored and rotated.join_code not in stored


def test_concurrent_joins_of_one_learner_make_one_membership(classes) -> None:
    made = classes.create(teacher_id="t1", tenant_id=None, name="Mine")
    results, errors = [], []
    start = threading.Barrier(8)

    def join() -> None:
        try:
            start.wait()
            results.append(classes.join("c", made.join_code, tenant_id=None))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=join) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sum(created for _, created in results) == 1
    assert len({e.joined_at for e, _ in results}) == 1
    assert [m.learner_id for m in classes.repository.members(made.class_id)] == ["c"]


def test_class_tables_are_added_to_an_existing_sqlite_database(tmp_path) -> None:
    """A database made before this change gains the two tables on start (create_all makes missing tables)."""
    from sqlalchemy import create_engine, inspect

    from app.storage.db import create_db, dispose
    from app.storage.orm import Base

    url = f"sqlite:///{tmp_path / 'old.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[t for name, t in Base.metadata.tables.items()
                                             if name not in ("classes", "class_members")])
    assert not {"classes", "class_members"} & set(inspect(engine).get_table_names())
    engine.dispose()
    factory = create_db(url)
    try:
        assert {"classes", "class_members"} <= set(inspect(factory.kw["bind"]).get_table_names())
    finally:
        dispose(factory)
