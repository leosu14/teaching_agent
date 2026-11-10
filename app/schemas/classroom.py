"""Classes and rosters: a teacher's class and the learners enrolled in it (MVP item 4).

A class belongs to one teacher (the gateway user id that created it) and, when the deployment has tenants, to the
creator's tenant. Learners join with the class's join code, so a learner (or whoever acts for them) consents to the
teacher seeing their data. Only a hash of the join code is stored: the plain code is shown once, when it is made."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import Schema


class Classroom(Schema):
    """The stored class. `teacher_id` is the creating principal's user id; `tenant_id` is theirs too."""

    class_id: str
    name: str
    teacher_id: str
    tenant_id: str | None = None
    created_at: datetime


class ClassMember(Schema):
    learner_id: str
    joined_at: datetime


class CreateClass(Schema):
    name: str = Field(min_length=1, max_length=200)


class ClassView(Schema):
    """The teacher's view of their class: its roster, never the join code (only its creation or rotation shows it)."""

    class_id: str
    name: str
    created_at: datetime
    learners: list[ClassMember] = Field(default_factory=list)


class ClassJoinCode(Schema):
    """A class with a freshly made join code. The code is not stored, so it is shown here and nowhere else."""

    class_id: str
    name: str
    created_at: datetime
    join_code: str
    learners: list[ClassMember] = Field(default_factory=list)


class JoinClass(Schema):
    join_code: str = Field(min_length=1, max_length=64)


class Enrollment(Schema):
    """A learner's view of a class they are in: the class, not its other learners or its teacher."""

    class_id: str
    name: str
    learner_id: str
    joined_at: datetime
