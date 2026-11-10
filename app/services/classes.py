"""Classes and rosters (MVP item 4): a teacher creates a class, learners join it with its join code, and the teacher
may then read (never change) the data of the learners on its roster. The API's authorization boundary asks
`teaches` whenever a teacher reads a learner outside their own gateway scope.

Join codes are random (96 bits), shown once when made, and stored only as a SHA-256 hash. Rotating a code retires the
old one at once; learners already on the roster stay."""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable
from datetime import datetime

from app.learner.memory import LearnerMemoryService
from app.schemas.classroom import ClassJoinCode, ClassMember, Classroom, ClassView, Enrollment
from app.schemas.common import new_id, utcnow
from app.storage.repositories import SqlClassRepository


class ClassNotFound(Exception):
    """No such class for this caller, or no class has this join code: the same 404 either way."""

    def __init__(self) -> None:
        super().__init__("not found")


def new_join_code() -> str:
    return secrets.token_urlsafe(12)  # 16 characters, 96 random bits


def hash_join_code(code: str) -> str:
    return hashlib.sha256(code.strip().encode()).hexdigest()


class ClassService:
    def __init__(self, repository: SqlClassRepository, memory: LearnerMemoryService,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self.repository = repository
        self._memory = memory
        self._clock = clock

    # --- the teacher's side ---------------------------------------------------------------------------------------

    def create(self, *, teacher_id: str, tenant_id: str | None, name: str) -> ClassJoinCode:
        classroom = Classroom(class_id=new_id("cls"), name=name.strip() or name, teacher_id=teacher_id,
                              tenant_id=tenant_id, created_at=self._clock())
        code = new_join_code()
        self.repository.create(classroom, hash_join_code(code))
        return ClassJoinCode(class_id=classroom.class_id, name=classroom.name, created_at=classroom.created_at,
                             join_code=code)

    def owned(self, class_id: str, *, teacher_id: str, tenant_id: str | None) -> Classroom:
        """The class, when this teacher (in this tenant) created it; ClassNotFound otherwise."""
        found = self.repository.get(class_id)
        if found is None or found.teacher_id != teacher_id or found.tenant_id != tenant_id:
            raise ClassNotFound()
        return found

    def for_teacher(self, *, teacher_id: str, tenant_id: str | None) -> list[ClassView]:
        return [self.view(c) for c in self.repository.for_teacher(teacher_id, tenant_id)]

    def view(self, classroom: Classroom) -> ClassView:
        return ClassView(class_id=classroom.class_id, name=classroom.name, created_at=classroom.created_at,
                         learners=self.repository.members(classroom.class_id))

    def rotate_join_code(self, classroom: Classroom) -> ClassJoinCode:
        code = new_join_code()
        if not self.repository.set_join_code(classroom.class_id, hash_join_code(code)):
            raise ClassNotFound()
        view = self.view(classroom)
        return ClassJoinCode(**view.model_dump(), join_code=code)

    def teaches(self, *, teacher_id: str, tenant_id: str | None, learner_id: str) -> bool:
        return self.repository.teaches(teacher_id, tenant_id, learner_id)

    # --- the learner's side ---------------------------------------------------------------------------------------

    def join(self, learner_id: str, join_code: str, *, tenant_id: str | None) -> tuple[Enrollment, bool]:
        """Enroll the learner in the class the code names (idempotent: a second join returns the same enrollment).
        A wrong code, a retired one and a class of another tenant are all ClassNotFound."""
        self._memory.get(learner_id)  # an unknown learner is a 404 (UnknownLearner)
        joined = self.repository.join(hash_join_code(join_code), learner_id, self._clock(), tenant_id)
        if joined is None:
            raise ClassNotFound()
        classroom, member, created = joined
        return self._enrollment(classroom, member), created

    def enrollments(self, learner_id: str, *, tenant_id: str | None) -> list[Enrollment]:
        return [self._enrollment(c, m) for c, m in self.repository.enrollments(learner_id)
                if c.tenant_id == tenant_id]

    def member(self, class_id: str, learner_id: str) -> ClassMember | None:
        return self.repository.member(class_id, learner_id)

    def remove(self, class_id: str, learner_id: str) -> None:
        if not self.repository.remove(class_id, learner_id):
            raise ClassNotFound()

    @staticmethod
    def _enrollment(classroom: Classroom, member: ClassMember) -> Enrollment:
        return Enrollment(class_id=classroom.class_id, name=classroom.name, learner_id=member.learner_id,
                          joined_at=member.joined_at)
