"""Lesson approval before release (Important item 2).

A class's teacher may require approving lessons before its learners see them (`classes.lesson_approval`). The
decision is taken when a lesson is requested: `request` (called by the TaskService when it creates a lesson task,
inside the request's learner-writes guard) stores a PENDING release for the first class of the learner that requires
approval. No release row means released, so nothing changes for learners in no such class or for lessons requested
before the setting was turned on.

A lesson is *held* while its release is PENDING or REJECTED and its learner is still on that class's roster: a
learner who leaves, or is removed, is not held by that class any more. Every learner surface of a held lesson asks
`check` (or `held`) and refuses (`LessonHeld`, 409): the reading, a teaching session, an evaluation, an attempt on its
items, its artifacts' content; a learning cycle observes it and waits BLOCKED (PENDING) or fails (REJECTED).

The teacher decides through the class routes (`taught_class`): a decision is final, a compare-and-set on PENDING."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from app.artifacts.service import ArtifactService
from app.learner.memory import LearnerMemoryService, UnknownLearner
from app.schemas.classroom import Classroom
from app.schemas.common import utcnow
from app.schemas.events import EventType
from app.schemas.lease import learner_writes
from app.schemas.review import (
    HELD_MESSAGE,
    REJECTED_MESSAGE,
    LessonHeld,
    LessonRelease,
    LessonReleaseView,
    ReleaseConflict,
    ReleaseStatus,
)
from app.schemas.task import Task, TaskStatus
from app.storage.repositories import NotFound, SqlClassRepository, SqlLessonReleaseRepository, SqlTaskRepository


class ReleaseNotFound(Exception):
    """No release of this lesson for this class, or its learner left the class: the uniform 404."""

    def __init__(self) -> None:
        super().__init__("not found")


class LessonReleaseService:
    def __init__(self, repository: SqlLessonReleaseRepository, classes: SqlClassRepository, tasks: SqlTaskRepository,
                 artifacts: ArtifactService, memory: LearnerMemoryService, events=None,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self.repository = repository
        self._classes = classes
        self._tasks = tasks
        self._artifacts = artifacts
        self._memory = memory
        self._events = events
        self._clock = clock

    # --- requesting and the gate ---------------------------------------------------------------------------------

    def request(self, task: Task) -> LessonRelease | None:
        """A PENDING release for a newly created lesson task when a class of its learner requires approval. Called
        inside the creating request's learner-writes guard; idempotent (a task keeps its first release)."""
        existing = self.repository.get(task.task_id)
        if existing is not None:
            return existing
        class_id = self._classes.approving_class(task.learner_id)
        if class_id is None:
            return None
        release = self.repository.request(LessonRelease(
            task_id=task.task_id, learner_id=task.learner_id, class_id=class_id, status=ReleaseStatus.PENDING,
            requested_at=self._clock()))
        self._emit(EventType.LESSON_RELEASE_REQUESTED, release)
        return release

    def held(self, task_id: str) -> ReleaseStatus | None:
        """PENDING or REJECTED while the lesson is held from its learner; None when it is released."""
        release = self.repository.get(task_id)
        if release is None or release.status == ReleaseStatus.APPROVED:
            return None
        if self._classes.member(release.class_id, release.learner_id) is None:
            return None  # the learner left the class (or was removed): its teacher no longer holds their lessons
        return release.status

    def status(self, task_id: str) -> ReleaseStatus | None:
        """What the task view shows: APPROVED, or PENDING / REJECTED while held; None without a release (or when its
        learner left the class)."""
        release = self.repository.get(task_id)
        if release is not None and release.status == ReleaseStatus.APPROVED:
            return release.status
        return self.held(task_id)

    def check(self, task_id: str) -> None:
        status = self.held(task_id)
        if status is not None:
            raise LessonHeld(REJECTED_MESSAGE if status == ReleaseStatus.REJECTED else HELD_MESSAGE)

    def check_artifact(self, artifact_id: str) -> None:
        """`check` for the lesson task that stored the artifact (an unknown artifact is left to the caller)."""
        try:
            artifact = self._artifacts.get(artifact_id)
        except (NotFound, KeyError):
            return
        self.check(artifact.task_id)

    def rejected(self, task_id: str) -> bool:
        return self.held(task_id) == ReleaseStatus.REJECTED

    # --- the teacher's side --------------------------------------------------------------------------------------

    def for_class(self, classroom: Classroom, status: ReleaseStatus | None = None) -> list[LessonReleaseView]:
        """The class's releases of learners still on its roster, newest first."""
        members = {m.learner_id for m in self._classes.members(classroom.class_id)}
        names: dict[str, str] = {}
        out = []
        for release in self.repository.for_class(classroom.class_id, status):
            if release.learner_id not in members:
                continue
            view = self._view(release, names)
            if view is not None:
                out.append(view)
        return out

    def release_for(self, classroom: Classroom, task_id: str) -> LessonRelease:
        """The release of this lesson for this class, its learner still on the roster; ReleaseNotFound otherwise."""
        release = self.repository.get(task_id)
        if release is None or release.class_id != classroom.class_id \
                or self._classes.member(classroom.class_id, release.learner_id) is None:
            raise ReleaseNotFound()
        return release

    def view(self, classroom: Classroom, task_id: str) -> LessonReleaseView:
        view = self._view(self.release_for(classroom, task_id), {})
        if view is None:
            raise ReleaseNotFound()
        return view

    def decide(self, classroom: Classroom, task_id: str, status: ReleaseStatus, *, teacher_id: str,
               note: str | None) -> LessonReleaseView:
        """Approve or reject a completed lesson. Final: the same decision again returns the release as it is; the
        other decision, or a lesson that has not completed, is ReleaseConflict."""
        assert status in (ReleaseStatus.APPROVED, ReleaseStatus.REJECTED)
        release = self.release_for(classroom, task_id)
        if release.status == ReleaseStatus.PENDING:
            task = self._task(task_id)
            if task.status != TaskStatus.COMPLETED:
                raise ReleaseConflict(f"the lesson is {task.status.value}: it can be decided on once it is completed")
            note = (note or "").strip() or None
            with learner_writes(release.learner_id, task_id):  # refused once the learner's data is deleted
                won = self.repository.decide(task_id, status, at=self._clock(), by=teacher_id, note=note)
            release = self.release_for(classroom, task_id)
            if won:
                self._emit(EventType.LESSON_RELEASE_APPROVED if status == ReleaseStatus.APPROVED
                           else EventType.LESSON_RELEASE_REJECTED, release)
        if release.status != status:
            raise ReleaseConflict(f"the lesson was already {release.status.value.lower()}")
        return self.view(classroom, task_id)

    # --- parts ---------------------------------------------------------------------------------------------------

    def _task(self, task_id: str) -> Task:
        try:
            return self._tasks.get(task_id)
        except NotFound:
            raise ReleaseNotFound() from None

    def _view(self, release: LessonRelease, names: dict[str, str]) -> LessonReleaseView | None:
        try:
            task = self._tasks.get(release.task_id)
        except NotFound:
            return None
        if release.learner_id not in names:
            try:
                names[release.learner_id] = self._memory.get(release.learner_id).display_name or release.learner_id
            except UnknownLearner:
                names[release.learner_id] = release.learner_id
        lesson = self._artifacts.find(task.task_id, "lesson") if task.status == TaskStatus.COMPLETED else None
        request = task.plan.lesson_request if task.plan is not None else None
        return LessonReleaseView(
            task_id=task.task_id, lesson_id=lesson.artifact_id if lesson is not None else None,
            learner_id=release.learner_id, display_name=names[release.learner_id],
            topic=request.topic if request is not None else task.request[:200],
            subject=request.subject if request is not None else "", task_status=task.status.value,
            status=release.status, requested_at=release.requested_at, decided_at=release.decided_at,
            note=release.note)

    def _emit(self, type: str, release: LessonRelease) -> None:
        """Ids and the status only: never the teacher's note."""
        if self._events is not None:
            self._events.emit(type, task_id=release.task_id, learner_id=release.learner_id,
                              class_id=release.class_id, status=release.status.value)
