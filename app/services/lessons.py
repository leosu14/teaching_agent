"""Lesson material: a generated lesson with everything that was produced with it. Shared by the services that work on
a finished lesson (interactive teaching, assessment)."""

from __future__ import annotations

from dataclasses import dataclass

from app.artifacts.service import ArtifactService
from app.runtime.workflows.lesson_generation import WORKFLOW_ID as LESSON_WORKFLOW
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.lesson import LessonContent, LessonRequest
from app.schemas.pedagogy import LessonFocus, PedagogicalPlan
from app.schemas.research import ResearchBundle
from app.schemas.task import Task, TaskStatus
from app.services.tasks import TaskService
from app.storage.repositories import NotFound


class LessonNotReady(ValueError):
    """The lesson task exists but has not completed (409 where a caller maps it)."""


@dataclass(frozen=True)
class LessonMaterial:
    task: Task
    artifact: Artifact
    lesson: LessonContent
    request: LessonRequest
    research: ResearchBundle | None
    focus: LessonFocus | None


def load_lesson(artifacts: ArtifactService, tasks: TaskService, lesson_id: str) -> LessonMaterial:
    """A LESSON artifact id, or the id of the completed lesson task that produced it."""
    try:
        artifact = artifacts.get(lesson_id)
    except (NotFound, KeyError):
        artifact = None
    if artifact is not None:
        if artifact.type != ArtifactType.LESSON:
            raise NotFound(f"artifact {lesson_id} is not a lesson")
        task = tasks.get(artifact.task_id)
    else:
        try:
            task = tasks.get(lesson_id)
        except NotFound:
            raise NotFound(f"lesson {lesson_id} not found") from None
        if task.plan is None or task.plan.workflow_id != LESSON_WORKFLOW:
            raise NotFound(f"task {lesson_id} is not a lesson")
        if task.status != TaskStatus.COMPLETED:
            raise LessonNotReady(f"lesson task {lesson_id} is {task.status.value}, not completed")
        artifact = artifacts.find(task.task_id, "lesson")
        if artifact is None:
            raise NotFound(f"lesson task {lesson_id} has no lesson")
    assert task.plan is not None
    research_artifact = artifacts.find(task.task_id, "research_bundle")
    plan_artifact = artifacts.find(task.task_id, "pedagogical_plan")
    plan = (PedagogicalPlan.model_validate_json(artifacts.read(plan_artifact.artifact_id))
            if plan_artifact is not None else None)
    return LessonMaterial(
        task=task, artifact=artifact,
        lesson=LessonContent.model_validate_json(artifacts.read(artifact.artifact_id)),
        request=task.plan.lesson_request,
        research=(ResearchBundle.model_validate_json(artifacts.read(research_artifact.artifact_id))
                  if research_artifact is not None else None),
        focus=plan.focus if plan is not None else None)
