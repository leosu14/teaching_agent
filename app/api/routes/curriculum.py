"""Learning goals, curricula and the next learning action."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Response
from pydantic import Field

from app.api.deps import container
from app.schemas.common import Schema
from app.schemas.curriculum import (
    Curriculum,
    CurriculumVersion,
    CurriculumWarning,
    GoalInput,
    GoalUpdate,
    NextLearningAction,
)
from app.schemas.learner import LearningGoal
from app.schemas.task import TaskError, TaskStatus
from app.services.container import Container
from app.storage.repositories import NotFound

router = APIRouter(tags=["curriculum"])


class GoalChange(Schema):
    goal: LearningGoal
    curriculum: Curriculum | None = None  # the replanned curriculum, when the change replanned it


class BuildCurriculum(Schema):
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)


class CurriculumBuild(Schema):
    task_id: str
    status: TaskStatus
    new_version: bool = False  # False: the curriculum was already up to date (nothing new stored)
    curriculum: Curriculum | None = None
    warnings: list[CurriculumWarning] = Field(default_factory=list)
    errors: list[TaskError] = Field(default_factory=list)


@router.post("/learners/{learner_id}/goals", response_model=LearningGoal, status_code=201)
async def create_goal(learner_id: str, body: GoalInput, response: Response,
                      c: Container = Depends(container)) -> LearningGoal:
    goal, created = await c.curriculum_service.create_goal(learner_id, body)
    if not created:
        response.status_code = 200
    return goal


@router.get("/learners/{learner_id}/goals", response_model=list[LearningGoal])
def list_goals(learner_id: str, c: Container = Depends(container)) -> list[LearningGoal]:
    return c.curriculum_service.goals(learner_id)


@router.get("/learners/{learner_id}/next-action", response_model=NextLearningAction)
async def next_action(learner_id: str, as_of: datetime | None = None,
                      c: Container = Depends(container)) -> NextLearningAction:
    return await c.curriculum_service.next_action(learner_id, as_of=as_of)


@router.get("/goals/{goal_id}", response_model=LearningGoal)
def get_goal(goal_id: str, c: Container = Depends(container)) -> LearningGoal:
    return c.curriculum_service.goal(goal_id)


@router.patch("/goals/{goal_id}", response_model=GoalChange)
async def update_goal(goal_id: str, body: GoalUpdate, c: Container = Depends(container)) -> GoalChange:
    goal, task = await c.curriculum_service.update_goal(goal_id, body, user_id="anonymous")
    return GoalChange(goal=goal, curriculum=c.curriculum_service.curriculum(goal_id) if task is not None else None)


@router.post("/goals/{goal_id}/curriculum", response_model=CurriculumBuild, status_code=201)
async def build_curriculum(goal_id: str, response: Response, body: BuildCurriculum | None = None,
                           c: Container = Depends(container)) -> CurriculumBuild:
    """Plan the goal's curriculum (or replan it: a new version only when the path changed)."""
    task = await c.curriculum_service.build_curriculum(goal_id, user_id=(body or BuildCurriculum()).user_id)
    result = task.result
    plan = result.curriculum if result is not None else None
    if task.status == TaskStatus.FAILED:
        response.status_code = 422
    elif plan is not None and not plan.changed:
        response.status_code = 200
    return CurriculumBuild(task_id=task.task_id, status=task.status, new_version=bool(plan and plan.changed),
                           curriculum=c.curriculum_service.curriculum(goal_id),
                           warnings=plan.warnings if plan is not None else [], errors=task.errors)


@router.get("/goals/{goal_id}/curriculum", response_model=Curriculum)
def get_curriculum(goal_id: str, c: Container = Depends(container)) -> Curriculum:
    found = c.curriculum_service.curriculum(goal_id)
    if found is None:
        raise NotFound(f"goal {goal_id} has no curriculum yet")
    return found


@router.get("/goals/{goal_id}/curriculum/versions", response_model=list[CurriculumVersion])
def curriculum_versions(goal_id: str, c: Container = Depends(container)) -> list[CurriculumVersion]:
    return c.curriculum_service.versions(goal_id)
