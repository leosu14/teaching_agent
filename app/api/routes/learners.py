from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import container
from app.schemas.common import Schema
from app.schemas.learner import LearnerProfile, LearnerProfileInput, LearnerProgress
from app.schemas.lesson import DiagnosticAnswers, LearnerAnswer
from app.schemas.task import Task
from app.services.container import Container

router = APIRouter(prefix="/learners", tags=["learners"])


class AssessmentSubmission(Schema):
    task_id: str
    answers: list[LearnerAnswer]


@router.put("/{learner_id}", response_model=LearnerProfile)
def upsert_learner(learner_id: str, body: LearnerProfileInput, c: Container = Depends(container)) -> LearnerProfile:
    return c.learner_service.upsert(learner_id, body)


@router.get("/{learner_id}", response_model=LearnerProfile)
def get_learner(learner_id: str, c: Container = Depends(container)) -> LearnerProfile:
    return c.learner_service.get(learner_id)


@router.get("/{learner_id}/progress", response_model=LearnerProgress)
def learner_progress(learner_id: str, c: Container = Depends(container)) -> LearnerProgress:
    return c.learner_service.progress(learner_id)


@router.post("/{learner_id}/assessment", response_model=Task)
async def submit_assessment(learner_id: str, body: AssessmentSubmission, c: Container = Depends(container)) -> Task:
    return await c.task_service.submit_assessment_for(learner_id, body.task_id, DiagnosticAnswers(answers=body.answers))
