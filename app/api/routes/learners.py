from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from app.api.authorization import Access, access, require_learner
from app.api.deps import container
from app.schemas.common import Schema
from app.schemas.learner import LearnerProfileInput, LearnerProfileView, LearnerProgress
from app.schemas.privacy import LearnerDataDeletion
from app.schemas.lesson import DiagnosticAnswers, LearnerAnswer
from app.schemas.task import TaskView
from app.services.container import Container

router = APIRouter(prefix="/learners", tags=["learners"])


class AssessmentSubmission(Schema):
    task_id: str
    answers: list[LearnerAnswer]


@router.put("/{learner_id}", response_model=LearnerProfileView, dependencies=[Depends(require_learner)])
def upsert_learner(learner_id: str, body: LearnerProfileInput,
                   c: Container = Depends(container)) -> LearnerProfileView:
    return LearnerProfileView.of(c.learner_service.upsert(learner_id, body))


@router.get("/{learner_id}", response_model=LearnerProfileView, dependencies=[Depends(require_learner)])
def get_learner(learner_id: str, c: Container = Depends(container)) -> LearnerProfileView:
    return LearnerProfileView.of(c.learner_service.get(learner_id))


@router.delete("/{learner_id}", response_model=LearnerDataDeletion, dependencies=[Depends(require_learner)])
def delete_learner(learner_id: str, c: Container = Depends(container)) -> LearnerDataDeletion:
    """Delete everything stored about the learner (MVP item 9): profile, mastery and evidence, goals and curricula,
    lessons and their files, sessions, evaluations, answers and grades, learning cycles, events and class
    memberships. Only the learner's own scope may do this: a teacher of their class gets 403 (they may read, never
    change), anyone else the uniform 404. The spend ledger (ids and amounts only) is kept so the spending caps still
    hold. 409 while the learner has a lesson queued or running, or a learning cycle being driven: nothing is deleted
    then. Deleting a learner that has no data left returns an empty receipt, so a retry is safe."""
    return c.learner_deletion.delete(learner_id)


@router.get("/{learner_id}/progress", response_model=LearnerProgress, dependencies=[Depends(require_learner)])
def learner_progress(learner_id: str, c: Container = Depends(container)) -> LearnerProgress:
    return c.learner_service.progress(learner_id)


@router.post("/{learner_id}/assessment", response_model=TaskView, status_code=202,
             dependencies=[Depends(require_learner)])
def submit_assessment(learner_id: str, body: AssessmentSubmission, response: Response,
                      a: Access = Depends(access), c: Container = Depends(container)) -> TaskView:
    """The diagnostic answers of a WAITING lesson task: validated, then queued like POST /tasks/{id}/answers."""
    a.check(c.ownership.task(body.task_id))
    task, job = c.task_service.enqueue_assessment_for(learner_id, body.task_id,
                                                      DiagnosticAnswers(answers=body.answers))
    response.headers["Location"] = f"/tasks/{task.task_id}"
    return TaskView.of(task, job)
