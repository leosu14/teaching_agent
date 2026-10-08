"""Assessment items and graded attempts: exact matching first, rubric and semantic grading for free text."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from app.api.auth import AUTHOR, Principal
from app.api.authorization import authorize, owned_attempt, owned_item, principal, require_role
from app.api.deps import container
from app.schemas.assessment import (
    AssessmentItem,
    AttemptResult,
    AttemptSubmission,
    AttemptView,
    PublicGrade,
    RegisterAssessmentItem,
)
from app.services.container import Container

router = APIRouter(tags=["assessment"])


@router.post("/assessment-items", response_model=AssessmentItem, status_code=201)
async def register_item(body: RegisterAssessmentItem, response: Response, p: Principal = Depends(principal),
                        c: Container = Depends(container)) -> AssessmentItem:
    """An item on a lesson (and its rubric), for its author: only principals with the author role may register one,
    on a lesson in their learner scope. The answer key is never served to learners (attempts and grades do not carry
    it). Registering the identical item again returns it; different content under the same id is a conflict."""
    require_role(p, AUTHOR)
    authorize(p, c.ownership.lesson(body.item.lesson_id) if body.item.lesson_id else None)
    taken = c.ownership.assessment_item(body.item.assessment_item_id)
    if taken is not None:
        authorize(p, taken)  # an id taken on another learner's lesson: not found, like any foreign resource
    service = c.assessment_service
    if service.repository.item(body.item.assessment_item_id) is not None:
        response.status_code = 200
    return service.register(body)


@router.get("/assessment-items/{item_id}/attempts", response_model=list[AttemptView],
            dependencies=[Depends(owned_item)])
async def list_attempts(item_id: str, learner_id: str, p: Principal = Depends(principal),
                        c: Container = Depends(container)) -> list[AttemptView]:
    """Every attempt of the learner on the item, in order (none is ever overwritten)."""
    authorize(p, learner_id)
    return c.assessment_service.attempts(item_id, learner_id)


@router.post("/assessment-items/{item_id}/attempts", response_model=AttemptResult, status_code=201,
             dependencies=[Depends(owned_item)])
async def submit_attempt(item_id: str, body: AttemptSubmission, response: Response,
                         p: Principal = Depends(principal), c: Container = Depends(container)) -> AttemptResult:
    """Grade an answer. Idempotent per `attempt_id`: the same answer returns the stored grade, a different answer
    under the same id is a conflict."""
    authorize(p, body.learner_id)
    if body.attempt_id is not None and (owner := c.ownership.assessment_attempt(body.attempt_id)) is not None:
        authorize(p, owner)  # another learner's attempt id is never replayed or compared against
    result = await c.assessment_service.submit_attempt(item_id, body)
    if result.replayed:
        response.status_code = 200
    return result


@router.get("/assessment-attempts/{attempt_id}", response_model=AttemptView, dependencies=[Depends(owned_attempt)])
async def get_attempt(attempt_id: str, c: Container = Depends(container)) -> AttemptView:
    return c.assessment_service.attempt(attempt_id)


@router.get("/assessment-attempts/{attempt_id}/grade", response_model=PublicGrade,
            dependencies=[Depends(owned_attempt)])
async def get_grade(attempt_id: str, c: Container = Depends(container)) -> PublicGrade:
    return c.assessment_service.grade_for(attempt_id)
