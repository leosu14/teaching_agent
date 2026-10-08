"""Learning cycles: execute the curriculum's next learning action end to end (opt-in: lessons, sessions and
evaluations work exactly as before without one)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from app.api.auth import Principal
from app.api.authorization import owned_cycle, principal, require_learner
from app.api.deps import container
from app.schemas.artifact import ArtifactView
from app.schemas.events import Event
from app.schemas.learning_cycle import CycleResponse, LearningCycleView, StartLearningCycle
from app.services.container import Container

router = APIRouter(tags=["learning-cycles"])


@router.post("/learners/{learner_id}/learning-cycles", response_model=LearningCycleView, status_code=201,
             dependencies=[Depends(require_learner)])
async def start_cycle(learner_id: str, body: StartLearningCycle, response: Response,
                      p: Principal = Depends(principal), c: Container = Depends(container)) -> LearningCycleView:
    """Start the learner's next cycle: the curriculum selects the action, the cycle runs it until the learner is
    needed (WAITING, with the prompt) or it ends. The same idempotency key returns (and continues) the same cycle;
    another key while a cycle is active is a 409. The cycle's user is the authenticated principal (a `user_id` in the
    body is ignored), so another user reusing the key is a 409 too."""
    view = await c.learning_cycle_service.start(learner_id, body.model_copy(update={"user_id": p.user_id}))
    if not view.created:
        response.status_code = 200
    return view


@router.get("/learners/{learner_id}/learning-cycles", response_model=list[LearningCycleView],
            dependencies=[Depends(require_learner)])
async def list_cycles(learner_id: str, c: Container = Depends(container)) -> list[LearningCycleView]:
    return await c.learning_cycle_service.for_learner(learner_id)


@router.get("/learning-cycles/{cycle_id}", response_model=LearningCycleView,
            dependencies=[Depends(owned_cycle)])
async def get_cycle(cycle_id: str, c: Container = Depends(container)) -> LearningCycleView:
    return await c.learning_cycle_service.get(cycle_id)


@router.post("/learning-cycles/{cycle_id}/responses", response_model=LearningCycleView,
             dependencies=[Depends(owned_cycle)])
async def respond(cycle_id: str, body: CycleResponse, c: Container = Depends(container)) -> LearningCycleView:
    """The learner's response to the current prompt (`answers` for a question sheet, `answer` for a session
    question). Idempotent per `client_response_id`."""
    return await c.learning_cycle_service.respond(cycle_id, body)


@router.post("/learning-cycles/{cycle_id}/resume", response_model=LearningCycleView,
             dependencies=[Depends(owned_cycle)])
async def resume_cycle(cycle_id: str, c: Container = Depends(container)) -> LearningCycleView:
    """Continue the cycle; a BLOCKED cycle retries its step through the step's own recovery."""
    return await c.learning_cycle_service.resume(cycle_id)


@router.post("/learning-cycles/{cycle_id}/cancel", response_model=LearningCycleView,
             dependencies=[Depends(owned_cycle)])
async def cancel_cycle(cycle_id: str, c: Container = Depends(container)) -> LearningCycleView:
    return await c.learning_cycle_service.cancel(cycle_id)


@router.get("/learning-cycles/{cycle_id}/events", response_model=list[Event],
            dependencies=[Depends(owned_cycle)])
def cycle_events(cycle_id: str, c: Container = Depends(container)) -> list[Event]:
    return c.learning_cycle_service.events(cycle_id)


@router.get("/learning-cycles/{cycle_id}/artifacts", response_model=list[ArtifactView],
            dependencies=[Depends(owned_cycle)])
def cycle_artifacts(cycle_id: str, c: Container = Depends(container)) -> list[ArtifactView]:
    return [ArtifactView.of(a) for a in c.learning_cycle_service.artifacts(cycle_id)]
