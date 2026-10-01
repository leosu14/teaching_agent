"""SlidePlannerAgent: decides WHAT each slide of an approved lesson shows.

The model proposes a deck of structured slides that reference lesson sections, research citation ids and
existing IMAGE_ASSET artifact ids. Code assigns the deck id and the lesson metadata, then checks the proposal
with the deterministic validator through the ToolManager and sends any errors back to the model. The agent
never builds or renders files, searches for or generates images, or touches storage.
"""

from __future__ import annotations

import hashlib
import json

from app.agents.base import Agent, AgentContext, AgentSpec
from app.schemas.common import ModelTier
from app.schemas.events import EventType
from app.schemas.presentation import (
    SlideDeckPlan,
    SlideDeckProposal,
    SlidePlanningRequest,
    SlidePlanValidationReport,
)


def deck_id_for(data: SlidePlanningRequest, proposal: SlideDeckProposal) -> str:
    """Deterministic: planning the same lesson the same way yields the same deck (and reuses its artifacts)."""
    body = json.dumps({"lesson": data.lesson.title, "sections": [s.section_id for s in data.lesson.sections],
                       "proposal": proposal.model_dump(mode="json", exclude={"rationale"})}, sort_keys=True)
    return "deck_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


class SlidePlannerAgent(Agent[SlidePlanningRequest, SlideDeckPlan]):
    spec = AgentSpec(
        id="slide_planner",
        name="Slide Planner",
        description="Plans the slide deck of an approved lesson: slide sequence and types, structured content, "
                    "placement of existing image assets and the research citations behind each slide.",
        input_model=SlidePlanningRequest,
        output_model=SlideDeckPlan,
        tier=ModelTier.STANDARD,
        tools=("slide_plan.validate",),
    )
    instructions = "Plan the slides for this approved lesson."

    async def run(self, data: SlidePlanningRequest, ctx: AgentContext) -> SlideDeckPlan:
        scope = ctx.scope
        scope.emit(EventType.SLIDE_PLANNING_STARTED, lesson_title=data.lesson.title,
                   sections=len(data.lesson.sections), images=len(data.image_assets),
                   citations=len(data.research.citations), max_slides=data.max_slides)
        try:
            deck, report = await self._plan(data, ctx)
        except Exception as exc:
            scope.emit(EventType.PRESENTATION_FAILED, stage="slide_planning", error=f"{type(exc).__name__}: {exc}"[:1000])
            raise
        scope.emit(EventType.SLIDE_PLAN_CREATED, deck_id=deck.deck_id, slides=len(deck.slides),
                   slide_types=[s.slide_type.value for s in deck.slides],
                   image_artifact_ids=deck.image_artifact_ids(), citation_ids=deck.citation_ids(),
                   valid=report.valid)
        return deck

    async def _plan(self, data: SlidePlanningRequest, ctx: AgentContext) -> tuple[SlideDeckPlan, SlidePlanValidationReport]:
        corrections: list[str] = []
        for attempt in range(1, self.spec.validation_retries + 2):
            planning = data.planning_input(corrections)
            proposal = await self.generate(planning, ctx, source=planning, output_model=SlideDeckProposal)
            assert isinstance(proposal, SlideDeckProposal)
            deck = SlideDeckPlan(
                deck_id=deck_id_for(data, proposal), title=proposal.title,
                language=data.request.language_of_instruction, level=data.lesson.level, topic=data.request.topic,
                objective="; ".join(data.plan.objectives), slides=proposal.slides,
                metadata={"lesson_title": data.lesson.title, "subject": data.request.subject,
                          "sections": [s.section_id for s in data.lesson.sections], "rationale": proposal.rationale},
            )
            report = await self.use_tool("slide_plan.validate", data.validation_request(deck), ctx)
            assert isinstance(report, SlidePlanValidationReport)
            if report.valid:
                return deck, report
            corrections = [e.describe() for e in report.errors]
            ctx.scope.emit(EventType.AGENT_VALIDATION_FAILED, attempt=attempt, stage="slide_plan",
                           error="; ".join(corrections)[:2000])
        # Still invalid: return it anyway. The workflow's validation gate fails the task before anything is built.
        return deck, report
