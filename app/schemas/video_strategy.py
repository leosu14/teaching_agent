"""Input of the video strategy, the generative-video part of visual planning.

Kept apart from app.schemas.generative_video because it brings together the lesson, presentation, pedagogy and
video schemas (which themselves use the generative-video schemas).
"""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import Schema
from app.schemas.generative_video import GeneratedVideoConfig, ProviderVideoLimits, VideoCandidate
from app.schemas.lesson import LessonContent
from app.schemas.pedagogy import GapAction, KnowledgeGapSet, PlanBrief
from app.schemas.presentation import SlideDeckPlan
from app.schemas.video import ImageAssetInput, VideoConfig
from app.schemas.visual import VisualPlan


class GapSignal(Schema):
    """One knowledge gap as the strategy uses it: the concept, its urgency and the planned action. No learner
    identifier, history or mastery value: nothing here can reach a provider prompt."""

    concept_id: str
    priority: float = Field(ge=0, le=1)
    action: GapAction

    @classmethod
    def from_gaps(cls, gaps: KnowledgeGapSet) -> list[GapSignal]:
        return [cls(concept_id=g.concept.concept_id, priority=g.priority, action=g.recommended_action)
                for g in gaps.gaps]


class VideoStrategyRequest(Schema):
    """Everything the strategy plans from: the approved lesson, its pedagogical plan and gaps, the slide plan, the
    images it already has and how long each slide's narration is."""

    lesson: LessonContent
    lesson_artifact_id: str | None = None
    pedagogical_plan: PlanBrief | None = None
    gaps: list[GapSignal] = Field(default_factory=list)
    deck: SlideDeckPlan
    visual_plan: VisualPlan | None = None
    image_assets: list[ImageAssetInput] = Field(default_factory=list)
    narration_seconds: dict[str, float] = Field(default_factory=dict)  # slide id -> seconds of narration
    language: str
    config: GeneratedVideoConfig = Field(default_factory=GeneratedVideoConfig)
    limits: ProviderVideoLimits | None = None  # None: the configured provider's (filled in by the tool)
    video: VideoConfig = Field(default_factory=VideoConfig)  # the platform format clips are normalised to
    suggestions: list[VideoCandidate] = Field(default_factory=list)
