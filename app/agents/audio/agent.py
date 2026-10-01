"""AudioPlannerAgent: decides WHAT is spoken over each slide of an approved, rendered presentation.

The model proposes narration segments mapped to slide ids from the slides' speakable text (titles, content lines,
speaker notes, questions and answers; never images, captions, tables or citations). Code picks the voice from the
TTS provider's catalog through the ToolManager, assigns deterministic ids, order, language, rate and a planning
estimate of each duration, then checks the plan with the deterministic validator through the ToolManager and sends
any errors back to the model. The agent never synthesizes speech, touches storage or creates artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from app.agents.base import Agent, AgentContext, AgentError, AgentSpec
from app.schemas.audio import (
    AudioPlan,
    AudioPlanningRequest,
    AudioPlanProposal,
    AudioPlanValidationReport,
    AudioSegment,
    Voice,
    VoiceCatalog,
    VoiceQuery,
    estimate_duration,
)
from app.schemas.common import ModelTier
from app.schemas.events import EventType


def audio_plan_id_for(data: AudioPlanningRequest, voice: str, segments: list[AudioSegment]) -> str:
    """Deterministic: planning the same narration the same way yields the same plan (and reuses its artifacts)."""
    body = json.dumps({"deck": data.deck.deck_id, "language": data.language, "voice": voice,
                       "segments": [s.model_dump(mode="json", exclude={"order", "expected_duration"})
                                    for s in segments]}, sort_keys=True, ensure_ascii=False)
    return "ap_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


class AudioPlannerAgent(Agent[AudioPlanningRequest, AudioPlan]):
    spec = AgentSpec(
        id="audio_planner",
        name="Audio Planner",
        description="Plans the narration of an approved presentation: which slide text is spoken, as concise "
                    "segments mapped to slides in order, in the lesson's language and with a catalog voice.",
        input_model=AudioPlanningRequest,
        output_model=AudioPlan,
        tier=ModelTier.STANDARD,
        tools=("tts.voices", "audio_plan.validate"),
    )
    instructions = "Plan the narration for this approved presentation."

    async def run(self, data: AudioPlanningRequest, ctx: AgentContext) -> AudioPlan:
        scope = ctx.scope
        scope.emit(EventType.AUDIO_PLANNING_STARTED, deck_id=data.deck.deck_id, slides=len(data.deck.slides),
                   language=data.language, voice=data.voice_id,
                   presentation_artifact_id=data.presentation_artifact_id)
        try:
            voice = await self._voice(data, ctx)
            plan, report = await self._plan(data, voice, ctx)
        except Exception as exc:
            scope.emit(EventType.AUDIO_FAILED, stage="planning", error=f"{type(exc).__name__}: {exc}"[:1000])
            raise
        scope.emit(EventType.AUDIO_PLAN_CREATED, audio_plan_id=plan.audio_plan_id, segments=len(plan.segments),
                   slides=len({s.slide_id for s in plan.segments}), language=plan.language, voice=plan.voice,
                   sources=dict(sorted(Counter(s.source_type.value for s in plan.segments).items())),
                   expected_duration=plan.expected_duration(), valid=report.valid)
        return plan

    async def _voice(self, data: AudioPlanningRequest, ctx: AgentContext) -> Voice:
        """The configured voice if it speaks the language, else the provider's first voice for it. Never a silent
        substitute for a configured voice that does not fit."""
        catalog = await self.use_tool("tts.voices", VoiceQuery(language=data.language), ctx)
        assert isinstance(catalog, VoiceCatalog)
        if data.voice_id is not None:
            voice = next((v for v in catalog.voices if v.voice_id == data.voice_id), None)
            if voice is None:
                raise AgentError(f"voice '{data.voice_id}' of provider '{catalog.provider}' does not speak "
                                 f"{data.language} or does not exist")
            return voice
        if not catalog.voices:
            raise AgentError(f"TTS provider '{catalog.provider}' has no voice for {data.language}")
        return catalog.voices[0]

    def _build(self, data: AudioPlanningRequest, voice: Voice, proposal: AudioPlanProposal) -> AudioPlan:
        per_slide: Counter[str] = Counter()
        segments = []
        for order, p in enumerate(proposal.segments, start=1):
            per_slide[p.slide_id] += 1
            segments.append(AudioSegment(
                segment_id=f"{p.slide_id}_a{per_slide[p.slide_id]}"[:96], order=order, slide_id=p.slide_id,
                source_type=p.source_type, source_ref=p.source_ref, text=p.text, language=data.language,
                voice=voice.voice_id, speaking_rate=data.speaking_rate, pitch=data.pitch,
                pause_before=p.pause_before, pause_after=p.pause_after,
                expected_duration=estimate_duration(p.text, data.speaking_rate), required=p.required,
            ))
        return AudioPlan(
            audio_plan_id=audio_plan_id_for(data, voice.voice_id, segments), task_id=data.task_id,
            deck_id=data.deck.deck_id, language=data.language, voice=voice.voice_id, segments=segments,
            metadata={"presentation_artifact_id": data.presentation_artifact_id, "lesson_title": data.lesson.title,
                      "voice_name": voice.display_name,
                      "rationale": proposal.rationale},
        )

    async def _plan(self, data: AudioPlanningRequest, voice: Voice,
                    ctx: AgentContext) -> tuple[AudioPlan, AudioPlanValidationReport]:
        corrections: list[str] = []
        for attempt in range(1, self.spec.validation_retries + 2):
            planning = data.planning_input(corrections)
            proposal = await self.generate(planning, ctx, source=planning, output_model=AudioPlanProposal)
            assert isinstance(proposal, AudioPlanProposal)
            plan = self._build(data, voice, proposal)
            report = await self.use_tool("audio_plan.validate", data.validation_request(plan), ctx)
            assert isinstance(report, AudioPlanValidationReport)
            if report.valid:
                return plan, report
            corrections = [e.describe() for e in report.errors]
            ctx.scope.emit(EventType.AGENT_VALIDATION_FAILED, attempt=attempt, stage="audio_plan",
                           error="; ".join(corrections)[:2000])
        # Still invalid: return it anyway. The workflow's validation gate fails the task before any audio is made.
        return plan, report
