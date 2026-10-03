"""The video strategy: the generative-video part of visual planning.

For an approved lesson it decides, section by section and with deterministic rules, whether a short generated clip
would teach better than the slide and its images, and plans the clips it selects (VideoSegmentPlanSet). Suggestions
(e.g. from a model) only add weight; the policy decides.

Rules, in order, each recorded as the section's decision:
1. Practice, assessment and review sections never get a clip: they already work as text.
2. A section needs a slide that presents it (one clip per slide).
3. Short factual text never gets a clip.
4. The section must describe something that moves: a physical or scientific process, a historical scene, a movement
   across geography, a pronunciation (mouth) demonstration, or a visual story. Cues come from a per-language lexicon
   (English ships; other languages can be added) and from the visual plan's picture types. Grammar, definitions and
   references count against a clip.
5. When the section already has an image, only a strong case gets a clip: the existing image is preferred.
6. Its duration is the slide's narration length, clamped to the configured limits and to a duration the provider
   offers (never an arbitrary one).
7. The budget (segments, seconds and, when the provider's price is known, cost) takes the highest-priority clips
   first; urgent knowledge gaps raise priority. Optional clips over budget are skipped with a warning; required
   ones are reported so the workflow's failure policy can act. A budget is never exceeded silently.

Every planned clip has a fallback: the slide's existing IMAGE_ASSET, else the slide itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable

from app.observability.scope import ExecutionScope
from app.schemas.events import EventType
from app.schemas.generative_video import (
    InsertionStrategy,
    ProviderPreferences,
    ProviderVideoLimits,
    SegmentFallback,
    VideoBudgetSummary,
    VideoPurpose,
    VideoSegmentDecision,
    VideoSegmentPlan,
    VideoSegmentPlanSet,
)
from app.schemas.lesson import LessonSection
from app.schemas.presentation import SlidePlan, SlideType
from app.schemas.video_strategy import VideoStrategyRequest
from app.schemas.visual import VisualType
from app.tools.base import Tool, ToolError
from app.tools.visual.video_prompts import VideoPromptBuilder, sentences

STRATEGY = "video-strategy/1"
SELECT_THRESHOLD = 2.0  # net score a section needs for a clip
STRONG_CASE = 3.0  # net score needed when the section already has an image
MIN_WORDS = 12  # explanation + examples shorter than this is short factual text
TEXT_PURPOSES = {"assessment": "exercise_text_sufficient", "free_practice": "exercise_text_sufficient",
                 "guided_practice": "exercise_text_sufficient", "review": "review_text_sufficient"}
PREFERRED_SLIDES = (SlideType.EXPLANATION, SlideType.EXAMPLE, SlideType.COMPARISON)

# Cue lexicons per language (BCP 47 primary subtag). A cue matches a word that starts with it, or a phrase.
CUES: dict[str, dict[VideoPurpose, tuple[str, ...]]] = {
    "en": {
        VideoPurpose.PHYSICAL_PROCESS: (
            "moves", "moving", "movement", "motion", "flows", "flowing", "melt", "boil", "freez", "rotat", "spin",
            "falls", "falling", "roll", "erupt", "expand", "contract", "pressure", "force", "collid", "pour",
            "mechanism", "machine", "engine", "lever", "pulley", "gear", "vibrat"),
        VideoPurpose.SCIENTIFIC_PROCESS: (
            "process", "evaporat", "condens", "precipitat", "photosynthes", "cell division", "mitosis", "reaction",
            "molecul", "circulat", "cycle", "erosion", "orbit", "digest", "respirat", "germinat", "metamorph",
            "convection", "heats", "cools", "transform"),
        VideoPurpose.HISTORICAL_SCENE: (
            "century", "ancient", "empire", "battle", "revolution", "historic", "medieval", "dynasty", "pharaoh",
            "colonial", "civilization", "civilisation", "conquest"),
        VideoPurpose.GEOGRAPHICAL_MOVEMENT: (
            "migrat", "route", "voyage", "expedition", "trade route", "continent", "drift", "tectonic", "currents",
            "crossed", "across the"),
        VideoPurpose.PRONUNCIATION: (
            "pronunciation", "pronounc", "articulat", "mouth", "tongue", "lips", "vowel sound", "phoneme",
            "intonation"),
        VideoPurpose.VISUAL_STORYTELLING: (
            "story", "storytelling", "scene", "narrative", "character", "once upon", "sequence of events"),
    },
}
UNNECESSARY: dict[str, tuple[str, ...]] = {
    "en": ("grammar", "conjugat", "definition", "is defined as", "means", "vocabulary", "spelling", "glossary",
           "reference", "tense", "agreement", "endings"),
}
# Picture types in the visual plan that hint at movement, and the purpose they hint at.
VISUAL_HINTS = {VisualType.DIAGRAM: VideoPurpose.SCIENTIFIC_PROCESS, VisualType.MAP: VideoPurpose.GEOGRAPHICAL_MOVEMENT}
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _matches(text: str, words: list[str], cue: str) -> bool:
    if " " in cue:
        return cue in text
    return any(w.startswith(cue) for w in words)


def lexicon(language: str) -> tuple[dict[VideoPurpose, tuple[str, ...]], tuple[str, ...]]:
    """English cues plus the lesson language's own, when a lexicon exists for it."""
    base = (language or "en").split("-")[0].lower()
    cues = {p: tuple(dict.fromkeys([*CUES["en"].get(p, ()), *CUES.get(base, {}).get(p, ())])) for p in VideoPurpose}
    unnecessary = tuple(dict.fromkeys([*UNNECESSARY["en"], *UNNECESSARY.get(base, ())]))
    return cues, unnecessary


def aspect_of(width: int, height: int) -> str:
    g = math.gcd(width, height)
    return f"{width // g}:{height // g}"


class VideoStrategy:
    name = STRATEGY

    def __init__(self, prompts: VideoPromptBuilder | None = None) -> None:
        self.prompts = prompts or VideoPromptBuilder()

    def plan_id(self, data: VideoStrategyRequest) -> str:
        body = json.dumps({
            "strategy": STRATEGY, "prompt_builder": self.prompts.name,
            "lesson": hashlib.sha256(data.lesson.model_dump_json().encode("utf-8")).hexdigest(),
            "deck": data.deck.deck_id, "images": sorted((i.artifact_id, i.checksum) for i in data.image_assets),
            "narration": data.narration_seconds, "language": data.language,
            "config": data.config.model_dump(mode="json"), "limits": data.limits.model_dump(mode="json"),
            "video": [data.video.width, data.video.height, data.video.fps],
            "gaps": [g.model_dump(mode="json") for g in data.gaps],
            "suggestions": [s.model_dump(mode="json") for s in data.suggestions],
        }, sort_keys=True)
        return "vsp_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

    def plan(self, data: VideoStrategyRequest) -> VideoSegmentPlanSet:
        if data.limits is None:
            raise ValueError("the video strategy needs the provider's limits")
        config, limits = data.config, data.limits
        cues, unnecessary = lexicon(data.language)
        suggested = {s.lesson_section_id: s for s in data.suggestions}
        gap_priority = {g.concept_id: g.priority for g in data.gaps
                        if g.action in ("introduce", "reteach", "prerequisite_first")}
        images = {i.artifact_id for i in data.image_assets}
        aspect = aspect_of(data.video.width, data.video.height)
        decisions: list[VideoSegmentDecision] = []
        candidates: list[tuple[float, int, VideoSegmentDecision, LessonSection, SlidePlan, float]] = []
        taken: set[str] = set()
        for order, section in enumerate(data.lesson.sections):
            decision = self._decide(section, data, cues, unnecessary, suggested, taken, images)
            if decision.selected:
                slide = next(s for s in data.deck.slides if s.slide_id == decision.slide_id)
                duration = self._duration(narration=data.narration_seconds.get(slide.slide_id), data=data)
                if aspect not in limits.aspect_ratios:
                    decision = decision.model_copy(update={"selected": False, "skip_reason": "format_unsupported"})
                elif duration is None:
                    decision = decision.model_copy(update={"selected": False, "skip_reason": "duration_unsupported"})
                else:
                    taken.add(slide.slide_id)
                    priority = round(decision.score * 10 + gap_priority.get(section.concept_id, 0.0) * 10)
                    candidates.append((priority, order, decision, section, slide, duration))
            decisions.append(decision)

        segments, warnings, over_required = [], [], []
        # a provider that runs locally (the mock) costs nothing; a network provider's price must be configured
        price = config.price_per_second_usd if limits.requires_network else 0.0
        used_seconds, used_cost, cost_known = 0.0, 0.0, price is not None
        if not cost_known and config.max_cost_usd is not None and candidates:
            warnings.append(f"The video provider's price is unknown (VIDEO_GENERATION_PRICE_PER_SECOND is not set), so "
                            f"MAX_VIDEO_GENERATION_COST_USD={config.max_cost_usd:g} cannot be checked; generated video "
                            "is limited by MAX_GENERATED_VIDEO_SEGMENTS and MAX_GENERATED_VIDEO_SECONDS instead.")
        by_section = {d.lesson_section_id: i for i, d in enumerate(decisions)}
        for priority, _order, decision, section, slide, duration in sorted(candidates, key=lambda c: (-c[0], c[1])):
            cost = round(duration * price, 6) if price is not None else None
            reason = None
            if len(segments) + 1 > config.max_segments:
                reason, limit = "budget_segments", f"MAX_GENERATED_VIDEO_SEGMENTS={config.max_segments}"
            elif used_seconds + duration > config.max_total_seconds + 1e-9:
                reason, limit = "budget_seconds", f"MAX_GENERATED_VIDEO_SECONDS={config.max_total_seconds:g}"
            elif cost is not None and config.max_cost_usd is not None and used_cost + cost > config.max_cost_usd + 1e-9:
                reason, limit = "budget_cost", f"MAX_VIDEO_GENERATION_COST_USD={config.max_cost_usd:g}"
            if reason is not None:
                decisions[by_section[section.section_id]] = decision.model_copy(
                    update={"selected": False, "skip_reason": reason})
                if config.required:
                    over_required.append(section.section_id)
                    warnings.append(f"A required generated clip for section {section.section_id} does not fit the "
                                    f"budget ({limit}).")
                else:
                    warnings.append(f"Skipped the optional generated clip for section {section.section_id}: it does "
                                    f"not fit the budget ({limit}); the slide and its images are used instead.")
                continue
            used_seconds = round(used_seconds + duration, 3)
            if cost is not None:
                used_cost = round(used_cost + cost, 6)
            segments.append(self._segment(section, slide, decision, duration, priority, cost, data))

        segments.sort(key=lambda s: next(i for i, sec in enumerate(data.lesson.sections)
                                         if sec.section_id == s.lesson_section_id))
        return VideoSegmentPlanSet(
            plan_id=self.plan_id(data), lesson_title=data.lesson.title, language=data.language,
            deck_id=data.deck.deck_id, provider=limits.provider, strategy=STRATEGY, decisions=decisions,
            segments=segments, warnings=warnings, over_budget_required=over_required,
            budget=VideoBudgetSummary(segments=len(segments), seconds=used_seconds,
                                      estimated_cost_usd=used_cost if cost_known else None, cost_known=cost_known,
                                      max_segments=config.max_segments, max_seconds=config.max_total_seconds,
                                      max_cost_usd=config.max_cost_usd))

    def _decide(self, section: LessonSection, data: VideoStrategyRequest, cues, unnecessary, suggested,
                taken: set[str], images: set[str]) -> VideoSegmentDecision:
        base = {"lesson_section_id": section.section_id, "suggested": section.section_id in suggested}
        if section.purpose in TEXT_PURPOSES:
            return VideoSegmentDecision(**base, selected=False, skip_reason=TEXT_PURPOSES[section.purpose],
                                        reasons=[f"section purpose is {section.purpose}"])
        slide = self._slide_for(section.section_id, data)
        if slide is None:
            return VideoSegmentDecision(**base, selected=False, skip_reason="no_slide")
        base["slide_id"] = slide.slide_id
        if slide.slide_id in taken:
            return VideoSegmentDecision(**base, selected=False, skip_reason="duplicate_slide")
        text = " ".join([section.heading, section.explanation, *section.examples]).lower()
        words = WORD.findall(text)
        if len(WORD.findall(" ".join([section.explanation, *section.examples]))) < MIN_WORDS:
            return VideoSegmentDecision(**base, selected=False, skip_reason="short_factual_text",
                                        reasons=[f"{len(words)} words"])
        scores: dict[VideoPurpose, float] = {}
        reasons: dict[VideoPurpose, list[str]] = {}
        for purpose, terms in cues.items():
            hits = [t for t in terms if _matches(text, words, t)]
            if hits:
                scores[purpose] = float(min(len(hits), 4))
                reasons[purpose] = [f"cue '{t}'" for t in hits[:4]]
        for requirement in (data.visual_plan.requirements if data.visual_plan else []):
            hint = VISUAL_HINTS.get(requirement.visual_type)
            if requirement.lesson_section_id == section.section_id and hint is not None:
                scores[hint] = scores.get(hint, 0.0) + 0.5
                reasons.setdefault(hint, []).append(f"visual plan has a {requirement.visual_type.value}")
        suggestion = suggested.get(section.section_id)
        if suggestion is not None:
            scores[suggestion.purpose] = scores.get(suggestion.purpose, 0.0) + 1.0
            reasons.setdefault(suggestion.purpose, []).append("suggested as a video candidate")
        penalties = [t for t in unnecessary if _matches(text, words, t)]
        penalty = float(min(len(penalties), 2))
        if not scores:
            return VideoSegmentDecision(**base, selected=False, score=-penalty,
                                        skip_reason="definition_or_grammar" if penalties else "no_visual_motion",
                                        reasons=[f"counter-cue '{t}'" for t in penalties[:2]])
        purpose = max(scores, key=lambda p: (scores[p], -list(VideoPurpose).index(p)))
        score = round(scores[purpose] - penalty, 2)
        why = [*reasons[purpose], *(f"counter-cue '{t}'" for t in penalties[:2])]
        if score < SELECT_THRESHOLD:
            return VideoSegmentDecision(**base, selected=False, purpose=purpose, score=score, reasons=why,
                                        skip_reason="definition_or_grammar" if penalties else "no_visual_motion")
        has_image = bool(section.visuals) or any(b.artifact_id in images for b in slide.blocks("image"))
        if has_image and score < STRONG_CASE:
            return VideoSegmentDecision(**base, selected=False, purpose=purpose, score=score,
                                        reasons=[*why, "the section already has an image"],
                                        skip_reason="existing_image_sufficient")
        return VideoSegmentDecision(**base, selected=True, purpose=purpose, score=score, reasons=why)

    @staticmethod
    def _slide_for(section_id: str, data: VideoStrategyRequest) -> SlidePlan | None:
        slides = [s for s in data.deck.slides if section_id in s.section_refs]
        preferred = [s for s in slides if s.slide_type in PREFERRED_SLIDES]
        return (preferred or slides or [None])[0]

    @staticmethod
    def _duration(*, narration: float | None, data: VideoStrategyRequest) -> float | None:
        config, limits = data.config, data.limits
        assert limits is not None
        low, high = config.min_segment_seconds, min(config.max_segment_seconds, limits.max_duration)
        if low > high:
            return None
        target = min(max(narration if narration else high, low), high)
        if limits.durations is None:
            return round(target, 3)
        offered = [d for d in limits.durations if low - 1e-9 <= d <= high + 1e-9]
        if not offered:
            return None
        return min(offered, key=lambda d: (abs(d - target), d))

    def _segment(self, section: LessonSection, slide: SlidePlan, decision: VideoSegmentDecision, duration: float,
                 priority: int, cost: float | None, data: VideoStrategyRequest) -> VideoSegmentPlan:
        assert decision.purpose is not None
        requirement = next((r for r in (data.visual_plan.requirements if data.visual_plan else [])
                            if r.lesson_section_id == section.section_id), None)
        if decision.purpose == VideoPurpose.PRONUNCIATION and section.examples:
            description = f"A speaker's mouth slowly and clearly articulating: {section.examples[0]}"
        elif requirement is not None and decision.purpose not in (VideoPurpose.PRONUNCIATION,):
            description = f"{requirement.description} {sentences(section.explanation, 1)}"
        else:
            description = sentences(section.explanation, 2)
        content = self.prompts.content(subject=section.heading, description=description, purpose=decision.purpose,
                                       language=data.language)
        image = next((b.artifact_id for b in slide.blocks("image")
                      if b.artifact_id in {i.artifact_id for i in data.image_assets}), None)
        fallback = (SegmentFallback(kind="image_asset", slide_id=slide.slide_id, artifact_id=image) if image
                    else SegmentFallback(kind="slide", slide_id=slide.slide_id))
        strategy = (InsertionStrategy.INSET if decision.purpose == VideoPurpose.PRONUNCIATION
                    else data.config.strategy)
        seed = int(hashlib.sha256(f"{section.section_id}:{content.model_dump_json()}".encode()).hexdigest()[:8], 16) \
            if data.limits is not None and data.limits.supports_seed else None
        segment_id = re.sub(r"[^A-Za-z0-9_-]+", "_", f"gv_{section.section_id}")[:64]
        return VideoSegmentPlan(
            segment_id=segment_id, lesson_section_id=section.section_id, slide_id=slide.slide_id,
            purpose=decision.purpose, prompt=self.prompts.build(content, duration=duration), content=content,
            duration=duration, aspect_ratio=aspect_of(data.video.width, data.video.height),
            width=data.video.width, height=data.video.height, fps=data.video.fps, priority=priority,
            required=data.config.required, provider_preferences=ProviderPreferences(seed=seed),
            insertion_strategy=strategy, audio="muted", fallback=fallback, estimated_cost_usd=cost)


class VideoStrategyTool(Tool[VideoStrategyRequest, VideoSegmentPlanSet]):
    name = "visual.video_strategy"
    description = ("Decide deterministically which lesson sections benefit from a short generated video clip and "
                   "plan those clips (prompt, duration, insertion, fallback) within the configured budget.")
    input_model = VideoStrategyRequest
    output_model = VideoSegmentPlanSet

    def __init__(self, limits: Callable[[], ProviderVideoLimits] | None = None,
                 strategy: VideoStrategy | None = None) -> None:
        self._limits = limits
        self._strategy = strategy or VideoStrategy()

    async def run(self, data: VideoStrategyRequest, scope: ExecutionScope) -> VideoSegmentPlanSet:
        if data.limits is None:
            if self._limits is None:
                raise ToolError("no video generation provider is configured to plan clips for")
            data = data.model_copy(update={"limits": self._limits()})
        plan = self._strategy.plan(data)
        scope.emit(EventType.VIDEO_STRATEGY_COMPLETED, tool=self.name, plan_id=plan.plan_id,
                   provider=plan.provider, sections=len(plan.decisions), segments=len(plan.segments),
                   seconds=plan.budget.seconds, estimated_cost_usd=plan.budget.estimated_cost_usd,
                   decisions={d.lesson_section_id: d.skip_reason or f"selected:{d.purpose.value}"
                              for d in plan.decisions if d.selected or d.skip_reason},
                   over_budget_required=plan.over_budget_required)
        return plan
