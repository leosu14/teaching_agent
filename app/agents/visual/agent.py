"""VisualAgent: plans the visuals of an approved lesson, then searches or generates, selects, validates and
stores each image as an IMAGE_ASSET artifact.

The model only describes what each visual must show (a VisualPlan, checked against the lesson); it never
produces URLs or attribution. It plans from an answer-free brief (briefing.py), never the lesson itself, and a plan
whose text states an exercise or check-question answer is rejected: its search queries and generation prompts are the
only text that reaches image search and generation. Everything else goes through tools on the ToolManager: the agent
never touches providers, the object store or the database. A visual that cannot be produced is recorded as a failure
with every attempt; whether a failed required visual stops the lesson is the workflow's visual policy.
"""

from __future__ import annotations

import hashlib
import json

from app.agents.base import Agent, AgentContext, AgentSpec, OutputRejected
from app.agents.visual.briefing import VisualBrief, build_brief
from app.observability.budget import BudgetExceededError
from app.schemas.artifact import Artifact, ArtifactBatch, ArtifactDraft, ArtifactType, StoredArtifacts
from app.schemas.common import ModelTier
from app.schemas.events import EventType
from app.schemas.lesson import LessonContent, VisualRequest
from app.schemas.visual import (
    ANSWER_ISOLATED_PLANNING,
    FetchedImage,
    ImageAsset,
    ImageAssetRequest,
    ImageFetchRequest,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageSearchRequest,
    ImageSearchResponse,
    ImageSelectionRequest,
    ImageSelectionResult,
    ImageValidationReport,
    ImageValidationRequest,
    SelectionRecord,
    VisualAssetRef,
    VisualAttempt,
    VisualFailure,
    VisualPlan,
    VisualPlanProposal,
    VisualRequirement,
    VisualResult,
    VisualType,
    aspect_value,
)
from app.tools.base import ToolError, ToolNotFound, ToolPermissionError

GENERATION_LONG_SIDE = 1280
GENERATION_STYLE = {
    VisualType.DIAGRAM: "clean flat educational diagram",
    VisualType.CHART: "clean labelled chart",
    VisualType.ILLUSTRATION: "flat educational illustration",
    VisualType.ICON: "simple flat icon",
    VisualType.GENERATED_VISUAL: "educational illustration",
}
NEGATIVE_PROMPT = "watermark, logo, illegible text"
# The plan text that reaches image search and generation, or the stored plan and asset metadata.
ANSWER_CHECKED_FIELDS = ("purpose", "concept", "description", "search_query", "generation_prompt")


def generation_size(aspect_ratio: str) -> tuple[int, int]:
    ratio = aspect_value(aspect_ratio)
    if ratio >= 1:
        return GENERATION_LONG_SIDE, round(GENERATION_LONG_SIDE / ratio)
    return round(GENERATION_LONG_SIDE * ratio), GENERATION_LONG_SIDE


def plan_id_for(lesson: LessonContent, proposal: VisualPlanProposal) -> str:
    """Deterministic: re-planning the same lesson the same way yields the same plan (and reuses its artifacts)."""
    body = json.dumps({"lesson": lesson.title, "sections": [s.section_id for s in lesson.sections],
                       "proposal": proposal.model_dump(mode="json")}, sort_keys=True)
    return "vp_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


class VisualAgent(Agent[VisualRequest, VisualResult]):
    spec = AgentSpec(
        id="visual",
        name="Visual",
        description="Plans the visuals of an approved lesson, finds or generates each image, selects and validates "
                    "it, and stores it as an attributed IMAGE_ASSET artifact.",
        input_model=VisualRequest,
        output_model=VisualResult,
        tier=ModelTier.STANDARD,
        tools=("image.search", "image.select", "image.fetch", "image.generate", "image.validate",
               "image.create_asset", "artifact.store"),
        permissions=frozenset({"network", "media:generate", "artifact:write", "artifact:read"}),
        timeout_seconds=300.0,
    )
    instructions = "Plan the visuals for this approved lesson."

    # --- planning ------------------------------------------------------------------------------

    async def plan(self, data: VisualRequest, ctx: AgentContext, brief: VisualBrief | None = None) -> VisualPlan:
        brief = brief or build_brief(data)
        if data.max_visuals == 0:
            proposal = VisualPlanProposal(rationale="Visuals are disabled for this lesson.")
        else:
            proposal = await self.generate(brief.planning, ctx, source=brief, output_model=VisualPlanProposal)
            assert isinstance(proposal, VisualPlanProposal)
        # Code names every visual (its section's ref number and its type, both system values) and puts the lesson's
        # section ids back in place of the refs: nothing the model wrote becomes an identifier.
        requirements: list[VisualRequirement] = []
        for r in proposal.requirements:
            base = f"v{r.lesson_section_id.removeprefix('s')}_{r.visual_type.value}"
            taken = {q.visual_id for q in requirements}
            visual_id = next(v for v in (base, *(f"{base}_{n}" for n in range(2, 100))) if v not in taken)
            requirements.append(VisualRequirement(**r.model_dump(exclude={"lesson_section_id"}), visual_id=visual_id,
                                                  lesson_section_id=brief.sections[r.lesson_section_id]))
        return VisualPlan(plan_id=plan_id_for(data.lesson, proposal), lesson_title=brief.planning.lesson_title or "",
                          requirements=requirements, rationale=proposal.rationale)

    def check(self, output, source) -> None:
        if not isinstance(output, VisualPlanProposal):
            return
        assert isinstance(source, VisualBrief)
        planning = source.planning
        # Rejections name a requirement by its position and a field by its name: never a value the model wrote.
        unknown = [i for i, r in enumerate(output.requirements, start=1) if r.lesson_section_id not in source.sections]
        if unknown:
            raise OutputRejected(f"requirements {unknown} reference unknown lesson sections; use a section's ref")
        if len(output.requirements) > planning.max_visuals:
            raise OutputRejected(f"{len(output.requirements)} visuals proposed; at most {planning.max_visuals} allowed")
        leaking = [f"requirement {i}.{name}" for i, r in enumerate(output.requirements, start=1)
                   for name in ANSWER_CHECKED_FIELDS if source.leaks(getattr(r, name))]
        if leaking:
            raise OutputRejected(f"{', '.join(leaking)} state the answer to one of the lesson's exercises or check "
                                 "questions; describe the concept without it")
        if source.leaks(output.rationale):
            raise OutputRejected("the rationale states the answer to one of the lesson's exercises or check questions")
        searched, generated = output.source_counts()
        if planning.max_searched_images is not None and searched > planning.max_searched_images:
            raise OutputRejected(f"{searched} visuals may search for an image (fallbacks included); the image budget "
                                 f"allows {planning.max_searched_images}: drop searched visuals or their search_query")
        if planning.max_generated_images is not None and generated > planning.max_generated_images:
            raise OutputRejected(f"{generated} visuals may generate an image (fallbacks included); the image budget "
                                 f"allows {planning.max_generated_images}: drop generated visuals or their "
                                 "generation_prompt")

    # --- the visual loop -----------------------------------------------------------------------

    async def run(self, data: VisualRequest, ctx: AgentContext) -> VisualResult:
        scope = ctx.scope
        scope.emit(EventType.VISUAL_STARTED, lesson_title=data.lesson.title, sections=len(data.lesson.sections),
                   max_visuals=data.max_visuals)
        brief = build_brief(data)
        plan = await self.plan(data, ctx, brief)
        # Images can be marked answer-isolated only when every model-derived field of the plan passed the checks;
        # each image is then checked again on everything of it that is persisted and learner-visible (_create).
        isolation = brief if brief.isolated(plan) else None
        scope.emit(EventType.VISUAL_PLAN_CREATED, plan_id=plan.plan_id, requirements=[
            {"visual_id": r.visual_id, "lesson_section_id": r.lesson_section_id, "visual_type": r.visual_type.value,
             "preferred_source": r.preferred_source, "required": r.required} for r in plan.requirements])
        stored = await self.use_tool("artifact.store", ArtifactBatch(drafts=[ArtifactDraft(
            key="visual_plan", name="visual_plan", type=ArtifactType.VISUAL_PLAN, media_type="application/json",
            content=plan.model_dump_json(indent=2), parent_ids=data.parent_artifact_ids,
            metadata={"plan_id": plan.plan_id, "visuals": len(plan.requirements),
                      "required": sum(r.required for r in plan.requirements)},
        )]), ctx)
        assert isinstance(stored, StoredArtifacts)
        plan_artifact_id = stored.by_key["visual_plan"]

        assets: list[VisualAssetRef] = []
        failures: list[VisualFailure] = []
        artifacts = list(stored.artifacts)
        for req in plan.requirements:
            outcome = await self._fulfil(req, plan, plan_artifact_id, data, ctx, isolation)
            if isinstance(outcome, VisualFailure):
                failures.append(outcome)
            else:
                ref, artifact = outcome
                assets.append(ref)
                artifacts.append(artifact)

        warnings = [
            f"{'Required' if f.required else 'Optional'} visual {f.visual_id} for section {f.lesson_section_id} "
            f"was not produced: {f.reason}" for f in failures
        ]
        status = "failed" if any(f.required for f in failures) else "partial" if failures else "complete"
        result = VisualResult(status=status, plan=plan, plan_artifact_id=plan_artifact_id, artifacts=artifacts,
                              assets=assets, failures=failures, warnings=warnings)
        summary = {"plan_id": plan.plan_id, "status": status, "assets": len(assets),
                   "failures": [{"visual_id": f.visual_id, "required": f.required, "reason": f.reason}
                                for f in failures]}
        scope.emit(EventType.VISUAL_FAILED if status == "failed" else EventType.VISUAL_COMPLETED, **summary)
        return result

    async def _fulfil(self, req: VisualRequirement, plan: VisualPlan, plan_artifact_id: str, data: VisualRequest,
                      ctx: AgentContext,
                      isolation: VisualBrief | None) -> tuple[VisualAssetRef, Artifact] | VisualFailure:
        attempts: list[VisualAttempt] = []
        for source in req.sources():
            if source == "search":
                ref = await self._search(req, plan, plan_artifact_id, data, ctx, attempts, isolation)
            else:
                ref = await self._generate(req, plan, plan_artifact_id, ctx, attempts, isolation)
            if ref is not None:
                return ref
        return VisualFailure(visual_id=req.visual_id, lesson_section_id=req.lesson_section_id, required=req.required,
                             reason=attempts[-1].message if attempts else "no source could provide it",
                             attempts=attempts)

    async def _tool(self, name: str, payload, ctx: AgentContext, attempts: list[VisualAttempt], *, source: str,
                    stage: str, image_id: str | None = None):
        """Call a tool; a tool failure becomes a recorded attempt (None). Wiring errors still raise. A budget refusal
        is this visual's failure too, as for a generated video segment: the ledger has already refused the request,
        and the visual failure policy decides whether the lesson can do without it."""
        try:
            return await self.use_tool(name, payload, ctx)
        except (ToolPermissionError, ToolNotFound):
            raise
        except BudgetExceededError as exc:
            attempts.append(VisualAttempt(source=source, stage="budget", message=str(exc)[:1000], image_id=image_id))
            return None
        except ToolError as exc:
            report = getattr(exc, "report", None)
            attempts.append(VisualAttempt(source=source, stage=stage, message=str(exc)[:1000], image_id=image_id,
                                          errors=report.errors if report is not None else []))
            return None

    async def _search(self, req: VisualRequirement, plan: VisualPlan, plan_artifact_id: str, data: VisualRequest,
                      ctx: AgentContext, attempts: list[VisualAttempt],
                      isolation: VisualBrief | None) -> tuple[VisualAssetRef, Artifact] | None:
        scope = ctx.scope
        assert req.search_query is not None
        scope.emit(EventType.IMAGE_SEARCH_STARTED, visual_id=req.visual_id, query=req.search_query)
        response = await self._tool("image.search", ImageSearchRequest(
            query=req.search_query, visual_type=req.visual_type, aspect_ratio=req.aspect_ratio,
            language=data.language), ctx, attempts, source="search", stage="search")
        if response is None:
            scope.emit(EventType.IMAGE_SEARCH_COMPLETED, visual_id=req.visual_id, ok=False, error=attempts[-1].message)
            return None
        assert isinstance(response, ImageSearchResponse)
        scope.emit(EventType.IMAGE_SEARCH_COMPLETED, visual_id=req.visual_id, ok=True, provider=response.provider,
                   results=len(response.results), image_ids=[r.image_id for r in response.results])
        if not response.results:
            attempts.append(VisualAttempt(source="search", stage="search",
                                          message=f"no image found for '{req.search_query}'"))
            return None

        selection = await self._tool("image.select", ImageSelectionRequest(requirement=req,
                                                                           candidates=response.results),
                                     ctx, attempts, source="search", stage="select")
        if selection is None:
            return None
        assert isinstance(selection, ImageSelectionResult)
        rejected = [{"image_id": r.candidate.image_id, "stage": "select", "reason": "; ".join(r.reasons)}
                    for r in selection.ranked if not r.eligible]
        eligible = selection.eligible()
        if not eligible:
            attempts.append(VisualAttempt(source="search", stage="select", message=(
                f"none of {len(selection.ranked)} candidates is eligible: "
                + "; ".join(f"{r['image_id']}: {r['reason']}" for r in rejected))))
            return None

        for ranked in eligible[:data.max_candidates]:
            c = ranked.candidate
            scope.emit(EventType.IMAGE_SELECTED, visual_id=req.visual_id, image_id=c.image_id, rank=ranked.rank,
                       score=ranked.score, title=c.title, provider=c.provider, source_url=c.source_url)
            fetched = await self._tool("image.fetch", ImageFetchRequest(result=c), ctx, attempts,
                                       source="search", stage="fetch", image_id=c.image_id)
            if fetched is None:
                rejected.append({"image_id": c.image_id, "stage": "fetch", "reason": attempts[-1].message})
                continue
            assert isinstance(fetched, FetchedImage)
            attribution = c.attribution()
            report = await self._validate(req, ImageValidationRequest(
                object=fetched.object, declared_width=c.width, declared_height=c.height, declared_format=c.format,
                expected_aspect_ratio=req.aspect_ratio, attribution=attribution,
                attribution_required=req.attribution_required, origin="search",
            ), ctx, attempts, source="search", image_id=c.image_id)
            if report is None:
                rejected.append({"image_id": c.image_id, "stage": "validate", "reason": attempts[-1].message})
                continue
            assert report.measured is not None
            asset = ImageAsset(
                asset_id="img_" + fetched.object.checksum[:16], visual_id=req.visual_id, plan_id=plan.plan_id,
                lesson_section_id=req.lesson_section_id, visual_type=req.visual_type, purpose=req.purpose,
                description=req.description, origin="search", object=fetched.object, width=report.measured.width,
                height=report.measured.height, format=report.measured.media_type, attribution=attribution,
                selection=SelectionRecord(selector=selection.selector, rank=ranked.rank, score=ranked.score,
                                          signals=ranked.signals, candidates=len(selection.ranked),
                                          rejected=list(rejected)),
                validation=report,
            )
            ref = await self._create(req, asset, plan_artifact_id, ctx, attempts, isolation)
            if ref is not None:
                return ref
            rejected.append({"image_id": c.image_id, "stage": "store", "reason": attempts[-1].message})
        return None

    async def _generate(self, req: VisualRequirement, plan: VisualPlan, plan_artifact_id: str, ctx: AgentContext,
                        attempts: list[VisualAttempt],
                        isolation: VisualBrief | None) -> tuple[VisualAssetRef, Artifact] | None:
        scope = ctx.scope
        assert req.generation_prompt is not None
        width, height = generation_size(req.aspect_ratio)
        seed = int(hashlib.sha256(req.generation_prompt.encode("utf-8")).hexdigest()[:8], 16) % 2**31
        scope.emit(EventType.IMAGE_GENERATION_STARTED, visual_id=req.visual_id, width=width, height=height,
                   seed=seed, aspect_ratio=req.aspect_ratio)
        generated = await self._tool("image.generate", ImageGenerationRequest(
            prompt=req.generation_prompt, negative_prompt=NEGATIVE_PROMPT, aspect_ratio=req.aspect_ratio,
            width=width, height=height, style=GENERATION_STYLE.get(req.visual_type), seed=seed,
        ), ctx, attempts, source="generate", stage="generate")
        if generated is None:
            return None
        assert isinstance(generated, ImageGenerationResult)
        scope.emit(EventType.IMAGE_GENERATION_COMPLETED, visual_id=req.visual_id, asset_id=generated.asset_id,
                   provider=generated.provider, model=generated.model, checksum=generated.storage.checksum,
                   reused_object=generated.storage.reused, prompt_hash=generated.metadata.prompt_hash,
                   usage=generated.usage.model_dump())
        report = await self._validate(req, ImageValidationRequest(
            object=generated.storage, declared_width=generated.width, declared_height=generated.height,
            declared_format=generated.format, expected_aspect_ratio=req.aspect_ratio, attribution=generated.metadata,
            attribution_required=req.attribution_required, origin="generated",
        ), ctx, attempts, source="generate", image_id=generated.asset_id)
        if report is None:
            return None
        assert report.measured is not None
        asset = ImageAsset(
            asset_id=generated.asset_id, visual_id=req.visual_id, plan_id=plan.plan_id,
            lesson_section_id=req.lesson_section_id, visual_type=req.visual_type, purpose=req.purpose,
            description=req.description, origin="generated", object=generated.storage,
            width=report.measured.width, height=report.measured.height, format=report.measured.media_type,
            attribution=generated.metadata, validation=report,
        )
        return await self._create(req, asset, plan_artifact_id, ctx, attempts, isolation)

    async def _validate(self, req: VisualRequirement, payload: ImageValidationRequest, ctx: AgentContext,
                        attempts: list[VisualAttempt], *, source: str, image_id: str) -> ImageValidationReport | None:
        report = await self._tool("image.validate", payload, ctx, attempts, source=source, stage="validate",
                                  image_id=image_id)
        if report is None:
            return None
        assert isinstance(report, ImageValidationReport)
        if report.valid:
            return report
        codes = sorted({e.code for e in report.errors})
        attempts.append(VisualAttempt(source=source, stage="validate", image_id=image_id, errors=report.errors,
                                      message=f"image {image_id} failed validation: {', '.join(codes)}"))
        ctx.scope.emit(EventType.IMAGE_VALIDATION_FAILED, visual_id=req.visual_id, image_id=image_id, source=source,
                       errors=[e.model_dump(exclude_none=True) for e in report.errors])
        return None

    async def _create(self, req: VisualRequirement, asset: ImageAsset, plan_artifact_id: str, ctx: AgentContext,
                      attempts: list[VisualAttempt],
                      isolation: VisualBrief | None) -> tuple[VisualAssetRef, Artifact] | None:
        source = "search" if asset.origin == "search" else "generate"
        name = f"image_{req.visual_id}"
        if isolation is not None:  # marked only if its whole learner-visible record, marker included, is answer-free
            marked = asset.model_copy(update={"planning": ANSWER_ISOLATED_PLANNING})
            if isolation.persisted_clean(name, marked):
                asset = marked
        artifact = await self._tool("image.create_asset", ImageAssetRequest(
            name=name, asset=asset, attribution_required=req.attribution_required,
            expected_aspect_ratio=req.aspect_ratio, parent_ids=[plan_artifact_id],
        ), ctx, attempts, source=source, stage="store", image_id=asset.asset_id)
        if artifact is None:
            if attempts[-1].errors:
                ctx.scope.emit(EventType.IMAGE_VALIDATION_FAILED, visual_id=req.visual_id, image_id=asset.asset_id,
                               source=source, errors=[e.model_dump(exclude_none=True) for e in attempts[-1].errors])
            return None
        ctx.scope.emit(EventType.IMAGE_ASSET_CREATED, visual_id=req.visual_id, artifact_id=artifact.artifact_id,
                       asset_id=asset.asset_id, origin=asset.origin, lesson_section_id=req.lesson_section_id,
                       checksum=asset.object.checksum, uri=asset.object.uri, media_type=asset.format,
                       width=asset.width, height=asset.height)
        assert isinstance(artifact, Artifact)
        return VisualAssetRef(
            visual_id=req.visual_id, lesson_section_id=req.lesson_section_id, artifact_id=artifact.artifact_id,
            asset_id=asset.asset_id, visual_type=req.visual_type, origin=asset.origin, purpose=req.purpose,
            description=req.description, uri=asset.object.uri, checksum=asset.object.checksum,
            media_type=asset.format, width=asset.width, height=asset.height,
        ), artifact
