"""Deterministic slide plan validation. A deck that fails here is never built or rendered."""

from __future__ import annotations

from pydantic import ValidationError

from app.observability.scope import ExecutionScope
from app.schemas.events import EventType
from app.schemas.presentation import (
    IMAGE_LAYOUTS,
    LAYOUTS_ALLOWING_IMAGES,
    MAX_BLOCKS_PER_SLIDE,
    MAX_WORDS_PER_SLIDE,
    REQUIRED_BLOCK,
    SlideDeckPlan,
    SlideLayout,
    SlidePlanIssue,
    SlidePlanValidationReport,
    SlidePlanValidationRequest,
    SlideType,
    block_words,
)
from app.tools.base import Tool, ToolError

CHECKS = [
    "schema", "slide_count", "unique_slide_ids", "ordering", "title_slide", "non_empty_slides", "section_refs",
    "image_refs", "citation_refs", "question_refs", "layout", "slide_type", "density",
]


class SlidePlanInvalid(ToolError):
    """The slide plan failed validation. `report` holds the structured errors."""

    def __init__(self, report: SlidePlanValidationReport) -> None:
        super().__init__("slide plan failed validation: " + "; ".join(e.describe() for e in report.errors))
        self.report = report


class SlidePlanValidator:
    name = "slide-plan-validator/1"

    def validate(self, request: SlidePlanValidationRequest) -> SlidePlanValidationReport:
        try:
            deck = SlideDeckPlan.model_validate(request.deck)
        except ValidationError as exc:
            errors = [SlidePlanIssue(code="invalid_schema", field=".".join(str(p) for p in err["loc"]) or None,
                                     message=err["msg"]) for err in exc.errors()]
            return SlidePlanValidationReport(valid=False, deck_id=request.deck.get("deck_id"), errors=errors,
                                             checks=["schema"], validator=self.name)
        errors = self._check(deck, request)
        return SlidePlanValidationReport(valid=not errors, deck_id=deck.deck_id, errors=errors, checks=CHECKS,
                                         validator=self.name, deck=deck)

    def _check(self, deck: SlideDeckPlan, req: SlidePlanValidationRequest) -> list[SlidePlanIssue]:
        errors: list[SlidePlanIssue] = []

        def issue(code, message, slide=None, field=None) -> None:
            errors.append(SlidePlanIssue(code=code, message=message, slide_id=slide.slide_id if slide else None,
                                         field=field))

        if len(deck.slides) > req.max_slides:
            issue("too_many_slides", f"{len(deck.slides)} slides; at most {req.max_slides} allowed")
        seen: set[str] = set()
        for slide in deck.slides:
            if slide.slide_id in seen:
                issue("duplicate_slide_id", f"slide id {slide.slide_id} is used more than once", slide)
            seen.add(slide.slide_id)
        orders = [s.order for s in deck.slides]
        if orders != list(range(1, len(deck.slides) + 1)):
            issue("slide_order", f"slide orders must be 1..{len(deck.slides)} in sequence, got {orders}")
        if deck.slides[0].slide_type != SlideType.TITLE:
            issue("missing_title_slide", "the first slide must be a title slide", deck.slides[0], "slide_type")

        sections, citations = set(req.section_ids), set(req.citation_ids)
        images, questions = set(req.image_artifact_ids), set(req.question_ids)
        asked: set[str] = set()
        for slide in deck.slides:
            if slide.slide_type != SlideType.TITLE and not slide.content_blocks:
                issue("empty_slide", "a content slide needs at least one content block", slide, "content_blocks")
            for sid in slide.section_refs:
                if sid not in sections:
                    issue("unknown_section", f"lesson section {sid} does not exist", slide, "section_refs")

            placed = [b.artifact_id for b in slide.blocks("image")]
            for aid in dict.fromkeys([*slide.visual_refs, *placed]):
                if aid not in images:
                    issue("unknown_image", f"{aid} is not an IMAGE_ASSET of this lesson", slide, "visual_refs")
            for aid in placed:
                if aid not in slide.visual_refs:
                    issue("image_not_declared", f"image block {aid} is not listed in visual_refs", slide,
                          "content_blocks")
            for aid in slide.visual_refs:
                if aid not in placed:
                    issue("unplaced_visual", f"visual_ref {aid} has no image block", slide, "visual_refs")

            cited = [*slide.citation_refs, *(c for b in slide.blocks("citations") for c in b.citation_ids)]
            for cid in dict.fromkeys(cited):
                if cid not in citations:
                    issue("unknown_citation", f"citation {cid} is not in the research bundle", slide,
                          "citation_refs")

            for block in slide.blocks("question"):
                if block.question_id not in questions:
                    issue("unknown_question", f"question {block.question_id} is not in the lesson", slide,
                          "content_blocks")
                asked.add(block.question_id)
            for block in slide.blocks("answer"):
                if block.question_id not in questions:
                    issue("unknown_question", f"answer to unknown question {block.question_id}", slide,
                          "content_blocks")
                elif block.question_id not in asked:
                    issue("answer_before_question", f"question {block.question_id} is answered before it is asked",
                          slide, "content_blocks")

            if placed and slide.layout not in LAYOUTS_ALLOWING_IMAGES:
                issue("layout_mismatch", f"layout {slide.layout.value} has no image region", slide, "layout")
            if slide.layout in IMAGE_LAYOUTS and not placed:
                issue("layout_mismatch", f"layout {slide.layout.value} needs an image block", slide, "layout")
            if len(placed) > 1:
                issue("layout_mismatch", "a slide places at most one image", slide, "content_blocks")
            text_blocks = [b for b in slide.content_blocks if b.kind != "image"]
            if slide.layout == SlideLayout.TWO_COLUMN and len(slide.content_blocks) < 2:
                issue("layout_mismatch", "a two-column slide needs two blocks", slide, "layout")
            if slide.layout == SlideLayout.FULL_IMAGE and len(text_blocks) > 1:
                issue("layout_mismatch", "a full-image slide has room for one caption block", slide, "layout")

            required = REQUIRED_BLOCK.get(slide.slide_type)
            if required and not slide.blocks(required):
                issue("type_mismatch", f"a {slide.slide_type.value} slide needs a {required} block", slide,
                      "slide_type")

            if len(slide.content_blocks) > MAX_BLOCKS_PER_SLIDE:
                issue("overcrowded", f"{len(slide.content_blocks)} blocks; at most {MAX_BLOCKS_PER_SLIDE}", slide,
                      "content_blocks")
            words = sum(block_words(b) for b in slide.content_blocks)
            if words > MAX_WORDS_PER_SLIDE:
                issue("overcrowded", f"{words} words of slide text; at most {MAX_WORDS_PER_SLIDE}", slide,
                      "content_blocks")
        return errors


class SlidePlanValidationTool(Tool[SlidePlanValidationRequest, SlidePlanValidationReport]):
    name = "slide_plan.validate"
    description = "Validate a slide deck plan deterministically: ids, ordering, references to lesson sections, " \
                  "IMAGE_ASSET artifacts, citations and questions, layouts, slide types and density."
    input_model = SlidePlanValidationRequest
    output_model = SlidePlanValidationReport

    def __init__(self, validator: SlidePlanValidator | None = None) -> None:
        self._validator = validator or SlidePlanValidator()

    async def run(self, data: SlidePlanValidationRequest, scope: ExecutionScope) -> SlidePlanValidationReport:
        report = self._validator.validate(data)
        if not data.enforce:
            return report
        if not report.valid:
            scope.emit(EventType.PRESENTATION_FAILED, tool=self.name, stage="validation", deck_id=report.deck_id,
                       errors=[e.model_dump(exclude_none=True) for e in report.errors])
            raise SlidePlanInvalid(report)
        assert report.deck is not None
        scope.emit(EventType.SLIDE_PLAN_VALIDATED, tool=self.name, deck_id=report.deck_id,
                   slides=len(report.deck.slides), checks=report.checks)
        return report
