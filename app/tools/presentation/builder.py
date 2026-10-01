"""PresentationBuilder: a validated SlideDeckPlan -> the renderer-independent Presentation.

It resolves what the plan only references: citation ids through the research bundle (Citation -> Evidence ->
Source) and image artifact ids through their IMAGE_ASSET metadata (bytes location, checksum, size, alt text and
attribution). It places every element in a named region of the slide's semantic layout; geometry, fonts and file
formats are the renderer's job. It never searches for or generates an image.
"""

from __future__ import annotations

import hashlib
import json

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.schemas.artifact import ArtifactType
from app.schemas.events import EventType
from app.schemas.presentation import (
    AnswerBlock,
    BulletBlock,
    BulletsElement,
    CitationBlock,
    ContentBlock,
    FooterElement,
    ImageBlock,
    ImageElement,
    Presentation,
    PresentationBuildRequest,
    PresentationConfig,
    PresentationElement,
    PresentationReference,
    PresentationSlide,
    QuestionBlock,
    Region,
    SlideDeckPlan,
    SlideLayout,
    SlidePlan,
    TableBlock,
    TableElement,
    TextBlock,
    TextElement,
    TitleElement,
    VocabularyBlock,
)
from app.schemas.research import ResearchBundle
from app.schemas.visual import ImageAsset
from app.tools.base import Tool, ToolError


class PresentationBuildError(ToolError):
    pass


def image_credit(asset: ImageAsset) -> str:
    """The attribution line for a placed image, from the IMAGE_ASSET metadata only."""
    a = asset.attribution
    if a.kind == "generated":
        return f"Generated image ({a.provider}/{a.model})"
    if a.attribution_text:
        return a.attribution_text
    licence = a.license.name if a.license else None
    return ", ".join(p for p in (a.title, a.creator, a.publisher, licence) if p)


class PresentationBuilder:
    name = "presentation-builder/1"

    def build(self, deck: SlideDeckPlan, *, config: PresentationConfig, research: ResearchBundle,
              images: dict[str, ImageAsset]) -> Presentation:
        numbers: dict[str, int] = {}
        references: list[PresentationReference] = []
        for number, cid in enumerate(deck.citation_ids(), start=1):
            try:
                citation, evidence, source = research.resolve(cid)
            except LookupError as exc:
                raise PresentationBuildError(f"citation {cid} does not resolve in the research bundle") from exc
            numbers[cid] = number
            references.append(PresentationReference(number=number, citation_id=cid, evidence_id=evidence.evidence_id,
                                                    source_id=source.source_id, text=citation.reference()))
        missing = [a for a in deck.image_artifact_ids() if a not in images]
        if missing:
            raise PresentationBuildError(f"image artifacts {missing} were not provided")
        titles = {r.citation_id: research.resolve(r.citation_id)[0].title for r in references}

        slides = [self._slide(s, config, numbers, titles, references, images) for s in deck.slides]
        body = json.dumps({"deck": deck.model_dump(mode="json"), "config": config.model_dump(mode="json"),
                           "builder": self.name}, sort_keys=True)
        return Presentation(
            presentation_id="pres_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16], deck_id=deck.deck_id,
            title=deck.title, language=deck.language, config=config, slides=slides, references=references,
            builder=self.name,
            metadata={"level": deck.level, "topic": deck.topic, "objective": deck.objective},
        )

    def _slide(self, slide: SlidePlan, config: PresentationConfig, numbers: dict[str, int], titles: dict[str, str],
               references: list[PresentationReference], images: dict[str, ImageAsset]) -> PresentationSlide:
        counter = iter(range(1, 1000))

        def eid(kind: str) -> str:
            return f"{slide.slide_id}.{kind}{next(counter)}"

        elements: list[PresentationElement] = [TitleElement(element_id=eid("title"), region="title", text=slide.title)]
        if slide.subtitle:
            elements.append(TextElement(element_id=eid("subtitle"), region="subtitle", text=slide.subtitle,
                                        role="subtitle"))
        has_image = any(b.kind == "image" for b in slide.content_blocks)
        text_blocks = [b for b in slide.content_blocks if b.kind != "image"]
        for block in slide.content_blocks:
            region = self._region(slide.layout, block, text_blocks, has_image)
            elements += self._elements(block, region, eid, numbers, references, images)

        placed = [b.artifact_id for b in slide.blocks("image")]
        cited = [*slide.citation_refs, *(c for b in slide.blocks("citations") for c in b.citation_ids)]
        footer = config.footer
        parts = [footer.text] if footer.text else []
        if footer.show_citations and slide.citation_refs:
            parts.append("Sources: " + "; ".join(f"[{numbers[c]}] {titles[c]}" for c in slide.citation_refs))
        if footer.show_image_credits and placed:
            parts.append("Image: " + "; ".join(image_credit(images[a]) for a in placed))
        if parts:
            elements.append(FooterElement(element_id=eid("footer"), text=" | ".join(parts),
                                          citation_ids=list(slide.citation_refs) if footer.show_citations else [],
                                          image_artifact_ids=placed if footer.show_image_credits else []))
        return PresentationSlide(
            slide_id=slide.slide_id, order=slide.order, slide_type=slide.slide_type, layout=slide.layout,
            elements=elements, notes=slide.speaker_notes, citation_ids=list(dict.fromkeys(cited)),
            image_artifact_ids=placed, section_ids=list(slide.section_refs),
        )

    @staticmethod
    def _region(layout: SlideLayout, block: ContentBlock, text_blocks: list[ContentBlock], has_image: bool) -> Region:
        if layout == SlideLayout.TWO_COLUMN:
            if block.kind == "image":
                return "right"
            return "left" if has_image or block is text_blocks[0] else "right"
        if block.kind == "image":
            return "image"
        if layout == SlideLayout.FULL_IMAGE:
            return "caption"
        return "body"

    @staticmethod
    def _elements(block: ContentBlock, region: Region, eid, numbers: dict[str, int],
                  references: list[PresentationReference], images: dict[str, ImageAsset]) -> list[PresentationElement]:
        match block:
            case TextBlock():
                return [TextElement(element_id=eid("text"), region=region, text=block.text)]
            case BulletBlock():
                return [BulletsElement(element_id=eid("bullets"), region=region, items=list(block.items))]
            case TableBlock():
                return [TableElement(element_id=eid("table"), region=region, headers=list(block.headers),
                                     rows=[list(r) for r in block.rows])]
            case VocabularyBlock():
                with_examples = any(e.example for e in block.entries)
                headers = ["Term", "Meaning", *(["Example"] if with_examples else [])]
                rows = [[e.term, e.meaning, *([e.example or ""] if with_examples else [])] for e in block.entries]
                return [TableElement(element_id=eid("table"), region=region, headers=headers, rows=rows)]
            case QuestionBlock():
                out: list[PresentationElement] = [TextElement(element_id=eid("question"), region=region,
                                                              text=block.prompt, role="question")]
                if block.choices:
                    out.append(BulletsElement(element_id=eid("choices"), region=region, items=list(block.choices),
                                              numbered=True))
                return out
            case AnswerBlock():
                text = block.answer + (f" ({block.explanation})" if block.explanation else "")
                return [TextElement(element_id=eid("answer"), region=region, text=text, role="answer")]
            case CitationBlock():
                by_id = {r.citation_id: r for r in references}
                items = [f"[{numbers[c]}] {by_id[c].text}" for c in block.citation_ids]
                return [BulletsElement(element_id=eid("references"), region=region, items=items)]
            case ImageBlock():
                asset = images[block.artifact_id]
                out = [ImageElement(
                    element_id=eid("image"), region=region, artifact_id=block.artifact_id, asset_id=asset.asset_id,
                    object_uri=asset.object.uri, checksum=asset.object.checksum, media_type=asset.format,
                    width=asset.width, height=asset.height, alt_text=asset.description,
                )]
                if block.caption:
                    out.append(TextElement(element_id=eid("caption"), region="caption", text=block.caption,
                                           role="caption"))
                return out
        raise PresentationBuildError(f"unsupported content block {block.kind}")


class PresentationBuildTool(Tool[PresentationBuildRequest, Presentation]):
    name = "presentation.build"
    description = "Build the renderer-independent presentation from a validated slide plan, resolving citations " \
                  "from the research bundle and placed images from their IMAGE_ASSET artifacts."
    input_model = PresentationBuildRequest
    output_model = Presentation
    permissions = frozenset({"artifact:read"})

    def __init__(self, artifacts: ArtifactService, builder: PresentationBuilder | None = None) -> None:
        self._artifacts = artifacts
        self._builder = builder or PresentationBuilder()

    async def run(self, data: PresentationBuildRequest, scope: ExecutionScope) -> Presentation:
        deck = data.deck
        scope.emit(EventType.PRESENTATION_BUILD_STARTED, tool=self.name, deck_id=deck.deck_id, slides=len(deck.slides),
                   slide_plan_artifact_id=data.slide_plan_artifact_id)
        try:
            presentation = self._builder.build(deck, config=data.config, research=data.research,
                                               images=self._images(deck))
        except (ToolError, LookupError, ValueError) as exc:
            scope.emit(EventType.PRESENTATION_FAILED, tool=self.name, stage="build", deck_id=deck.deck_id,
                       error=str(exc)[:1000])
            if isinstance(exc, PresentationBuildError):
                raise
            raise PresentationBuildError(f"cannot build presentation: {exc}") from exc
        scope.emit(EventType.PRESENTATION_BUILD_COMPLETED, tool=self.name, deck_id=deck.deck_id,
                   presentation_id=presentation.presentation_id, slides=len(presentation.slides),
                   elements=presentation.element_count(), image_artifact_ids=presentation.image_artifact_ids(),
                   citation_ids=[r.citation_id for r in presentation.references])
        return presentation

    def _images(self, deck: SlideDeckPlan) -> dict[str, ImageAsset]:
        images = {}
        for artifact_id in deck.image_artifact_ids():
            artifact = self._artifacts.get(artifact_id)
            if artifact.type != ArtifactType.IMAGE_ASSET:
                raise PresentationBuildError(f"{artifact_id} is a {artifact.type.value}, not an IMAGE_ASSET")
            images[artifact_id] = ImageAsset.model_validate(artifact.metadata)
        return images
