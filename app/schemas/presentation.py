"""Presentation schemas. Domain- and renderer-independent.

Three representations, kept apart on purpose:

- SlideDeckPlan: the semantic plan (WHAT appears on each slide). Structured content blocks that reference
  lesson sections, research citation ids and IMAGE_ASSET artifact ids; no geometry, fonts or file formats.
- Presentation: the renderer-independent intermediate built from a validated plan. Each slide is a list of
  elements placed in named layout regions; citations and image references are resolved, still no geometry.
- RenderedPresentation (in the provider contract): the concrete file a renderer produced from a Presentation.

Chains: Slide -> Citation -> Evidence -> Source (textual, through the research bundle) and, separately,
Slide -> IMAGE_ASSET -> source or generation metadata (visual attribution). A slide only holds ids; the
metadata is resolved from the research bundle and the IMAGE_ASSET artifacts when the presentation is built.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from app.schemas.common import Schema
from app.schemas.lesson import LessonContent, LessonPlan, LessonRequest
from app.schemas.research import ResearchBundle
from app.schemas.visual import VisualAssetRef, VisualPlan, VisualType, aspect_matches, aspect_value

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,64}$")]

MAX_BULLETS_PER_BLOCK = 6
MAX_WORDS_PER_BULLET = 20
MAX_BLOCKS_PER_SLIDE = 4
MAX_WORDS_PER_SLIDE = 120
MAX_TABLE_ROWS = 8
MAX_TABLE_COLUMNS = 4
MAX_SLIDES = 30


# --- Slide types and layouts ------------------------------------------------------------------


class SlideType(str, Enum):
    """What a slide is for. Add members as new kinds are needed; nothing here is subject-specific."""

    TITLE = "title"
    OBJECTIVES = "objectives"
    EXPLANATION = "explanation"
    EXAMPLE = "example"
    COMPARISON = "comparison"
    VOCABULARY = "vocabulary"
    EXERCISE = "exercise"
    ANSWER = "answer"
    SUMMARY = "summary"
    REFERENCES = "references"


class SlideLayout(str, Enum):
    """Semantic layouts. A renderer maps each one to concrete geometry; the plan never carries coordinates."""

    TITLE = "title"
    TITLE_CONTENT = "title_content"
    TWO_COLUMN = "two_column"
    IMAGE_TEXT = "image_text"
    FULL_IMAGE = "full_image"
    EXERCISE = "exercise"
    SUMMARY = "summary"


IMAGE_LAYOUTS = frozenset({SlideLayout.IMAGE_TEXT, SlideLayout.FULL_IMAGE})
LAYOUTS_ALLOWING_IMAGES = IMAGE_LAYOUTS | {SlideLayout.TWO_COLUMN}

# The block a slide type cannot do without (title, objectives, explanation, ... are free-form).
REQUIRED_BLOCK: dict[SlideType, str] = {
    SlideType.VOCABULARY: "vocabulary",
    SlideType.EXERCISE: "question",
    SlideType.ANSWER: "answer",
    SlideType.REFERENCES: "citations",
}


def word_count(text: str) -> int:
    return len(text.split())


# --- Content blocks ---------------------------------------------------------------------------


class TextBlock(Schema):
    kind: Literal["text"] = "text"
    text: Text


class BulletBlock(Schema):
    kind: Literal["bullets"] = "bullets"
    items: list[Text] = Field(min_length=1, max_length=MAX_BULLETS_PER_BLOCK)

    @field_validator("items")
    @classmethod
    def _short(cls, items: list[str]) -> list[str]:
        for item in items:
            if word_count(item) > MAX_WORDS_PER_BULLET:
                raise ValueError(f"bullet exceeds {MAX_WORDS_PER_BULLET} words: {item[:40]}...")
        return items


class TableBlock(Schema):
    kind: Literal["table"] = "table"
    headers: list[Text] = Field(min_length=1, max_length=MAX_TABLE_COLUMNS)
    rows: list[list[Text]] = Field(min_length=1, max_length=MAX_TABLE_ROWS)

    @model_validator(mode="after")
    def _rectangular(self) -> TableBlock:
        if any(len(row) != len(self.headers) for row in self.rows):
            raise ValueError("every table row needs one cell per header")
        return self


class QuestionBlock(Schema):
    """An exercise or check question from the lesson, referenced by its id."""

    kind: Literal["question"] = "question"
    question_id: Identifier
    prompt: Text
    choices: list[Text] = Field(default_factory=list, max_length=MAX_BULLETS_PER_BLOCK)


class AnswerBlock(Schema):
    kind: Literal["answer"] = "answer"
    question_id: Identifier
    answer: Text
    explanation: str = ""


class VocabularyEntry(Schema):
    term: Text
    meaning: Text
    example: str | None = None


class VocabularyBlock(Schema):
    kind: Literal["vocabulary"] = "vocabulary"
    entries: list[VocabularyEntry] = Field(min_length=1, max_length=MAX_TABLE_ROWS)


class CitationBlock(Schema):
    """The full references for citation ids from the research bundle (used on a references slide)."""

    kind: Literal["citations"] = "citations"
    citation_ids: list[Identifier] = Field(min_length=1)


class ImageBlock(Schema):
    """Places an existing IMAGE_ASSET artifact. Only its id: size, alt text and attribution stay on the artifact."""

    kind: Literal["image"] = "image"
    artifact_id: Identifier
    caption: str | None = None


ContentBlock = Annotated[
    TextBlock | BulletBlock | TableBlock | QuestionBlock | AnswerBlock | VocabularyBlock | CitationBlock | ImageBlock,
    Field(discriminator="kind"),
]


def block_words(block: ContentBlock) -> int:
    match block:
        case TextBlock():
            return word_count(block.text)
        case BulletBlock():
            return sum(word_count(i) for i in block.items)
        case TableBlock():
            return sum(word_count(c) for c in block.headers) + sum(word_count(c) for r in block.rows for c in r)
        case QuestionBlock():
            return word_count(block.prompt) + sum(word_count(c) for c in block.choices)
        case AnswerBlock():
            return word_count(block.answer) + word_count(block.explanation)
        case VocabularyBlock():
            return sum(word_count(e.term) + word_count(e.meaning) + word_count(e.example or "") for e in block.entries)
        case ImageBlock():
            return word_count(block.caption or "")
    return 0  # citation lists are rendered as references, not counted as slide text


# --- The semantic plan ------------------------------------------------------------------------


class SlidePlan(Schema):
    slide_id: Identifier
    order: int = Field(ge=1)
    slide_type: SlideType
    title: Text
    subtitle: str | None = None
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    visual_refs: list[Identifier] = Field(default_factory=list)  # IMAGE_ASSET artifact ids this slide places
    citation_refs: list[Identifier] = Field(default_factory=list)  # research citation ids supporting this slide
    section_refs: list[Identifier] = Field(default_factory=list)  # lesson section ids this slide presents
    speaker_notes: str = ""
    layout: SlideLayout
    duration_hint: int = Field(default=60, ge=5, le=1800)  # seconds

    def blocks(self, kind: str) -> list[ContentBlock]:
        return [b for b in self.content_blocks if b.kind == kind]


class SlideDeckProposal(Schema):
    """What the model proposes. The system assigns the deck id and the deck's lesson metadata."""

    title: Text
    slides: list[SlidePlan] = Field(min_length=1)
    rationale: str = ""


class SlideDeckPlan(Schema):
    deck_id: Identifier
    title: Text
    language: Text
    level: Text
    topic: Text
    objective: Text
    slides: list[SlidePlan] = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)

    def image_artifact_ids(self) -> list[str]:
        return list(dict.fromkeys(b.artifact_id for s in self.slides for b in s.blocks("image")))

    def citation_ids(self) -> list[str]:
        """Every citation id the deck uses, in order of first appearance."""
        ids = [c for s in self.slides for c in
               [*s.citation_refs, *(c for b in s.blocks("citations") for c in b.citation_ids)]]
        return list(dict.fromkeys(ids))


# --- Slide planning input -----------------------------------------------------------------------


class SlideVisualOption(Schema):
    """An IMAGE_ASSET the planner may place. What it shows, never where its bytes are."""

    artifact_id: str
    visual_id: str
    lesson_section_id: str
    visual_type: VisualType
    purpose: str
    description: str
    width: int
    height: int


class CitationOption(Schema):
    citation_id: str
    title: str
    publisher: str | None = None
    section_ids: list[str] = Field(default_factory=list)  # the lesson sections citing it


class SlidePlanningInput(Schema):
    """What the model sees when planning slides."""

    topic: str
    language: str
    level: str
    plan: LessonPlan
    lesson: LessonContent
    citations: list[CitationOption] = Field(default_factory=list)
    visuals: list[SlideVisualOption] = Field(default_factory=list)
    max_slides: int = Field(ge=2)
    corrections: list[str] = Field(default_factory=list)  # validation errors of the previous proposal


class SlidePlanningRequest(Schema):
    """Input of the SlidePlannerAgent: an approved lesson and what it may draw on."""

    request: LessonRequest
    plan: LessonPlan
    lesson: LessonContent
    research: ResearchBundle
    visual_plan: VisualPlan | None = None
    image_assets: list[VisualAssetRef] = Field(default_factory=list)
    max_slides: int = Field(default=20, ge=2, le=MAX_SLIDES)

    def planning_input(self, corrections: list[str] | None = None) -> SlidePlanningInput:
        cited: dict[str, list[str]] = {}
        for section in self.lesson.sections:
            for cid in section.citations:
                cited.setdefault(cid, []).append(section.section_id)
        return SlidePlanningInput(
            topic=self.request.topic, language=self.request.language_of_instruction, level=self.lesson.level,
            plan=self.plan, lesson=self.lesson,
            citations=[CitationOption(citation_id=c.citation_id, title=c.title, publisher=c.publisher,
                                      section_ids=cited.get(c.citation_id, []))
                       for c in self.research.citations],
            visuals=[SlideVisualOption(artifact_id=a.artifact_id, visual_id=a.visual_id,
                                       lesson_section_id=a.lesson_section_id, visual_type=a.visual_type,
                                       purpose=a.purpose, description=a.description, width=a.width,
                                       height=a.height) for a in self.image_assets],
            max_slides=self.max_slides, corrections=corrections or [],
        )

    def question_ids(self) -> list[str]:
        return [e.exercise_id for e in self.lesson.exercises] + [q.question_id for q in self.lesson.check_questions]

    def validation_request(self, deck: SlideDeckPlan, *, enforce: bool = False) -> SlidePlanValidationRequest:
        return SlidePlanValidationRequest(
            deck=deck.model_dump(mode="json"), section_ids=[s.section_id for s in self.lesson.sections],
            citation_ids=[c.citation_id for c in self.research.citations],
            image_artifact_ids=[a.artifact_id for a in self.image_assets], question_ids=self.question_ids(),
            max_slides=self.max_slides, enforce=enforce,
        )


# --- Deterministic validation ------------------------------------------------------------------

SlidePlanIssueCode = Literal[
    "invalid_schema", "no_slides", "too_many_slides", "duplicate_slide_id", "slide_order", "missing_title_slide",
    "empty_slide", "unknown_section", "unknown_image", "image_not_declared", "unplaced_visual", "unknown_citation",
    "unknown_question", "answer_before_question", "layout_mismatch", "type_mismatch", "overcrowded",
]


class SlidePlanIssue(Schema):
    code: SlidePlanIssueCode
    message: str
    slide_id: str | None = None
    field: str | None = None

    def describe(self) -> str:
        where = f"slide {self.slide_id}" if self.slide_id else "deck"
        return f"[{self.code}] {where}{f' ({self.field})' if self.field else ''}: {self.message}"


class SlidePlanValidationRequest(Schema):
    """A deck (as raw JSON, so malformed blocks or slide types are reported, not crashed on) and the ids it may
    reference. With `enforce`, an invalid deck is an error instead of a report."""

    deck: dict
    section_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    image_artifact_ids: list[str] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)
    max_slides: int = Field(default=MAX_SLIDES, ge=1)
    enforce: bool = False


class SlidePlanValidationReport(Schema):
    valid: bool
    deck_id: str | None = None
    errors: list[SlidePlanIssue] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)  # every check that ran
    validator: str
    deck: SlideDeckPlan | None = None  # the parsed deck, when it parsed

    @model_validator(mode="after")
    def _consistent(self) -> SlidePlanValidationReport:
        if self.valid == bool(self.errors):
            raise ValueError("a report is valid exactly when it has no errors")
        if self.valid and self.deck is None:
            raise ValueError("a valid report carries the deck")
        return self


# --- Configuration and theme (renderer-independent) ----------------------------------------------


class TypographyScale(Schema):
    """Font sizes in points."""

    title: float = Field(default=36, gt=0)
    subtitle: float = Field(default=22, gt=0)
    heading: float = Field(default=30, gt=0)
    body: float = Field(default=20, gt=0)
    caption: float = Field(default=12, gt=0)
    footer: float = Field(default=10, gt=0)


class ThemeFonts(Schema):
    heading: str = "Calibri"
    body: str = "Calibri"


class ThemeSpacing(Schema):
    """Distances in points."""

    margin: float = Field(default=36, ge=0)
    gutter: float = Field(default=24, ge=0)
    title_height: float = Field(default=64, gt=0)
    footer_height: float = Field(default=30, ge=0)
    block_gap: float = Field(default=10, ge=0)
    paragraph_after: float = Field(default=6, ge=0)


class ThemeColors(Schema):
    """Hex RGB without '#'."""

    background: str = Field(default="FFFFFF", pattern=r"^[0-9A-Fa-f]{6}$")
    text: str = Field(default="1F2933", pattern=r"^[0-9A-Fa-f]{6}$")
    accent: str = Field(default="2563EB", pattern=r"^[0-9A-Fa-f]{6}$")
    muted: str = Field(default="6B7280", pattern=r"^[0-9A-Fa-f]{6}$")


class PresentationTheme(Schema):
    name: str = "default"
    fonts: ThemeFonts = Field(default_factory=ThemeFonts)
    typography: TypographyScale = Field(default_factory=TypographyScale)
    spacing: ThemeSpacing = Field(default_factory=ThemeSpacing)
    colors: ThemeColors = Field(default_factory=ThemeColors)
    border_width: float = Field(default=0.0, ge=0)  # points, around images, where the renderer supports it
    corner_radius: float = Field(default=0.0, ge=0)  # points, where the renderer supports it


class FooterConfig(Schema):
    show_slide_numbers: bool = True
    show_citations: bool = True
    show_image_credits: bool = True
    text: str | None = None  # a fixed footer line, e.g. a course name


ASPECT_SIZES = {"16:9": (960.0, 540.0), "4:3": (720.0, 540.0)}  # points: 13.333x7.5in and 10x7.5in


class PresentationConfig(Schema):
    aspect_ratio: str = "16:9"
    width: float = Field(default=960.0, gt=0)  # points
    height: float = Field(default=540.0, gt=0)  # points
    language: str = "en"
    theme: PresentationTheme = Field(default_factory=PresentationTheme)
    footer: FooterConfig = Field(default_factory=FooterConfig)

    @model_validator(mode="after")
    def _size_matches_ratio(self) -> PresentationConfig:
        aspect_value(self.aspect_ratio)
        if not aspect_matches(round(self.width), round(self.height), self.aspect_ratio):
            raise ValueError(f"{self.width}x{self.height}pt does not have aspect ratio {self.aspect_ratio}")
        return self

    @classmethod
    def for_aspect(cls, aspect_ratio: str = "16:9", **kwargs) -> PresentationConfig:
        width, height = ASPECT_SIZES[aspect_ratio]
        return cls(aspect_ratio=aspect_ratio, width=width, height=height, **kwargs)


# --- The intermediate, renderer-independent presentation ----------------------------------------

Region = Literal["title", "subtitle", "body", "left", "right", "image", "caption", "footer"]


class TitleElement(Schema):
    kind: Literal["title"] = "title"
    element_id: str
    region: Region
    text: str


class TextElement(Schema):
    kind: Literal["text"] = "text"
    element_id: str
    region: Region
    text: str
    role: Literal["body", "subtitle", "question", "answer", "caption"] = "body"


class BulletsElement(Schema):
    kind: Literal["bullets"] = "bullets"
    element_id: str
    region: Region
    items: list[str] = Field(min_length=1)
    numbered: bool = False


class TableElement(Schema):
    kind: Literal["table"] = "table"
    element_id: str
    region: Region
    headers: list[str] = Field(min_length=1)
    rows: list[list[str]] = Field(min_length=1)


class ImageElement(Schema):
    """An IMAGE_ASSET placed on a slide, resolved from the artifact when the presentation is built."""

    kind: Literal["image"] = "image"
    element_id: str
    region: Region
    artifact_id: str
    asset_id: str
    object_uri: str  # where the object store keeps the bytes; the renderer reads them, never generates any
    checksum: str  # sha256 the bytes must match
    media_type: str
    width: int
    height: int
    alt_text: str


class FooterElement(Schema):
    kind: Literal["footer"] = "footer"
    element_id: str
    region: Region = "footer"
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    image_artifact_ids: list[str] = Field(default_factory=list)  # images credited in this footer


PresentationElement = Annotated[
    TitleElement | TextElement | BulletsElement | TableElement | ImageElement | FooterElement,
    Field(discriminator="kind"),
]


class PresentationSlide(Schema):
    slide_id: str
    order: int = Field(ge=1)
    slide_type: SlideType
    layout: SlideLayout
    elements: list[PresentationElement] = Field(min_length=1)
    notes: str = ""
    citation_ids: list[str] = Field(default_factory=list)
    image_artifact_ids: list[str] = Field(default_factory=list)
    section_ids: list[str] = Field(default_factory=list)

    def element(self, kind: str) -> list[PresentationElement]:
        return [e for e in self.elements if e.kind == kind]


class PresentationReference(Schema):
    """A citation as it appears on the slides: its number in this deck and the resolved reference."""

    number: int = Field(ge=1)
    citation_id: str
    evidence_id: str
    source_id: str
    text: str


class Presentation(Schema):
    presentation_id: str
    deck_id: str
    title: str
    language: str
    config: PresentationConfig
    slides: list[PresentationSlide] = Field(min_length=1)
    references: list[PresentationReference] = Field(default_factory=list)
    builder: str
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ordered(self) -> Presentation:
        if [s.order for s in self.slides] != list(range(1, len(self.slides) + 1)):
            raise ValueError("slides must be ordered 1..n")
        ids = [e.element_id for s in self.slides for e in s.elements]
        if len(ids) != len(set(ids)):
            raise ValueError("element ids must be unique")
        return self

    def element_count(self) -> int:
        return sum(len(s.elements) for s in self.slides)

    def image_artifact_ids(self) -> list[str]:
        return list(dict.fromkeys(a for s in self.slides for a in s.image_artifact_ids))


class PresentationBuildRequest(Schema):
    deck: SlideDeckPlan
    slide_plan_artifact_id: str
    config: PresentationConfig = Field(default_factory=PresentationConfig)
    research: ResearchBundle  # citations are resolved Citation -> Evidence -> Source from here


class PresentationRenderRequest(Schema):
    presentation: Presentation
    name: str = "presentation"
    slide_plan_artifact_id: str
    lesson_artifact_id: str
    parent_ids: list[str] = Field(default_factory=list)


class PresentationArtifactMetadata(Schema):
    """Metadata of a PRESENTATION artifact. The file itself is a content-addressed object in the object store;
    the artifact id (not a filesystem path) is its identity."""

    presentation_id: str
    deck_id: str
    title: str
    language: str
    slide_plan_artifact_id: str
    lesson_artifact_id: str
    renderer: str
    format: str
    media_type: str
    checksum: str  # sha256 of the file
    size_bytes: int
    object_key: str  # storage-relative key in the object store
    slides: int
    elements: int
    image_artifact_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    layouts: dict[str, int] = Field(default_factory=dict)
