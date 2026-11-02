"""Learner-facing artifact content: which artifact bytes may be served, and a streamed reader for them.

Artifacts can carry answer keys: a deck can have answer slides, a lesson's exercises and check questions hold the
answers that grade practice, an open teaching question is stored as an ASSESSMENT_ITEM. Serving content is therefore
block-by-default: a type is served only when it is in `SERVABLE` (with the media types it may be served as), and
every `ArtifactType` is classified in `BLOCKED` or `SERVABLE` (a test enforces it, so a new type stays blocked until
someone decides). A blocked artifact gets the same "not found" as one that does not exist.

Only raster IMAGE_ASSETs are servable (issue #26). The visual planner writes the search query or generation prompt an
image comes from, and it now plans from an answer-free brief (app/agents/visual/briefing.py): no exercises or check
questions, any text stating one of their answers dropped, and a plan stating one rejected. An image is served only
when its metadata records that planning (`PROVENANCE`): images made before the fix were planned from the whole lesson.
An SVG is text (it could spell an answer out and carries script), so it stays blocked. Allowing a type is a
deliberate, reviewed change."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import BinaryIO

from app.artifacts.service import EXTENSIONS, ArtifactService
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.visual import ANSWER_ISOLATED_PLANNING
from app.storage.repositories import NotFound

log = logging.getLogger(__name__)

CHUNK = 64 * 1024
_SAFE = re.compile(r"[^A-Za-z0-9_-]")  # the download name is the artifact id, made header-safe

# Type -> the media types it may be served as (see the module docstring). Never image/svg+xml.
SERVABLE: Mapping[ArtifactType, frozenset[str]] = {
    ArtifactType.IMAGE_ASSET: frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"}),
}
# Type -> what its stored metadata must prove before it is served, on top of SERVABLE.
PROVENANCE: Mapping[ArtifactType, Callable[[Artifact], bool]] = {
    ArtifactType.IMAGE_ASSET: lambda a: a.metadata.get("planning") == ANSWER_ISOLATED_PLANNING,
}

_LESSON_ANSWERS = "built from the lesson, whose exercises and check questions carry their answers"
_DECK = "built from the slide plan, which can hold answer slides"
_ANSWER_INPUTS = "written by a model that was given the lesson's exercise and check-question answers"
_INTERNAL = "internal planning or learner record, not learner material"
# Type -> why its content is not served.
BLOCKED: Mapping[ArtifactType, str] = {
    ArtifactType.LESSON: "holds exercises and check questions with their answers",
    ArtifactType.EXERCISE: "holds exercise answers",
    ArtifactType.ASSESSMENT_ITEM: "holds the item's answer key (an open teaching question is stored as one)",
    ArtifactType.ASSESSMENT_RUBRIC: "holds rubric criteria and expected answers",
    ArtifactType.ASSESSMENT_GRADE: "holds grading internals (matched answers, uncertainty reasons)",
    ArtifactType.ASSESSMENT_FEEDBACK: "grader-written text checked only on the PublicGrade path",
    ArtifactType.LEARNER_EVALUATION: "holds evaluation questions with their expected answers",
    ArtifactType.LEARNING_EVIDENCE: "holds graded answers with their expected answers",
    ArtifactType.INTERACTION_EVIDENCE: "holds graded session answers with their expected answers",
    ArtifactType.TEACHING_SESSION: "holds the session's open question and its answer key",
    ArtifactType.TEACHING_TURN: "holds turn state, including the open question's answer key",
    ArtifactType.TEACHING_SESSION_SUMMARY: "summarises graded questions and their answers",
    ArtifactType.SLIDE: _DECK,
    ArtifactType.SLIDE_PLAN: _DECK,
    ArtifactType.PRESENTATION: _DECK,
    ArtifactType.SCRIPT: _LESSON_ANSWERS,
    ArtifactType.AUDIO_PLAN: _LESSON_ANSWERS,
    ArtifactType.AUDIO_ASSET: "narration of the deck, which can hold answer slides",
    ArtifactType.PRESENTATION_TIMELINE: _DECK,
    ArtifactType.SUBTITLE: "narration text of the deck, which can hold answer slides",
    ArtifactType.VIDEO_PLAN: _DECK,
    ArtifactType.VIDEO: "rendered from the deck, which can hold answer slides",
    ArtifactType.VIDEO_SEGMENT_PLAN: _ANSWER_INPUTS,
    ArtifactType.GENERATED_VIDEO_ASSET: _ANSWER_INPUTS,
    ArtifactType.VISUAL_PLAN: _INTERNAL,
    ArtifactType.CODE: "unclassified generated content",
    ArtifactType.REPORT: "review report quoting the lesson draft",
    ArtifactType.LESSON_PLAN: _INTERNAL,
    ArtifactType.RESEARCH_BUNDLE: _INTERNAL,
    ArtifactType.LEARNER_MODEL: _INTERNAL,
    ArtifactType.KNOWLEDGE_GAPS: _INTERNAL,
    ArtifactType.PEDAGOGICAL_PLAN: _INTERNAL,
    ArtifactType.LEARNING_RECOMMENDATION: _INTERNAL,
    ArtifactType.LEARNING_GOAL: _INTERNAL,
    ArtifactType.CURRICULUM: _INTERNAL,
    ArtifactType.CURRICULUM_VERSION: _INTERNAL,
    ArtifactType.LEARNING_OBJECTIVE: _INTERNAL,
    ArtifactType.LEARNING_ACTION: _INTERNAL,
    ArtifactType.LEARNING_CYCLE: _INTERNAL,
}


class ArtifactContentNotFound(Exception):
    """No such artifact, or its content is not servable: deliberately the same answer (404)."""

    def __init__(self) -> None:
        super().__init__("not found")


class ArtifactContentUnavailable(Exception):
    """The artifact may be served but its stored object is missing or unreadable. The message is fixed: no location,
    key or digest reaches the caller (operators get the artifact id in the log)."""

    def __init__(self) -> None:
        super().__init__("artifact content is unavailable")


@dataclass(frozen=True)
class ArtifactContent:
    """An open, servable artifact: what the response needs and nothing else (no uri, hash or size)."""

    media_type: str
    filename: str
    stream: BinaryIO

    def chunks(self) -> Iterator[bytes]:
        """The content in chunks; closes the stream when done, also when the client goes away."""
        try:
            while chunk := self.stream.read(CHUNK):
                yield chunk
        finally:
            self.stream.close()


class ArtifactContentService:
    def __init__(self, artifacts: ArtifactService,
                 servable: Mapping[ArtifactType, frozenset[str]] = SERVABLE,
                 provenance: Mapping[ArtifactType, Callable[[Artifact], bool]] = PROVENANCE) -> None:
        self._artifacts = artifacts
        self._servable = servable
        self._provenance = provenance

    def servable(self, artifact: Artifact) -> bool:
        proven = self._provenance.get(artifact.type)
        return (artifact.media_type in self._servable.get(artifact.type, frozenset())
                and (proven is None or proven(artifact)))

    def open(self, artifact_id: str) -> ArtifactContent:
        """The content of a servable artifact, opened for streaming. The caller has already checked ownership."""
        try:
            artifact = self._artifacts.get(artifact_id)
        except (NotFound, KeyError):
            raise ArtifactContentNotFound() from None
        if not self.servable(artifact):
            raise ArtifactContentNotFound()
        try:
            stream = self._artifacts.open(artifact)
        except (OSError, ValueError):
            log.warning("artifact %s: stored object missing or unreadable", artifact.artifact_id)
            raise ArtifactContentUnavailable() from None
        return ArtifactContent(media_type=artifact.media_type, stream=stream,
                               filename=_SAFE.sub("_", artifact.artifact_id) + EXTENSIONS.get(artifact.media_type, ".bin"))
