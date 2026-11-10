"""The learner's reading of a generated lesson: the lesson player's text, built by code from the stored lesson.

The LESSON artifact is blocked for learners (it holds exercises and check questions with their answers), and the
deck, narration and video are blocked too (they can hold answer slides). The reading is the learner-safe view of
the teaching text:

- the sections' explanations, examples and analogies (the text the teacher also teaches from in a session);
- never the exercises, check questions, narration, plan, research bundle or review;
- text is checked sentence by sentence (examples, analogies, headings and image descriptions as a whole) against
  every answer the learner is graded on with this lesson: the lesson's exercise and check-question answers and the
  assessment items registered on the lesson, always; and, while any question is open for the learner (a teaching
  session's pending question, a task waiting on a diagnostic or evaluation sheet), that question's answer too, so the
  reading is never an open book on a question being answered;
- images only when the artifact content policy would serve them (answer-isolated planning).

What was held back is counted (`withheld`), so the app can say so. Answers never leave this module.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import urlsplit

from app.agents.visual.briefing import answer_keys
from app.artifacts.service import ArtifactService
from app.schemas.artifact import ArtifactType
from app.schemas.reading import LessonReading, ReadingImage, ReadingReference, ReadingSection
from app.schemas.task import TaskStatus
from app.services.artifact_content import ArtifactContentService
from app.services.assessment import AssessmentService
from app.services.lessons import LessonMaterial, load_lesson
from app.services.tasks import TaskService
from app.services.teaching import TeachingSessionService
from app.storage.repositories import NotFound
from app.utils.text import normalize, reveals

ANSWER_FIELDS = frozenset({"expected_answer", "accepted_answers", "acceptable_answers"})
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str | None) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if s.strip()]


def answer_strings(value) -> list[str]:
    """Every string under an answer-key field, at any depth of a dumped record."""
    out: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k in ANSWER_FIELDS:
                out += [v] if isinstance(v, str) else [x for x in v or [] if isinstance(x, str)]
            else:
                out += answer_strings(v)
    elif isinstance(value, list):
        for v in value:
            out += answer_strings(v)
    return out


class _Filter:
    def __init__(self, answers: Iterable[str]) -> None:
        self.answers = sorted({a for a in answers if normalize(a)})
        self.withheld = 0

    def text(self, value: str | None) -> str | None:
        if not value:
            return None
        if reveals(value, self.answers):
            self.withheld += 1
            return None
        return value

    def sentences(self, value: str | None) -> list[str]:
        return [s for s in sentences(value) if self.text(s) is not None]

    def items(self, values: Iterable[str]) -> list[str]:
        return [v for v in values if self.text(v) is not None]


class LessonReadingService:
    def __init__(self, *, artifacts: ArtifactService, tasks: TaskService, teaching: TeachingSessionService,
                 assessment: AssessmentService, content: ArtifactContentService) -> None:
        self._artifacts = artifacts
        self._tasks = tasks
        self._teaching = teaching
        self._assessment = assessment
        self._content = content

    def open_answers(self, learner_id: str) -> list[str]:
        """The answers of every question open for the learner now: pending teaching-session questions and the sheets
        of tasks waiting for the learner (their internal records hold the keys the waiting sheet is graded with)."""
        answers: list[str] = []
        for session in self._teaching.sessions(learner_id):
            q = session.state.pending_question
            if not session.terminal and q is not None:
                answers += [q.expected_answer, *q.accepted_answers]
        for task in self._tasks.list_for_learner(learner_id):
            if task.status == TaskStatus.WAITING:
                answers += answer_strings(task.model_dump(mode="json", include={"workflow", "waiting"}))
        return answers

    def reading(self, lesson_id: str) -> LessonReading:
        m = load_lesson(self._artifacts, self._tasks, lesson_id)
        items = self._assessment.items_for_lesson(m.artifact.artifact_id, m.task.task_id)
        always = [*answer_keys(m.lesson), *(a for i in items for a in [i.expected_answer, *i.acceptable_answers])]
        open_now = [a for a in self.open_answers(m.task.learner_id) if normalize(a)]
        reading = self._build(m, _Filter([*always, *open_now]))
        if open_now:  # did an open question hold anything back? Then the app offers the reading again afterwards
            reading.question_open = reading.withheld > self._build(m, _Filter(always)).withheld
        return reading

    def _build(self, m: LessonMaterial, f: _Filter) -> LessonReading:
        lesson = m.lesson
        sections = []
        for s in lesson.sections:
            section = ReadingSection(
                section_id=s.section_id, purpose=s.purpose, heading=f.text(s.heading),
                paragraphs=f.sentences(s.explanation), examples=f.items(s.examples), analogy=f.text(s.analogy),
                images=self._images(s.visuals, m.task.task_id, f))
            if section.paragraphs or section.examples or section.analogy or section.images:
                sections.append(section)
        references: list[ReadingReference] = []
        for c in lesson.references:
            r = _reference(c.title, c.publisher, c.url, f)
            if r is not None and r not in references:
                references.append(r)
        return LessonReading(
            lesson_id=m.artifact.artifact_id, task_id=m.task.task_id, title=f.text(lesson.title),
            level=f.text(lesson.level), objectives=f.items(o.description for o in lesson.objectives),
            introduction=f.sentences(lesson.introduction), sections=sections, summary=f.sentences(lesson.summary),
            references=references, withheld=f.withheld)

    def _images(self, visuals, task_id: str, f: _Filter) -> list[ReadingImage]:
        out = []
        for v in visuals:
            try:
                artifact = self._artifacts.get(v.artifact_id)
            except (NotFound, KeyError):
                continue
            if artifact.type != ArtifactType.IMAGE_ASSET or artifact.task_id != task_id \
                    or not self._content.servable(artifact):
                f.withheld += 1
                continue
            out.append(ReadingImage(artifact_id=artifact.artifact_id, description=f.text(v.description)))
        return out


def _reference(title: str, publisher: str | None, url: str | None, f: _Filter) -> ReadingReference | None:
    shown = f.text(title)
    if shown is None:
        return None
    link = url if url and urlsplit(url).scheme in ("http", "https") and f.text(url) else None
    return ReadingReference(title=shown, publisher=f.text(publisher), url=link)

