"""The visual planner's brief: the only lesson material the visual model sees.

An image is found or generated from the search query or generation prompt the planner writes, so anything in the
planner's input can end up in a learner-downloadable image. The brief is therefore built from learner-facing teaching
text only (title, subject, topic, level, objectives and the teaching sections), never from the lesson's exercises or
check questions, their answers or explanations, the lesson plan or the research bundle. Section ids are replaced by
opaque refs. Any brief field that states an exercise or check-question answer (`reveals`, whole words after
normalisation) is dropped, and a section whose concept name states one is left out: it cannot be illustrated without
showing the answer. The answers stay with the agent's code, which also rejects a plan that states one.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.lesson import LessonContent, SectionPurpose, VisualPlanningInput, VisualRequest, VisualSectionBrief
from app.schemas.visual import VisualPlan
from app.utils.text import normalize, reveals

# Sections whose text teaches. Practice and assessment sections can hold worked answers, so they get no visuals.
TEACHING_PURPOSES: frozenset[SectionPurpose] = frozenset({"explanation", "example", "review"})


def answer_keys(lesson: LessonContent) -> list[str]:
    """Every answer the lesson grades practice with: exercise and check-question answers."""
    answers = [e.answer for e in lesson.exercises] + [q.answer for q in lesson.check_questions]
    return sorted({a for a in answers if normalize(a)})


@dataclass(frozen=True)
class VisualBrief:
    """The planner's input, plus what only code may know: the real section behind each ref and the answers."""

    planning: VisualPlanningInput
    sections: dict[str, str]  # ref -> lesson section id
    answers: tuple[str, ...]

    def leaks(self, text: str | None) -> bool:
        return bool(text) and reveals(text, self.answers)

    def isolated(self, plan: VisualPlan) -> bool:
        """Does no model-derived text of the finished plan state an answer? Checked again after the model's output
        passed `VisualAgent.check`, over every text field (ids included) except the lesson's own section ids and
        the plan id (a digest): only then are the plan's images marked answer-isolated."""
        texts = [plan.lesson_title, plan.rationale]
        for r in plan.requirements:
            texts += [v for k, v in r.model_dump(mode="json").items()
                      if isinstance(v, str) and k != "lesson_section_id"]
        return not any(self.leaks(t) for t in texts)


def build_brief(data: VisualRequest) -> VisualBrief:
    answers = answer_keys(data.lesson)

    def safe(text: str | None) -> str | None:
        return text if text and not reveals(text, answers) else None

    names = {c.concept_id: c.name for c in data.plan.concepts}
    objective = data.research.objective
    sections: dict[str, str] = {}
    briefs: list[VisualSectionBrief] = []
    for s in data.lesson.sections:
        if s.purpose not in TEACHING_PURPOSES:
            continue
        concept = safe(names.get(s.concept_id, s.heading))
        if concept is None:  # the concept itself is an answer: any picture of it would show the answer
            continue
        ref = f"s{len(briefs) + 1}"
        sections[ref] = s.section_id
        briefs.append(VisualSectionBrief(
            ref=ref, concept=concept, purpose=s.purpose, heading=safe(s.heading), explanation=safe(s.explanation),
            examples=[e for e in s.examples if safe(e)], analogy=safe(s.analogy)))
    planning = VisualPlanningInput(
        lesson_title=safe(data.lesson.title), subject=safe(objective.subject), topic=safe(objective.topic),
        level=safe(data.lesson.level),
        objectives=[o.description for o in data.lesson.objectives if safe(o.description)],
        sections=briefs, max_visuals=data.max_visuals, max_generated_images=data.max_generated_images,
        max_searched_images=data.max_searched_images)
    return VisualBrief(planning=planning, sections=sections, answers=tuple(answers))
