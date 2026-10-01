"""Deterministic stand-ins for model reasoning, one per agent.

They derive every answer from the structured input the agent sent (learner snapshot,
knowledge-base entries, research facts), so they are subject-independent: all subject
matter comes from the corpus the tools retrieved. They return plain dicts; the agent
validates them exactly as it validates real model output.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from app.providers.llm.base import LLMRequest
from app.providers.llm.mock import Responder
from app.schemas.evaluation import (
    AssessmentPlan,
    AssessmentQuestion,
    ConceptOutcome,
    EvaluationInput,
    EvaluationStep,
    LearnerEvaluationResult,
    LearningRecommendation,
)
from app.schemas.learner import AnswerEvaluation
from app.schemas.research import (
    EvidenceExtraction,
    EvidenceExtractionInput,
    ExtractedEvidence,
    KeyFinding,
    ProposedFinding,
)
from app.schemas.lesson import (
    ConceptEstimate,
    ConceptRef,
    DiagnosticInput,
    DiagnosticItem,
    DiagnosticQuestion,
    DiagnosticResult,
    DiagnosticStep,
    Exercise,
    CheckQuestion,
    InterpreterInput,
    LessonContent,
    LessonPlan,
    LessonRequest,
    LessonSection,
    PlanStep,
    PlannedConcept,
    PlannedExercise,
    PlannerInput,
    ReviewCriterion,
    ReviewerInput,
    ReviewIssue,
    ReviewResult,
    Slide,
    SlideDeckPlan,
    SlideInput,
    TeacherInput,
    Verdict,
    VisualSpec,
    MAX_BULLETS_PER_SLIDE,
    MAX_WORDS_PER_BULLET,
)

LEVEL_RE = re.compile(r"\b(A1|A2|B1|B2|C1|C2)\b", re.IGNORECASE)
TOPIC_RE = re.compile(r"\b(?:about|on|regarding)\s+(.+?)[\s.!?]*$", re.IGNORECASE)
KNOWN_THRESHOLD = 0.7
GAP_THRESHOLD = 0.5


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def clip_words(text: str, limit: int = MAX_WORDS_PER_BULLET) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit])


# --- request interpreter -------------------------------------------------------------------


def interpret(request: LLMRequest) -> dict:
    p = InterpreterInput.model_validate(request.input_payload)
    text = p.request.strip()
    level_match = LEVEL_RE.search(text)
    topic_match = TOPIC_RE.search(text)
    topic = (topic_match.group(1) if topic_match else text).strip().lower()
    level = level_match.group(1).upper() if level_match else None
    subject: str | None = None
    framework: str | None = "cefr" if level else None

    for state in p.learner.subjects:
        mentioned = state.subject.lower() in text.lower()
        if mentioned or (framework is not None and state.framework_id == framework):
            subject, framework = state.subject, state.framework_id
            level = level or state.target_level
            break
    if subject is None:
        subject = "language" if framework == "cefr" else topic
        framework = framework or "mastery"

    return LessonRequest(
        raw_request=text,
        subject=subject,
        topic=topic,
        framework_id=framework,
        target_level=level,
        language_of_instruction=p.learner.preferences.language_of_instruction,
        capabilities=["lesson.text", "lesson.review", "slides.plan"],
    ).model_dump(mode="json")


# --- knowledge diagnostic --------------------------------------------------------------------


def _evidence_score(correct: bool, difficulty: float) -> float:
    return 0.55 + 0.4 * difficulty if correct else 0.35 * (1 - difficulty)


def _evaluate(p: DiagnosticInput) -> list[AnswerEvaluation]:
    evaluations: list[AnswerEvaluation] = []
    for rnd in p.rounds:
        answers = {a.question_id: a.answer for a in rnd.answers}
        for item in rnd.items:
            given = answers.get(item.question.question_id, "")
            accepted = {normalize(item.expected_answer), *(normalize(a) for a in item.accepted_answers)}
            correct = normalize(given) in accepted
            evaluations.append(
                AnswerEvaluation(
                    question_id=item.question.question_id,
                    concept_id=item.question.concept_id,
                    answer=given,
                    expected=item.expected_answer,
                    correct=correct,
                    difficulty=item.question.difficulty,
                    feedback="Correct." if correct else f"Expected: {item.expected_answer}",
                )
            )
    return evaluations


def _level_for(p: DiagnosticInput, average: float) -> str:
    levels = p.snapshot.framework_levels
    target = p.request.target_level or p.snapshot.target_level or levels[0]
    index = levels.index(target) if target in levels else 0
    if average >= 0.55:
        return levels[index]
    return levels[max(0, index - 1)]


def _conclude(p: DiagnosticInput, concepts: list[ConceptRef], estimates: list[ConceptEstimate],
              evaluations: list[AnswerEvaluation], source: str) -> DiagnosticResult:
    by_id = {e.concept_id: e for e in estimates}
    gaps = [c.concept_id for c in concepts if by_id[c.concept_id].mastery < GAP_THRESHOLD]
    known = [c.concept_id for c in concepts if by_id[c.concept_id].mastery >= KNOWN_THRESHOLD]
    gap_set = set(gaps)
    start = next(
        (c.concept_id for c in concepts if c.concept_id in gap_set and not gap_set & set(c.prerequisites)),
        gaps[0] if gaps else concepts[0].concept_id,
    )
    average = sum(e.mastery for e in estimates) / len(estimates)
    return DiagnosticResult(
        source=source,
        estimated_level=_level_for(p, average),
        concept_mastery=estimates,
        known=known,
        gaps=gaps,
        starting_point=start,
        evaluations=evaluations,
        rationale=f"Average estimated mastery {average:.2f} across {len(concepts)} concepts; {len(gaps)} gaps.",
    )


def diagnose(request: LLMRequest) -> dict:
    p = DiagnosticInput.model_validate(request.input_payload)
    concepts = [entry.concept for entry in p.concepts]
    ids = [c.concept_id for c in concepts]

    if not p.rounds and p.snapshot.has_evidence_for(ids, p.memory_confidence_threshold):
        memory = {c.concept_id: c for c in p.snapshot.concept_mastery}
        estimates = [
            ConceptEstimate(concept_id=cid, mastery=memory[cid].mastery, confidence=memory[cid].confidence)
            for cid in ids
        ]
        result = _conclude(p, concepts, estimates, [], "memory")
        return DiagnosticStep(status="complete", concepts=concepts, result=result).model_dump(mode="json")

    if not p.rounds:
        items = []
        for entry in p.concepts:
            if not entry.probes:
                continue
            probe = min(entry.probes, key=lambda pr: abs(pr.difficulty - 0.5))
            items.append(_item(p.round_number, entry.concept.concept_id, probe))
        return DiagnosticStep(status="ask", concepts=concepts, items=items).model_dump(mode="json")

    evaluations = _evaluate(p)
    latest_ids = {i.question.question_id for i in p.rounds[-1].items}
    latest = [e for e in evaluations if e.question_id in latest_ids]

    if len(p.rounds) < p.max_rounds:
        # Adapt: for each concept missed in the last round, ask an easier question not yet asked.
        asked = {normalize(i.question.prompt) for r in p.rounds for i in r.items}
        followups = []
        for ev in latest:
            if ev.correct:
                continue
            entry = next(e for e in p.concepts if e.concept.concept_id == ev.concept_id)
            easier = sorted(
                (pr for pr in entry.probes if pr.difficulty < ev.difficulty and normalize(pr.prompt) not in asked),
                key=lambda pr: -pr.difficulty,
            )
            if easier:
                followups.append(_item(p.round_number, ev.concept_id, easier[0]))
        if followups:
            return DiagnosticStep(
                status="ask", concepts=concepts, items=followups, evaluations=latest
            ).model_dump(mode="json")

    per_concept: dict[str, list[float]] = defaultdict(list)
    for ev in evaluations:
        per_concept[ev.concept_id].append(_evidence_score(ev.correct, ev.difficulty))
    memory = {c.concept_id: c for c in p.snapshot.concept_mastery}
    estimates = []
    for cid in ids:
        scores = per_concept.get(cid)
        if scores:
            estimates.append(ConceptEstimate(
                concept_id=cid, mastery=round(sum(scores) / len(scores), 4),
                confidence=round(min(0.95, 0.35 + 0.2 * len(scores)), 4),
            ))
        else:
            prior = memory.get(cid)
            estimates.append(ConceptEstimate(
                concept_id=cid, mastery=prior.mastery if prior else 0.3,
                confidence=prior.confidence if prior else 0.1,
            ))
    result = _conclude(p, concepts, estimates, evaluations, "assessment")
    return DiagnosticStep(
        status="complete", concepts=concepts, evaluations=latest, result=result
    ).model_dump(mode="json")


def _item(round_number: int, concept_id: str, probe) -> DiagnosticItem:
    return DiagnosticItem(
        question=DiagnosticQuestion(
            question_id=f"r{round_number}_{concept_id}",
            concept_id=concept_id,
            prompt=probe.prompt,
            difficulty=probe.difficulty,
        ),
        expected_answer=probe.answer,
        accepted_answers=probe.accepted,
    )


# --- research (evidence extraction) ---------------------------------------------------------

STOPWORDS = frozenset(
    "a an and are as at be by for from how in is it its of on or that the this to with what when which who "
    "about talk using use you your not than into".split()
)
PARAGRAPH_RE = re.compile(r"\S(?:.|\n(?!\s*\n))*")
MARKER_RE = re.compile(r"\s*\b(Example|Practice|Answer):\s*")


def _terms(text: str) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if len(w) > 2 and w not in STOPWORDS}


def _parse_paragraph(text: str) -> dict[str, str | None]:
    """'Claim. Example: ... Practice: ... Answer: ...' -> parts. Only the claim is required."""
    parts = MARKER_RE.split(text)
    out: dict[str, str | None] = {"claim": parts[0].strip(), "Example": None, "Practice": None, "Answer": None}
    for label, value in zip(parts[1::2], parts[2::2]):
        out[label] = value.strip() or None
    return out


def research(request: LLMRequest) -> dict:
    """Quote each source paragraph under the research target it is most clearly about (ties are skipped)."""
    p = EvidenceExtractionInput.model_validate(request.input_payload)
    background = _terms(f"{p.objective.subject} {p.objective.topic}")
    targets = {
        t.target_id: _terms(f"{t.name} {t.description} {t.target_id.replace('.', ' ').replace('_', ' ')}") - background
        for t in p.objective.targets
    }
    evidence: list[ExtractedEvidence] = []
    findings: list[ProposedFinding] = []
    per_target: dict[str, int] = defaultdict(int)
    for src in p.sources:
        field, text = ("content", src.content) if src.content else ("snippet", src.snippet)
        for match in PARAGRAPH_RE.finditer(text):
            paragraph = match.group(0).rstrip()
            words = _terms(paragraph)
            scores = sorted(((len(words & terms), tid) for tid, terms in targets.items()), reverse=True)
            best, target_id = scores[0]
            if best == 0 or (len(scores) > 1 and scores[1][0] == best):
                continue
            if per_target[target_id] >= p.max_findings_per_target:
                continue
            per_target[target_id] += 1
            ref = f"e{len(evidence) + 1}"
            evidence.append(ExtractedEvidence(
                ref=ref, source_id=src.source_id, target_id=target_id, text=paragraph, field=field,
                start=match.start(), end=match.start() + len(paragraph),
                relevance=round(min(1.0, best / max(1, len(targets[target_id]))), 3),
            ))
            parts = _parse_paragraph(paragraph)
            findings.append(ProposedFinding(
                target_id=target_id, statement=parts["claim"], example=parts["Example"],
                practice_prompt=parts["Practice"], practice_answer=parts["Answer"], evidence_refs=[ref],
            ))
    covered = sum(1 for t in targets if per_target[t])
    return EvidenceExtraction(
        evidence=evidence, findings=findings,
        summary=f"{len(evidence)} quoted passages from {len({e.source_id for e in evidence})} sources "
                f"covering {covered}/{len(targets)} targets.",
    ).model_dump(mode="json")


# --- curriculum planner ----------------------------------------------------------------------


def plan(request: LLMRequest) -> dict:
    p = PlannerInput.model_validate(request.input_payload)
    known = set(p.diagnostic.known)
    gaps = [cid for cid in p.diagnostic.gaps]
    order = gaps + [c.concept_id for c in p.concepts if c.concept_id not in gaps and c.concept_id not in known]
    if not order:
        order = [c.concept_id for c in p.concepts]
    names = {c.concept_id: c.name for c in p.concepts}
    facts_by_concept: dict[str, list[KeyFinding]] = defaultdict(list)
    for fact in p.research.key_findings:
        facts_by_concept[fact.target_id].append(fact)

    planned = [
        PlannedConcept(
            concept_id=cid,
            name=names[cid],
            rationale="Identified as a gap by the diagnostic." if cid in gaps else "Not yet secure; consolidate.",
            strategy="Worked examples, then guided practice." if cid in gaps else "Brief recap, then practice.",
            examples=[f.example for f in facts_by_concept[cid] if f.example][:2],
        )
        for cid in order
    ]
    review = [cid for cid in p.snapshot.due_for_review if cid not in order and cid in names]
    sequence = [PlanStep(step_id="st1", concept_id=None, activity="warm_up", minutes=3)]
    for cid in review:
        sequence.append(PlanStep(step_id=f"st{len(sequence) + 1}", concept_id=cid, activity="review", minutes=3))
    for cid in order:
        sequence.append(PlanStep(step_id=f"st{len(sequence) + 1}", concept_id=cid, activity="explain",
                                 minutes=6 if cid in gaps else 4))
        sequence.append(PlanStep(step_id=f"st{len(sequence) + 1}", concept_id=cid, activity="practice", minutes=4))
    sequence.append(PlanStep(step_id=f"st{len(sequence) + 1}", concept_id=None, activity="assess", minutes=5))

    exercises = []
    for cid in order:
        practice = next((f for f in facts_by_concept[cid] if f.practice_prompt), None)
        if practice:
            exercises.append(PlannedExercise(
                exercise_id=f"ex_{cid}", concept_id=cid, kind="short_answer", prompt=practice.practice_prompt
            ))

    lesson_plan = LessonPlan(
        title=f"{p.request.topic.title()} ({p.diagnostic.estimated_level})",
        level=p.diagnostic.estimated_level,
        objectives=[f"Practise: {names[cid]}" for cid in order],
        prerequisites=sorted({pre for c in p.concepts if c.concept_id in order for pre in c.prerequisites}),
        concepts=planned,
        review_concepts=review,
        sequence=sequence,
        estimated_minutes=sum(s.minutes for s in sequence),
        teaching_strategy="Start from the diagnosed gaps, teach with examples first, practise each concept immediately.",
        exercises=exercises,
        assessment=[f"Check question on {names[cid]}" for cid in order],
        remediation=[f"If missed, revisit {names[cid]} with simpler examples" for cid in gaps],
        extensions=[f"Apply {p.request.topic} in a short free-response task"],
    )
    return lesson_plan.model_dump(mode="json")


# --- teacher -------------------------------------------------------------------------------


def make_teacher(first_draft_defects: bool) -> Responder:
    def teach(request: LLMRequest) -> dict:
        p = TeacherInput.model_validate(request.input_payload)
        research = p.research

        sections = []
        for concept in p.plan.concepts:
            facts = research.findings_for(concept.concept_id)[:3]
            explanation = " ".join(f.statement for f in facts) or concept.rationale
            examples = [f.example for f in facts if f.example]
            narration = f"{concept.name}. {explanation}"
            if examples:
                narration += f" For example: {examples[0]}"
            sections.append(LessonSection(
                section_id=f"sec_{concept.concept_id}",
                concept_id=concept.concept_id,
                heading=concept.name,
                explanation=explanation,
                examples=examples,
                narration=narration,
                citations=list(dict.fromkeys(c for f in facts for c in research.citation_ids_for(f))),
            ))
        if first_draft_defects and p.revision is None and sections:
            # Simulates a common model failure (dropped citations) so the review loop is exercised.
            sections[-1] = sections[-1].model_copy(update={"citations": []})

        answers = {f.practice_prompt: f.practice_answer for f in research.key_findings if f.practice_prompt}
        exercises = [
            Exercise(exercise_id=ex.exercise_id, concept_id=ex.concept_id, kind=ex.kind, prompt=ex.prompt,
                     answer=answers.get(ex.prompt) or "", explanation="Compare with the lesson examples.")
            for ex in p.plan.exercises
        ]
        checks = [
            CheckQuestion(question_id=f"cq_{ex.concept_id}", concept_id=ex.concept_id, prompt=ex.prompt,
                          answer=ex.answer)
            for ex in exercises
        ]
        names = ", ".join(c.name for c in p.plan.concepts)
        content = LessonContent(
            title=p.plan.title,
            level=p.plan.level,
            introduction=f"In this lesson you will work on: {names}.",
            sections=sections,
            exercises=exercises,
            check_questions=checks,
            summary=f"You studied {names}. Review the examples and try the exercises again tomorrow.",
        )
        return content.model_dump(mode="json")

    return teach


# --- reviewer ------------------------------------------------------------------------------


def review(request: LLMRequest) -> dict:
    p = ReviewerInput.model_validate(request.input_payload)
    resolvable = {c.citation_id for c in p.research.citations}
    researched = {f.target_id for f in p.research.key_findings}
    sections = {s.concept_id: s for s in p.content.sections}
    issues: list[ReviewIssue] = []

    def add(criterion: ReviewCriterion, severity: str, location: str, problem: str, fix: str) -> None:
        issues.append(ReviewIssue(issue_id=f"i{len(issues) + 1}", criterion=criterion, severity=severity,
                                  location=location, problem=problem, suggested_fix=fix))

    for concept in p.plan.concepts:
        if concept.concept_id not in sections:
            add(ReviewCriterion.COMPLETENESS, "major", concept.concept_id,
                f"Planned concept '{concept.name}' has no section.", "Add a section for it.")
    for section in p.content.sections:
        if not section.citations and section.concept_id not in researched:
            add(ReviewCriterion.SOURCE_QUALITY, "minor", section.section_id,
                "Unverified: research found no evidence for this concept, so the section cites no source.",
                "Verify the section once research evidence is available.")
        elif not section.citations:
            add(ReviewCriterion.SOURCE_QUALITY, "major", section.section_id,
                "Section makes factual claims without citing a source.", "Cite the research findings it uses.")
        elif not set(section.citations) <= resolvable:
            add(ReviewCriterion.HALLUCINATION_RISK, "critical", section.section_id,
                "Section cites a source that is unknown or unreliable.", "Cite only reliable research sources.")
        if not section.examples:
            add(ReviewCriterion.PEDAGOGICAL_QUALITY, "minor", section.section_id,
                "Section has no example.", "Add a concrete example.")
    exercise_ids = {e.exercise_id: e for e in p.content.exercises}
    for planned in p.plan.exercises:
        ex = exercise_ids.get(planned.exercise_id)
        if ex is None:
            add(ReviewCriterion.EXERCISE_QUALITY, "major", planned.exercise_id,
                "Planned exercise is missing.", "Include every planned exercise.")
        elif not ex.answer:
            add(ReviewCriterion.EXERCISE_QUALITY, "major", planned.exercise_id,
                "Exercise has no answer key.", "Provide the expected answer.")

    scores = {c: 1.0 for c in ReviewCriterion}
    for issue in issues:
        penalty = {"minor": 0.1, "major": 0.3, "critical": 0.5}[issue.severity]
        scores[issue.criterion] = round(max(0.0, scores[issue.criterion] - penalty), 3)
    blocking = [i for i in issues if i.severity != "minor"]
    result = ReviewResult(
        verdict=Verdict.REVISION_REQUIRED if blocking else Verdict.APPROVED,
        scores=scores,
        issues=issues,
        summary=f"{len(blocking)} blocking and {len(issues) - len(blocking)} minor issues.",
    )
    return result.model_dump(mode="json")


# --- slides --------------------------------------------------------------------------------


def slides(request: LLMRequest) -> dict:
    p = SlideInput.model_validate(request.input_payload)
    lesson = p.lesson
    deck: list[Slide] = [
        Slide(slide_id="sl1", kind="title", heading=lesson.title,
              bullets=[f"Level {lesson.level}", f"{p.plan.estimated_minutes} minutes"],
              speaker_notes=lesson.introduction),
        Slide(slide_id="sl2", kind="objectives", heading="Objectives",
              bullets=[clip_words(o) for o in p.plan.objectives[:MAX_BULLETS_PER_SLIDE]]),
    ]
    for section in lesson.sections:
        deck.append(Slide(
            slide_id=f"sl{len(deck) + 1}", kind="explanation", heading=section.heading,
            bullets=[clip_words(s) for s in sentences(section.explanation)[:4]],
            visual=VisualSpec(kind="image", description=f"Illustration for: {section.heading}"),
            narration_section_id=section.section_id, speaker_notes=section.narration,
        ))
        if section.examples:
            deck.append(Slide(
                slide_id=f"sl{len(deck) + 1}", kind="example", heading=f"{section.heading}: examples",
                bullets=[clip_words(e) for e in section.examples[:MAX_BULLETS_PER_SLIDE]],
                narration_section_id=section.section_id,
            ))
    if lesson.exercises:
        deck.append(Slide(
            slide_id=f"sl{len(deck) + 1}", kind="exercise", heading="Practice",
            bullets=[clip_words(e.prompt) for e in lesson.exercises[:MAX_BULLETS_PER_SLIDE]],
        ))
    deck.append(Slide(
        slide_id=f"sl{len(deck) + 1}", kind="summary", heading="Summary",
        bullets=[clip_words(s.heading) for s in lesson.sections[:MAX_BULLETS_PER_SLIDE]],
        speaker_notes=lesson.summary,
    ))
    return SlideDeckPlan(title=lesson.title, slides=deck).model_dump(mode="json")


# --- learner evaluation --------------------------------------------------------------------

WEAK_MASTERY = 0.6


def evaluate_learner(request: LLMRequest) -> dict:
    p = EvaluationInput.model_validate(request.input_payload)
    step = _assess(p) if p.stage == "assess" else _grade(p)
    return step.model_dump(mode="json")


def _assess(p: EvaluationInput) -> EvaluationStep:
    mastery = {c.concept_id: c.mastery for c in p.snapshot.concept_mastery}
    exercises: dict[str, Exercise] = {}
    for ex in p.lesson.exercises:
        exercises.setdefault(ex.concept_id, ex)
    questions: list[AssessmentQuestion] = []
    for section in p.lesson.sections:
        cid = section.concept_id
        objective = next((o for o in p.plan.objectives if section.heading.lower() in o.lower()), p.plan.objectives[0])
        weak = mastery.get(cid, 0.0) < WEAK_MASTERY
        ex = exercises.get(cid)
        if ex is not None and ex.answer:
            questions.append(AssessmentQuestion(
                question_id=f"sa_{cid}", concept_id=cid, objective=objective, kind="short_answer",
                prompt=ex.prompt, difficulty=0.5, expected_answer=ex.answer,
            ))
        distractors = [s.examples[0] for s in p.lesson.sections if s.concept_id != cid and s.examples]
        if (weak or ex is None) and section.examples and distractors:
            # Weaker concepts get a second, recognition-level question built from the lesson's own examples.
            correct = section.examples[0]
            questions.append(AssessmentQuestion(
                question_id=f"mc_{cid}", concept_id=cid, objective=objective, kind="multiple_choice",
                prompt=f"Which of these is an example of: {section.heading}?",
                choices=sorted({correct, *distractors[:2]}), difficulty=0.3, expected_answer=correct,
            ))
    plan = AssessmentPlan(
        title=f"Check: {p.lesson.title}", level=p.lesson.level,
        objectives=sorted({q.objective for q in questions}), questions=questions,
        total_points=float(len(questions)),
        rationale="One application question per taught concept; weaker concepts also get a recognition question.",
    )
    return EvaluationStep(stage="assess", assessment=plan)


def _grade(p: EvaluationInput) -> EvaluationStep:
    assert p.assessment is not None and p.response is not None
    answers = {a.question_id: a.answer for a in p.response.answers}
    names = {s.concept_id: s.heading for s in p.lesson.sections}
    evaluations = []
    for q in p.assessment.questions:
        given = answers.get(q.question_id, "")
        accepted = {normalize(q.expected_answer), *(normalize(a) for a in q.accepted_answers)}
        correct = bool(given) and normalize(given) in accepted
        evaluations.append(AnswerEvaluation(
            question_id=q.question_id, concept_id=q.concept_id, answer=given, expected=q.expected_answer,
            correct=correct, difficulty=q.difficulty,
            feedback="Correct." if correct else f"Expected: {q.expected_answer}",
        ))
    points = {q.question_id: q.points for q in p.assessment.questions}
    concepts: list[ConceptOutcome] = []
    for cid in dict.fromkeys(q.concept_id for q in p.assessment.questions):
        mine = [e for e in evaluations if e.concept_id == cid]
        possible = sum(points[e.question_id] for e in mine)
        earned = sum(points[e.question_id] for e in mine if e.correct)
        score = earned / possible
        status = "mastered" if score == 1 else "partial" if score >= 0.5 else "gap"
        concepts.append(ConceptOutcome(concept_id=cid, name=names.get(cid, cid), questions=len(mine),
                                       points_earned=earned, points_possible=possible, score=round(score, 4),
                                       status=status))
    mastered = [c.concept_id for c in concepts if c.status == "mastered"]
    partial = [c.concept_id for c in concepts if c.status == "partial"]
    gaps = [c.concept_id for c in concepts if c.status == "gap"]
    level = p.request.target_level or p.lesson.level
    label = lambda ids: ", ".join(names.get(i, i) for i in ids)  # noqa: E731
    if gaps:
        rec = LearningRecommendation(
            action="reteach", focus_concepts=gaps, review_concepts=partial,
            rationale=f"{len(gaps)} concept(s) are still gaps after the lesson.",
            suggested_request=f"Create a new {level} lesson that reteaches {label(gaps)} in {p.request.topic}.")
    elif partial:
        rec = LearningRecommendation(
            action="review", focus_concepts=partial,
            rationale="Some concepts are only partly secure; a short review should consolidate them.",
            suggested_request=f"Create a short {level} review of {label(partial)} in {p.request.topic}.")
    else:
        assessed = {c.concept_id for c in concepts}
        upcoming = [cid for cid in p.snapshot.recommended_next if cid not in assessed][:3]
        rec = LearningRecommendation(
            action="advance", focus_concepts=upcoming,
            rationale="Every assessed concept is mastered.",
            suggested_request=f"Create a new {level} lesson that builds on {p.request.topic}.")
    earned = sum(c.points_earned for c in concepts)
    possible = sum(c.points_possible for c in concepts)
    result = LearnerEvaluationResult(
        score=round(earned / possible, 4), points_earned=earned, points_possible=possible,
        evaluations=evaluations, concepts=concepts, mastered=mastered, partial=partial, gaps=gaps,
        recommendation=rec,
    )
    return EvaluationStep(stage="evaluate", result=result)


def default_responders(*, first_draft_defects: bool = True) -> dict[str, Responder]:
    return {
        "request_interpreter": interpret,
        "knowledge_diagnostic": diagnose,
        "research": research,
        "curriculum_planner": plan,
        "teacher": make_teacher(first_draft_defects),
        "content_reviewer": review,
        "slide_generation": slides,
        "learner_evaluation": evaluate_learner,
    }
