"""End-to-end demo of the adaptive pedagogical engine.

    python scripts/run_adaptive_demo.py

A B1 Spanish learner with a calibrated placement (preterite 0.52, opinions 0.20, imperfect 0.78; see
fixtures/adaptive/learner.json) and the goal "Improve conversational Spanish":

  1. prints the learner model and the engine's first next-lesson recommendation;
  2. runs an adaptive lesson: the diagnostic only probes what memory does not already know, its answers become
     evidence, and the knowledge gaps and the pedagogical plan are computed from the learner model by code;
  3. prints the lesson's objectives and the purpose of each section;
  4. evaluates the lesson with the fixture learner's answers: evidence -> deterministic mastery update;
  5. prints mastery before and after, the feedback and the next recommendation, which must differ from the first.

Every number comes from the deterministic engine; models only word the lesson. Exits 1 if the recommendation
did not change.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config.settings import Settings  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.schemas.common import utcnow  # noqa: E402
from app.schemas.evaluation import AssessmentSheet, LearnerEvaluationReport  # noqa: E402
from app.schemas.learner import LearnerProfileInput, LearningEvidence, LearningGoal  # noqa: E402
from app.schemas.lesson import DiagnosticAnswers, DiagnosticQuestionSheet, LearnerAnswer, LessonContent  # noqa: E402
from app.schemas.pedagogy import (  # noqa: E402
    KnowledgeGapSet,
    LearnerModel,
    NextLearningRecommendation,
    PedagogicalPlan,
)
from app.schemas.task import Task, TaskStatus  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402

FIXTURES = REPO_ROOT / "fixtures" / "adaptive"
RULE = "=" * 72


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def placement_evidence(learner_id: str, placement: list[dict]) -> list[LearningEvidence]:
    """The fixture's practice results and calibrated placement for each concept, as immutable evidence."""
    now = utcnow()
    evidence = []
    for p in placement:
        cid, start = p["concept_id"], now - timedelta(days=p["days_ago"])
        items = [(f"placement-practice-{i}", "exercise", correctness, 1.0 if correctness == "correct" else 0.0, diff)
                 for i, (correctness, diff) in enumerate(p["practice"], start=1)]
        items.append(("placement", "manual", "partial", p["placement"], 0.5))
        for i, (ref, source, correctness, score, difficulty) in enumerate(items):
            evidence.append(LearningEvidence(
                evidence_id=LearningEvidence.id_for(learner_id, source, ref, cid), learner_id=learner_id,
                concept_id=cid, source_type=source, source_ref=ref, correctness=correctness, score=score,
                difficulty=difficulty, timestamp=start + timedelta(minutes=i)))
    return evidence


def _answers(sheet_questions, by_concept: dict, key) -> list[LearnerAnswer]:
    return [LearnerAnswer(question_id=q.question_id, answer=key(by_concept.get(q.concept_id, ""), q))
            for q in sheet_questions]


def _json(container: Container, task: Task, name: str):
    artifact = next(a for a in container.task_service.artifacts(task.task_id) if a.name == name)
    return container.artifacts.read(artifact.artifact_id)


def print_model(out, title: str, model: LearnerModel, concept_ids: list[str]) -> None:
    out(f"\n{title}")
    for cid in concept_ids:
        state = model.state(cid)
        if state is None or not model.has_evidence(cid):
            out(f"  {cid:<30} unknown")
            continue
        review = state.next_review_at.date().isoformat() if state.next_review_at else "-"
        out(f"  {cid:<30} mastery {state.mastery:.2f}  confidence {state.confidence:.2f}  "
            f"evidence {state.evidence_count}  next review {review}")


def print_recommendation(out, title: str, rec: NextLearningRecommendation) -> None:
    out(f"\n{title}")
    if rec.goal_achieved:
        out("  goal achieved")
    out(f"  learn next:          {', '.join(rec.recommended_concepts) or '-'}")
    out(f"  prerequisite review: {', '.join(rec.prerequisite_review) or '-'}")
    out(f"  spaced review:       {', '.join(rec.review_concepts) or '-'}")
    out(f"  activities:          {', '.join(rec.suggested_activity_types) or '-'}  ({rec.estimated_duration} min)")
    out(f"  why: {rec.reason}")


def _key(rec: NextLearningRecommendation) -> tuple:
    return (tuple(rec.recommended_concepts), tuple(rec.prerequisite_review), tuple(rec.review_concepts))


async def run_adaptive_demo(container: Container, *, out=print) -> tuple[Task | None, bool]:
    fixture = _fixture("learner.json")
    diagnostic_key = _fixture("answers.json")
    evaluation_key = _fixture("evaluation_answers.json")
    learner_id = fixture["learner_id"]
    learners = container.learner_service
    learners.upsert(learner_id, LearnerProfileInput.model_validate(fixture["profile"]))
    goal = learners.set_goal(LearningGoal.model_validate({**fixture["goal"], "learner_id": learner_id}))
    domain, framework = goal.domain, "cefr"
    await learners.record_evidence(learner_id, domain, placement_evidence(learner_id, fixture["placement"]))
    concept_ids = goal.target_concepts

    out(RULE + "\nStep 1: the learner and the goal\n" + RULE)
    out(f"learner {learner_id}, goal {goal.goal_id!r}: {goal.description} ({goal.target_level})")
    before = await learners.model(learner_id, domain, framework, goal.target_level)
    print_model(out, "learner model (from the placement evidence):", before, concept_ids)
    first = await learners.recommend(learner_id, goal.goal_id, framework)
    print_recommendation(out, "recommendation #1 (before the lesson):", first)

    out("\n" + RULE + "\nStep 2: an adaptive lesson\n" + RULE)
    out(f"request: {fixture['request']!r}")
    task = await container.task_service.create_and_run(request=fixture["request"], learner_id=learner_id,
                                                       user_id="demo-user")
    while task.status == TaskStatus.WAITING:
        assert task.waiting is not None
        sheet = DiagnosticQuestionSheet.model_validate(task.waiting.prompt)
        rounds = diagnostic_key["rounds"]
        by_concept = rounds[sheet.round_number - 1] if sheet.round_number <= len(rounds) else {}
        answers = _answers(sheet.questions, by_concept, lambda a, q: a)
        out(f"\ndiagnostic round {sheet.round_number} (only what memory does not already know):")
        for q, a in zip(sheet.questions, answers):
            out(f"  [{q.concept_id}] {q.prompt}\n     learner answers: {a.answer!r}")
        task = await container.task_service.submit_assessment(task.task_id, DiagnosticAnswers(answers=answers))
    if task.status != TaskStatus.COMPLETED:
        for err in task.errors:
            out(f"error: [{err.kind}] {err.message}")
        return task, False

    gaps = KnowledgeGapSet.model_validate_json(_json(container, task, "knowledge_gaps"))
    plan = PedagogicalPlan.model_validate_json(_json(container, task, "pedagogical_plan"))
    lesson = LessonContent.model_validate_json(_json(container, task, "lesson"))
    after_diagnostic = LearnerModel.model_validate_json(_json(container, task, "learner_model"))
    print_model(out, "learner model after the diagnostic evidence:", after_diagnostic, concept_ids)
    out("\nknowledge gaps (deterministic priority):")
    for g in gaps.gaps:
        unmet = f"  needs {', '.join(g.unmet_prerequisites)}" if g.unmet_prerequisites else ""
        out(f"  {g.priority:.3f}  {g.concept.concept_id:<30} {g.band:<13} {g.recommended_action}{unmet}")
    out(f"\npedagogical plan {plan.plan_id} ({plan.strategy_id} strategy, "
        f"{plan.estimated_duration}/{plan.available_minutes} min):")
    out(f"  targets: {', '.join(plan.target_concepts)}")
    out(f"  prerequisite review: {', '.join(plan.prerequisite_concepts) or '-'}  "
        f"spaced review: {', '.join(plan.review_concepts) or '-'}")
    for step in plan.sequencing:
        activity = next(a for a in plan.activities if a.activity_id == step.activity_id)
        out(f"  {step.order:>2}. {step.phase:<19} {activity.type:<16} {activity.difficulty:<13} "
            f"{step.minutes} min  {activity.concept_ids[0]}")
    out(f"  rationale: {plan.rationale}")
    out(f"\nlesson {lesson.title!r}:")
    for o in lesson.objectives:
        out(f"  objective {o.objective_id}: {o.description} (target mastery {o.target_mastery:.2f})")
    for s in lesson.sections:
        out(f"  section {s.section_id:<6} {s.purpose:<12} {s.concept_id:<30} objectives {', '.join(s.objective_ids)}")

    out("\n" + RULE + "\nStep 3: evaluate the lesson\n" + RULE)
    evaluation = await container.task_service.start_evaluation(task.task_id, user_id="demo-user")
    if evaluation.status != TaskStatus.WAITING or evaluation.waiting is None:
        return evaluation, False
    sheet = AssessmentSheet.model_validate(evaluation.waiting.prompt)
    answers = _answers(sheet.questions, evaluation_key["answers"], lambda a, q: (a or {}).get(q.kind, ""))
    for q, a in zip(sheet.questions, answers):
        choices = f"  choices: {q.choices}" if q.choices else ""
        out(f"  [{q.kind}] {q.prompt}{choices}\n     learner answers: {a.answer!r}")
    evaluation = await container.task_service.submit_answers(
        evaluation.task_id, {"answers": [a.model_dump(mode="json") for a in answers]})
    if evaluation.status != TaskStatus.COMPLETED or evaluation.result is None:
        for err in evaluation.errors:
            out(f"error: [{err.kind}] {err.message}")
        return evaluation, False
    report = LearnerEvaluationReport.model_validate_json(_json(container, evaluation, "learner_evaluation"))
    out(f"\nscore {report.score:.0%} ({report.points_earned:g}/{report.points_possible:g})")

    after = await learners.model(learner_id, domain, framework, goal.target_level)
    out("\nmastery before the lesson -> after the evaluation:")
    for cid in concept_ids:
        b = before.mastery_of(cid) if before.has_evidence(cid) else None
        a = after.mastery_of(cid) if after.has_evidence(cid) else None
        out(f"  {cid:<30} {'unknown' if b is None else f'{b:.2f}':>7} -> {'unknown' if a is None else f'{a:.2f}'}")
    feedback = evaluation.result.feedback
    if feedback is not None:
        out(f"\nfeedback: {feedback.summary}")

    second = evaluation.result.next_recommendation
    assert second is not None
    print_recommendation(out, "recommendation #2 (after the evaluation):", second)
    changed = _key(first) != _key(second)
    out("\n" + RULE)
    out(f"recommendation changed with the learner's mastery: {'yes' if changed else 'NO'}")
    return evaluation, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database and artifacts (default: a fresh temporary directory)")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args()
    configure_logging("INFO" if args.verbose else "WARNING")

    with tempfile.TemporaryDirectory(prefix="teaching-agent-adaptive-demo-") as tmp:
        container = build_container(Settings(data_dir=args.data_dir or Path(tmp), corpus_dir=FIXTURES))
        try:
            task, changed = asyncio.run(run_adaptive_demo(container))
        finally:
            container.close()
    return 0 if task is not None and task.status == TaskStatus.COMPLETED and changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
