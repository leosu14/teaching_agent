"""End-to-end demo of learner goals and the curriculum engine.

    python scripts/run_curriculum_demo.py

A Spanish learner (fixtures/curriculum/learner.json) with a calibrated placement on four concepts:

    A es.present        0.90                      (A2)
    B es.preterite      0.35   requires A         (B1)
    C es.past_contrast  0.10   requires B         (B1)
    D es.subjunctive    0.20   requires B         (B1)
    E es.argument       -      requires C and D   (B2)

  1. creates the learner;  2. creates the goal "Reach B1 Spanish" (and again: same goal, no duplicate);
  3. builds the curriculum (a planning task; building again stores no second version);
  4-6. prints the objectives, the prerequisites and the mastery state;
  7. selects the next learning action: B, never C or D while B is below the prerequisite threshold;
  8. starts the lesson from that objective; 9. evaluates it with the fixture learner's answers;
  10-11. the mastery update and the recalculated curriculum progress;
  12. a different next action (C or D, now that B is secure); 13. curriculum progress;
  then changes the goal to B2 (a new curriculum version, the old one still readable), adds a second goal with a
  target date that cannot be met (a structured warning, nothing compressed), selects across both goals, and
  14. completes the Spanish goal from exercise evidence: the completion rule, not a model, completes it.

Every decision comes from the deterministic engine; models only word objectives and lessons. Exits 1 if any check
fails.
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
from app.schemas.curriculum import (  # noqa: E402
    Curriculum,
    GoalInput,
    GoalUpdate,
    LearningActionType,
    NextLearningAction,
)
from app.schemas.evaluation import AssessmentSheet  # noqa: E402
from app.schemas.learner import GoalStatus, LearnerProfileInput, LearningEvidence  # noqa: E402
from app.schemas.lesson import DiagnosticAnswers, DiagnosticQuestionSheet, LearnerAnswer  # noqa: E402
from app.schemas.pedagogy import PedagogicalPlan  # noqa: E402
from app.schemas.task import Task, TaskStatus  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402

FIXTURES = REPO_ROOT / "fixtures" / "curriculum"
RULE = "=" * 72
A, B, C, D, E = "es.present", "es.preterite", "es.past_contrast", "es.subjunctive", "es.argument"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class Checks:
    def __init__(self, out) -> None:
        self.out = out
        self.failed: list[str] = []

    def __call__(self, ok: bool, label: str) -> bool:
        self.out(f"  [{'ok' if ok else 'FAIL'}] {label}")
        if not ok:
            self.failed.append(label)
        return ok


def placement_evidence(learner_id: str, placement: list[dict]) -> dict[str, list[LearningEvidence]]:
    """The fixture's practice results and calibrated placement per concept, as immutable evidence, by domain."""
    now = utcnow()
    by_domain: dict[str, list[LearningEvidence]] = {}
    for p in placement:
        cid, start = p["concept_id"], now - timedelta(days=p["days_ago"])
        items = [(f"placement-practice-{i}", "exercise", correctness, 1.0 if correctness == "correct" else 0.0, diff)
                 for i, (correctness, diff) in enumerate(p["practice"], start=1)]
        items.append(("placement", "manual", "partial", p["placement"], 0.5))
        for i, (ref, source, correctness, score, difficulty) in enumerate(items):
            by_domain.setdefault(p["domain"], []).append(LearningEvidence(
                evidence_id=LearningEvidence.id_for(learner_id, source, ref, cid), learner_id=learner_id,
                concept_id=cid, source_type=source, source_ref=ref, correctness=correctness, score=score,
                difficulty=difficulty, timestamp=start + timedelta(minutes=i)))
    return by_domain


def exercise_evidence(learner_id: str, concept_id: str, count: int, tag: str) -> list[LearningEvidence]:
    now = utcnow()
    return [LearningEvidence(
        evidence_id=LearningEvidence.id_for(learner_id, "exercise", f"{tag}-{i}", concept_id), learner_id=learner_id,
        concept_id=concept_id, source_type="exercise", source_ref=f"{tag}-{i}", correctness="correct", score=1.0,
        difficulty=0.7, timestamp=now + timedelta(seconds=i)) for i in range(count)]


def print_curriculum(out, cur: Curriculum) -> None:
    out(f"curriculum {cur.curriculum_id} version {cur.version} ({cur.version_id}), {cur.versions} version(s), "
        f"{cur.status.value}")
    out(f"  {'#':>2}  {'concept':<18} {'role':<12} {'mode':<8} {'target':>6} {'now':>5} {'evid':>4}  status")
    for o in cur.objectives:
        p = cur.progress.of(o.objective_id)
        out(f"  {o.order:>2}  {o.concept_id:<18} {o.role:<12} {o.mode:<8} {o.target_mastery:>6.2f} "
            f"{p.current_mastery:>5.2f} {p.evidence_count:>2}/{p.evidence_required}  {p.status.value}"
            + (f"  (needs {', '.join(p.unmet_prerequisites)})" if p.unmet_prerequisites else ""))
        out(f"      {o.description}")


def print_action(out, title: str, action: NextLearningAction) -> None:
    out(f"\n{title}")
    out(f"  action: {action.action.value}  objective: {action.concept_id or '-'}  goal: {action.goal_id or '-'}")
    out(f"  why: {action.reason}")
    for c in action.candidates:
        out(f"    {'eligible' if c.eligible else 'blocked ':<8} {c.action.value:<8} {c.concept_id:<18} "
            f"score {c.score:.3f}  ({c.goal_id})")


async def answer_diagnostic(container: Container, task: Task, out) -> Task:
    """A curriculum lesson's diagnostic only asks what memory does not already know; answer it from the probes."""
    probes = {}
    for doc in _fixture("knowledge_base.json"):
        concept = doc["metadata"]["concept"]["concept_id"]
        probes.update({p["prompt"]: p["answer"] for p in doc["metadata"]["probes"]} | {concept: ""})
    while task.status == TaskStatus.WAITING and task.waiting is not None and task.waiting.kind == "diagnostic_answers":
        sheet = DiagnosticQuestionSheet.model_validate(task.waiting.prompt)
        answers = [LearnerAnswer(question_id=q.question_id, answer=probes.get(q.prompt, "")) for q in sheet.questions]
        out(f"  diagnostic round {sheet.round_number}: {len(answers)} question(s) answered")
        task = await container.task_service.submit_assessment(task.task_id, DiagnosticAnswers(answers=answers))
    return task


async def lesson_and_evaluation(container: Container, action: NextLearningAction, key: dict, out,
                                check: Checks) -> Task | None:
    service = container.curriculum_service
    lesson = await service.start_lesson(action, user_id="demo-user")
    lesson = await answer_diagnostic(container, lesson, out)
    if not check(lesson.status == TaskStatus.COMPLETED, f"the {action.action.value} lesson completed"):
        for err in lesson.errors:
            out(f"  error: [{err.kind}] {err.message}")
        return None
    arts = {a.name: a for a in container.task_service.artifacts(lesson.task_id)}
    plan = PedagogicalPlan.model_validate_json(container.artifacts.read(arts["pedagogical_plan"].artifact_id))
    out(f"  lesson task {lesson.task_id}: plan targets {plan.target_concepts}, "
        f"prerequisite review {plan.prerequisite_concepts or '-'}, spaced review {plan.review_concepts or '-'}")
    out(f"  focus: {plan.focus.action if plan.focus else '-'} {plan.focus.concept_id if plan.focus else ''} "
        f"({plan.focus.description if plan.focus else ''})")
    for step in plan.sequencing:
        activity = next(a for a in plan.activities if a.activity_id == step.activity_id)
        out(f"    {step.order:>2}. {step.phase:<19} {activity.type:<16} {step.minutes} min  {activity.concept_ids[0]}")
    objective_artifact = plan.focus.objective_artifact_id if plan.focus else None
    lineage = [a.type.value for a in container.artifacts.lineage(arts["lesson"].artifact_id)]
    check(plan.focus is not None and plan.focus.concept_id == action.concept_id
          and plan.target_concepts == [action.concept_id], "the lesson targets exactly the selected objective")
    check(objective_artifact is not None and objective_artifact in arts["lesson"].parent_ids
          and "LEARNING_GOAL" in lineage and "CURRICULUM_VERSION" in lineage,
          "lesson lineage: LEARNING_GOAL -> CURRICULUM_VERSION -> LEARNING_OBJECTIVE -> LESSON")

    evaluation = await container.task_service.start_evaluation(lesson.task_id, user_id="demo-user")
    if not check(evaluation.status == TaskStatus.WAITING and evaluation.waiting is not None,
                 "the evaluation waits for the learner's answers"):
        return None
    sheet = AssessmentSheet.model_validate(evaluation.waiting.prompt)
    answers = [LearnerAnswer(question_id=q.question_id, answer=key.get(q.concept_id, {}).get(q.kind, ""))
               for q in sheet.questions]
    for q, a in zip(sheet.questions, answers):
        out(f"    [{q.kind}] {q.prompt}  -> {a.answer!r}")
    evaluation = await container.task_service.submit_answers(
        evaluation.task_id, {"answers": [a.model_dump(mode="json") for a in answers]})
    if not check(evaluation.status == TaskStatus.COMPLETED and evaluation.result is not None,
                 "the evaluation completed"):
        for err in evaluation.errors:
            out(f"  error: [{err.kind}] {err.message}")
        return None
    return evaluation


async def run_curriculum_demo(container: Container, *, out=print) -> bool:
    check = Checks(out)
    fixture = _fixture("learner.json")
    key = _fixture("evaluation_answers.json")["answers"]
    learner_id = fixture["learner_id"]
    service = container.curriculum_service

    out(RULE + "\nStep 1: create the learner\n" + RULE)
    container.learner_service.upsert(learner_id, LearnerProfileInput.model_validate(fixture["profile"]))
    for domain, evidence in placement_evidence(learner_id, fixture["placement"]).items():
        await container.learner_service.record_evidence(learner_id, domain, evidence)
    out(f"learner {learner_id}: placement evidence recorded for {len(fixture['placement'])} concepts")
    no_goal = await service.next_action(learner_id)
    check(no_goal.action == LearningActionType.WAIT, "without goals the learner has no curriculum action (WAIT)")

    out("\n" + RULE + "\nStep 2: create the learning goal\n" + RULE)
    goal, created = await service.create_goal(learner_id, GoalInput.model_validate(fixture["goal"]))
    again, created_again = await service.create_goal(learner_id, GoalInput.model_validate(fixture["goal"]))
    out(f"goal {goal.goal_id}: {goal.title!r} ({goal.domain}, target {goal.target_level}, priority {goal.priority})")
    out(f"  target concepts (from the knowledge base up to {goal.target_level}): {', '.join(goal.target_concepts)}")
    check(created and not created_again and again.goal_id == goal.goal_id,
          "the same idempotency key returns the same goal (no duplicate)")
    check(goal.target_concepts == [A, B, C, D], "B1 targets every B1-and-below concept, not the B2 one")

    out("\n" + RULE + "\nStep 3: build the curriculum\n" + RULE)
    build = await service.build_curriculum(goal.goal_id, user_id="demo-user")
    if not check(build.status == TaskStatus.COMPLETED and build.result is not None and build.result.curriculum,
                 "the curriculum planning task completed"):
        for err in build.errors:
            out(f"  error: [{err.kind}] {err.message}")
        return False
    plan = build.result.curriculum
    out(f"planning task {build.task_id}: version {plan.version} ({plan.version_id}), reasons "
        f"{[r.value for r in plan.reasons]}")
    out(f"  rationale (model-worded): {plan.rationale}")
    out(f"  proposal review: accepted {plan.proposal_review.accepted}, unresolved {plan.proposal_review.unresolved}")
    rebuild = await service.build_curriculum(goal.goal_id, user_id="demo-user")
    same = rebuild.result.curriculum if rebuild.result else None
    check(same is not None and not same.changed and same.version_id == plan.version_id
          and len(service.versions(goal.goal_id)) == 1, "building again stores no second version (same content)")

    out("\n" + RULE + "\nStep 4: curriculum objectives\n" + RULE)
    cur = service.curriculum(goal.goal_id)
    assert cur is not None
    print_curriculum(out, cur)
    check([o.concept_id for o in cur.objectives] == [A, B, C, D], "objectives in prerequisite order: A, B, C, D")

    out("\n" + RULE + "\nStep 5: prerequisites (from the knowledge base, never model text)\n" + RULE)
    for o in cur.objectives:
        out(f"  {o.concept_id:<18} requires {', '.join(o.prerequisites) or '-'}")

    out("\n" + RULE + "\nStep 6: mastery\n" + RULE)
    for o in cur.objectives:
        p = cur.progress.of(o.objective_id)
        out(f"  {o.concept_id:<18} {p.current_mastery:.2f}  ({p.evidence_count} evidence)")

    out("\n" + RULE + "\nStep 7: next learning action\n" + RULE)
    first = await service.next_action(learner_id)
    print_action(out, "next action #1:", first)
    check(first.action == LearningActionType.LEARN and first.concept_id == B,
          "LEARN B: the weakest unblocked objective")
    check(not any(c.concept_id in (C, D) and c.eligible for c in first.candidates),
          "C and D are not eligible while B is below the prerequisite threshold")
    repeat = await service.next_action(learner_id, as_of=first.as_of)
    check(repeat.model_dump(exclude={"candidates"}) == first.model_dump(exclude={"candidates"}),
          "the same state at the same time selects the same action")

    out("\n" + RULE + "\nStep 8-9: a lesson from the objective, then its evaluation\n" + RULE)
    before = service.curriculum(goal.goal_id)
    evaluation = await lesson_and_evaluation(container, first, key, out, check)
    if evaluation is None:
        return False

    out("\n" + RULE + "\nStep 10: mastery update (deterministic, from the graded answers)\n" + RULE)
    for change in evaluation.result.mastery_changes:
        out(f"  {change.concept_id:<18} {change.before:.2f} -> {change.after:.2f}  ({change.reason})")

    out("\n" + RULE + "\nStep 11: recalculated curriculum progress\n" + RULE)
    after = service.curriculum(goal.goal_id)
    assert before is not None and after is not None
    for o in after.objectives:
        b, a = before.progress.of(o.objective_id), after.progress.of(o.objective_id)
        out(f"  {o.concept_id:<18} {b.current_mastery:.2f} {b.status.value:<12} -> {a.current_mastery:.2f} "
            f"{a.status.value}")
    check(after.progress.of(after.objectives[1].objective_id).current_mastery
          > before.progress.of(before.objectives[1].objective_id).current_mastery, "B's mastery rose")

    out("\n" + RULE + "\nStep 12: the next action changes with the learner's mastery\n" + RULE)
    second = evaluation.result.learning_action
    assert second is not None
    print_action(out, "next action #2 (selected by the evaluation):", second)
    check(second.action != first.action or second.concept_id != first.concept_id, "a different next action")
    check(second.concept_id in (C, D), "now C or D: B is secure enough to build on")

    out("\n" + RULE + "\nStep 13: curriculum progress\n" + RULE)
    out(f"  {after.progress.mastered}/{after.progress.required} required objectives mastered "
        f"({after.progress.percent_complete:.0f}%), completion rule {after.progress.completion_rule}")

    out("\n" + RULE + "\nGoal change: B1 -> B2 (a new curriculum version; history is kept)\n" + RULE)
    v1 = service.versions(goal.goal_id)[-1]
    goal, replan = await service.update_goal(goal.goal_id, GoalUpdate(target_level="B2"), user_id="demo-user")
    versions = service.versions(goal.goal_id)
    out(f"  replanning task {replan.task_id if replan else '-'}; versions: "
        + ", ".join(f"v{v.version} {v.version_id} {[r.value for r in v.reasons]}" for v in versions))
    check(replan is not None and len(versions) == 2 and versions[-1].version_id != v1.version_id,
          "the goal change replanned the curriculum as version 2 with a new id")
    check(service.engine.repository.version(v1.version_id) is not None and versions[0].objectives == v1.objectives,
          "version 1 is still readable, unchanged")
    check([o.concept_id for o in versions[-1].objectives] == [A, B, C, D, E], "version 2 adds E (B2)")

    out("\n" + RULE + "\nA second goal with a target date, and selection across goals\n" + RULE)
    second_goal = dict(fixture["second_goal"])
    days = second_goal.pop("target_date_days")
    second_goal["target_date"] = (utcnow() + timedelta(days=days)).isoformat()
    maths, _ = await service.create_goal(learner_id, GoalInput.model_validate(second_goal))
    maths_build = await service.build_curriculum(maths.goal_id, user_id="demo-user")
    warnings = maths_build.result.curriculum.warnings if maths_build.result and maths_build.result.curriculum else []
    for w in warnings:
        out(f"  warning {w.code}: {w.message}")
        out(f"    {w.details}")
    check(any(w.code == "deadline_infeasible" for w in warnings),
          "an unreachable target date is a structured warning (the plan is not compressed)")
    across = await service.next_action(learner_id)
    print_action(out, "next action across both goals:", across)
    check(across.goal_id in (goal.goal_id, maths.goal_id) and across.priority is not None,
          "one goal and objective selected, with a deterministic reason and score")

    out("\n" + RULE + "\nStep 14: goal completion (the completion rule, never a model)\n" + RULE)
    for cid in (B, C, D, E):
        await container.learner_service.record_evidence(learner_id, "spanish",
                                                        exercise_evidence(learner_id, cid, 4, "mastery-practice"))
    done = await service.next_action(learner_id)
    print_action(out, "next action after the learner mastered every objective:", done)
    goal = service.goal(goal.goal_id)
    cur = service.curriculum(goal.goal_id)
    assert cur is not None
    out(f"  goal status: {goal.status.value}; curriculum {cur.status.value}; "
        f"{cur.progress.mastered}/{cur.progress.required} mastered")
    check(done.action == LearningActionType.COMPLETE and done.goal_id == goal.goal_id,
          "COMPLETE: every required objective is mastered")
    check(goal.status == GoalStatus.COMPLETED, "the goal is stored as completed")
    later = await service.next_action(learner_id, as_of=utcnow() + timedelta(days=120))
    print_action(out, "next action four months later:", later)
    check(any(c.action == LearningActionType.REVIEW and c.goal_id == goal.goal_id for c in later.candidates)
          and not any(c.goal_id == goal.goal_id and c.action != LearningActionType.REVIEW for c in later.candidates),
          "the completed goal's mastered objectives come back only as REVIEW")

    out("\n" + RULE)
    out("all checks passed" if not check.failed else f"{len(check.failed)} check(s) failed: {check.failed}")
    return not check.failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database and artifacts (default: a fresh temporary directory)")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args()
    configure_logging("INFO" if args.verbose else "WARNING")

    with tempfile.TemporaryDirectory(prefix="teaching-agent-curriculum-demo-") as tmp:
        container = build_container(Settings(data_dir=args.data_dir or Path(tmp), corpus_dir=FIXTURES))
        try:
            ok = asyncio.run(run_curriculum_demo(container))
        finally:
            container.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
