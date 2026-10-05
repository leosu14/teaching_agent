"""End-to-end demo of the semantic assessment and rubric engine.

    python scripts/run_assessment_demo.py

The curriculum demo's Spanish learner has secured the preterite, so the curriculum's next objective is the
preterite/imperfect contrast; its lesson is generated, and three assessment items on it are registered
(fixtures/assessment/items.json). A scripted learner then answers through the assessment service:

  1. an exact-match incorrect answer ("hablo" for "hablé"): INCORRECT by deterministic matching, no model call;
  2. incorrect answers with misconceptions: a free-text answer the (mock) semantic grader finds habitual-past
     reasoning in, and a known wrong form ("hablaba") caught by a deterministic rule; both recorded as evidence;
  3. the next learning action changes: three misses in a row make the curriculum reteach the assessed concept;
  4. an exact-match correct answer ("Hablé." for "hablé"): normalisation alone, no model call;
  5. a semantically equivalent free-text answer ("Porque la acción ya terminó." for "The action is completed."):
     credited by the semantic grader, validated, every criterion met, CORRECT; the curriculum moves on again;
  6. a partial-credit answer: one of two rubric criteria met, PARTIAL with a score of 0.4;
  7. an answer the grader cannot account for: low confidence, UNCERTAIN, no mastery change, a retry recommended;
     the retry is another attempt (the first is kept);
  8. rubric aggregation on a deterministic rubric: the sum of score x weight, no model call;
  9. the deterministic mastery update made by learner memory's existing updater from every graded answer; a
     duplicate submission (same attempt id) is replayed and changes nothing, a different answer under it conflicts.

The semantic grader is the mock provider's deterministic responder, so the whole run is repeated on a fresh data
directory and the grades must be identical. Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_curriculum_demo as curriculum_demo  # noqa: E402
import run_interactive_demo as interactive_demo  # noqa: E402

from app.observability.logging import configure_logging  # noqa: E402
from app.schemas.assessment import (  # noqa: E402
    AssessmentItem,
    AssessmentRubric,
    AttemptConflict,
    AttemptResult,
    AttemptSubmission,
    RegisterAssessmentItem,
)
from app.services.container import Container, build_container  # noqa: E402

FIXTURES = REPO_ROOT / "fixtures" / "assessment"
RULE = "=" * 72
Checks = curriculum_demo.Checks
ASSESSMENT_EVENTS = ("assessment.", "misconception.")


def fixture() -> dict:
    return json.loads((FIXTURES / "items.json").read_text(encoding="utf-8"))


def register(container: Container, lesson_task: str, objective_id: str | None) -> dict[str, dict]:
    """The fixture's items on the lesson; ids are prefixed with the lesson so a data directory can hold several."""
    items = {}
    for entry in fixture()["items"]:
        raw = dict(entry["item"])
        key = raw["assessment_item_id"]
        raw.update(assessment_item_id=f"{lesson_task}.{key}", lesson_id=lesson_task, objective_id=objective_id)
        rubric = None
        if "rubric" in entry:
            rubric = AssessmentRubric.model_validate(
                {**entry["rubric"], "rubric_id": f"{lesson_task}.{entry['rubric']['rubric_id']}"})
            raw["rubric_id"] = rubric.rubric_id
        item = container.assessment_service.register(
            RegisterAssessmentItem(item=AssessmentItem.model_validate(raw), rubric=rubric))
        items[key] = {"item": item, "rubric": rubric, "answers": entry["answers"]}
    return items


def mastery(container: Container, learner_id: str) -> float:
    return container.memory.get(learner_id).concepts[curriculum_demo.C].mastery


def print_result(out, label: str, r: AttemptResult, *, usage=None) -> None:
    g = r.grade
    out(f"  learner: {r.attempt.learner_answer!r}  (attempt {r.attempt.attempt_number})")
    out(f"  -> {g.outcome.value}  score {g.score:g}/{g.max_score:g}  confidence {g.confidence:g}  "
        f"graded by {g.grader_type.value}")
    for c in g.criterion_results:
        out(f"     criterion {c.criterion_id:<10} {c.score:g} x {c.weight:g}  {c.rationale}")
    for m in g.misconceptions:
        out(f"     misconception {m.type} ({m.source}, {m.confidence:g}): {m.description}")
    f = g.feedback
    for s in f.strengths:
        out(f"     + {s}")
    for e in f.errors:
        out(f"     - {e}")
    if f.explanation:
        out(f"     why: {f.explanation}")
    if f.next_hint:
        out(f"     next: {f.next_hint}")
    if usage is not None and usage.llm_calls:
        out(f"     grader: {usage.llm_calls} call(s), {usage.provider}/{usage.model}, "
            f"{usage.input_tokens}+{usage.output_tokens} tokens, ${usage.estimated_cost_usd or 0:.6f}")
    for change in r.mastery_changes:
        out(f"     mastery {change['concept_id']}: {change['before']:.3f} -> {change['after']:.3f}")


async def run_assessment_demo(data_dir: Path, *, out=print) -> tuple[bool, list]:
    """Runs the scenario; returns whether every check passed and the grades' fingerprints."""
    check = Checks(out)
    fingerprints: list = []
    container = build_container(interactive_demo.settings(data_dir))
    try:
        out(RULE + "\nThe learner, the curriculum and the lesson\n" + RULE)
        lesson_task = await interactive_demo.prepare(container, out, check)
        lesson = container.task_service.get(lesson_task)
        learner_id = lesson.learner_id
        assert learner_id is not None
        goal = container.memory.goals(learner_id)[0]
        curriculum = container.curriculum_service.curriculum(goal.goal_id)
        assert curriculum is not None
        objective = next(o for o in curriculum.objectives if o.concept_id == curriculum_demo.C)
        items = register(container, lesson_task, objective.objective_id)
        out(f"  registered {len(items)} assessment items on the lesson (objective {objective.concept_id})")
        service = container.assessment_service
        counter = Counter()

        async def submit(key: str, say: str, attempt_id: str | None = None) -> AttemptResult:
            counter[key] += 1
            entry = items[key]
            r = await service.submit_attempt(entry["item"].assessment_item_id, AttemptSubmission(
                learner_id=learner_id, answer=entry["answers"][say],
                attempt_id=attempt_id or f"{lesson_task}.{key}.{counter[key]}"))
            stored = service.stored_grade(r.attempt.attempt_id)
            print_result(out, say, r, usage=stored.grader)
            fingerprints.append((key, say, r.grade.outcome.value, r.grade.score, r.grade.confidence,
                                 [(c.criterion_id, c.score) for c in r.grade.criterion_results],
                                 [m.type for m in r.grade.misconceptions]))
            return r

        def calls(r: AttemptResult) -> int:
            usage = service.stored_grade(r.attempt.attempt_id).grader
            return usage.llm_calls if usage else 0

        conjugate, why, story = items["conjugate"], items["why_preterite"], items["explain_story"]
        start = await container.curriculum_service.next_action(learner_id)

        out("\n" + RULE + "\n1. Exact match, incorrect\n" + RULE)
        out(f"  question: {conjugate['item'].prompt}")
        r1 = await submit("conjugate", "wrong")
        check(r1.grade.outcome.value == "INCORRECT" and r1.grade.grader_type.value == "EXACT" and calls(r1) == 0,
              "'hablo' does not match 'hablé': INCORRECT by deterministic matching, no model call")

        out("\n" + RULE + "\n2. Incorrect, with a misconception\n" + RULE)
        out(f"  question: {why['item'].prompt}")
        out(f"  expected meaning: {why['item'].expected_answer}")
        r2 = await submit("why_preterite", "misconception")
        check(r2.grade.outcome.value == "INCORRECT" and r2.grade.grader_type.value == "SEMANTIC"
              and [m.type for m in r2.grade.misconceptions] == ["preterite_as_habitual"],
              "'una acción habitual que duraba' is INCORRECT and shows the habitual-preterite misconception")
        r2b = await submit("conjugate", "known")
        check(r2b.grade.grader_type.value == "RULE" and calls(r2b) == 0
              and [(m.type, m.source) for m in r2b.grade.misconceptions] == [("imperfect_for_completed_action",
                                                                               "rule")],
              "'hablaba' is a known error: INCORRECT by rule with its misconception, no model call")
        recorded = [e for e in container.memory.evidence(learner_id)
                    if e.metadata.get("attempt_id") in (r2.attempt.attempt_id, r2b.attempt.attempt_id)]
        check(len(recorded) == 2 and all(e.correctness == "incorrect" and e.metadata["misconceptions"]
                                         for e in recorded),
              "each misconception is recorded with its learning evidence; mastery moves only by the graded score")

        out("\n" + RULE + "\n3. The next learning action changes\n" + RULE)
        reteach = await container.curriculum_service.next_action(learner_id)
        out(f"  before the answers: {start.action.value} {start.concept_id}")
        out(f"  now:                {reteach.action.value} {reteach.concept_id}  ({reteach.reason[:110]}...)")
        check(start.concept_id != curriculum_demo.C and reteach.concept_id == curriculum_demo.C
              and reteach.action.value == "LEARN" and "in a row" in reteach.reason
              and r2b.learning_action is not None and r2b.learning_action["concept_id"] == curriculum_demo.C,
              "three incorrect graded answers in a row switch the next action to reteaching the assessed concept")

        out("\n" + RULE + "\n4. Exact match, correct\n" + RULE)
        out(f"  question: {conjugate['item'].prompt}")
        r4 = await submit("conjugate", "exact")
        check(r4.grade.outcome.value == "CORRECT" and r4.grade.grader_type.value == "EXACT" and calls(r4) == 0
              and r4.attempt.attempt_number == 3,
              "'Hablé.' matches 'hablé' after normalisation: CORRECT, no model call; attempts 1 and 2 are kept")

        out("\n" + RULE + "\n5. Semantic equivalence\n" + RULE)
        out(f"  question: {why['item'].prompt}")
        r5 = await submit("why_preterite", "semantic")
        check(r5.grade.outcome.value == "CORRECT" and r5.grade.grader_type.value == "SEMANTIC"
              and r5.grade.score == 1.0 and calls(r5) == 1,
              "'Porque la acción ya terminó.' gets semantic credit: CORRECT, one validated grader call")
        check(bool(r5.grade.feedback.citations), "the feedback cites the lesson it was grounded in")
        moved = await container.curriculum_service.next_action(learner_id)
        out(f"  next learning action: {moved.action.value} {moved.concept_id}")
        check(moved.concept_id == start.concept_id, "the streak is broken: the curriculum moves on again")

        out("\n" + RULE + "\n6. Partial credit\n" + RULE)
        r6 = await submit("why_preterite", "partial")
        check(r6.grade.outcome.value == "PARTIAL" and r6.grade.score == 0.4,
              "a bounded time without completion meets one criterion: PARTIAL, 0.6 x 0 + 0.4 x 1 = 0.4")

        out("\n" + RULE + "\n7. Uncertain\n" + RULE)
        mastery_before = mastery(container, learner_id)
        r7 = await submit("why_preterite", "uncertain")
        check(r7.grade.outcome.value == "UNCERTAIN" and r7.attempt.retry_recommended
              and not r7.attempt.mastery_updated and not r7.mastery_changes
              and mastery(container, learner_id) == mastery_before,
              "an answer the grader cannot account for is UNCERTAIN: no evidence, no mastery change, retry")
        r7b = await submit("why_preterite", "semantic")
        attempts = service.attempts(why["item"].assessment_item_id, learner_id)
        check(r7b.grade.outcome.value == "CORRECT"
              and [a.outcome.value for a in attempts] == ["INCORRECT", "CORRECT", "PARTIAL", "UNCERTAIN", "CORRECT"],
              "the retry is attempt 5; every attempt is kept, the UNCERTAIN one too")

        out("\n" + RULE + "\n8. Rubric aggregation\n" + RULE)
        out(f"  question: {story['item'].prompt}")
        r8 = await submit("explain_story", "rubric")
        expected = sum(c.score * c.weight for c in r8.grade.criterion_results)
        check(r8.grade.grader_type.value == "RUBRIC" and calls(r8) == 0 and abs(r8.grade.score - 0.8) < 1e-9
              and abs(expected - 0.8) < 1e-9 and r8.grade.outcome.value == "CORRECT",
              "deterministic rubric: 1 x 0.5 + 1 x 0.3 + 0 x 0.2 = 0.8 meets the passing threshold, no model call")

        out("\n" + RULE + "\n9. Deterministic mastery update and idempotency\n" + RULE)
        graded = (r1, r2, r2b, r4, r5, r6, r7b, r8)
        for r in graded:
            for c in r.mastery_changes:
                out(f"  {r.grade.outcome.value:<9} {r.grade.grader_type.value:<8} mastery {c['concept_id']}: "
                    f"{c['before']:.3f} -> {c['after']:.3f}")
        changes = [c for r in graded for c in r.mastery_changes if c["concept_id"] == curriculum_demo.C]
        check(len(changes) == len(graded) and all(r.attempt.mastery_updated for r in graded)
              and changes[3]["after"] > changes[3]["before"] and changes[0]["after"] < changes[0]["before"],
              "every graded attempt updated mastery through learner memory's updater (the UNCERTAIN one did not)")
        evidence_count = len(container.memory.evidence(learner_id))
        mastery_now = mastery(container, learner_id)
        again = await service.submit_attempt(why["item"].assessment_item_id, AttemptSubmission(
            learner_id=learner_id, answer=why["answers"]["semantic"], attempt_id=r5.attempt.attempt_id))
        check(again.replayed and again.grade.grade_id == r5.grade.grade_id
              and len(container.memory.evidence(learner_id)) == evidence_count
              and mastery(container, learner_id) == mastery_now,
              "the same attempt submitted again is replayed: same grade, no new evidence, mastery unchanged")
        try:
            await service.submit_attempt(why["item"].assessment_item_id, AttemptSubmission(
                learner_id=learner_id, answer="Otra respuesta.", attempt_id=r5.attempt.attempt_id))
            conflict = False
        except AttemptConflict:
            conflict = True
        check(conflict, "a different answer under the same attempt id is a deterministic conflict")
        progress = r8.objective_progress or {}
        out(f"  objective {progress.get('concept_id')}: {progress.get('status')}, mastery "
            f"{progress.get('current_mastery')} of {progress.get('target_mastery')}")
        check(progress.get("concept_id") == curriculum_demo.C, "the objective's progress is recalculated")

        out("\n" + RULE + "\nArtifacts and events\n" + RULE)
        arts = container.task_service.artifacts(lesson_task)
        grade_art = next(a for a in arts if a.type.value == "ASSESSMENT_GRADE"
                         and a.metadata.get("attempt_id") == r5.attempt.attempt_id)
        up = [a.type.value for a in container.artifacts.lineage(grade_art.artifact_id)]
        out("  lineage of a grade: " + " -> ".join(
            t for t in ("LEARNING_GOAL", "CURRICULUM_VERSION", "LEARNING_OBJECTIVE", "LESSON", "ASSESSMENT_RUBRIC",
                        "ASSESSMENT_ITEM") if t in up) + " -> ASSESSMENT_GRADE")
        check(all(t in up for t in ("LESSON", "ASSESSMENT_ITEM", "ASSESSMENT_RUBRIC")),
              "lineage: LESSON -> ITEM -> GRADE and RUBRIC -> GRADE")
        evidence_art = next(a for a in arts if a.type.value == "LEARNING_EVIDENCE"
                            and a.metadata.get("attempt_id") == r5.attempt.attempt_id)
        check(grade_art.artifact_id in evidence_art.parent_ids, "GRADE -> LEARNING_EVIDENCE")
        events = [e for e in container.task_service.events(lesson_task) if e.type.startswith(ASSESSMENT_EVENTS)]
        counts = Counter(e.type for e in events)
        out("  events: " + ", ".join(f"{t} x{n}" for t, n in sorted(counts.items())))
        check(counts["assessment.started"] == 9 and counts["assessment.completed"] == 9
              and counts["assessment.graded"] == 8 and counts["assessment.uncertain"] == 1
              and counts["misconception.detected"] == 2 and len({e.event_id for e in events}) == len(events),
              "every attempt emitted started, graded (or uncertain) and completed once; the replay emitted nothing")
        usage = [service.stored_grade(a.attempt_id).grader for e in items.values()
                 for a in service.attempts(e["item"].assessment_item_id, learner_id)]
        graded_by_model = [u for u in usage if u is not None and u.llm_calls]
        out(f"  semantic grader: {sum(u.llm_calls for u in graded_by_model)} call(s) for {len(usage)} attempts, "
            f"{sum(u.input_tokens + u.output_tokens for u in graded_by_model)} tokens, "
            f"${sum(u.estimated_cost_usd or 0 for u in graded_by_model):.6f}")
        check(len(usage) == 9 and len(graded_by_model) == 5 and all(u.llm_calls == 1 for u in graded_by_model),
              "only the 5 free-text answers called the grader (once each); exact, rule and rubric grading none")
    finally:
        container.close()
    out("\n" + RULE)
    out("all checks passed" if not check.failed else f"{len(check.failed)} check(s) failed: {check.failed}")
    return not check.failed, fingerprints


def main(argv: list[str] | None = None, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database and artifacts (default: a fresh temporary directory)")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args(argv)
    configure_logging("INFO" if args.verbose else "WARNING")
    with tempfile.TemporaryDirectory(prefix="teaching-agent-assessment-demo-") as tmp:
        ok, first = asyncio.run(run_assessment_demo(args.data_dir or Path(tmp) / "run", out=out))
    with tempfile.TemporaryDirectory(prefix="teaching-agent-assessment-demo-") as tmp:
        again, second = asyncio.run(run_assessment_demo(Path(tmp), out=lambda _line: None))
    same = again and first == second
    out(f"  [{'ok' if same else 'FAIL'}] a second run on a fresh data directory produces identical grades")
    return 0 if ok and same else 1


if __name__ == "__main__":
    raise SystemExit(main())
