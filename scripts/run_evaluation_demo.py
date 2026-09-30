"""End-to-end demo of the post-lesson evaluation loop.

    python scripts/run_evaluation_demo.py

Runs the lesson demo, starts a LearnerEvaluationAgent task for the completed lesson, prints the
assessment while the task is WAITING, submits the fixture learner's answers, and prints the
evaluation, the mastery changes and the next-learning recommendation. It uses the same TaskService
as the API (POST /tasks/{id}/evaluation, then POST /tasks/{id}/answers).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config.settings import Settings  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.schemas.evaluation import AssessmentResponse, AssessmentSheet, LearnerEvaluationReport  # noqa: E402
from app.schemas.lesson import LearnerAnswer  # noqa: E402
from app.schemas.task import Task, TaskStatus  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402
from scripts.run_demo import FIXTURES, run_demo  # noqa: E402


def simulated_answers(sheet: AssessmentSheet, answer_key: dict) -> AssessmentResponse:
    """The fixture learner answers each question by its concept and kind."""
    by_concept = answer_key["answers"]
    return AssessmentResponse(answers=[
        LearnerAnswer(question_id=q.question_id, answer=by_concept.get(q.concept_id, {}).get(q.kind, ""))
        for q in sheet.questions
    ])


def _mastery(container: Container, task: Task, concept_ids: list[str]) -> dict[str, float]:
    concepts = container.learner_service.get(task.learner_id).concepts
    return {cid: concepts[cid].mastery if cid in concepts else 0.0 for cid in concept_ids}


async def run_evaluation_demo(container: Container, *, out=print) -> Task:
    answer_key = json.loads((FIXTURES / "evaluation_answers.json").read_text(encoding="utf-8"))
    out("=" * 72 + "\nStep 1: complete a lesson\n" + "=" * 72)
    lesson = await run_demo(container, out=out)
    if lesson.status != TaskStatus.COMPLETED:
        out(f"\nlesson did not complete ({lesson.status.value}); nothing to evaluate")
        return lesson
    out(f"\nlesson completed: {lesson.task_id}")

    out("\n" + "=" * 72 + "\nStep 2: evaluate what the learner took away\n" + "=" * 72)
    task = await container.task_service.start_evaluation(lesson.task_id, user_id="demo-user")
    out(f"evaluation task: {task.task_id}  status: {task.status.value}")
    if task.status != TaskStatus.WAITING or task.waiting is None:
        return _finish(container, task, {}, out)

    sheet = AssessmentSheet.model_validate(task.waiting.prompt)
    concept_ids = list(dict.fromkeys(q.concept_id for q in sheet.questions))
    before = _mastery(container, task, concept_ids)
    out(f"\nassessment generated: {sheet.title!r}, {len(sheet.questions)} question(s)")
    out(f"task is {task.status.value} for {task.waiting.kind}; the learner answers:")
    answers = simulated_answers(sheet, answer_key)
    for q, a in zip(sheet.questions, answers.answers):
        choices = f"  choices: {q.choices}" if q.choices else ""
        out(f"  [{q.kind}] {q.prompt}{choices}\n     learner answers: {a.answer!r}")

    task = await container.task_service.submit_answers(task.task_id, answers.model_dump(mode="json"))
    out(f"\nanswers submitted; task status: {task.status.value}")
    return _finish(container, task, before, out)


def _finish(container: Container, task: Task, before: dict[str, float], out) -> Task:
    for err in task.errors:
        out(f"error: [{err.kind}] {err.message}")
    if task.status != TaskStatus.COMPLETED or task.result is None:
        return task
    artifact = next(a for a in task.result.artifacts if a.name == "learner_evaluation")
    _, raw = container.task_service.artifact_content(artifact.artifact_id)
    report = LearnerEvaluationReport.model_validate_json(raw)

    out("\nevaluation result:")
    out(f"  score {report.score:.0%} ({report.points_earned:g}/{report.points_possible:g} points)")
    for e in report.evaluations:
        out(f"  {e.question_id:<34} {'correct' if e.correct else 'incorrect'}")
    for c in report.concepts:
        out(f"  concept {c.concept_id:<30} {c.status:<8} {c.score:.0%}")

    after = _mastery(container, task, list(before))
    out("\nmastery before -> after:")
    for cid in before:
        out(f"  {cid:<30} {before[cid]:.2f} -> {after[cid]:.2f}")
    out(f"\nremaining gaps:  {', '.join(report.remaining_gaps) or 'none'}")
    out(f"partial:         {', '.join(report.partial) or 'none'}")
    out(f"mastered:        {', '.join(report.mastered) or 'none'}")
    rec = report.recommendation
    out(f"\nnext recommendation: {rec.action} (focus: {', '.join(rec.focus_concepts) or '-'})")
    out(f"  {rec.rationale}")
    out(f"  suggested request: {rec.suggested_request!r}")
    out(f"\nevaluation artifact: {artifact.uri}")
    out(f"evaluation cost: ${task.cost.actual_cost_usd:.6f} over {task.cost.llm_calls} LLM call(s)")
    out(f"events recorded: {len(container.task_service.events(task.task_id))} "
        f"(GET /tasks/{task.task_id}/events)")
    return task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database and artifacts (default: a fresh temporary directory)")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args()
    configure_logging("INFO" if args.verbose else "WARNING")

    with tempfile.TemporaryDirectory(prefix="teaching-agent-eval-demo-") as tmp:
        container = build_container(Settings(data_dir=args.data_dir or Path(tmp)))
        try:
            task = asyncio.run(run_evaluation_demo(container))
        finally:
            container.close()
        if args.data_dir is None:
            print("\n(artifacts were written to a temporary directory; pass --data-dir to keep them)")
    return 0 if task.status == TaskStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
