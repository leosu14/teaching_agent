"""End-to-end demo of the first vertical slice.

    python scripts/run_demo.py

Creates a learner, asks for "Create an A2 lesson about football.", answers the adaptive diagnostic
with the fixture learner's answers, and prints the outcome. It uses the same TaskService as the API.
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
from app.schemas.learner import LearnerProfileInput  # noqa: E402
from app.schemas.lesson import DiagnosticAnswers, DiagnosticQuestionSheet, LearnerAnswer  # noqa: E402
from app.schemas.task import Task, TaskStatus  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402

FIXTURES = REPO_ROOT / "fixtures" / "demo"


def simulated_answers(sheet: DiagnosticQuestionSheet, answer_key: dict) -> DiagnosticAnswers:
    """The fixture learner answers each question by concept for the given round."""
    by_concept = answer_key["rounds"][sheet.round_number - 1]
    return DiagnosticAnswers(answers=[
        LearnerAnswer(question_id=q.question_id, answer=by_concept.get(q.concept_id, ""))
        for q in sheet.questions
    ])


async def run_demo(container: Container, *, out=print) -> Task:
    fixture = json.loads((FIXTURES / "learner.json").read_text(encoding="utf-8"))
    answer_key = json.loads((FIXTURES / "answers.json").read_text(encoding="utf-8"))
    learner_id = fixture["learner_id"]
    container.learner_service.upsert(learner_id, LearnerProfileInput.model_validate(fixture["profile"]))

    out(f"Request: {fixture['request']!r} for learner {learner_id}")
    task = await container.task_service.create_and_run(request=fixture["request"], learner_id=learner_id,
                                                       user_id="demo-user")
    while task.status == TaskStatus.WAITING:
        assert task.waiting is not None
        sheet = DiagnosticQuestionSheet.model_validate(task.waiting.prompt)
        answers = simulated_answers(sheet, answer_key)
        out(f"\nWAITING for diagnostic round {sheet.round_number}:")
        for q, a in zip(sheet.questions, answers.answers):
            out(f"  Q: {q.prompt}\n     learner answers: {a.answer!r}")
        task = await container.task_service.submit_assessment(task.task_id, answers)

    report(container, task, out)
    return task


def report(container: Container, task: Task, out=print) -> None:
    out("\n" + "=" * 72)
    out(f"task_id:       {task.task_id}")
    out(f"final status:  {task.status.value}")
    if task.errors:
        for err in task.errors:
            out(f"error:         [{err.kind}] {err.message}")
    if task.plan:
        req = task.plan.lesson_request
        out(f"interpreted:   subject={req.subject} topic={req.topic} framework={req.framework_id} "
            f"level={req.target_level} workflow={task.plan.workflow_id}")

    out("\nworkflow steps:")
    for step in task.steps():
        duration = f"{step.duration_ms:.1f} ms" if step.duration_ms is not None else "-"
        out(f"  {step.node_id:<22} {step.status.value:<10} attempts={step.attempts} {duration}")

    if task.result:
        r = task.result
        out(f"\nlesson:        {r.title} (estimated level {r.estimated_level})")
        out(f"review:        {r.review_verdict} after {r.revisions} revision(s)")
        names = {a.artifact_id: a.name for a in r.artifacts}
        out("\ngenerated artifacts:")
        for a in r.artifacts:
            parents = ", ".join(names.get(p, p) for p in a.parent_ids) or "-"
            out(f"  {a.type.value:<11} {a.name:<17} v{a.version}  parents: {parents}")
            out(f"              {a.uri}")
        out("\nlearner mastery changes:")
        for c in r.mastery_changes:
            out(f"  {c.concept_id:<30} {c.before:.2f} -> {c.after:.2f}  ({c.reason})")

    cost = task.cost
    out("\ntoken usage:")
    out(f"  input={cost.token_usage.input_tokens} output={cost.token_usage.output_tokens} "
        f"total={cost.token_usage.total_tokens} llm_calls={cost.llm_calls}")
    for agent_id, line in sorted(cost.by_agent.items()):
        out(f"  {agent_id:<22} calls={line.calls} tokens={line.usage.total_tokens} ${line.cost_usd:.6f}")
    out(f"estimated cost: ${cost.estimated_cost_usd:.6f}")
    out(f"actual cost:    ${cost.actual_cost_usd:.6f}")
    events = container.task_service.events(task.task_id)
    out(f"events recorded: {len(events)} (GET /tasks/{task.task_id}/events)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database and artifacts (default: a fresh temporary directory)")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args()
    configure_logging("INFO" if args.verbose else "WARNING")

    with tempfile.TemporaryDirectory(prefix="teaching-agent-demo-") as tmp:
        data_dir = args.data_dir or Path(tmp)
        container = build_container(Settings(data_dir=data_dir))
        try:
            task = asyncio.run(run_demo(container))
        finally:
            container.close()
        if args.data_dir is None:
            print("\n(artifacts were written to a temporary directory; pass --data-dir to keep them)")
    return 0 if task.status == TaskStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
