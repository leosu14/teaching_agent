"""Learning cycles: one curriculum action executed end to end, then the next.

A Spanish learner (the curriculum demo's) has the active goal "Reach B1 Spanish". With
no curriculum built by hand and no step stitched together here, the demo only starts a cycle and answers what the
cycle asks:

  1. the learner has an active goal; the cycle builds its curriculum and the curriculum selects the next action;
  2. the cycle starts (LEARN the preterite) and runs the lesson workflow;
  3. the lesson's diagnostic needs the learner: the cycle is WAITING and the request returns;
  4. the learner answers; the lesson completes and the interactive session starts: WAITING again;
  5. the learner answers the session's questions (graded by the AssessmentService); the first one is wrong;
  6. the process restarts in the middle of the session; the cycle is read back and continues where it was;
  7. the session completes: evidence is stored, mastery is updated by learner memory, the curriculum recalculates,
     and the cycle completes with the next action;
  8. replays: the same response and the same start key execute nothing twice;
  9. the remaining objectives are secured and a new cycle verifies the goal's completion rule (COMPLETE).

It prints a concise trace, runs everything twice and exits 1 unless every check passes and both runs agree.

    python scripts/run_learning_cycle_demo.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_curriculum_demo as curriculum_demo  # noqa: E402

from app.config.settings import Settings  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.schemas.curriculum import GoalInput  # noqa: E402
from app.schemas.learner import LearnerProfileInput  # noqa: E402
from app.schemas.learning_cycle import (  # noqa: E402
    CycleResponse,
    CycleStatus,
    LearnerPrompt,
    LearningCycleView,
    StartLearningCycle,
)
from app.schemas.lesson import LearnerAnswer  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402

FIXTURES = REPO_ROOT / "fixtures" / "learning_cycle"
RULE = "=" * 72
Checks = curriculum_demo.Checks
CUE = re.compile(r"\(([^()]+)\)\s*$")


def scenario() -> dict:
    return json.loads((FIXTURES / "scenario.json").read_text(encoding="utf-8"))


def settings(data_dir: Path, **overrides) -> Settings:
    # The interactive demo's short session: two correct answers in a row raise the difficulty, one miss lowers it.
    return Settings(data_dir=data_dir, corpus_dir=curriculum_demo.FIXTURES,
                    **{"teaching_increase_after": 2, "teaching_decrease_after": 1, **overrides})


class ScriptedLearner:
    """Answers what a cycle asks from the knowledge base's probes: by the exact prompt (diagnostic) or by the verb cue
    in parentheses (session and evaluation questions). `wrong_first` is the first session answer."""

    def __init__(self, wrong_first: str | None = None, *, prefix: str = "r") -> None:
        self.prefix = prefix
        self.by_prompt: dict[str, str] = {}
        self.by_cue: dict[str, str] = {}
        self.by_concept: dict[str, str] = {}
        for doc in curriculum_demo._fixture("knowledge_base.json"):
            concept = doc["metadata"]["concept"]["concept_id"]
            for p in doc["metadata"]["probes"]:
                self.by_prompt[p["prompt"]] = p["answer"]
                self.by_concept.setdefault(concept, p["answer"])
                if (m := CUE.search(p["prompt"])) is not None:
                    self.by_cue[f"({m.group(1)})"] = p["answer"]
        self.by_cue.update(scenario().get("extra_cues", {}))
        self.wrong_first = wrong_first
        self.sent: list[CycleResponse] = []

    def answer(self, prompt: str, concept_id: str | None = None) -> str:
        if prompt in self.by_prompt:
            return self.by_prompt[prompt]
        for cue, right in self.by_cue.items():
            if cue in prompt:
                return right
        return self.by_concept.get(concept_id or "", "no sé")

    def respond(self, prompt: LearnerPrompt) -> CycleResponse:
        rid = f"{self.prefix}{len(self.sent) + 1}"
        if prompt.kind == "SESSION_QUESTION":
            assert prompt.question is not None
            answer, self.wrong_first = self.wrong_first or self.answer(prompt.question.prompt), None
            response = CycleResponse(client_response_id=rid, answer=answer)
        else:
            response = CycleResponse(client_response_id=rid, answers=[
                LearnerAnswer(question_id=q["question_id"], answer=self.answer(q["prompt"], q.get("concept_id")))
                for q in prompt.questions])
        self.sent.append(response)
        return response


async def prepare(container: Container, *, secured: list[str] | None = None) -> str:
    """The curriculum demo's learner and goal (no curriculum yet), with the scenario's concepts secured."""
    fx = curriculum_demo._fixture("learner.json")
    learner_id = fx["learner_id"]
    container.learner_service.upsert(learner_id, LearnerProfileInput.model_validate(fx["profile"]))
    for domain, evidence in curriculum_demo.placement_evidence(learner_id, fx["placement"]).items():
        await container.learner_service.record_evidence(learner_id, domain, evidence)
    for concept in scenario()["secured_concepts"] if secured is None else secured:
        await container.learner_service.record_evidence(
            learner_id, "spanish", curriculum_demo.exercise_evidence(learner_id, concept, 4, "secured"))
    await container.curriculum_service.create_goal(learner_id, GoalInput.model_validate(fx["goal"]))
    return learner_id


def describe(view: LearningCycleView) -> str:
    steps = ", ".join(f"{s.kind.value}{' (reused)' if s.reused else ''}:{s.status.value}" for s in view.steps) or "-"
    prompt = ""
    if view.prompt is not None:
        p = view.prompt
        prompt = f" -> {p.kind}: " + (p.question.prompt if p.question else f"{len(p.questions)} question(s)")
    return f"{view.status.value:<9} [{steps}]{prompt}"


async def answer_until_done(container: Container, view: LearningCycleView, learner: ScriptedLearner, out, *,
                            stop_after: int | None = None) -> LearningCycleView:
    """Answer every prompt until the cycle ends (or `stop_after` responses)."""
    sent = 0
    while view.status == CycleStatus.WAITING and view.prompt is not None and (stop_after is None or sent < stop_after):
        response = learner.respond(view.prompt)
        view = await container.learning_cycle_service.respond(view.cycle_id, response)
        sent += 1
        shown = response.answer if response.answer is not None else f"{len(response.answers or [])} answer(s)"
        out(f"  learner {response.client_response_id}: {shown:<14} cycle {describe(view)}")
    return view


def artifact_path(container: Container, start: str, target: str | None) -> list[str]:
    """The artifact types on the shortest parent path from `start` to `target` (or to the LEARNING_GOAL)."""
    paths = {start: [start]}
    frontier = [start]
    while frontier:
        current = frontier.pop(0)
        art = container.artifacts.get(current)
        if current == target or (target is None and art.type.value == "LEARNING_GOAL"):
            return [container.artifacts.get(a).type.value for a in paths[current]]
        for parent in art.parent_ids:
            if parent not in paths:
                paths[parent] = [*paths[current], parent]
                frontier.append(parent)
    return []


def mastery(container: Container, learner_id: str, concept: str) -> float:
    c = container.memory.get(learner_id).concepts.get(concept)
    return round(c.mastery, 3) if c else 0.0


async def run_learning_cycle_demo(data_dir: Path, *, out=print) -> tuple[bool, dict]:
    check = Checks(out)
    fx = scenario()
    container = build_container(settings(data_dir))
    summary: dict = {}
    try:
        out(RULE + "\n1-3. The learner, the goal, the next action, the cycle\n" + RULE)
        learner_id = await prepare(container)
        goals = container.curriculum_service.goals(learner_id)
        check(len(goals) == 1 and goals[0].is_active and container.curriculum_service.curriculum(goals[0].goal_id)
              is None, "the learner has one active goal and no curriculum yet")
        cycles = container.learning_cycle_service
        view = await cycles.start(learner_id, StartLearningCycle(idempotency_key="cycle-1", user_id="demo"))
        a = view.action
        out(f"cycle {view.cycle_id}: {a.action.value} {a.concept_id} (curriculum v{a.curriculum_version})")
        out(f"  why: {a.reason}")
        out(f"  cycle {describe(view)}")
        check(view.created and a.action.value == "LEARN" and a.concept_id == curriculum_demo.B,
              f"the curriculum selects LEARN {curriculum_demo.B}")
        check(view.status == CycleStatus.WAITING and view.prompt is not None
              and view.prompt.kind == "DIAGNOSTIC_QUESTIONS", "the lesson's diagnostic needs the learner: WAITING")
        check(all("answer" not in q for q in view.prompt.questions), "the prompt carries no answer key")

        out("\n" + RULE + "\n4-5. The learner answers; the lesson completes; the session teaches\n" + RULE)
        learner = ScriptedLearner(wrong_first=fx["wrong_first_answer"])
        view = await answer_until_done(container, view, learner, out, stop_after=3)
        check(view.steps[0].status.value == "COMPLETED" and view.steps[0].child_id is not None,
              "the lesson workflow completed")
        check(view.status == CycleStatus.WAITING and view.prompt is not None
              and view.prompt.kind == "SESSION_QUESTION", "the session waits for the learner")
        lesson_task = view.steps[0].child_id

        out("\n" + RULE + "\n6. A process restart in the middle of the session\n" + RULE)
        before = view
        container.close()
        container = build_container(settings(data_dir))
        cycles = container.learning_cycle_service
        view = await cycles.get(before.cycle_id)
        out(f"  read back: cycle {describe(view)}")
        check(view.status == before.status and view.prompt == before.prompt and view.steps == before.steps,
              "the cycle continues at the same question after the restart")

        out("\n" + RULE + "\n7. The session completes: evidence, mastery, curriculum, next action\n" + RULE)
        mastery_before = mastery(container, learner_id, curriculum_demo.B)
        view = await answer_until_done(container, view, learner, out)
        check(view.status == CycleStatus.COMPLETED and view.outcome is not None, "the cycle completed")
        o = view.outcome
        assert o is not None
        mastery_after = mastery(container, learner_id, curriculum_demo.B)
        out(f"  result {o.result.value}: {len(o.learning_evidence_ids)} evidence item(s), mastery "
            f"{curriculum_demo.B} {mastery_before} -> {mastery_after}")
        if o.objective_progress:
            p = o.objective_progress
            out(f"  objective progress: {p['status']} mastery {p['current_mastery']:.2f} of {p['target_mastery']:.2f},"
                f" evidence {p['evidence_count']}/{p['evidence_required']}")
        nxt = o.next_action
        out(f"  next action: {nxt.action.value if nxt else '-'} {nxt.concept_id if nxt else ''}")
        evidence = [e for e in container.memory.evidence(learner_id) if e.evidence_id in set(o.learning_evidence_ids)]
        check(len(evidence) == len(o.learning_evidence_ids) > 0, "the session's graded answers are stored as evidence")
        check(any(c["concept_id"] == curriculum_demo.B for c in o.mastery_changes) and mastery_after != mastery_before,
              "mastery was updated by learner memory")
        fresh = await container.curriculum_service.next_action(learner_id)
        check(nxt is not None and (nxt.action, nxt.concept_id) == (fresh.action, fresh.concept_id),
              "the next action is the curriculum's own")

        out("\n" + RULE + "\n8. Replays execute nothing twice\n" + RULE)
        tasks = len(container.task_service.list_for_learner(learner_id))
        evidence_count = len(container.memory.evidence(learner_id))
        events = Counter(e.type for e in cycles.events(view.cycle_id))
        replay = await cycles.start(learner_id, StartLearningCycle(idempotency_key="cycle-1", user_id="demo"))
        check(replay.cycle_id == view.cycle_id and not replay.created and replay.status == CycleStatus.COMPLETED,
              "the same start key returns the same completed cycle")
        again = await cycles.respond(view.cycle_id, learner.sent[-1])
        check(again.replayed and again.status == CycleStatus.COMPLETED, "the last response, sent again, is a replay")
        check(len(container.task_service.list_for_learner(learner_id)) == tasks
              and len(container.memory.evidence(learner_id)) == evidence_count
              and Counter(e.type for e in cycles.events(view.cycle_id)) == events,
              "no new lesson, evidence or event")
        out("  events: " + ", ".join(f"{t.split('.', 1)[1]} x{n}" for t, n in sorted(events.items())))
        check(all(n == 1 for t, n in events.items() if t in {"learning_cycle.started", "learning_cycle.action_selected",
                                                               "learning_cycle.action_completed",
                                                               "learning_cycle.completed"}),
              "each lifecycle event once")

        out("\n" + RULE + "\nArtifact lineage\n" + RULE)
        outcome_art = container.artifacts.get(o.artifact_ids["outcome"])
        evidence_art = o.artifact_ids["learning_evidence"]
        path = artifact_path(container, evidence_art, view.artifact_ids["cycle"]) + \
            artifact_path(container, view.artifact_ids["cycle"], None)[1:]
        out("  " + " -> ".join(reversed(path)))
        check(path[-4:] == ["LEARNING_CYCLE", "LEARNING_OBJECTIVE", "CURRICULUM_VERSION", "LEARNING_GOAL"]
              and "LESSON" in path and "ASSESSMENT_GRADE" not in path[:1],
              "Goal -> Curriculum -> Objective -> Learning Cycle -> Lesson -> ... -> Evidence")
        grades = [a for a in container.task_service.artifacts(lesson_task) if a.type.value == "ASSESSMENT_GRADE"]
        check(bool(grades) and all(o.artifact_ids["lesson"] in {x.artifact_id for x in container.artifacts.lineage(
            g.artifact_id)} for g in grades), "every session grade derives from the cycle's lesson")
        check(view.artifact_ids["cycle"] in outcome_art.parent_ids and o.artifact_ids["lesson"] in outcome_art.parent_ids,
              "the cycle's outcome derives from the cycle and its lesson")
        check(lesson_task == view.steps[0].child_id, "one lesson for the cycle")

        out("\n" + RULE + "\n9. The goal completes: COMPLETE verifies the completion rule\n" + RULE)
        for concept in (curriculum_demo.A, curriculum_demo.B, curriculum_demo.C, curriculum_demo.D):
            await container.learner_service.record_evidence(
                learner_id, "spanish", curriculum_demo.exercise_evidence(learner_id, concept, 6, "mastered"))
        tasks = len(container.task_service.list_for_learner(learner_id))
        done = await cycles.start(learner_id, StartLearningCycle(idempotency_key="cycle-2", user_id="demo"))
        out(f"cycle {done.cycle_id}: {done.action.action.value} -> {describe(done)}")
        check(done.action.action.value == "COMPLETE" and done.status == CycleStatus.COMPLETED and done.outcome
              and done.outcome.result.value == "GOAL_COMPLETED" and done.outcome.goal_complete,
              "COMPLETE: the completion rule holds and the cycle completes")
        check(len(container.task_service.list_for_learner(learner_id)) == tasks, "COMPLETE generates no lesson")
        summary = {"actions": [view.action.action.value, done.action.action.value],
                   "steps": [s.kind.value for s in view.steps], "mastery": [mastery_before, mastery_after],
                   "evidence": len(o.learning_evidence_ids), "next": nxt.concept_id if nxt else None,
                   "events": dict(events)}
    finally:
        container.close()
    out("\n" + RULE)
    out("all checks passed" if not check.failed else f"{len(check.failed)} check(s) failed: {check.failed}")
    return not check.failed, summary


async def main_async(data_dir: Path | None) -> bool:
    results = []
    for run in (1, 2):
        with tempfile.TemporaryDirectory(prefix="teaching-agent-cycle-demo-") as tmp:
            out = print if run == 1 else (lambda _line: None)
            if run == 2:
                print("\nsecond run (silent) ...")
            ok, summary = await run_learning_cycle_demo(data_dir or Path(tmp) if run == 1 else Path(tmp), out=out)
            results.append((ok, summary))
    same = results[0][1] == results[1][1]
    print(f"second run: {'all checks passed' if results[1][0] else 'FAILED'}; "
          f"{'identical' if same else 'DIFFERENT'} actions, steps, mastery and events")
    return all(ok for ok, _ in results) and same


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the first run's database and artifacts (default: a temporary directory)")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args()
    configure_logging("INFO" if args.verbose else "WARNING")
    return 0 if asyncio.run(main_async(args.data_dir)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
