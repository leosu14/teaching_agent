"""End-to-end demo of an interactive teaching session.

    python scripts/run_interactive_demo.py

The curriculum demo's Spanish learner has secured the preterite, so the curriculum's next objective is the
preterite/imperfect contrast. The lesson for it is generated and its diagnostic answered, then an interactive session
runs on it with a scripted learner (fixtures/interactive/session.json):

  1. the teacher explains and asks a medium question; the learner answers incorrectly: difficulty 2 -> 1, a
     conceptual hint, a misconception candidate recorded as evidence;
  2. the learner answers correctly after the hint: difficulty kept, the hint level recorded with the answer;
  3. an easier question answered correctly: difficulty 1 -> 2;
     the session is paused (an answer is refused), the process "restarts" (a new container on the same data
     directory), the session resumes exactly where it was;
  4. the learner asks a question the lesson covers: a grounded answer citing the lesson;
     and one it does not cover: a structured limitation, no invented citation;
  5. the final check is answered correctly: the objective is demonstrated and the session completes;
     a duplicate answer is replayed, never applied twice;
  6. the summary, the mastery update made by learner memory's existing updater, the objective's progress and the
     curriculum's next action; the artifact lineage and the session's events, each exactly once.

Every decision (correctness, difficulty, hints, completion, mastery) is deterministic; the model only words turns.
Exits 1 if any check fails.
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

from app.config.settings import Settings  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.schemas.curriculum import GoalInput  # noqa: E402
from app.schemas.learner import LearnerProfileInput  # noqa: E402
from app.schemas.task import TaskStatus  # noqa: E402
from app.schemas.teaching import (  # noqa: E402
    AnswerResult,
    LearnerInput,
    PublicQuestion,
    StartTeachingSession,
    TeachingSessionStatus,
    TeachingTurn,
)
from app.services.container import Container, build_container  # noqa: E402
from app.teaching.errors import InvalidSessionTransition  # noqa: E402

FIXTURES = REPO_ROOT / "fixtures" / "interactive"
RULE = "=" * 72
Checks = curriculum_demo.Checks
TEACHING_EVENTS = ("teaching_session.", "teaching_turn.", "learner_answer.", "hint.", "misconception.",
                   "difficulty.")


def scenario() -> dict:
    return json.loads((FIXTURES / "session.json").read_text(encoding="utf-8"))


def settings(data_dir: Path) -> Settings:
    # A short session for the demo: two answers in a row move the difficulty up, one miss moves it down.
    return Settings(data_dir=data_dir, corpus_dir=curriculum_demo.FIXTURES, teaching_increase_after=2,
                    teaching_decrease_after=1)


def answer_for(question: PublicQuestion, key: dict, say: str) -> str:
    """The scripted learner answers by the question's verb cue."""
    for cue, answers in key.items():
        if cue in question.prompt:
            return answers[say]
    raise LookupError(f"no scripted answer for {question.prompt!r}")


def print_turns(out, turns: list[TeachingTurn]) -> None:
    for t in turns:
        extra = ""
        if t.metadata.get("citations"):
            extra = f"  [cites {', '.join(t.metadata['citations'])}]"
        if t.metadata.get("grounded") is False:
            extra = f"  [not grounded: {t.metadata.get('limitation')}]"
        out(f"  teacher {t.turn_type.value:<13} {t.content}{extra}")


def print_result(out, r: AnswerResult) -> None:
    graded = "" if r.correct is None else f"  -> {'correct' if r.correct else 'incorrect'}"
    out(f"  learner {r.learner_turn.turn_type.value:<13} {r.learner_turn.content}{graded}")
    if r.difficulty_change is not None:
        out(f"  difficulty {r.difficulty_change.before} -> {r.difficulty_change.after} "
            f"({r.difficulty_change.reason})")
    for e in r.evidence:
        detail = e.misconception or (f"hint level {e.hint_level}" if e.hint_level else "")
        out(f"  evidence {e.evidence_type.value:<15} correct={e.correct} difficulty={e.difficulty} {detail}".rstrip())
    print_turns(out, r.teacher_turns)


async def prepare(container: Container, out, check: Checks) -> str:
    """The curriculum demo's learner and goal, with the preterite secured; returns the lesson's task id."""
    fx = curriculum_demo._fixture("learner.json")
    learner_id = fx["learner_id"]
    container.learner_service.upsert(learner_id, LearnerProfileInput.model_validate(fx["profile"]))
    for domain, evidence in curriculum_demo.placement_evidence(learner_id, fx["placement"]).items():
        await container.learner_service.record_evidence(learner_id, domain, evidence)
    for concept in scenario()["secured_concepts"]:
        await container.learner_service.record_evidence(
            learner_id, "spanish", curriculum_demo.exercise_evidence(learner_id, concept, 4, "secured"))
    goal, _ = await container.curriculum_service.create_goal(learner_id, GoalInput.model_validate(fx["goal"]))
    await container.curriculum_service.build_curriculum(goal.goal_id, user_id="demo")
    action = await container.curriculum_service.next_action(learner_id)
    out(f"next learning action: {action.action.value} {action.concept_id}")
    check(action.concept_id == curriculum_demo.C, f"the curriculum selects {curriculum_demo.C}")
    lesson = await container.curriculum_service.start_lesson(action, user_id="demo")
    lesson = await curriculum_demo.answer_diagnostic(container, lesson, out)
    check(lesson.status == TaskStatus.COMPLETED, "the lesson is generated")
    return lesson.task_id


async def run_interactive_demo(data_dir: Path, *, out=print) -> bool:
    check = Checks(out)
    fx = scenario()
    key, script = fx["answer_key"], fx["script"]
    container = build_container(settings(data_dir))
    try:
        out(RULE + "\nThe lesson\n" + RULE)
        lesson_task = await prepare(container, out, check)

        out("\n" + RULE + "\nStart an interactive session\n" + RULE)
        started = await container.teaching_service.start(lesson_task, StartTeachingSession(idempotency_key="demo"))
        sid = started.session_id
        out(f"session {sid}: objective {started.objective.concept_id}, difficulty {started.difficulty}")
        print_turns(out, started.teacher_turns)
        check(started.status == TeachingSessionStatus.WAITING_FOR_LEARNER and started.difficulty == 2,
              "the session waits for the learner on a medium question")
        again = await container.teaching_service.start(lesson_task, StartTeachingSession(idempotency_key="demo"))
        check(again.session_id == sid and not again.created, "starting again with the same key returns the session")

        results: list[AnswerResult] = []

        async def step(container: Container, i: int) -> AnswerResult:
            item = script[i]
            out(f"\nturn {i + 1}: {item['expect']}")
            view = await container.teaching_service.view(sid)
            if item["kind"] == "answer":
                text = answer_for(view.state.waiting_question, key, item["say"])
                data = LearnerInput(answer=text, client_turn_id=f"demo-{i + 1}")
            else:
                data = LearnerInput(answer=item["say"], kind="question", client_turn_id=f"demo-{i + 1}")
            r = await container.teaching_service.submit(sid, data)
            print_result(out, r)
            results.append(r)
            return r

        r1 = await step(container, 0)
        check(r1.correct is False and r1.difficulty_change is not None
              and (r1.difficulty_change.before, r1.difficulty_change.after) == (2, 1),
              "an incorrect medium answer lowers the difficulty 2 -> 1")
        check(any(t.turn_type.value == "HINT" for t in r1.teacher_turns), "the teacher gives a hint, not the answer")
        check(any(e.evidence_type.value == "MISCONCEPTION" for e in r1.evidence),
              "the validated misconception candidate is stored as evidence")
        r2 = await step(container, 1)
        answered = [e for e in r2.evidence if e.evidence_type.value == "ANSWER"]
        check(r2.correct is True and r2.difficulty_change is None and answered and answered[0].hint_level == 1,
              "correct after a hint: difficulty kept, the hint level recorded with the answer")
        r3 = await step(container, 2)
        check(r3.correct is True and r3.difficulty_change is not None and r3.difficulty_change.after == 2,
              "two correct answers raise the difficulty 1 -> 2")

        out("\n" + RULE + "\nPause, restart the process, resume\n" + RULE)
        paused = await container.teaching_service.pause(sid)
        out(f"paused: {paused.status.value}")
        try:
            await container.teaching_service.submit(sid, LearnerInput(answer="sonó", client_turn_id="demo-paused"))
            refused = False
        except InvalidSessionTransition as exc:
            refused = True
            out(f"answer while paused refused: {exc}")
        check(refused, "a paused session refuses answers")
        before = await container.teaching_service.view(sid)
        container.close()
        out("process restarted: a new container on the same data directory")
        container = build_container(settings(data_dir))
        resumed = await container.teaching_service.resume(sid)
        check(resumed.status == TeachingSessionStatus.WAITING_FOR_LEARNER
              and [t.turn_id for t in resumed.turns] == [t.turn_id for t in before.turns]
              and resumed.state.difficulty == before.state.difficulty
              and resumed.state.waiting_question == before.state.waiting_question,
              f"resumed after the restart with the same {len(resumed.turns)} turns, difficulty and question")

        out("\n" + RULE + "\nLearner questions\n" + RULE)
        r4 = await step(container, 3)
        t4 = r4.teacher_turns[0]
        check(t4.metadata.get("grounded") is True and bool(t4.metadata.get("citations")),
              "a question the lesson covers is answered with the lesson's citations")
        r5 = await step(container, 4)
        t5 = r5.teacher_turns[0]
        check(t5.metadata.get("grounded") is False and not t5.metadata.get("citations")
              and bool(t5.metadata.get("limitation")),
              "a question outside the lesson gets a structured limitation and no citation")

        out("\n" + RULE + "\nThe final check\n" + RULE)
        r6 = await step(container, 5)
        check(r6.status == TeachingSessionStatus.COMPLETED
              and r6.completion_reason is not None and r6.completion_reason.value == "OBJECTIVE_DEMONSTRATED",
              "the deterministic completion rule completes the session: objective demonstrated")
        evidence_before = len(container.teaching_service.evidence(sid))
        replay = await container.teaching_service.submit(sid, LearnerInput(answer=answer_for(
            r1.waiting_question, key, "correct"), client_turn_id="demo-2"))
        check(replay.replayed and replay.learner_turn.turn_id == r2.learner_turn.turn_id
              and len(container.teaching_service.evidence(sid)) == evidence_before,
              "a duplicate answer (same client_turn_id) is replayed, not applied twice")

        view = await container.teaching_service.view(sid)
        summary, outcome = view.summary, view.outcome
        assert summary is not None and outcome is not None
        out("\n" + RULE + "\nSummary\n" + RULE)
        out(f"  {summary.narrative}")
        out(f"  strengths: {'; '.join(summary.strengths)}")
        out(f"  difficulties: {'; '.join(summary.difficulties)}")
        out(f"  misconceptions: {'; '.join(summary.misconceptions) or '-'}")
        out(f"  difficulty trajectory: {' -> '.join(map(str, summary.difficulty_trajectory))}  "
            f"hints: {summary.hints_used}  completion: {summary.completion_reason.value}  "
            f"recommended: {summary.recommended_next_action}")
        check(summary.difficulty_trajectory == [2, 1, 2] and summary.hints_used == 1
              and summary.correct_answers == 3 and summary.incorrect_answers == 1,
              "the summary matches what happened")

        out("\n" + RULE + "\nMastery, objective progress, next action\n" + RULE)
        for change in outcome.mastery_changes:
            out(f"  mastery {change['concept_id']}: {change['before']:.2f} -> {change['after']:.2f}")
        check(not outcome.practice and len(outcome.learning_evidence_ids) == 4
              and any(c["concept_id"] == curriculum_demo.C and c["after"] > c["before"]
                      for c in outcome.mastery_changes),
              "the 4 graded answers became learning evidence; learner memory's updater raised the mastery")
        progress = outcome.objective_progress or {}
        out(f"  objective {progress.get('concept_id')}: {progress.get('status')}, mastery "
            f"{progress.get('current_mastery')} of {progress.get('target_mastery')}")
        check(progress.get("concept_id") == curriculum_demo.C, "the objective's progress is recalculated")
        action = outcome.learning_action or {}
        out(f"  next action: {action.get('action')} {action.get('concept_id')}")
        out(f"  why: {action.get('reason')}")
        check(bool(action.get("action")), "the curriculum engine selects the next learning action")

        out("\n" + RULE + "\nArtifacts and events\n" + RULE)
        arts = container.task_service.artifacts(lesson_task)
        evidence_art = next(a for a in arts if a.type.value == "INTERACTION_EVIDENCE")
        up = [a.type.value for a in container.artifacts.lineage(evidence_art.artifact_id)]
        down = [container.artifacts.get(outcome.artifact_ids[k]).type.value
                for k in ("summary", "learning_evidence", "learner_model", "learning_action")]
        chain = [t for t in ("LEARNING_GOAL", "CURRICULUM_VERSION", "LEARNING_OBJECTIVE", "LESSON", "TEACHING_SESSION",
                             "TEACHING_TURN") if t in up]
        out("  lineage of an evidence item: " + " -> ".join([*chain, "INTERACTION_EVIDENCE"])
            + f"  ({len(up)} ancestors in all)")
        out("  after completion: " + " -> ".join(down))
        check(all(t in up for t in ("TEACHING_TURN", "TEACHING_SESSION", "LESSON", "LEARNING_OBJECTIVE",
                                    "CURRICULUM_VERSION", "LEARNING_GOAL")),
              "evidence lineage: GOAL -> CURRICULUM_VERSION -> OBJECTIVE -> LESSON -> SESSION -> TURN -> EVIDENCE")
        names = Counter(a.name for a in arts)
        check(all(n == 1 for n in names.values()), "every artifact stored once")
        events = [e for e in container.task_service.events(lesson_task) if e.type.startswith(TEACHING_EVENTS)]
        counts = Counter(e.type for e in events)
        out("  events: " + ", ".join(f"{t} x{n}" for t, n in sorted(counts.items())))
        check(len({e.event_id for e in events}) == len(events) and counts["teaching_session.completed"] == 1
              and counts["teaching_session.paused"] == 1 and counts["teaching_session.resumed"] == 1,
              "every session event emitted once, across the restart")
        turns = container.teaching_service.turns(sid)
        check([t.sequence for t in turns] == list(range(1, len(turns) + 1)), f"{len(turns)} turns, no gaps")
    finally:
        container.close()

    out("\n" + RULE)
    out("all checks passed" if not check.failed else f"{len(check.failed)} check(s) failed: {check.failed}")
    return not check.failed


def main(argv: list[str] | None = None, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database and artifacts (default: a fresh temporary directory)")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args(argv)
    configure_logging("INFO" if args.verbose else "WARNING")
    with tempfile.TemporaryDirectory(prefix="teaching-agent-interactive-demo-") as tmp:
        ok = asyncio.run(run_interactive_demo(args.data_dir or Path(tmp), out=out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
