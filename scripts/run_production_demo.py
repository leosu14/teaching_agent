"""Production end-to-end run: one real-provider lesson, from the task to the final MP4.

    python scripts/run_production_demo.py --dry-run          # check configuration, providers, budget and stages
    python scripts/run_production_demo.py                     # show configuration and budget, then stop (safe)
    python scripts/run_production_demo.py \\
        --level B1 --topic "Climate change" --language es --learner-id demo-user --output ./output --confirm

The run creates an ordinary lesson task (the same workflow, agents and tools as the API and the other demos) and
drives it: diagnostic -> research -> planning -> teacher/reviewer -> visuals -> presentation -> TTS -> timeline ->
video -> learner memory. It writes production_run.json (machine-readable report, no secrets) into --output and
prints a summary. Canonical artifacts stay in the object store under the data directory (TA_DATA_DIR).

Safety:
- production needs TEACHING_AGENT_MODE=production and a real provider for each capability the workflow uses;
  otherwise the command fails before anything is called;
- without --confirm (or --yes) nothing is called: the configuration and budget are shown and the command exits;
- --dry-run validates everything and makes no provider call at all;
- every run has a budget (MAX_LLM_REQUESTS, MAX_GENERATED_IMAGES, ...): exceeding it stops the task.

Same command again: an identical completed run is reused (no provider call); an identical interrupted or failed run
is resumed from its last checkpoint. --fresh forces a new run; --resume TASK_ID continues a specific task.

--mock rehearses the exact same run offline on the mock providers (no network, no credentials, no confirmation).

Evaluation: after the lesson, the post-lesson assessment is created (one LLM step) unless --skip-evaluation. Answer
it later with --evaluate LESSON_TASK_ID --answers answers.json (or interactively on a terminal).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config.providers import MOCK, ProviderSettings  # noqa: E402
from app.config.routing import ConfigError  # noqa: E402
from app.config.settings import Settings  # noqa: E402
from app.observability.logging import ProductionLogWriter, configure_logging  # noqa: E402
from app.schemas.common import utcnow  # noqa: E402
from app.schemas.lesson import DiagnosticAnswers, DiagnosticQuestionSheet, LearnerAnswer  # noqa: E402
from app.schemas.production import ProductionPlan, ProductionReport, ProductionTask  # noqa: E402
from app.schemas.task import Task, TaskStatus  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402
from app.services.production import RunOutcome  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_UNHEALTHY = 0, 1, 2, 3
REPORT_NAME = "production_run.json"


# --- configuration --------------------------------------------------------------------------------------------

def mock_providers() -> ProviderSettings:
    """--mock: offline, every capability on its mock, whatever the environment says."""
    return ProviderSettings(teaching_agent_mode="offline", teaching_agent_offline=True, llm_provider=None,
                            llm_model=None, llm_fallback_provider=None, llm_fallback_model=None, llm_routes={},
                            tts_provider=MOCK, tts_fallback_provider=None, image_provider=MOCK,
                            image_fallback_provider=None, image_search_provider=MOCK, search_provider=MOCK,
                            search_fallback_provider=None)


def settings_for(args: argparse.Namespace) -> Settings:
    values: dict = {"log_json": True}
    if args.data_dir is not None:
        values["data_dir"] = args.data_dir
    if args.mock:
        values["providers"] = mock_providers()
    return Settings(**values)


# --- answering ------------------------------------------------------------------------------------------------

def load_diagnostic_answers(path: Path | None):
    """Answers from a file: {"<question_id or concept_id>": "answer"}, or the fixture format {"rounds": [...]}
    (per round, concept id -> answer). None: ask on a terminal, else answer nothing (a learner who does not know)."""
    data = json.loads(path.read_text(encoding="utf-8")) if path else None
    interactive = data is None and sys.stdin.isatty()
    skipped: list[str] = []

    async def answer(sheet: DiagnosticQuestionSheet) -> DiagnosticAnswers:
        answers = []
        for q in sheet.questions:
            if data is not None:
                by = data["rounds"][sheet.round_number - 1] if "rounds" in data else data
                text = by.get(q.question_id, by.get(q.concept_id, ""))
            elif interactive:
                text = input(f"[diagnostic round {sheet.round_number}] {q.prompt}\n> ").strip()
            else:
                text = ""
                skipped.append(q.question_id)
            answers.append(LearnerAnswer(question_id=q.question_id, answer=text))
        return DiagnosticAnswers(answers=answers)

    return answer, skipped


def evaluation_answers(task: Task, path: Path | None) -> dict:
    assert task.waiting is not None
    questions = task.waiting.prompt["questions"]
    if path is not None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "answers" in data and isinstance(data["answers"], list):
            return data
        return {"answers": [{"question_id": q["question_id"], "answer": data.get(q["question_id"], "")}
                            for q in questions]}
    if not sys.stdin.isatty():
        raise SystemExit("evaluation answers are needed: pass --answers FILE (or run on a terminal)")
    return {"answers": [{"question_id": q["question_id"], "answer": input(f"{q['prompt']}\n> ").strip()}
                        for q in questions]}


# --- output ---------------------------------------------------------------------------------------------------

def print_plan(plan: ProductionPlan, *, mock: bool) -> None:
    print("=" * 72)
    print(f"Task:      {plan.lesson_request.raw_request}")
    print(f"Level:     {plan.task.level} ({plan.lesson_request.framework_id})")
    print(f"Topic:     {plan.task.topic}")
    print(f"Language:  {plan.task.language}   Subject: {plan.lesson_request.subject}   Learner: {plan.task.learner_id}")
    print(f"Mode:      {plan.mode}" + ("  (--mock rehearsal: mock providers, no network)" if mock else ""))
    print(f"Run key:   {plan.run_key}")
    print("\nProviders (credentials are never printed):")
    for p in plan.providers:
        flag = "required" if p.required else "not needed"
        print(f"  {p.capability:<16} {p.provider:<9} model={p.model or '-':<24} "
              f"{'real' if p.real else 'mock':<4} {flag}" + (f" fallback={','.join(p.fallbacks)}" if p.fallbacks else ""))
    routes = sorted(set(plan.llm_routes.values()))
    print(f"  LLM routes: {', '.join(routes)}")
    b = plan.budget
    print("\nBudget (per task; the task stops before exceeding a limit):")
    print(f"  LLM requests {b.max_llm_requests}  LLM tokens {b.max_llm_tokens}  searches {b.max_search_requests}  "
          f"generated images {b.max_generated_images}  searched images {b.max_searched_images}")
    print(f"  TTS characters {b.max_tts_characters}  TTS seconds {b.max_tts_seconds:g}  "
          f"cost {'$%.2f' % b.max_cost_usd if b.max_cost_usd is not None else 'no limit (unit limits apply)'}")
    estimate = plan.estimated_llm_cost_usd
    print(f"  estimated LLM cost: {'$%.4f' % estimate if estimate is not None else 'unknown (models have no price)'}")
    g = plan.video_generation
    if g is not None:
        print("\nVideo generation:")
        if not g.requested:
            print("  not requested (pass --generated-video to ask for generated segments)")
        print(f"  provider: {g.provider}" + (f"  model={g.model}" if g.model else "")
              + f"  {'real' if g.real else 'mock'}" + ("" if g.enabled or not g.requested else
                                                       "  (disabled: settings or budget allow no segment)"))
        print(f"  segments: up to {g.max_segments}, {g.segment_seconds[0]:g}-{g.segment_seconds[1]:g}s each, "
              f"at most {g.max_seconds:g}s per lesson")
        print(f"  estimated duration: up to {g.estimated_seconds:g}s")
        cost = ("unknown (VIDEO_GENERATION_PRICE_PER_SECOND is not set; count and duration limits apply)"
                if g.estimated_cost_usd is None else f"up to ${g.estimated_cost_usd:.2f}")
        print(f"  estimated cost: {cost}"
              + (f"  (limit ${g.max_cost_usd:.2f})" if g.max_cost_usd is not None else ""))
        print(f"  fallback policy: {'required' if g.required else 'optional'} segments; a failed "
              f"{'required segment fails the task' if g.required and g.failure_policy == 'fail' else 'segment continues'}"
              f" with {g.fallback}")
    print(f"\nKnowledge base: {plan.knowledge_concepts} concepts for {plan.lesson_request.subject}/{plan.task.topic}")
    print(f"Learning loop: {' -> '.join(plan.learning_loop)}")
    print(f"Curriculum stage ({len(plan.curriculum_stages)} workflow nodes, for a learner with a goal): "
          f"{' -> '.join(plan.curriculum_stages)}")
    t = plan.interactive_teaching
    if t is not None:
        print(f"Interactive teaching ({'available' if t.available else 'NOT available'}, opt-in per lesson, "
              f"not run by a dry run): {' -> '.join(t.loop)}")
        print(f"  teacher agent {t.agent_id} on {t.llm_route}; difficulty {t.policy['min_difficulty']}-"
              f"{t.policy['max_difficulty']}, harder after {t.policy['increase_after_successes']} correct, easier "
              f"after {t.policy['decrease_after_failures']} incorrect, hints up to level {t.policy['max_hint_level']}")
        print(f"  endpoints: {', '.join(t.endpoints)}")
    print(f"Stages ({len(plan.stages)} workflow nodes): {' -> '.join(plan.stages)}")
    for w in plan.warnings:
        print(f"WARNING: {w}")
    for p in plan.problems:
        print(f"PROBLEM: {p}")


def print_summary(report: ProductionReport) -> None:
    s, u = report.summary, report.usage
    print("\n" + "=" * 72)
    print(f"Task      {report.task_id}" + ("  (reused identical run)" if report.reused else "")
          + ("  (resumed)" if report.resumed else ""))
    print(f"Level     {report.level}\nTopic     {report.topic}\nStatus    {report.status}")
    print(f"\nResearch:\n  sources {s['research']['sources']}  evidence {s['research']['evidence']}  "
          f"citations {s['research']['citations']}")
    print(f"Lesson:\n  {s['lesson']['title']}: {s['lesson']['sections']} sections, review {s['lesson']['review']} "
          f"after {s['lesson']['revisions']} revision(s)")
    print(f"Visual:\n  images {s['visual']['images']} (generated {s['visual']['generated']}, "
          f"searched {s['visual']['searched']})")
    print(f"Presentation:\n  slides {s['presentation']['slides']}")
    print(f"Audio:\n  {s['audio']['segments']} segments, duration {s['audio']['duration']:.3f}s")
    v = report.final_video
    print("Video:\n  " + (f"duration {v.duration:.3f}s, resolution {v.width}x{v.height}, {v.size_bytes} bytes"
                          if v else "none"))
    cost = f"${report.estimated_cost_usd:.6f}" + ("" if report.cost_complete else
                                                   f" (lower bound: no price for {', '.join(u.unpriced)})")
    print(f"Usage:\n  LLM tokens {u.llm_tokens} ({u.llm_requests} requests)  searches {u.search_requests}  "
          f"images {u.image_generations}  TTS {u.tts_characters} chars / {u.tts_seconds:.1f}s  "
          f"video render {u.video_render_seconds:.1f}s\n  estimated cost {cost}")
    print("Artifacts:")
    for a in report.artifact_graph:
        print(f"  {a.type.value:<22} {a.artifact_id}  node={a.node_id or '-'}")
    print(f"Graph:     {'complete' if report.graph_check.complete else 'INCOMPLETE ' + str(report.graph_check.missing_types)}")
    for w in report.warnings:
        print(f"WARNING: {w}")
    for e in report.errors:
        print(f"ERROR [{e.category or e.kind}] at {e.node_id}: {e.message[:400]}")
    if report.evaluation_task_id:
        print(f"\nEvaluation task {report.evaluation_task_id} is waiting for the learner's answers:\n"
              f"  python scripts/run_production_demo.py --evaluate {report.task_id} --answers answers.json")


# --- commands -------------------------------------------------------------------------------------------------

async def evaluate(container: Container, args: argparse.Namespace) -> int:
    production = container.production
    evaluation = production.evaluation_for(args.evaluate)
    if evaluation is None or evaluation.status != TaskStatus.WAITING:
        evaluation = await production.start_evaluation(args.evaluate)
    if evaluation.status != TaskStatus.WAITING:
        print(f"evaluation could not start: {evaluation.status.value} {evaluation.errors}")
        return EXIT_FAILED
    done = await production.submit_evaluation(evaluation.task_id, evaluation_answers(evaluation, args.answers))
    print(f"Evaluation {done.task_id}: {done.status.value}")
    if done.result is not None:
        print(f"  score {done.result.score}  estimated level {done.result.estimated_level}")
        for c in done.result.mastery_changes:
            print(f"  {c.concept_id:<36} {c.before:.2f} -> {c.after:.2f}")
    return EXIT_OK if done.status == TaskStatus.COMPLETED else EXIT_FAILED


async def run(container: Container, args: argparse.Namespace) -> int:
    production = container.production
    task = ProductionTask(level=args.level, topic=args.topic, language=args.language, learner_id=args.learner_id,
                          subject=args.subject, generated_video=args.generated_video)
    plan = await production.plan(task, require_production=not args.mock)
    print_plan(plan, mock=args.mock)
    if args.dry_run:
        print("\nDry run: configuration, providers, budget and workflow checked; no provider was called.")
        if plan.mode == "offline" and not args.mock:
            print("Offline mode: the mock pipeline is ready; a production run is NOT possible until the problems "
                  "above are fixed.")
            return EXIT_OK
        return EXIT_OK if plan.ready else EXIT_CONFIG
    if not plan.ready:
        print("\nProduction run refused: fix the problems above (see docs/production.md). Nothing was called.")
        return EXIT_CONFIG
    if not (args.confirm or args.mock):
        print("\nNo provider was called. This run uses paid providers within the budget above; re-run with "
              "--confirm (or --yes) to start it.")
        return EXIT_OK

    container.events.subscribe(ProductionLogWriter(args.log_file.open("a", encoding="utf-8") if args.log_file
                                                   else sys.stderr))
    start = utcnow()
    health = await production.check_health(plan.required_capabilities)
    down = [h for h in health if not h.available]
    if down:
        for h in down:
            print(f"UNHEALTHY: {h.capability.value} provider '{h.provider}': {h.error}")
        print("Stopped before any generation: a required provider is unavailable.")
        return EXIT_UNHEALTHY

    answer, unanswered = load_diagnostic_answers(args.diagnostic_answers)
    if args.resume:
        outcome = await production.resume(args.resume, answer=answer)
    else:
        outcome: RunOutcome = await production.run(plan, answer=answer, fresh=args.fresh)
    if unanswered:
        outcome.warnings.append(f"{len(unanswered)} diagnostic question(s) were left unanswered (no terminal and "
                                "no --diagnostic-answers file)")
    evaluation_id = None
    if outcome.task.status == TaskStatus.COMPLETED and not args.skip_evaluation:
        existing = production.evaluation_for(outcome.task.task_id)
        evaluation = existing if existing is not None else await production.start_evaluation(outcome.task.task_id)
        evaluation_id = evaluation.task_id
    report = production.report(outcome, plan, start_time=start, health=health, evaluation_task_id=evaluation_id)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / REPORT_NAME).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    exported = production.export(report, args.output) if args.export else []
    print_summary(report)
    print(f"\nReport:    {args.output / REPORT_NAME}")
    for path in exported:
        print(f"Exported:  {path}")
    return EXIT_OK if report.status == TaskStatus.COMPLETED.value else EXIT_FAILED


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", default="B1", help="target level, e.g. B1 (CEFR) or intermediate (default: B1)")
    p.add_argument("--topic", default="Climate change", help="lesson topic (default: Climate change)")
    p.add_argument("--language", default="es", help="language of the lesson, BCP 47 (default: es)")
    p.add_argument("--subject", default=None, help="what is taught (default: the language, for a CEFR level)")
    p.add_argument("--generated-video", action="store_true",
                   help="ask for optional generated video segments (VIDEO_GENERATION_PROVIDER; budgets "
                        "MAX_GENERATED_VIDEO_SEGMENTS, MAX_GENERATED_VIDEO_SECONDS, MAX_VIDEO_GENERATION_COST_USD)")
    p.add_argument("--learner-id", default="demo-user", help="learner whose memory is used and updated")
    p.add_argument("--output", type=Path, default=Path("output"), help="where production_run.json is written")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="database and object store (default: TA_DATA_DIR, i.e. ./var); keep it to resume runs")
    p.add_argument("--dry-run", action="store_true", help="validate and show the plan; call no provider")
    p.add_argument("--confirm", "--yes", dest="confirm", action="store_true",
                   help="really run (paid provider calls within the budget)")
    p.add_argument("--mock", action="store_true", help="rehearse the same run offline on mock providers")
    p.add_argument("--fresh", action="store_true", help="start a new run even if an identical one exists")
    p.add_argument("--resume", metavar="TASK_ID", help="continue this interrupted or failed production task")
    p.add_argument("--skip-evaluation", action="store_true", help="do not create the post-lesson assessment")
    p.add_argument("--evaluate", metavar="LESSON_TASK_ID", help="evaluation mode: answer a lesson's assessment")
    p.add_argument("--answers", type=Path, default=None, help="evaluation answers (JSON) for --evaluate")
    p.add_argument("--diagnostic-answers", type=Path, default=None,
                   help="diagnostic answers (JSON); default: ask on a terminal, else leave unanswered")
    p.add_argument("--export", action="store_true", help="also copy the MP4, PPTX and VTT into --output")
    p.add_argument("--log-file", type=Path, default=None, help="write production logs here instead of stderr")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    configure_logging("WARNING")  # production log lines come from ProductionLogWriter, not the debug event log
    try:
        container = build_container(settings_for(args))
    except ConfigError as exc:
        print(f"Configuration is invalid (nothing was called):\n{exc}")
        return EXIT_CONFIG
    try:
        if args.evaluate:
            return asyncio.run(evaluate(container, args))
        return asyncio.run(run(container, args))
    except ConfigError as exc:
        print(f"Configuration problem (nothing was called): {exc}")
        return EXIT_CONFIG
    finally:
        container.close()


if __name__ == "__main__":
    raise SystemExit(main())
