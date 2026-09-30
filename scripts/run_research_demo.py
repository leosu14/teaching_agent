"""End-to-end demo of the Research / RAG slice.

    python scripts/run_research_demo.py

Runs the same lesson as run_demo.py (Diagnostic -> Research -> Planner -> Teacher -> Reviewer) with the
deterministic mock search provider, then shows what research did: the research request, the generated
queries, the mock search results, the selected and rejected sources, the extracted evidence, the
citations, the stored ResearchBundle, and how the lesson cites it. It uses the same TaskService as the API.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config.settings import Settings  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.schemas.lesson import LessonContent  # noqa: E402
from app.schemas.research import ResearchBundle, Source  # noqa: E402
from app.schemas.task import Task, TaskStatus  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402
from scripts.run_demo import run_demo  # noqa: E402


def _clip(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _heading(out, title: str) -> None:
    out("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def report_research(container: Container, task: Task, out=print) -> ResearchBundle | None:
    artifact = container.artifacts.find(task.task_id, "research_bundle")
    if artifact is None:
        out("no research bundle was stored")
        return None
    bundle = ResearchBundle.model_validate_json(container.artifacts.read(artifact.artifact_id))
    events = [e for e in container.task_service.events(task.task_id) if e.type.startswith("research.")]
    known: dict[str, Source] = {s.source_id: s for s in bundle.sources}
    known |= {r.source.source_id: r.source for r in bundle.rejected_sources}

    _heading(out, "1. Research request")
    obj = bundle.objective
    out(f"objective: {obj.description}")
    out(f"subject={obj.subject} topic={obj.topic} level={obj.level} language={obj.language}")
    for t in obj.targets:
        out(f"  target [{t.priority:<6}] {t.target_id:<28} {t.name}")

    _heading(out, "2. Generated queries")
    for q in bundle.queries:
        out(f"  {q.query_id}: {q.text!r}  (max {q.max_results} results, targets: {', '.join(q.target_ids)})")

    _heading(out, "3. Mock search results")
    for e in (e for e in events if e.type == "research.search_completed"):
        d = e.data
        if not d.get("ok"):
            out(f"  {d['query_id']} {e.tool:<12} ERROR {d.get('error')}")
            continue
        cached = " (cache hit)" if d.get("cached") else ""
        out(f"  {d['query_id']} {e.tool:<12} {d['results']} results{cached}")
        seen: set[str] = set()
        for hit in d.get("hits", []):
            sid, s = hit["source_id"], known.get(hit["source_id"])
            if sid in seen:
                out(f"      {sid}  duplicate: {hit['url']}")
                continue
            seen.add(sid)
            label = f"{s.title} [{s.publisher or '-'}, {s.source_type}]" if s else hit["url"]
            out(f"      {sid}  {_clip(label, 70)}")
    done = next((e.data for e in events if e.type == "research.completed"), {})
    if done:
        out(f"  {done['candidates']} candidates, {done['duplicates_removed']} duplicates removed before ranking")

    _heading(out, "4. Selected sources")
    scores = {e.data["source_id"]: e.data for e in events if e.type == "research.source_selected"}
    for s in bundle.sources:
        sc = scores.get(s.source_id, {})
        rel = s.reliability
        out(f"  #{sc.get('rank', '?')} {s.source_id} score={sc.get('score', 0):.3f} "
            f"reliability={rel.score if rel else None} ({rel.basis if rel else '-'})")
        out(f"      {s.title} | {s.publisher or '-'} | author={s.author or '-'} | published={s.published_at or '-'}")
        out(f"      {s.url}")
    out("  rejected:")
    for r in bundle.rejected_sources:
        out(f"    {r.source.source_id} {_clip(r.source.title, 40):<40} {r.reason}")

    _heading(out, "5. Extracted evidence")
    for ev in bundle.evidence:
        loc = f"{ev.location.field}[{ev.location.start}:{ev.location.end}]" if ev.location else "-"
        out(f"  {ev.evidence_id} {ev.target_id:<28} from {ev.source_id} {loc} relevance={ev.relevance}")
        out(f"      \"{_clip(ev.text)}\"")
    out("  key findings:")
    for f in bundle.key_findings:
        out(f"    {f.finding_id} {f.target_id:<28} {_clip(f.statement, 60)}  <- {', '.join(f.evidence_ids)}")

    _heading(out, "6. Citations")
    for c in bundle.citations:
        out(f"  [{c.citation_id}] {c.reference()}  ({c.locator}; evidence {c.evidence_id})")

    _heading(out, "7. ResearchBundle")
    out(f"research_id:  {bundle.research_id}")
    out(f"status:       {bundle.status}")
    out(f"summary:      {bundle.summary}")
    out(f"counts:       {len(bundle.queries)} queries, {len(bundle.sources)} sources, {len(bundle.evidence)} evidence, "
        f"{len(bundle.key_findings)} findings, {len(bundle.citations)} citations")
    out(f"warnings:     {bundle.warnings or 'none'}")
    out(f"artifact:     {artifact.type.value} v{artifact.version} {artifact.uri}")
    return bundle


def report_lesson(container: Container, task: Task, bundle: ResearchBundle, out=print) -> None:
    _heading(out, "8. Lesson generated from the bundle")
    artifact = container.artifacts.find(task.task_id, "lesson")
    lesson = LessonContent.model_validate_json(container.artifacts.read(artifact.artifact_id))
    names = {a.artifact_id: a.name for a in container.task_service.artifacts(task.task_id)}
    out(f"lesson: {lesson.title}  (parents: {', '.join(names[p] for p in artifact.parent_ids)})")
    for section in lesson.sections:
        out(f"  section {section.section_id}: {section.heading}")
        out(f"      {_clip(section.explanation)}")
        for cid in section.citations:
            citation, evidence, source = bundle.resolve(cid)
            out(f"      [{cid}] -> {evidence.evidence_id} -> {source.source_id} {_clip(source.title, 45)} "
                f"({citation.locator})")
    out(f"references attached to the lesson: {len(lesson.references)}")
    cost = task.cost
    out("\nusage:")
    out(f"  llm_calls={cost.llm_calls} research_llm_calls={cost.by_agent['research'].calls} "
        f"actual cost=${cost.actual_cost_usd:.6f}")
    for name, line in sorted(cost.by_service.items()):
        price = "not reported" if line.cost_usd is None else f"${line.cost_usd:.6f}"
        out(f"  {name:<18} calls={line.calls} cache_hits={line.cache_hits} results={line.results} cost={price}")


async def run_research_demo(container: Container, *, out=print) -> Task:
    _heading(out, "Lesson request -> Diagnostic -> Research -> Planner -> Teacher -> Reviewer")
    task = await run_demo(container, out=lambda line: None)
    out(f"task {task.task_id}: {task.status.value}")
    if task.status != TaskStatus.COMPLETED:
        for err in task.errors:
            out(f"error: [{err.kind}] {err.node_id}: {err.message}")
        return task
    bundle = report_research(container, task, out)
    if bundle is not None:
        report_lesson(container, task, bundle, out)
    return task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database and artifacts (default: a fresh temporary directory)")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args()
    configure_logging("INFO" if args.verbose else "WARNING")

    with tempfile.TemporaryDirectory(prefix="teaching-agent-research-demo-") as tmp:
        data_dir = args.data_dir or Path(tmp)
        container = build_container(Settings(data_dir=data_dir))
        try:
            task = asyncio.run(run_research_demo(container))
        finally:
            container.close()
        if args.data_dir is None:
            print("\n(artifacts were written to a temporary directory; pass --data-dir to keep them)")
    return 0 if task.status == TaskStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
