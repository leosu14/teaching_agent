"""End-to-end demo of the Visual / Image slice.

    python scripts/run_visual_demo.py

Runs the same lesson as run_demo.py (Diagnostic -> Research -> Planner -> Teacher -> Reviewer -> Visual) with the
deterministic mock image search and image generation providers, offline, then shows what the visual step did:
the lesson, the visual plan and its requirements, each image search and its ranked candidates, the selected
image, generated images, validation (including a candidate rejected for lying about its size), attribution,
the IMAGE_ASSET artifacts with their checksums, and the usage. It uses the same TaskService as the API.
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
from app.schemas.artifact import ArtifactType  # noqa: E402
from app.schemas.lesson import LessonContent  # noqa: E402
from app.schemas.task import Task, TaskStatus  # noqa: E402
from app.schemas.visual import ImageAsset, VisualPlan, VisualResult  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402
from scripts.run_demo import run_demo  # noqa: E402


def _clip(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _heading(out, title: str) -> None:
    out("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def report_visuals(container: Container, task: Task, out=print) -> None:
    arts = {a.name: a for a in container.task_service.artifacts(task.task_id)}
    events = [e for e in container.task_service.events(task.task_id) if e.type.startswith(("visual.", "image."))]
    lesson = LessonContent.model_validate_json(container.artifacts.read(arts["lesson"].artifact_id))
    plan_art = arts["visual_plan"]
    plan = VisualPlan.model_validate_json(container.artifacts.read(plan_art.artifact_id))
    result = VisualResult.model_validate(task.workflow.node_states["visual_policy"].output)

    _heading(out, "1. Lesson (approved by review)")
    out(f"{lesson.title} ({lesson.level}), review: {task.result.review_verdict} after {task.result.revisions} "
        "revision(s)")
    for s in lesson.sections:
        out(f"  {s.section_id}: {s.heading}")

    _heading(out, "2. Visual plan")
    out(f"plan_id: {plan.plan_id}  ({len(plan.requirements)} visuals)  {plan.rationale}")
    out(f"artifact: {plan_art.type.value} v{plan_art.version}  parents: "
        + ", ".join(container.artifacts.get(p).name for p in plan_art.parent_ids))

    _heading(out, "3. Visual requirements")
    for r in plan.requirements:
        need = "required" if r.required else "optional"
        out(f"  {r.visual_id:<16} {r.visual_type.value:<12} {r.preferred_source:<8} {need:<8} {r.aspect_ratio} "
            f"section={r.lesson_section_id}")
        out(f"      purpose: {_clip(r.purpose, 70)}")
        if r.search_query:
            out(f"      search_query: {r.search_query!r}")
        if r.generation_prompt:
            out(f"      generation_prompt: {_clip(r.generation_prompt, 70)!r}")
        out(f"      attribution_required={r.attribution_required}  sources to try: {' -> '.join(r.sources())}")

    _heading(out, "4. Image search, selection and validation")
    for e in events:
        d = e.data
        if e.type == "image.search_started":
            out(f"  [{d['visual_id']}] search {d['query']!r}")
        elif e.type == "image.search_completed":
            out(f"      {d['results']} candidates from {d.get('provider', '-')}" if d["ok"]
                else f"      search failed: {d['error']}")
        elif e.type == "image.selected":
            out(f"      selected #{d['rank']} {d['image_id']} score={d['score']:.3f} {_clip(d['title'], 50)!r}")
        elif e.type == "image.validation_failed":
            for err in d["errors"]:
                out(f"      VALIDATION FAILED {d['image_id']}: {err['code']} (expected {err.get('expected')}, "
                    f"got {err.get('actual')})")
        elif e.type == "image.generation_started":
            out(f"  [{d['visual_id']}] generate {d['width']}x{d['height']} seed={d['seed']}")
        elif e.type == "image.generation_completed":
            out(f"      generated {d['asset_id']} by {d['provider']}/{d['model']} prompt_hash={d['prompt_hash'][:16]}...")
        elif e.type == "image.asset_created":
            out(f"      IMAGE_ASSET {d['artifact_id']} {d['width']}x{d['height']} {d['media_type']}")
    for ref in result.assets:
        asset = ImageAsset.model_validate(container.artifacts.get(ref.artifact_id).metadata)
        if asset.selection:
            sel = asset.selection
            out(f"  {ref.visual_id}: chose rank {sel.rank}/{sel.candidates} (score {sel.score:.3f}); signals "
                + ", ".join(f"{k}={v:g}" for k, v in sel.signals.model_dump().items()))
            for r in sel.rejected:
                out(f"      rejected {r['image_id']} at {r['stage']}: {_clip(r['reason'], 70)}")

    _heading(out, "5. IMAGE_ASSET artifacts: validation, attribution, checksum")
    for ref in result.assets:
        art = container.artifacts.get(ref.artifact_id)
        asset = ImageAsset.model_validate(art.metadata)
        out(f"  {art.name}  ({art.type.value} v{art.version}, parent: {container.artifacts.get(art.parent_ids[0]).name})")
        out(f"      section={asset.lesson_section_id} type={asset.visual_type.value} origin={asset.origin} "
            f"{asset.width}x{asset.height} {asset.format} {asset.object.size_bytes} bytes")
        out(f"      checksum sha256={asset.object.checksum}")
        out(f"      object: {asset.object.uri}")
        out(f"      validation: {'valid' if asset.validation.valid else 'INVALID'} "
            f"({', '.join(asset.validation.checks)})")
        a = asset.attribution
        if a.kind == "search":
            lic = f"{a.license.name} <{a.license.url}>" if a.license else "none"
            out(f"      attribution: creator={a.creator or '-'} publisher={a.publisher or '-'} license={lic}")
            out(f"                   source={a.source_url}")
            out(f"                   credit={a.attribution_text or '(provider gave none)'}")
        else:
            out(f"      generated: provider={a.provider} model={a.model} at={a.generated_at.isoformat()} "
                f"seed={a.seed} style={a.style!r}")
            out(f"                 prompt_hash={a.prompt_hash}  (not an external source)")
    failures = result.failures or []
    out(f"  failures: {len(failures)}  warnings: {result.warnings or 'none'}")

    _heading(out, "6. Lesson sections reference their image assets")
    for s in lesson.sections:
        out(f"  {s.section_id}: " + ", ".join(f"{v.visual_id} -> {v.artifact_id}" for v in s.visuals))
        out(f"      citations (text, unchanged): {', '.join(s.citations) or '-'}")
    images = [a for a in arts.values() if a.type == ArtifactType.IMAGE_ASSET]
    out(f"lesson artifact parents: {', '.join(sorted(container.artifacts.get(p).name for p in arts['lesson'].parent_ids))}")
    out(f"distinct stored image objects: {len({a.uri for a in images})} for {len(images)} assets")

    _heading(out, "7. Cost and usage")
    cost = task.cost
    out(f"  visual agent: calls={cost.by_agent['visual'].calls} tokens={cost.by_agent['visual'].usage.total_tokens} "
        f"${cost.by_agent['visual'].cost_usd:.6f}")
    for name, line in sorted(cost.by_service.items()):
        if not name.startswith("image_"):
            continue
        price = "not reported" if line.cost_usd is None else f"${line.cost_usd:.6f}"
        units = ", ".join(f"{k}={v:g}" for k, v in sorted(line.units.items())) or "-"
        out(f"  {name:<24} calls={line.calls} results={line.results} cost={price} units: {units}")
    out(f"  task: llm_calls={cost.llm_calls} estimated=${cost.estimated_cost_usd:.6f} actual=${cost.actual_cost_usd:.6f}")


async def run_visual_demo(container: Container, *, out=print) -> Task:
    _heading(out, "Lesson request -> Diagnostic -> Research -> Planner -> Teacher -> Reviewer -> Visual")
    task = await run_demo(container, out=lambda line: None)
    out(f"task {task.task_id}: {task.status.value}")
    if task.status != TaskStatus.COMPLETED:
        for err in task.errors:
            out(f"error: [{err.kind}] {err.node_id}: {err.message}")
        return task
    report_visuals(container, task, out)
    return task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database and artifacts (default: a fresh temporary directory)")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args()
    configure_logging("INFO" if args.verbose else "WARNING")

    with tempfile.TemporaryDirectory(prefix="teaching-agent-visual-demo-") as tmp:
        data_dir = args.data_dir or Path(tmp)
        container = build_container(Settings(data_dir=data_dir))
        try:
            task = asyncio.run(run_visual_demo(container))
        finally:
            container.close()
        if args.data_dir is None:
            print("\n(artifacts were written to a temporary directory; pass --data-dir to keep them)")
    return 0 if task.status == TaskStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
