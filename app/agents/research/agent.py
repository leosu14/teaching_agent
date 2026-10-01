"""ResearchAgent: decides what a lesson needs researched, then searches, ranks, selects and extracts
traceable evidence into a ResearchBundle.

Search, retrieval and ranking are reached only through the ToolManager, and the model only through the
ModelRouter. Search failures are recorded in the bundle; the agent never fills a gap with invented
sources. Whether a failed research stops the lesson is the workflow's research policy, not this agent's.
"""

from __future__ import annotations

from collections import defaultdict

from app.agents.base import Agent, AgentContext, AgentSpec, OutputRejected
from app.schemas.common import ModelTier, new_id, utcnow
from app.schemas.events import EventType
from app.schemas.lesson import ResearchRequest
from app.schemas.research import (
    Citation,
    Evidence,
    EvidenceExtraction,
    EvidenceExtractionInput,
    EvidenceLocation,
    ExtractionMetadata,
    ExtractionSource,
    KeyFinding,
    RankCandidate,
    RejectedSource,
    ResearchBundle,
    ResearchError,
    ResearchObjective,
    ResearchTarget,
    SearchQuery,
    Source,
)
from app.tools.base import ToolError, ToolNotFound, ToolPermissionError
from app.tools.rag.retrieve import RetrieveOutput
from app.tools.research.rank import RankedSource, RankInput, RankOutput
from app.tools.web.search import SearchResponse

KB_RESULTS_PER_QUERY = 3


class ResearchAgent(Agent[ResearchRequest, ResearchBundle]):
    spec = AgentSpec(
        id="research",
        name="Research",
        description="Plans research queries for a lesson, searches, ranks and selects sources, and extracts "
                    "cited evidence and key findings into a traceable research bundle.",
        input_model=ResearchRequest,
        output_model=ResearchBundle,
        tier=ModelTier.STANDARD,
        tools=("search.web", "rag.retrieve", "research.rank"),
        permissions=frozenset({"network", "knowledge:read"}),
    )
    instructions = "Extract evidence and key findings for the research targets from these sources."

    # --- what to research ----------------------------------------------------------------------

    def objective(self, data: ResearchRequest) -> ResearchObjective:
        """Research what the lesson may teach: diagnosed gaps first, then concepts not yet known, then the rest."""
        gaps, known = set(data.diagnostic.gaps), set(data.diagnostic.known)
        priority = {c.concept_id: "gap" if c.concept_id in gaps else "review" if c.concept_id in known else "learn"
                    for c in data.concepts}
        rank = {"gap": 0, "learn": 1, "review": 2}
        ordered = sorted(data.concepts, key=lambda c: rank[priority[c.concept_id]])
        req = data.request
        level = data.diagnostic.estimated_level or req.target_level
        return ResearchObjective(
            description=f"Evidence for a lesson on {req.topic} ({req.subject}, level {level}): "
                        + ", ".join(c.name for c in ordered),
            subject=req.subject, topic=req.topic, level=level, language=req.language_of_instruction,
            targets=[ResearchTarget(target_id=c.concept_id, name=c.name, description=c.description,
                                    priority=priority[c.concept_id]) for c in ordered],
        )

    def plan_queries(self, objective: ResearchObjective, max_results: int) -> list[SearchQuery]:
        """One broad query for the topic, then one focused query per target in priority order."""
        common = {"language": objective.language, "subject": objective.subject, "domain": objective.topic,
                  "max_results": max_results}
        broad = " ".join(filter(None, [objective.topic, objective.subject, objective.level]))
        queries = [SearchQuery(query_id="q1", text=broad, target_ids=[t.target_id for t in objective.targets],
                               **common)]
        for target in objective.targets:
            queries.append(SearchQuery(query_id=f"q{len(queries) + 1}", text=f"{target.name} {objective.topic}",
                                       target_ids=[target.target_id], **common))
        return queries

    # --- the research loop ---------------------------------------------------------------------

    async def run(self, data: ResearchRequest, ctx: AgentContext) -> ResearchBundle:
        scope = ctx.scope
        objective = self.objective(data)
        scope.emit(EventType.RESEARCH_STARTED, subject=objective.subject, topic=objective.topic,
                   targets=[t.target_id for t in objective.targets])
        queries = self.plan_queries(objective, data.max_results_per_query)
        for q in queries:
            scope.emit(EventType.RESEARCH_QUERY_CREATED, query_id=q.query_id, text=q.text, target_ids=q.target_ids)

        candidates, errors = await self._search(queries, objective, ctx)
        if not candidates:
            return self._failed(objective, queries, errors, [], "no search or retrieval returned any source", ctx)

        try:
            ranking = await self.use_tool("research.rank", RankInput(candidates=candidates,
                                                                     language=objective.language), ctx)
        except (ToolPermissionError, ToolNotFound):
            raise
        except ToolError as exc:
            errors.append(ResearchError(stage="rank", message=str(exc), tool="research.rank"))
            return self._failed(objective, queries, errors, [], "ranking failed", ctx)
        assert isinstance(ranking, RankOutput)

        selected, rejected = self._select(ranking.ranked, data)
        for r in selected:
            src = r.result.source
            scope.emit(EventType.RESEARCH_SOURCE_SELECTED, source_id=src.source_id, url=src.url, rank=r.rank,
                       score=r.score, reliability=src.reliability.score if src.reliability else None)
        if not selected:
            errors.append(ResearchError(stage="select", message=(
                f"none of {len(ranking.ranked)} candidate sources met the selection criteria "
                f"(reliability >= {data.min_reliability}, relevant to a query)")))
            return self._failed(objective, queries, errors, rejected, "no source was reliable and relevant", ctx)

        payload = EvidenceExtractionInput(objective=objective, sources=[
            ExtractionSource(source_id=r.result.source_id, title=r.result.title, publisher=r.result.source.publisher,
                             snippet=r.result.snippet, content=r.result.content, relevance=r.signals.relevance)
            for r in selected
        ])
        extraction = await self.generate(payload, ctx, source=payload, output_model=EvidenceExtraction)
        assert isinstance(extraction, EvidenceExtraction)
        query_ids = {q.text: q.query_id for q in queries}
        bundle = self._assemble(objective, queries, selected, rejected, extraction, errors, query_ids)
        if bundle.status == "failed":
            scope.emit(EventType.RESEARCH_FAILED, research_id=bundle.research_id,
                       errors=[e.message for e in bundle.errors], warnings=bundle.warnings)
        else:
            scope.emit(EventType.RESEARCH_COMPLETED, research_id=bundle.research_id, status=bundle.status,
                       sources=len(bundle.sources), evidence=len(bundle.evidence),
                       findings=len(bundle.key_findings), citations=len(bundle.citations),
                       warnings=len(bundle.warnings), candidates=len(candidates),
                       duplicates_removed=len(ranking.duplicates))
        return bundle

    async def _search(self, queries: list[SearchQuery], objective: ResearchObjective,
                      ctx: AgentContext) -> tuple[list[RankCandidate], list[ResearchError]]:
        candidates: list[RankCandidate] = []
        errors: list[ResearchError] = []
        for q in queries:
            calls = (
                ("search.web", q, "search"),
                ("rag.retrieve", {"query": q.text, "k": KB_RESULTS_PER_QUERY,
                                  "filters": {"kind": "reference", "subject": objective.subject}}, "retrieve"),
            )
            for tool, payload, stage in calls:
                try:
                    out = await self.use_tool(tool, payload, ctx)
                except (ToolPermissionError, ToolNotFound):
                    raise
                except ToolError as exc:
                    errors.append(ResearchError(stage=stage, message=str(exc), query_id=q.query_id, tool=tool))
                    ctx.scope.emit(EventType.RESEARCH_SEARCH_COMPLETED, query_id=q.query_id, tool=tool,
                                   ok=False, error=str(exc)[:500])
                    continue
                assert isinstance(out, (SearchResponse, RetrieveOutput))
                cached = isinstance(out, SearchResponse) and out.cached
                ctx.scope.emit(EventType.RESEARCH_SEARCH_COMPLETED, query_id=q.query_id, tool=tool, ok=True,
                               results=len(out.results), cached=cached,
                               hits=[{"source_id": r.source_id, "url": r.url} for r in out.results])
                candidates += [RankCandidate(result=r, matched_queries=[q.text]) for r in out.results]
        return candidates, errors

    def _select(self, ranked: list[RankedSource],
                data: ResearchRequest) -> tuple[list[RankedSource], list[RejectedSource]]:
        selected: list[RankedSource] = []
        rejected: list[RejectedSource] = []
        for r in ranked:
            src = r.result.source
            rel = src.reliability
            if rel is None or rel.score is None:
                reason = f"reliability could not be assessed ({rel.basis if rel else 'no assessment'})"
            elif rel.score < data.min_reliability:
                reason = f"reliability {rel.score:.2f} is below {data.min_reliability:.2f} ({rel.basis})"
            elif r.signals.relevance == 0:
                reason = "not relevant to any research query"
            elif len(selected) >= data.max_sources:
                reason = f"ranked {r.rank}, below the top {data.max_sources}"
            else:
                selected.append(r)
                continue
            rejected.append(RejectedSource(source=src, reason=reason))
        return selected, rejected

    def _failed(self, objective: ResearchObjective, queries: list[SearchQuery], errors: list[ResearchError],
                rejected: list[RejectedSource], why: str, ctx: AgentContext) -> ResearchBundle:
        bundle = ResearchBundle(
            research_id=new_id("res"), objective=objective, status="failed", queries=queries,
            rejected_sources=rejected, errors=errors, warnings=[f"Research failed: {why}."],
            summary="No evidence was gathered.", generated_at=utcnow(),
        )
        ctx.scope.emit(EventType.RESEARCH_FAILED, research_id=bundle.research_id, reason=why,
                       errors=[e.message for e in errors])
        return bundle

    def _assemble(self, objective: ResearchObjective, queries: list[SearchQuery], selected: list[RankedSource],
                  rejected: list[RejectedSource], extraction: EvidenceExtraction, errors: list[ResearchError],
                  query_ids: dict[str, str]) -> ResearchBundle:
        extracted_at = utcnow()
        by_source = {r.result.source_id: r for r in selected}
        evidence: list[Evidence] = []
        ids: dict[str, str] = {}
        for item in extraction.evidence:
            ids[item.ref] = f"ev{len(evidence) + 1}"
            ranked = by_source[item.source_id]
            evidence.append(Evidence(
                evidence_id=ids[item.ref], source_id=item.source_id, target_id=item.target_id, text=item.text,
                relevance=item.relevance, location=EvidenceLocation(field=item.field, start=item.start, end=item.end),
                extraction=ExtractionMetadata(
                    method="model_quote", extractor=f"agent:{self.spec.id}", extracted_at=extracted_at,
                    query_ids=[query_ids[q] for q in ranked.matched_queries if q in query_ids],
                ),
            ))
        findings = [
            KeyFinding(finding_id=f"kf{i}", target_id=f.target_id, statement=f.statement, example=f.example,
                       practice_prompt=f.practice_prompt, practice_answer=f.practice_answer,
                       evidence_ids=[ids[ref] for ref in f.evidence_refs])
            for i, f in enumerate(extraction.findings, start=1)
        ]
        used = {e.source_id for e in evidence}
        sources: list[Source] = [r.result.source for r in selected if r.result.source_id in used]
        rejected = rejected + [RejectedSource(source=r.result.source, reason="no relevant evidence was extracted")
                               for r in selected if r.result.source_id not in used]
        citations = []
        for ev in evidence:
            src = by_source[ev.source_id].result.source
            loc = ev.location
            citations.append(Citation(
                citation_id=f"c{len(citations) + 1}", evidence_id=ev.evidence_id, source_id=src.source_id,
                title=src.title, url=src.url, publisher=src.publisher, retrieved_at=src.retrieved_at,
                locator=f"{loc.field}[{loc.start}:{loc.end}]" if loc else None,
            ))

        covered = defaultdict(int)
        for f in findings:
            covered[f.target_id] += 1
        warnings = [f"No evidence found for '{t.name}' ({t.target_id})." for t in objective.targets
                    if not covered[t.target_id]]
        warnings += [f"{e.stage} failed for {e.query_id or 'research'} via {e.tool}: {e.message}" for e in errors]
        if not evidence:
            status = "failed"
            warnings.insert(0, "Research failed: no evidence could be extracted from the selected sources.")
            sources, citations, findings = [], [], []
        else:
            status = "partial" if warnings else "complete"
        summary = extraction.summary or (
            f"{len(sources)} sources, {len(evidence)} evidence items and {len(findings)} findings covering "
            f"{sum(1 for t in objective.targets if covered[t.target_id])}/{len(objective.targets)} targets."
        )
        return ResearchBundle(
            research_id=new_id("res"), objective=objective, status=status, queries=queries, sources=sources,
            rejected_sources=rejected, evidence=evidence if status != "failed" else [], key_findings=findings,
            citations=citations, summary=summary, warnings=warnings, errors=errors, generated_at=extracted_at,
        )

    # --- semantic checks on the model's extraction ------------------------------------------------

    def check(self, output, source) -> None:
        if not isinstance(output, EvidenceExtraction):
            return
        assert isinstance(source, EvidenceExtractionInput)
        sources = {s.source_id: s for s in source.sources}
        targets = {t.target_id for t in source.objective.targets}
        refs: dict[str, str] = {}
        for ev in output.evidence:
            src = sources.get(ev.source_id)
            if src is None:
                raise OutputRejected(f"evidence {ev.ref} quotes source {ev.source_id}, which was not provided")
            if ev.target_id not in targets:
                raise OutputRejected(f"evidence {ev.ref} targets unknown research target {ev.target_id}")
            text = src.content if ev.field == "content" else src.snippet
            if text is None or text[ev.start:ev.end] != ev.text:
                raise OutputRejected(
                    f"evidence {ev.ref} is not a verbatim quote of {ev.source_id} {ev.field}[{ev.start}:{ev.end}]")
            if ev.ref in refs:
                raise OutputRejected(f"duplicate evidence ref {ev.ref}")
            refs[ev.ref] = ev.target_id
        per_target: dict[str, int] = defaultdict(int)
        for f in output.findings:
            missing = [r for r in f.evidence_refs if r not in refs]
            if missing:
                raise OutputRejected(f"finding for {f.target_id} cites unknown evidence {missing}")
            if any(refs[r] != f.target_id for r in f.evidence_refs):
                raise OutputRejected(f"finding for {f.target_id} cites evidence about another target")
            per_target[f.target_id] += 1
        over = [t for t, n in per_target.items() if n > source.max_findings_per_target]
        if over:
            raise OutputRejected(f"more than {source.max_findings_per_target} findings for {over}")
