"""Research building blocks: schemas, search provider and tool, cache, deduplication, ranking and the agent."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from app.agents.base import AgentContext
from app.agents.research.agent import ResearchAgent
from app.config.settings import REPO_ROOT
from app.providers.core.errors import ProviderError
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.llm.router import ModelRouter
from app.providers.ranking.heuristic import HeuristicRanker
from app.providers.retrieval.local import LocalKnowledgeBase
from app.providers.search.base import (
    ProviderSearchRequest,
    SearchHit,
    SearchPage,
    SearchProvider,
    SearchUsage,
)
from app.providers.search.mock import MockSearchProvider
from app.schemas.lesson import ConceptRef, DiagnosticResult, LessonRequest, ResearchRequest
from app.schemas.research import (
    Citation,
    Evidence,
    EvidenceLocation,
    ExtractionMetadata,
    KeyFinding,
    RankCandidate,
    RankingWeights,
    ResearchBundle,
    ResearchObjective,
    ResearchTarget,
    SearchQuery,
    SearchResult,
    Source,
)
from app.tools.base import ToolCaller, ToolError, ToolTransientError
from app.tools.manager import ToolManager
from app.tools.rag.retrieve import RetrievalTool
from app.tools.registry import ToolRegistry
from app.tools.research.cache import InMemoryResearchCache
from app.tools.research.dedup import deduplicate
from app.tools.research.rank import RankInput, RankSourcesTool
from app.tools.web.search import SearchTool
from app.utils.urls import canonical_url, source_id_for
from tests.unit.helpers import NOW, routing, scope

CORPUS = REPO_ROOT / "fixtures" / "demo"
CALLER = ToolCaller("test", frozenset({"search.web", "rag.retrieve", "research.rank"}),
                    frozenset({"network", "knowledge:read"}))


def src(url: str, **kw) -> Source:
    fields = {"title": f"Title of {url}", "publisher": "P", "source_type": "reference", "language": "en", **kw}
    return Source(source_id=source_id_for(url), url=url, canonical_url=canonical_url(url), retrieved_at=NOW,
                  retrieved_via="web", provider="test", **fields)


def result(url: str, *, snippet: str = "s", content: str | None = None, rank: int = 1, **kw) -> SearchResult:
    s = src(url, **kw)
    return SearchResult(source_id=s.source_id, title=s.title, url=url, snippet=snippet, content=content, rank=rank,
                        source=s)


def cand(url: str, query: str = "q", **kw) -> RankCandidate:
    return RankCandidate(result=result(url, **kw), matched_queries=[query])


def objective(*targets: str) -> ResearchObjective:
    return ResearchObjective(description="d", subject="s", topic="t",
                             targets=[ResearchTarget(target_id=t, name=t) for t in targets or ("c1",)])


def bundle(**kw) -> ResearchBundle:
    s = src("https://a.example/x")
    ev = Evidence(evidence_id="ev1", source_id=s.source_id, target_id="c1", text="claim", relevance=0.8,
                  location=EvidenceLocation(field="content", start=0, end=5),
                  extraction=ExtractionMetadata(method="m", extractor="x", extracted_at=NOW))
    cit = Citation(citation_id="c1", evidence_id="ev1", source_id=s.source_id, title=s.title, url=s.url,
                   publisher=s.publisher, retrieved_at=NOW)
    finding = KeyFinding(finding_id="kf1", target_id="c1", statement="claim", evidence_ids=["ev1"])
    fields = {"research_id": "r", "objective": objective(), "status": "complete", "sources": [s], "evidence": [ev],
              "key_findings": [finding], "citations": [cit], "generated_at": NOW, **kw}
    return ResearchBundle(**fields)


# --- schemas -----------------------------------------------------------------------------------


def test_equivalent_urls_share_a_canonical_form_and_source_id() -> None:
    variants = ["https://example.org/a/b", "http://example.org/a/b/", "https://WWW.Example.org/a/b#part",
                "https://example.org:443/a/b?utm_source=x&utm_medium=y"]
    assert len({canonical_url(u) for u in variants}) == 1
    assert len({source_id_for(u) for u in variants}) == 1
    assert source_id_for("https://example.org/a/c") != source_id_for(variants[0])
    assert canonical_url("https://example.org/p?b=2&a=1") == canonical_url("https://example.org/p?a=1&b=2")
    assert canonical_url("kb://doc-1") == "kb://doc-1"


def test_source_schema_keeps_unknown_metadata_empty() -> None:
    s = src("https://a.example/x")
    assert s.author is None and s.published_at is None and s.reliability is None
    with pytest.raises(ValidationError):
        Source.model_validate({**s.model_dump(), "made_up": 1})
    with pytest.raises(ValidationError):  # retrieval time is required for traceability
        Source.model_validate({k: v for k, v in s.model_dump().items() if k != "retrieved_at"})
    r = result("https://a.example/x")
    with pytest.raises(ValidationError, match="match its source"):
        SearchResult.model_validate({**r.model_dump(), "url": "https://other.example/"})


def test_evidence_and_citation_schemas() -> None:
    with pytest.raises(ValidationError, match="after start"):
        EvidenceLocation(field="content", start=5, end=5)
    with pytest.raises(ValidationError):  # evidence must keep its source reference
        Evidence(evidence_id="e", target_id="c", text="x", relevance=0.5,
                 extraction=ExtractionMetadata(method="m", extractor="x", extracted_at=NOW))
    cit = bundle().citations[0]
    assert cit.reference() == f"Title of https://a.example/x. P. https://a.example/x. retrieved {NOW.date().isoformat()}"


def test_bundle_enforces_traceability() -> None:
    good = bundle()
    citation, ev, source = good.resolve("c1")
    assert (citation.evidence_id, ev.source_id) == ("ev1", source.source_id)
    with pytest.raises(LookupError):
        good.resolve("c404")
    s = good.sources[0]
    with pytest.raises(ValidationError, match="deduplicated"):
        bundle(sources=[s, s])
    with pytest.raises(ValidationError, match="unknown or unselected source"):
        bundle(sources=[src("https://b.example/")])
    with pytest.raises(ValidationError, match="unknown evidence"):
        bundle(key_findings=[good.key_findings[0].model_copy(update={"evidence_ids": ["ev9"]})])
    with pytest.raises(ValidationError, match="does not match"):
        bundle(citations=[good.citations[0].model_copy(update={"source_id": "src_other"})])
    with pytest.raises(ValidationError, match="exactly one citation"):
        bundle(citations=[])
    with pytest.raises(ValidationError, match="needs evidence"):
        bundle(evidence=[], key_findings=[], citations=[])
    assert bundle(status="partial", sources=[], evidence=[], key_findings=[], citations=[]).evidence == []
    with pytest.raises(ValidationError, match="no sources or evidence"):
        bundle(status="failed", warnings=["w"])
    with pytest.raises(ValidationError, match="say why"):
        bundle(status="failed", sources=[], evidence=[], key_findings=[], citations=[])
    failed = bundle(status="failed", sources=[], evidence=[], key_findings=[], citations=[], warnings=["down"])
    assert failed.citations == []


def test_focused_bundle_keeps_citations_for_the_chosen_targets() -> None:
    good = bundle()
    assert good.focused(["c1"]) == good
    empty = good.focused(["other"])
    assert empty.citations == [] and empty.evidence == [] and empty.research_id == good.research_id
    assert empty.status == "partial" and "no evidence" in empty.warnings[-1]
    ResearchBundle.model_validate(empty.model_dump())  # still a valid bundle when sent to the next agent


def test_query_cache_key_ignores_ids_and_formatting() -> None:
    a = SearchQuery(query_id="q1", text="Football  Spanish", target_ids=["x"], language="en")
    b = SearchQuery(query_id="q9", text=" football spanish", language="en")
    assert a.cache_key() == b.cache_key()
    assert a.cache_key() != b.model_copy(update={"language": "es"}).cache_key()
    assert a.cache_key() != b.model_copy(update={"max_results": 9}).cache_key()


# --- provider and search tool ---------------------------------------------------------------------


class FakeProvider(SearchProvider):
    name = "fake"

    def __init__(self, hits: list[SearchHit] | None = None, error: Exception | None = None) -> None:
        self.hits = hits or [SearchHit(url="https://a.example/x", title="A", snippet="about football")]
        self.error = error
        self.calls = 0

    async def search(self, request: ProviderSearchRequest) -> SearchPage:
        self.calls += 1
        if self.error:
            raise self.error
        return SearchPage(hits=self.hits[:request.max_results], usage=SearchUsage(results=len(self.hits)))


def manager(*tools) -> ToolManager:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return ToolManager(reg)


async def test_mock_provider_is_deterministic_and_reports_only_supplied_metadata() -> None:
    provider = MockSearchProvider(CORPUS / "web_corpus.json")
    request = ProviderSearchRequest(query="football spanish preterite", max_results=10, language="en")
    first = await provider.search(request)
    assert first == await provider.search(request)
    by_url = {h.url: h for h in first.hits}
    forum = by_url["https://fan-forum.example.com/thread/1234"]
    assert forum.published_at is None and forum.author is None and forum.source_type == "forum"
    blog = by_url["https://language-school.example.com/blog/football-vocabulary"]
    assert blog.author == "Lucía Romero" and blog.published_at == date(2023, 9, 1)
    assert first.usage == SearchUsage(requests=1, results=len(first.hits), cost_usd=0.0)
    assert (await provider.search(request.model_copy(update={"language": "fr"}))).hits == []


async def test_search_tool_returns_traceable_results_and_records_usage() -> None:
    tool = SearchTool(MockSearchProvider(CORPUS / "web_corpus.json"), clock=lambda: NOW)
    sc, events = scope()
    out = await manager(tool).call(CALLER, "search.web", SearchQuery(text="football vocabulary", max_results=3), sc)
    assert [r.rank for r in out.results] == [1, 2, 3] and not out.cached
    for r in out.results:
        assert r.source_id == source_id_for(r.url) == r.source.source_id
        assert r.source.retrieved_at == NOW and r.source.provider == "mock" and r.source.retrieved_via == "web"
        assert r.content and r.snippet
    usage = sc.usage.summary.by_service["search:mock"]
    assert (usage.calls, usage.results, usage.cost_usd) == (1, 3, 0.0)
    assert [e.type for e in events] == ["tool.started", "tool.finished"]


async def test_search_tool_maps_provider_failures() -> None:
    transient = FakeProvider(error=ProviderError("rate limited"))
    with pytest.raises(ToolTransientError):
        await manager(SearchTool(transient)).call(CALLER, "search.web", {"text": "x"}, scope()[0])
    assert transient.calls == 3  # the tool's retry policy
    fatal = FakeProvider(error=ProviderError("bad api key", transient=False))
    with pytest.raises(ToolError, match="bad api key"):
        await manager(SearchTool(fatal)).call(CALLER, "search.web", {"text": "x"}, scope()[0])
    assert fatal.calls == 1


async def test_cache_serves_identical_traceable_results_and_can_be_invalidated() -> None:
    provider = FakeProvider()
    cache = InMemoryResearchCache()
    times = iter([NOW, NOW + timedelta(hours=1), NOW + timedelta(hours=2)])
    tool = SearchTool(provider, cache=cache, clock=lambda: next(times))
    m, sc = manager(tool), scope()[0]
    query = SearchQuery(query_id="q1", text="Football", target_ids=["a"])
    first = await m.call(CALLER, "search.web", query, sc)
    again = await m.call(CALLER, "search.web", query.model_copy(update={"query_id": "q7", "text": "football"}), sc)
    assert provider.calls == 1 and again.cached and not first.cached
    assert again.results == first.results  # same source ids, urls and original retrieval time
    assert again.results[0].source.retrieved_at == NOW
    line = sc.usage.summary.by_service["search:fake"]
    assert (line.calls, line.cache_hits, line.cost_usd) == (1, 1, None)  # the fake reports no cost: none invented
    cache.invalidate(query)
    fresh = await m.call(CALLER, "search.web", query, sc)
    assert provider.calls == 2 and not fresh.cached and fresh.results[0].source.retrieved_at == NOW + timedelta(hours=1)


def test_cache_get_set_invalidate_and_isolation() -> None:
    cache = InMemoryResearchCache()
    q = SearchQuery(text="x")
    assert cache.get(q) is None
    stored = [result("https://a.example/x")]
    cache.set(q, stored)
    got = cache.get(q)
    assert got == stored and got is not stored
    got[0].source.metadata["tampered"] = True
    assert "tampered" not in cache.get(q)[0].source.metadata
    cache.invalidate(q)
    assert cache.get(q) is None and len(cache) == 0


async def test_retrieval_tool_returns_knowledge_base_sources() -> None:
    tool = RetrievalTool(LocalKnowledgeBase(CORPUS / "knowledge_base.json"), clock=lambda: NOW)
    sc = scope()[0]
    out = await manager(tool).call(CALLER, "rag.retrieve",
                                   {"query": "preterite", "k": 2, "filters": {"kind": "reference"}}, sc)
    top = out.results[0]
    assert top.url == "kb://es-ref-preterite" and top.source.source_type == "knowledge_base"
    assert top.source.publisher == "Course knowledge base" and top.source.published_at is None
    assert top.source_id == source_id_for("kb://es-ref-preterite") and top.content
    assert sc.usage.summary.by_service["retrieval:local"].calls == 1


# --- deduplication ------------------------------------------------------------------------------


def test_exact_duplicate_urls_are_removed() -> None:
    out = deduplicate([cand("https://a.example/x"), cand("https://a.example/x"), cand("https://b.example/y")])
    assert [c.result.url for c in out.unique] == ["https://a.example/x", "https://b.example/y"]
    assert [d.reason for d in out.duplicates] == ["same source id"]


def test_repeated_results_and_equivalent_urls_collapse_to_one_source() -> None:
    out = deduplicate([cand("https://a.example/x", rank=1), cand("http://www.a.example/x/?utm_source=n#top", rank=2),
                       cand("https://a.example/x", rank=3)])
    assert len(out.unique) == 1 and len(out.duplicates) == 2
    assert out.unique[0].result.url == "https://a.example/x"  # the first occurrence is kept


def test_multiple_queries_returning_the_same_source_merge_their_queries() -> None:
    out = deduplicate([cand("https://a.example/x", "q one"), cand("https://b.example/y", "q one"),
                       cand("https://a.example/x", "q two"), cand("https://a.example/x", "q one")])
    by_url = {c.result.url: c.matched_queries for c in out.unique}
    assert by_url == {"https://a.example/x": ["q one", "q two"], "https://b.example/y": ["q one"]}


def test_mirrors_with_identical_content_or_title_are_duplicates() -> None:
    out = deduplicate([cand("https://a.example/x", content="Same body."), cand("https://mirror.example/x",
                                                                             content="same body", title="Other"),
                       cand("https://c.example/z", title="Guide", publisher="Pub"),
                       cand("https://d.example/z", title="guide", publisher="pub")])
    assert [c.result.url for c in out.unique] == ["https://a.example/x", "https://c.example/z"]
    assert {d.reason for d in out.duplicates} == {"identical content", "same title and publisher"}


# --- ranking ---------------------------------------------------------------------------------


async def rank(*candidates: RankCandidate, language: str | None = "en", weights: RankingWeights | None = None):
    payload = RankInput(candidates=list(candidates), language=language, as_of=NOW,
                        weights=weights or RankingWeights())
    return await manager(RankSourcesTool(HeuristicRanker())).call(CALLER, "research.rank", payload, scope()[0])


async def test_ranking_prefers_relevant_high_quality_fresh_matching_sources() -> None:
    q = "gustar opinions"
    out = await rank(
        cand("https://forum.example/1", q, content="gustar opinions (forum)", source_type="forum", publisher="F"),
        cand("https://ref.example/1", q, content="gustar opinions (ref)", source_type="reference", publisher="R"),
        cand("https://ref.example/2", q, content="unrelated text", source_type="reference", publisher="R2"),
        cand("https://ref.example/3", q, content="gustar opinions (fr)", source_type="reference", publisher="R3",
             language="fr"),
    )
    urls = [r.result.url for r in out.ranked]
    assert urls[0] == "https://ref.example/1"
    assert urls.index("https://ref.example/3") > 0  # language mismatch costs rank
    assert [r.rank for r in out.ranked] == [1, 2, 3, 4]
    top = out.ranked[0]
    assert top.signals.relevance == 1.0 and top.result.source.reliability.score == 0.95
    assert next(r for r in out.ranked if r.result.url.endswith("/2")).signals.relevance == 0.0
    assert out.ranker == "heuristic"


async def test_ranking_uses_freshness_only_when_dates_exist_and_marks_unknowns() -> None:
    old = cand("https://a.example/old", "x", content="x old", published_at=NOW.date() - timedelta(days=3652))
    new = cand("https://b.example/new", "x", content="x new", published_at=NOW.date())
    undated = cand("https://c.example/undated", "x", content="x undated", source_type="mystery", language=None)
    out = await rank(old, new, undated, weights=RankingWeights(relevance=0, quality=0, freshness=1, language=0,
                                                               novelty=0))
    by_url = {r.result.url: r for r in out.ranked}
    assert out.ranked[0].result.url.endswith("/new")
    assert by_url["https://a.example/old"].signals.freshness == 0.25  # two half-lives
    unknown = by_url["https://c.example/undated"]
    assert unknown.signals.freshness == 0.5 and "no publication date" in unknown.signals.notes
    assert unknown.result.source.reliability.score is None  # an unrated source type gets no invented score
    assert "language unknown" in unknown.signals.notes


async def test_ranking_is_deterministic_and_deduplicates_first() -> None:
    items = [cand("https://a.example/x", "x", content="x"), cand("https://b.example/y", "x", content="x y"),
             cand("https://a.example/x", "x", content="x")]
    first, second = await rank(*items), await rank(*items)
    assert first == second and len(first.ranked) == 2 and len(first.duplicates) == 1
    with pytest.raises(ValueError):
        HeuristicRanker().score([items[0]], None, RankingWeights(relevance=0, quality=0, freshness=0, language=0,
                                                                 novelty=0))


# --- the agent -----------------------------------------------------------------------------------


CONCEPTS = [ConceptRef.model_validate(d["metadata"]["concept"])
            for d in json.loads((CORPUS / "knowledge_base.json").read_text(encoding="utf-8"))
            if d["metadata"]["kind"] == "concept"]


def research_request(**kw) -> ResearchRequest:
    ids = [c.concept_id for c in CONCEPTS]
    diagnostic = DiagnosticResult(source="assessment", estimated_level="A2", concept_mastery=[], known=ids[:2],
                                  gaps=["es.football.opinions"], starting_point="es.football.opinions")
    request = LessonRequest(raw_request="r", subject="spanish", topic="football", framework_id="cefr",
                            target_level="A2", capabilities=["lesson.text"])
    return ResearchRequest(request=request, diagnostic=diagnostic, concepts=CONCEPTS, **kw)


class DownRetriever(LocalKnowledgeBase):
    async def retrieve(self, query, k, filters=None):
        raise ConnectionError("knowledge base offline")


def research_context(llm: MockLLMProvider, provider: SearchProvider | None = None, retriever=None):
    kb = LocalKnowledgeBase(CORPUS / "knowledge_base.json")
    tools = manager(SearchTool(provider or MockSearchProvider(CORPUS / "web_corpus.json"), clock=lambda: NOW),
                    RetrievalTool(retriever or kb, clock=lambda: NOW), RankSourcesTool(HeuristicRanker()))
    sc, events = scope()
    return AgentContext(router=ModelRouter(routing(("mock", "m1")), {"mock": llm}), tools=tools, scope=sc), events


async def test_research_agent_builds_a_traceable_bundle() -> None:
    llm = MockLLMProvider(default_responders())
    ctx, events = research_context(llm)
    agent = ResearchAgent()
    out = await agent.execute(research_request(), ctx)
    assert isinstance(out, ResearchBundle) and out.status == "complete" and not out.warnings

    # Gaps are researched first; one broad query plus one per concept.
    assert out.objective.targets[0].target_id == "es.football.opinions"
    assert [q.query_id for q in out.queries] == ["q1", "q2", "q3", "q4", "q5"]
    # The mirrored news article (www., utm, trailing slash, fragment) is one source; the forum is rejected.
    urls = [s.url for s in out.sources]
    assert len(urls) == len(set(urls)) and sum("match-report" in u for u in urls) == 1
    assert any(r.source.publisher == "Fan Forum" and "below 0.50" in r.reason for r in out.rejected_sources)
    assert {f.target_id for f in out.key_findings} == {c.concept_id for c in CONCEPTS}

    # Each evidence item quotes its source verbatim at the recorded location, and has one citation.
    corpus = {d["url"]: d["content"] for d in json.loads((CORPUS / "web_corpus.json").read_text(encoding="utf-8"))}
    kb = {f"kb://{d['doc_id']}": d["text"]
          for d in json.loads((CORPUS / "knowledge_base.json").read_text(encoding="utf-8"))}
    for ev in out.evidence:
        source = out.source(ev.source_id)
        text = corpus.get(source.url) or kb[source.url]
        assert text[ev.location.start:ev.location.end] == ev.text
        assert ev.extraction.extractor == "agent:research" and ev.extraction.query_ids
        assert out.citation_for(ev.evidence_id).source_id == source.source_id
    assert len(out.citations) == len(out.evidence)

    types = [e.type for e in events if e.type.startswith("research.")]
    assert types[0] == "research.started" and types[-1] == "research.completed"
    assert types.count("research.query_created") == 5
    assert types.count("research.search_completed") == 10  # web + knowledge base per query
    assert types.count("research.source_selected") == len(out.sources)
    order = [types.index(t) for t in ("research.started", "research.query_created", "research.search_completed",
                                       "research.source_selected", "research.completed")]
    assert order == sorted(order)
    assert llm.calls["research"] == 1


async def test_research_agent_rejects_non_verbatim_evidence() -> None:
    llm = MockLLMProvider(default_responders())
    blog = source_id_for("https://language-school.example.com/blog/football-vocabulary")
    llm.inject("research", json.dumps({
        "evidence": [{"ref": "e1", "source_id": blog, "target_id": "es.football.positions", "field": "content",
                      "text": "A claim the source never made.", "start": 0, "end": 30, "relevance": 0.9}],
        "findings": [{"target_id": "es.football.positions", "statement": "Invented.", "evidence_refs": ["e1"]}],
    }))
    ctx, events = research_context(llm)
    out = await ResearchAgent().execute(research_request(), ctx)
    assert out.status == "complete" and llm.calls["research"] == 2
    assert all("never made" not in e.text for e in out.evidence)
    failures = [e for e in events if e.type == "agent.validation_failed"]
    assert len(failures) == 1 and "not a verbatim quote" in failures[0].data["error"]


async def test_search_failure_is_recorded_and_no_source_is_invented() -> None:
    llm = MockLLMProvider(default_responders())
    ctx, events = research_context(llm, FakeProvider(error=ProviderError("down", transient=False)),
                                   DownRetriever(CORPUS / "knowledge_base.json"))
    out = await ResearchAgent().execute(research_request(), ctx)
    assert out.status == "failed" and out.sources == [] and out.evidence == [] and out.citations == []
    assert {e.stage for e in out.errors} == {"search", "retrieve"} and len(out.errors) == 10
    assert out.warnings and "Research failed" in out.warnings[0]
    assert "research.failed" in [e.type for e in events] and "research.completed" not in [e.type for e in events]
    assert llm.calls["research"] == 0  # nothing to extract from, so the model is never asked


async def test_partial_search_failure_keeps_real_sources_and_warns() -> None:
    llm = MockLLMProvider(default_responders())
    ctx, _ = research_context(llm, FakeProvider(error=ProviderError("down", transient=False)))
    out = await ResearchAgent().execute(research_request(), ctx)
    assert out.status == "partial"
    assert {s.retrieved_via for s in out.sources} == {"knowledge_base"}
    assert {f.target_id for f in out.key_findings} == {"es.football.opinions", "es.football.preterite_match"}
    assert any("No evidence found for 'Player positions'" in w for w in out.warnings)
    assert all(e.stage == "search" for e in out.errors)


def test_research_agent_uses_only_tools() -> None:
    spec = ResearchAgent.spec
    assert set(spec.tools) == {"search.web", "rag.retrieve", "research.rank"}
    assert spec.output_model is ResearchBundle and spec.input_model is ResearchRequest
