"""Research inside the lesson workflow: policy, events, artifact, downstream use and end-to-end traceability."""

from __future__ import annotations

import json

import pytest

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.retrieval.local import LocalKnowledgeBase
from app.providers.search.base import ProviderSearchRequest, SearchPage, SearchProvider, SearchProviderError
from app.schemas.artifact import ArtifactType
from app.schemas.lesson import LessonContent
from app.schemas.research import ResearchBundle
from app.schemas.task import TaskStatus
from app.services.container import build_container
from tests.conftest import FIXTURES, run_lesson


class DownSearch(SearchProvider):
    name = "down"

    async def search(self, request: ProviderSearchRequest) -> SearchPage:
        raise SearchProviderError("search API unavailable", transient=False)


class ReferencesDown(LocalKnowledgeBase):
    """The concept map still works (the diagnostic needs it); reference retrieval for research fails."""

    async def retrieve(self, query, k, filters=None):
        if filters and filters.get("kind") == "reference":
            raise ConnectionError("reference store offline")
        return await super().retrieve(query, k, filters)


@pytest.fixture
def make_container(tmp_path):
    made = []

    def make(*, down: bool = False, web_down: bool = False, llm: MockLLMProvider | None = None, **settings):
        kwargs = {}
        if down or web_down:
            kwargs["search_provider"] = DownSearch()
        if down:
            kwargs["retriever"] = ReferencesDown(FIXTURES / "knowledge_base.json")
        c = build_container(Settings(data_dir=tmp_path / f"d{len(made)}", log_json=False, **settings),
                            llm_providers={"mock": llm or MockLLMProvider(default_responders())}, **kwargs)
        made.append(c)
        return c

    yield make
    for c in made:
        c.close()


def research_of(container, task) -> ResearchBundle:
    artifact = container.artifacts.find(task.task_id, "research_bundle")
    assert artifact is not None and artifact.type == ArtifactType.RESEARCH_BUNDLE
    return ResearchBundle.model_validate_json(container.artifacts.read(artifact.artifact_id))


def lesson_of(container, task) -> LessonContent:
    artifact = container.artifacts.find(task.task_id, "lesson")
    return LessonContent.model_validate_json(container.artifacts.read(artifact.artifact_id))


async def test_end_to_end_lesson_has_traceable_research_references(make_container) -> None:
    llm = MockLLMProvider(default_responders())
    container = make_container(llm=llm)
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    order = [n for n in task.workflow.execution_order if n in
             {"diagnostic", "research", "research_policy", "store_research", "plan", "teach_review", "update_learner"}]
    assert order == ["diagnostic", "research", "research_policy", "store_research", "plan", "teach_review",
                     "update_learner"]

    research = research_of(container, task)
    lesson = lesson_of(container, task)
    assert research.status == "complete" and task.result.warnings == []
    # Lesson -> ResearchBundle -> Evidence -> Source, down to the exact quoted characters.
    corpus = {d["url"]: d["content"] for d in json.loads((FIXTURES / "web_corpus.json").read_text("utf-8"))}
    corpus |= {f"kb://{d['doc_id']}": d["text"]
               for d in json.loads((FIXTURES / "knowledge_base.json").read_text("utf-8"))}
    for section in lesson.sections:
        assert section.citations
        for cid in section.citations:
            citation, evidence, source = research.resolve(cid)
            assert evidence.target_id == section.concept_id
            assert corpus[source.url][evidence.location.start:evidence.location.end] == evidence.text
            assert citation.url == source.url and citation.retrieved_at == source.retrieved_at
    # The lesson carries its resolved references, identical to the bundle's citations.
    assert lesson.references == [c for c in research.citations
                                 if c.citation_id in {x for s in lesson.sections for x in s.citations}]


async def test_research_artifact_is_linked_and_stored_before_planning(make_container) -> None:
    container = make_container()
    task = await run_lesson(container)
    arts = {a.name: a for a in container.task_service.artifacts(task.task_id)}
    bundle_art = arts["research_bundle"]
    assert bundle_art.task_id == task.task_id and bundle_art.parent_ids == []
    assert bundle_art.metadata["status"] == "complete" and bundle_art.metadata["citations"] > 0
    assert bundle_art.artifact_id in task.artifact_ids
    assert arts["lesson_plan"].parent_ids == [bundle_art.artifact_id]
    assert bundle_art.artifact_id in arts["lesson"].parent_ids
    assert [a.name for a in container.artifacts.lineage(arts["slide_plan"].artifact_id)] == \
        ["lesson", "lesson_plan", "research_bundle"]
    events = container.task_service.events(task.task_id)
    created = next(i for i, e in enumerate(events) if e.type == "artifact.created"
                   and e.data["artifact_type"] == "RESEARCH_BUNDLE")
    plan_started = next(i for i, e in enumerate(events) if e.type == "node.started" and e.node_id == "plan")
    assert created < plan_started
    stored = research_of(container, task)
    assert stored == ResearchBundle.model_validate(task.workflow.node_states["research_policy"].output)


async def test_research_events_are_emitted_on_the_existing_bus(make_container) -> None:
    container = make_container()
    task = await run_lesson(container)
    events = [e for e in container.task_service.events(task.task_id) if e.type.startswith("research.")]
    types = [e.type for e in events]
    assert types[0] == "research.started" and types[-1] == "research.completed"
    for t in ("research.query_created", "research.search_completed", "research.source_selected"):
        assert t in types
    assert all(e.task_id == task.task_id and e.node_id == "research" and e.agent_id == "research" for e in events)
    completed = events[-1].data
    assert completed["status"] == "complete" and completed["citations"] == completed["evidence"] > 0


async def test_planner_and_teacher_receive_the_research(make_container) -> None:
    llm = MockLLMProvider(default_responders())
    container = make_container(llm=llm)
    task = await run_lesson(container)
    research = research_of(container, task)
    planner = next(r for r in llm.requests if r.agent_id == "curriculum_planner").input_payload["research"]
    assert ResearchBundle.model_validate(planner) == research  # the whole bundle
    plan = json.loads(container.artifacts.read(container.artifacts.find(task.task_id, "lesson_plan").artifact_id))
    taught = {c["concept_id"] for c in plan["concepts"]}
    for request in (r for r in llm.requests if r.agent_id == "teacher"):
        focused = ResearchBundle.model_validate(request.input_payload["research"])
        # Only the planned concepts' research, with every citation kept intact.
        assert {f.target_id for f in focused.key_findings} == taught
        assert focused.research_id == research.research_id
        assert all(c in research.citations for c in focused.citations)
        assert {c.evidence_id for c in focused.citations} == {e.evidence_id for e in focused.evidence}
    reviewer = next(r for r in llm.requests if r.agent_id == "content_reviewer").input_payload["research"]
    assert ResearchBundle.model_validate(reviewer) == research


async def test_mandatory_research_failure_fails_the_task(make_container) -> None:
    container = make_container(down=True, research_requirement="mandatory")
    task = await run_lesson(container)
    assert task.status == TaskStatus.FAILED
    assert task.errors[-1].node_id == "research_policy"
    assert "research is mandatory and failed" in task.errors[-1].message
    assert "search API unavailable" in task.errors[-1].message
    # The failed bundle is inspectable in the checkpoint; nothing was invented or stored.
    bundle = ResearchBundle.model_validate(task.workflow.node_states["research"].output)
    assert bundle.status == "failed" and bundle.sources == [] and bundle.errors
    assert container.task_service.artifacts(task.task_id) == []
    assert "research.failed" in [e.type for e in container.task_service.events(task.task_id)]


async def test_optional_research_failure_continues_with_an_explicit_warning(make_container) -> None:
    container = make_container(down=True, research_requirement="optional")
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    research = research_of(container, task)
    assert research.status == "failed" and research.sources == [] and research.citations == []
    assert any("optional" in w for w in research.warnings)
    assert task.result.warnings == research.warnings
    lesson = lesson_of(container, task)
    assert all(not s.citations for s in lesson.sections) and lesson.references == []
    review = json.loads(container.artifacts.read(container.artifacts.find(task.task_id, "review_report").artifact_id))
    assert any(i["problem"].startswith("Unverified") and i["severity"] == "minor"
               for i in review["final_review"]["issues"])


async def test_partial_research_is_used_with_warnings(make_container) -> None:
    container = make_container(web_down=True)
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    research = research_of(container, task)
    assert research.status == "partial" and {s.retrieved_via for s in research.sources} == {"knowledge_base"}
    assert task.result.warnings and any("search failed" in w for w in task.result.warnings)
    lesson = lesson_of(container, task)
    assert all(s.citations for s in lesson.sections)  # the taught concepts are covered by the knowledge base


async def test_research_usage_and_cache_across_lessons(make_container) -> None:
    container = make_container(diagnostic_memory_confidence=0.3)
    first = await run_lesson(container)
    assert first.status == TaskStatus.COMPLETED
    search = first.cost.by_service["search:mock"]
    assert search.calls == 5 and search.cache_hits == 0 and search.cost_usd == 0.0
    assert first.cost.by_service["retrieval:local"].cost_usd is None  # no cost reported, none invented
    assert first.cost.llm_calls == sum(line.calls for line in first.cost.by_agent.values())

    second = await container.task_service.create_and_run(request="Create an A2 lesson about football.",
                                                         learner_id=first.learner_id, user_id="u1")
    assert second.status == TaskStatus.COMPLETED
    assert second.cost.by_service["search:mock"].cache_hits == 5 and second.cost.by_service["search:mock"].calls == 0
    # Cached results keep their provenance: the same ids and the original retrieval time.
    a, b = research_of(container, first), research_of(container, second)
    web = lambda bundle: {s.source_id: s.retrieved_at for s in bundle.sources if s.retrieved_via == "web"}  # noqa: E731
    assert web(b) and web(b).items() <= web(a).items()


async def test_research_agent_retries_are_explicit_failures_not_silent(make_container) -> None:
    llm = MockLLMProvider(default_responders())
    llm.inject("research", *(["not json"] * 6))
    container = make_container(llm=llm, research_requirement="optional")
    task = await run_lesson(container)
    # A broken model is an agent failure, not a research outage: the optional policy does not hide it.
    assert task.status == TaskStatus.FAILED and task.errors[-1].node_id == "research"


async def test_taught_concepts_without_evidence_are_marked_unverified(make_container) -> None:
    # Only the grammar reference (0.95) passes this bar: it covers a known concept, not the taught gaps.
    container = make_container(research_min_reliability=0.92)
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    research = research_of(container, task)
    assert research.status == "partial"
    assert {s.source_type for s in research.sources} == {"reference"}
    assert any("No evidence found for 'Opinions with gustar and encantar'" in w for w in task.result.warnings)
    lesson = lesson_of(container, task)
    assert all(not s.citations for s in lesson.sections) and lesson.references == []
    review = json.loads(container.artifacts.read(container.artifacts.find(task.task_id, "review_report").artifact_id))
    unverified = [i for i in review["final_review"]["issues"] if i["problem"].startswith("Unverified")]
    assert len(unverified) == len(lesson.sections) and all(i["severity"] == "minor" for i in unverified)
