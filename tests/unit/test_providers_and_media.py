from __future__ import annotations

from app.config.settings import REPO_ROOT
from app.providers.retrieval.local import LocalKnowledgeBase
from app.providers.search.base import ProviderSearchRequest
from app.providers.search.mock import MockSearchProvider
from app.tools.rag.retrieve import ConceptMapTool
from tests.unit.helpers import scope

CORPUS = REPO_ROOT / "fixtures" / "demo"


async def test_search_and_retrieval_are_deterministic() -> None:
    search = MockSearchProvider(CORPUS / "web_corpus.json")
    request = ProviderSearchRequest(query="football spanish", max_results=3)
    first = await search.search(request)
    assert first == await search.search(request) and len(first.hits) == 3
    assert first.usage.requests == 1 and first.usage.results == 3 and first.usage.cost_usd == 0.0
    kb = LocalKnowledgeBase(CORPUS / "knowledge_base.json")
    refs = await kb.retrieve("preterite", 5, {"kind": "reference"})
    assert refs[0].doc_id == "es-ref-preterite" and all(p.metadata["kind"] == "reference" for p in refs)


async def test_concept_map_orders_prerequisites_first() -> None:
    tool = ConceptMapTool(LocalKnowledgeBase(CORPUS / "knowledge_base.json"))
    out = await tool.run(tool.input_model(subject="spanish", topic="football", level="A2"), scope()[0])
    ids = [e.concept.concept_id for e in out.concepts]
    assert len(ids) == 4
    assert ids.index("es.football.jugar_present") < ids.index("es.football.preterite_match")
