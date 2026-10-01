from __future__ import annotations

import json

from app.artifacts.service import ArtifactService
from app.config.settings import REPO_ROOT
from app.providers.retrieval.local import LocalKnowledgeBase
from app.providers.search.base import ProviderSearchRequest
from app.providers.search.mock import MockSearchProvider
from app.providers.video.base import SceneSpec
from app.providers.video.mock import MockVideoProvider
from app.storage.db import create_db, dispose
from app.storage.object_store import FilesystemObjectStore
from app.storage.repositories import SqlArtifactRepository
from app.tools.base import ToolCaller
from app.tools.manager import ToolManager
from app.schemas.artifact import ArtifactDraft, ArtifactType
from app.tools.media.tools import VideoRenderTool
from app.tools.rag.retrieve import ConceptMapTool
from app.tools.registry import ToolRegistry
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


async def test_video_tool_stores_artifacts_through_the_tool_manager(tmp_path) -> None:
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    artifacts = ArtifactService(SqlArtifactRepository(sessions), FilesystemObjectStore(tmp_path / "o"))
    registry = ToolRegistry()
    registry.register(VideoRenderTool(MockVideoProvider(), artifacts))
    manager = ToolManager(registry)
    caller = ToolCaller(caller_id="t", allowed_tools=frozenset(registry.names()),
                        permissions=frozenset({"media:generate", "artifact:write"}))
    sc, _ = scope("task1")
    script = artifacts.store_batch("task1", [ArtifactDraft(key="s", name="script", type=ArtifactType.SCRIPT,
                                                           media_type="text/markdown", content="uno")], sc)
    script_id = script.by_key["s"]

    scenes = [SceneSpec(scene_id="s1", narration="uno", visual_prompt="goal", duration_ms=1000).model_dump()]
    video = await manager.call(caller, "video.render", {"name": "video", "scenes": scenes,
                                                        "parent_ids": [script_id]}, sc)
    assert json.loads(artifacts.read(video.artifact_id))["scenes"][0]["scene_id"] == "s1"
    assert artifacts.graph("task1")[video.artifact_id] == [script_id]
    dispose(sessions)
