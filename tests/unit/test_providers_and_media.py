from __future__ import annotations

import io
import json
import wave

from app.artifacts.service import ArtifactService
from app.config.settings import REPO_ROOT
from app.providers.image.mock import MockImageProvider
from app.providers.retrieval.local import LocalKnowledgeBase
from app.providers.search.mock import CorpusSearchProvider
from app.providers.tts.mock import MockTTSProvider
from app.providers.video.base import SceneSpec
from app.providers.video.mock import MockVideoProvider
from app.storage.db import create_db, dispose
from app.storage.object_store import FilesystemObjectStore
from app.storage.repositories import SqlArtifactRepository
from app.tools.base import ToolCaller
from app.tools.manager import ToolManager
from app.tools.media.tools import ImageGenerationTool, SpeechSynthesisTool, VideoRenderTool
from app.tools.rag.retrieve import ConceptMapTool
from app.tools.registry import ToolRegistry
from tests.unit.helpers import scope

CORPUS = REPO_ROOT / "fixtures" / "demo"


async def test_search_and_retrieval_are_deterministic() -> None:
    search = CorpusSearchProvider(CORPUS / "web_corpus.json")
    first = await search.search("football spanish", 3)
    assert first == await search.search("football spanish", 3) and len(first) == 3
    kb = LocalKnowledgeBase(CORPUS / "knowledge_base.json")
    refs = await kb.retrieve("preterite", 5, {"kind": "reference"})
    assert refs[0].doc_id == "es-ref-preterite" and all(p.metadata["kind"] == "reference" for p in refs)


async def test_concept_map_orders_prerequisites_first() -> None:
    tool = ConceptMapTool(LocalKnowledgeBase(CORPUS / "knowledge_base.json"))
    out = await tool.run(tool.input_model(subject="spanish", topic="football", level="A2"), scope()[0])
    ids = [e.concept.concept_id for e in out.concepts]
    assert len(ids) == 4
    assert ids.index("es.football.jugar_present") < ids.index("es.football.preterite_match")


async def test_media_tools_store_artifacts_through_the_tool_manager(tmp_path) -> None:
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    artifacts = ArtifactService(SqlArtifactRepository(sessions), FilesystemObjectStore(tmp_path / "o"))
    registry = ToolRegistry()
    for tool in (ImageGenerationTool(MockImageProvider(), artifacts), SpeechSynthesisTool(MockTTSProvider(), artifacts),
                 VideoRenderTool(MockVideoProvider(), artifacts)):
        registry.register(tool)
    manager = ToolManager(registry)
    caller = ToolCaller(caller_id="t", allowed_tools=frozenset(registry.names()),
                        permissions=frozenset({"media:generate", "artifact:write"}))
    sc, _ = scope("task1")

    image = await manager.call(caller, "image.generate", {"name": "diagram", "prompt": "a goal"}, sc)
    again = await manager.call(caller, "image.generate", {"name": "diagram", "prompt": "a goal"}, sc)
    assert image.artifact_id == again.artifact_id  # identical generation is reused
    assert artifacts.read(image.artifact_id).startswith(b"<svg")

    audio = await manager.call(caller, "audio.synthesize", {"name": "narration", "text": "uno dos tres",
                                                            "parent_ids": [image.artifact_id]}, sc)
    assert audio.metadata["duration_ms"] == 1050 and len(audio.metadata["timings"]) == 3
    with wave.open(io.BytesIO(artifacts.read(audio.artifact_id))) as wav:
        assert wav.getnframes() > 0

    scenes = [SceneSpec(scene_id="s1", narration="uno", visual_prompt="goal", duration_ms=1000).model_dump()]
    video = await manager.call(caller, "video.render", {"name": "video", "scenes": scenes,
                                                        "parent_ids": [audio.artifact_id]}, sc)
    assert json.loads(artifacts.read(video.artifact_id))["scenes"][0]["scene_id"] == "s1"
    assert artifacts.graph("task1")[video.artifact_id] == [audio.artifact_id]
    dispose(sessions)
