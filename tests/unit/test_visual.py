"""Visual slice units: schemas, providers, tools (through the ToolManager), selection, validation, storage."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.artifacts.service import ArtifactService
from app.config.settings import REPO_ROOT
from app.providers.image.base import GeneratedImage, ImageGenerationProvider, ProviderImageRequest
from app.providers.image.mock import MockImageGenerationProvider
from app.providers.image_search.base import ImageSearchProviderError, ProviderImageSearchRequest
from app.providers.image_search.mock import MockImageSearchProvider
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.visual import (
    GenerationAttribution,
    ImageAsset,
    ImageGenerationRequest,
    ImageSearchResult,
    ImageSelectionResult,
    ImageUsage,
    ImageValidationReport,
    SelectionRecord,
    VisualPlan,
    VisualPlanProposal,
    VisualRequirement,
    VisualResult,
    VisualType,
)
from app.storage.db import create_db, dispose
from app.storage.object_store import FilesystemObjectStore
from app.storage.repositories import SqlArtifactRepository
from app.tools.base import ToolCaller, ToolError, ToolPermissionError, ToolTransientError
from app.tools.manager import ToolManager
from app.tools.registry import ToolRegistry
from app.tools.visual.assets import ImageAssetRejected, ImageAssetTool
from app.tools.visual.generate import ImageGenerationTool, prompt_hash
from app.tools.visual.search import ImageFetchTool, ImageSearchTool
from app.tools.visual.selection import ImageSelectionTool
from app.tools.visual.validation import ImageValidationTool
from app.utils.images import ImageProbeError, encode_png, probe_image
from tests.unit.helpers import NOW, scope

CATALOG = REPO_ROOT / "fixtures" / "demo" / "image_catalog.json"
ALL = frozenset({"network", "media:generate", "artifact:write", "artifact:read"})


def requirement(**over) -> VisualRequirement:
    base = dict(visual_id="v1", purpose="Context", lesson_section_id="s1", concept="Opinions",
                description="Fans at a match", visual_type="photo", preferred_source="search",
                search_query="football fans opinions", aspect_ratio="16:9", required=True, attribution_required=True)
    return VisualRequirement.model_validate({**base, **over})


class Env:
    def __init__(self, tmp_path, generation: ImageGenerationProvider | None = None) -> None:
        self.db = tmp_path / "db.sqlite"
        self.sessions = create_db(f"sqlite:///{self.db}")
        self.root = tmp_path / "objects"
        self.artifacts = ArtifactService(SqlArtifactRepository(self.sessions), FilesystemObjectStore(self.root))
        self.search_provider = MockImageSearchProvider(CATALOG)
        registry = ToolRegistry()
        for tool in (ImageSearchTool(self.search_provider, clock=lambda: NOW),
                     ImageFetchTool(self.search_provider, self.artifacts), ImageSelectionTool(),
                     ImageGenerationTool(generation or MockImageGenerationProvider(), self.artifacts, clock=lambda: NOW),
                     ImageValidationTool(self.artifacts), ImageAssetTool(self.artifacts)):
            registry.register(tool)
        self.tools = ToolManager(registry)
        self.caller = ToolCaller(caller_id="test", allowed_tools=frozenset(registry.names()), permissions=ALL)
        self.scope, self.events = scope("task1")

    async def call(self, name: str, payload):
        return await self.tools.call(self.caller, name, payload, self.scope)

    def object_files(self) -> list:
        return sorted(p for p in self.root.rglob("*") if p.is_file())


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    dispose(e.sessions)


# --- schemas ------------------------------------------------------------------------------


def test_visual_types_are_domain_independent_and_parse_provider_labels() -> None:
    assert {t.value for t in VisualType} >= {"photo", "illustration", "diagram", "chart", "map", "icon",
                                             "generated_visual"}
    assert VisualType.parse("Photograph") == VisualType.PHOTO
    assert VisualType.parse("vector") == VisualType.ILLUSTRATION
    assert VisualType.parse("hologram") is None and VisualType.parse(None) is None


def test_visual_requirement_rules() -> None:
    assert requirement().sources() == ["search"]  # a photo is never generated
    illustration = requirement(visual_type="illustration", generation_prompt="A drawing of fans")
    assert illustration.sources() == ["search", "generate"]
    assert requirement(visual_type="diagram", preferred_source="generate", generation_prompt="x",
                       search_query=None).sources() == ["generate"]
    bad = [
        {"search_query": None},  # searched without a query
        {"visual_type": "diagram", "preferred_source": "generate", "search_query": None},  # no prompt
        {"preferred_source": "generate", "generation_prompt": "a photo"},  # a photo must be searched
        {"visual_type": "map", "preferred_source": "generate", "generation_prompt": "a map"},
        {"visual_type": "generated_visual"},  # must be generated
        {"description": "see https://example.org/cat.png"},  # the model never writes URLs
        {"search_query": "www.example.org fans"},
        {"aspect_ratio": "wide"},
        {"visual_id": "has spaces"},
    ]
    for over in bad:
        with pytest.raises(ValidationError):
            requirement(**over)
    with pytest.raises(ValidationError, match="unique"):
        VisualPlanProposal(requirements=[requirement(), requirement()])


def test_visual_result_must_account_for_every_planned_visual() -> None:
    plan = VisualPlan(plan_id="vp1", lesson_title="L", requirements=[requirement()])
    plan_art = Artifact(artifact_id="a0", task_id="t", type=ArtifactType.VISUAL_PLAN, name="visual_plan",
                        uri="file:///x", media_type="application/json", content_hash="h", size_bytes=1, version=1,
                        provider="p")
    failure = {"visual_id": "v1", "lesson_section_id": "s1", "required": True, "reason": "x"}
    VisualResult(status="failed", plan=plan, plan_artifact_id="a0", artifacts=[plan_art], failures=[failure])
    with pytest.raises(ValidationError, match="status"):
        VisualResult(status="partial", plan=plan, plan_artifact_id="a0", artifacts=[plan_art], failures=[failure])
    with pytest.raises(ValidationError, match="every planned visual"):
        VisualResult(status="complete", plan=plan, plan_artifact_id="a0", artifacts=[plan_art])
    with pytest.raises(ValidationError, match="artifacts"):
        VisualResult(status="failed", plan=plan, plan_artifact_id="a0", artifacts=[], failures=[failure])


# --- image bytes ------------------------------------------------------------------------------


def test_png_encoder_is_deterministic_and_probe_reads_real_dimensions() -> None:
    png = encode_png(320, 180, [(1, 2, 3), (4, 5, 6)])
    assert png == encode_png(320, 180, [(1, 2, 3), (4, 5, 6)])
    assert probe_image(png).__dict__ == {"format": "png", "media_type": "image/png", "width": 320, "height": 180}
    gif = b"GIF89a" + struct.pack("<HH", 64, 32) + b"\x00" * 8
    assert (probe_image(gif).format, probe_image(gif).width, probe_image(gif).height) == ("gif", 64, 32)
    jpeg = b"\xff\xd8" + b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 90, 160) + b"\x00" * 12
    assert (probe_image(jpeg).width, probe_image(jpeg).height) == (160, 90)
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"></svg>'
    assert (probe_image(svg).media_type, probe_image(svg).width) == ("image/svg+xml", 400)
    with pytest.raises(ImageProbeError):
        probe_image(b"not an image")


# --- providers ------------------------------------------------------------------------------


async def test_mock_image_search_provider_is_deterministic_and_offline() -> None:
    provider = MockImageSearchProvider(CATALOG)
    request = ProviderImageSearchRequest(query="football fans opinions", max_results=3)
    first = await provider.search(request)
    assert first == await provider.search(request) and len(first.hits) == 3
    assert first.usage.requests == 1 and first.usage.results == 3 and first.usage.cost_usd == 0.0
    hit = first.hits[0]
    image = await provider.download(hit.provider_image_id, hit.url)
    assert image == await provider.download(hit.provider_image_id, hit.url)
    # This catalogue entry advertises 1600x900 but serves 1200x900, like a provider with wrong metadata.
    assert hit.provider_image_id == "commons-fans-opinions" and hit.metadata["mock_actual_size"] == [1200, 900]
    assert (probe_image(image.content).width, probe_image(image.content).height) == (1200, 900)
    with pytest.raises(ImageSearchProviderError):
        await provider.download("missing", "https://images.example.org/none.png")


async def test_mock_image_generation_provider_is_deterministic() -> None:
    provider = MockImageGenerationProvider()
    request = ProviderImageRequest(prompt="A diagram", width=640, height=360, seed=7)
    a, b = await provider.generate(request), await provider.generate(request)
    assert a == b and a.model == "mock-image-1" and a.seed == 7
    assert (probe_image(a.content).width, probe_image(a.content).height) == (640, 360)
    assert a.usage == ImageUsage(requests=1, images=1, megapixels=0.2304, cost_usd=0.0)
    other = await provider.generate(request.model_copy(update={"seed": 8}))
    assert other.content != a.content


# --- tools ----------------------------------------------------------------------------------


async def test_image_search_tool_normalises_results_through_the_tool_manager(env) -> None:
    no_network = ToolCaller(caller_id="x", allowed_tools=frozenset({"image.search"}))
    with pytest.raises(ToolPermissionError):
        await env.tools.call(no_network, "image.search", {"query": "fans"}, env.scope)

    out = await env.call("image.search", {"query": "football fans opinions", "visual_type": "photo",
                                          "aspect_ratio": "16:9"})
    assert out.provider == "mock" and [r.rank for r in out.results] == list(range(1, len(out.results) + 1))
    top = out.results[0]
    assert top.image_id == (await env.call("image.search", {"query": "football fans opinions"})).results[0].image_id
    assert top.visual_type == VisualType.PHOTO and top.license.name == "CC BY 4.0"
    assert (top.width, top.height, top.format) == (1600, 900, "image/png")
    assert top.retrieved_at == NOW and top.source_url.startswith("https://commons.example.org/")
    # Provider-specific fields stay in metadata; the agent never sees provider schemas.
    assert top.metadata["mock_actual_size"] == [1200, 900] and top.metadata["provider_kind"] == "photo"
    # Nothing is invented: the stock image has no creator or credit line, and stays that way.
    scoreboard = next(r for r in (await env.call("image.search", {"query": "scoreboard"})).results
                      if r.provider_image_id == "stock-scoreboard")
    assert scoreboard.creator is None and scoreboard.attribution_text is None
    line = env.scope.usage.summary.by_service["image_search:mock"]
    assert line.calls == 3 and line.cost_usd == 0.0 and line.units == {"requests": 3.0}


async def test_image_search_tool_maps_provider_failures(tmp_path) -> None:
    calls = []

    class Down(MockImageSearchProvider):
        def __init__(self, transient):
            super().__init__(CATALOG)
            self.transient = transient

        async def search(self, request):
            calls.append(request)
            raise ImageSearchProviderError("rate limited", transient=self.transient)

    for transient, error, attempts in ((True, ToolTransientError, 3), (False, ToolError, 1)):
        calls.clear()
        registry = ToolRegistry()
        registry.register(ImageSearchTool(Down(transient)))
        caller = ToolCaller(caller_id="t", allowed_tools=frozenset({"image.search"}), permissions=ALL)
        with pytest.raises(error, match="rate limited"):
            await ToolManager(registry).call(caller, "image.search", {"query": "fans"}, scope()[0])
        assert len(calls) == attempts


async def test_image_generation_tool_records_provenance_storage_and_usage(env) -> None:
    out = await env.call("image.generate", {"prompt": "Labelled diagram of the preterite", "aspect_ratio": "16:9",
                                            "width": 1280, "height": 720, "style": "flat", "seed": 3,
                                            "negative_prompt": "watermark"})
    meta = out.metadata
    assert (out.provider, out.model, out.format) == ("mock", "mock-image-1", "image/png")
    assert meta.prompt_hash == hashlib.sha256(b"Labelled diagram of the preterite").hexdigest() == prompt_hash(meta.prompt)
    assert meta.generated_at == NOW and meta.seed == 3 and meta.style == "flat" and meta.ignored_parameters == []
    data = env.artifacts.read_object(out.storage.uri)
    assert hashlib.sha256(data).hexdigest() == out.storage.checksum and out.asset_id == "img_" + out.storage.checksum[:16]
    line = env.scope.usage.summary.by_service["image_generation:mock"]
    assert line.calls == 1 and line.results == 1 and line.cost_usd == 0.0
    assert line.units == {"images": 1.0, "megapixels": 0.9216}
    with pytest.raises(ToolError, match="aspect ratio"):  # size and ratio must agree
        await env.call("image.generate", {"prompt": "x", "aspect_ratio": "16:9", "width": 1280, "height": 960})


async def test_image_generation_records_unsupported_parameters_as_ignored(tmp_path) -> None:
    class Plain(ImageGenerationProvider):
        name = "plain"

        async def generate(self, request: ProviderImageRequest) -> GeneratedImage:
            assert request.seed is None and request.negative_prompt is None and request.style is None
            return GeneratedImage(content=encode_png(request.width, request.height, [(9, 9, 9)]),
                                  media_type="image/png", width=request.width, height=request.height,
                                  model="plain-1", usage=ImageUsage(images=1))

    e = Env(tmp_path, generation=Plain())
    out = await e.call("image.generate", {"prompt": "p", "seed": 1, "negative_prompt": "n", "style": "s"})
    assert out.metadata.ignored_parameters == ["negative_prompt", "seed", "style"] and out.metadata.seed is None
    assert e.scope.usage.summary.by_service["image_generation:plain"].cost_usd is None  # none reported, none invented
    dispose(e.sessions)


async def test_selection_is_deterministic_and_keeps_all_source_metadata(env) -> None:
    req = requirement(search_query="Describing a match in the past football")
    results = (await env.call("image.search", {"query": req.search_query})).results
    first = await env.call("image.select", {"requirement": req, "candidates": results})
    again = await env.call("image.select", {"requirement": req, "candidates": list(reversed(results))})
    assert isinstance(first, ImageSelectionResult) and first == again
    assert [r.rank for r in first.ranked] == list(range(1, len(results) + 1))
    eligible = first.eligible()
    assert [r.score for r in eligible] == sorted((r.score for r in eligible), reverse=True)
    assert all(r.eligible for r in first.ranked[:len(eligible)])  # eligible candidates come first
    # The unlicensed upload is relevant but excluded because this visual requires attribution.
    upload = next(r for r in first.ranked if r.candidate.provider_image_id == "upload-old-match")
    assert not upload.eligible and "licence" in upload.reasons[0] and upload.signals.license == 0
    assert all(r.candidate in results for r in first.ranked)  # untouched candidates, all metadata kept
    relaxed = await env.call("image.select", {"requirement": requirement(search_query=req.search_query,
                                                                         attribution_required=False),
                                              "candidates": results})
    assert next(r for r in relaxed.ranked if r.candidate.provider_image_id == "upload-old-match").eligible
    # Signals: an exact visual-type and aspect-ratio match score 1; a square icon scores lower on aspect.
    icon = next(r for r in (await env.call("image.select", {
        "requirement": requirement(search_query="football icon"),
        "candidates": (await env.call("image.search", {"query": "football icon"})).results})).ranked
        if r.candidate.provider_image_id == "icons-football")
    assert icon.signals.aspect < 0.5 and icon.signals.visual_type == 0.0


def _search_attribution(**over) -> dict:
    return {"kind": "search", "provider": "mock", "image_id": "i1", "provider_image_id": "p1",
            "image_url": "https://images.example.org/p1.png", "source_url": "https://example.org/p1",
            "title": "T", "creator": "C", "license": {"name": "CC BY 4.0"}, "retrieved_at": NOW.isoformat(), **over}


def _validation(obj, **over) -> dict:
    return {"object": obj.model_dump(), "declared_width": 1600, "declared_height": 900, "declared_format": "image/png",
            "expected_aspect_ratio": "16:9", "attribution": _search_attribution(), "attribution_required": True,
            "origin": "search", **over}


async def test_validation_checks_dimensions_format_and_aspect_ratio(env) -> None:
    obj = env.artifacts.put_object(encode_png(1600, 900, [(1, 1, 1)]), "image/png")
    ok = await env.call("image.validate", _validation(obj))
    assert isinstance(ok, ImageValidationReport) and ok.valid and ok.errors == []
    assert {"content", "checksum", "format", "dimensions", "aspect_ratio", "provenance", "attribution"} <= set(ok.checks)
    assert (ok.measured.width, ok.measured.height, ok.measured.checksum) == (1600, 900, obj.checksum)

    def codes(report) -> set[str]:
        return {e.code for e in report.errors}

    lying = await env.call("image.validate", _validation(obj, declared_width=1920, declared_height=1080))
    assert codes(lying) == {"dimension_mismatch"} and lying.errors[0].expected == "1920x1080"
    assert codes(await env.call("image.validate", _validation(obj, declared_format="image/jpeg"))) == {"format_mismatch"}
    square = env.artifacts.put_object(encode_png(900, 900, [(1, 1, 1)]), "image/png")
    report = await env.call("image.validate", _validation(square, declared_width=900))
    assert codes(report) == {"aspect_ratio_mismatch"} and not report.valid
    assert (await env.call("image.validate", _validation(square, declared_width=900,
                                                         expected_aspect_ratio=None))).valid
    tiny = env.artifacts.put_object(encode_png(32, 18, [(1, 1, 1)]), "image/png")
    assert "too_small" in codes(await env.call("image.validate", _validation(tiny, declared_width=32,
                                                                             declared_height=18)))
    junk = env.artifacts.put_object(b"definitely not an image", "image/png")
    assert codes(await env.call("image.validate", _validation(junk))) == {"unreadable_image"}
    tampered = obj.model_copy(update={"checksum": "0" * 64})
    assert "checksum_mismatch" in codes(await env.call("image.validate", _validation(tampered)))


async def test_validation_checks_attribution_and_generation_metadata(env) -> None:
    obj = env.artifacts.put_object(encode_png(1600, 900, [(2, 2, 2)]), "image/png")

    async def codes(**over) -> set[str]:
        return {e.code for e in (await env.call("image.validate", _validation(obj, **over))).errors}

    assert await codes(attribution=_search_attribution(license=None)) == {"missing_license"}
    assert await codes(attribution=_search_attribution(creator=None)) == {"missing_attribution"}
    assert await codes(attribution=_search_attribution(license=None, creator=None), attribution_required=False) == set()
    assert await codes(attribution=_search_attribution(source_url="")) == {"missing_source"}

    generated = GenerationAttribution(provider="mock", model="m", generated_at=NOW, prompt="draw", width=1600,
                                      height=900, prompt_hash=prompt_hash("draw"))
    assert await codes(attribution=generated.model_dump(mode="json"), origin="generated") == set()
    assert await codes(attribution=generated.model_dump(mode="json")) == {"origin_mismatch"}
    wrong_hash = generated.model_copy(update={"prompt_hash": prompt_hash("something else")})
    assert await codes(attribution=wrong_hash.model_dump(mode="json"), origin="generated") == \
        {"invalid_generation_metadata"}
    wrong_size = generated.model_copy(update={"width": 1280, "height": 720})
    assert await codes(attribution=wrong_size.model_dump(mode="json"), origin="generated") == \
        {"invalid_generation_metadata"}


async def test_checksum_addressing_deduplicates_stored_objects(env) -> None:
    content = encode_png(640, 360, [(5, 6, 7)])
    first = env.artifacts.put_object(content, "image/png")
    second = env.artifacts.put_object(content, "image/png")
    assert first.checksum == hashlib.sha256(content).hexdigest() and first.size_bytes == len(content)
    assert (first.reused, second.reused) == (False, True) and first.uri == second.uri
    assert len(env.object_files()) == 1 and env.object_files()[0].name == f"{first.checksum}.png"
    # Identical generations land on the same object; a different seed is a different object.
    payload = {"prompt": "Diagram", "width": 640, "height": 360, "seed": 1}
    a, b = await env.call("image.generate", payload), await env.call("image.generate", payload)
    assert a.storage.uri == b.storage.uri and b.storage.reused and a.asset_id == b.asset_id
    c = await env.call("image.generate", {**payload, "seed": 2})
    assert c.storage.uri != a.storage.uri
    assert len(env.object_files()) == 3


def _asset(fetched_obj, result: ImageSearchResult, report: ImageValidationReport, visual_id="v1") -> ImageAsset:
    return ImageAsset(
        asset_id="img_" + fetched_obj.checksum[:16], visual_id=visual_id, plan_id="vp1", lesson_section_id="s1",
        visual_type="photo", purpose="Context", description="Fans", origin="search", object=fetched_obj,
        width=report.measured.width, height=report.measured.height, format=report.measured.media_type,
        attribution=result.attribution(), validation=report,
        selection=SelectionRecord(selector="deterministic-v1", rank=1, score=0.9, candidates=1,
                                  signals={"relevance": 1, "aspect": 1, "visual_type": 1, "license": 1,
                                           "source_quality": 1, "resolution": 1}),
    )


async def test_image_asset_artifact_points_at_the_object_and_keeps_metadata_out_of_binary(env) -> None:
    result = next(r for r in (await env.call("image.search", {"query": "supporters scarves"})).results
                  if r.provider_image_id == "commons-supporters-scarves")
    fetched = await env.call("image.fetch", {"result": result})
    report = await env.call("image.validate", _validation(fetched.object, attribution=result.attribution().model_dump(mode="json")))
    assert report.valid
    asset = _asset(fetched.object, result, report)
    art = await env.call("image.create_asset", {"name": "image_v1", "asset": asset, "attribution_required": True,
                                                "expected_aspect_ratio": "16:9"})
    assert isinstance(art, Artifact) and art.type == ArtifactType.IMAGE_ASSET
    assert (art.uri, art.content_hash, art.media_type) == (fetched.object.uri, fetched.object.checksum, "image/png")
    stored = ImageAsset.model_validate(art.metadata)
    assert stored.attribution.creator == "Luis Gómez" and stored.attribution.license.name == "CC BY-SA 4.0"
    assert stored.attribution.source_url == result.source_url and stored.attribution.attribution_text
    assert env.artifacts.read(art.artifact_id) == env.artifacts.read_object(fetched.object.uri)
    assert any(e.type == "artifact.created" and e.data["artifact_type"] == "IMAGE_ASSET" for e in env.events)

    # A second use of the same image is another artifact on the same stored object.
    other = await env.call("image.create_asset", {"name": "image_v2", "asset": _asset(fetched.object, result, report,
                                                                                       "v2"),
                                                  "attribution_required": True})
    assert other.artifact_id != art.artifact_id and other.uri == art.uri and len(env.object_files()) == 1

    # SQLite holds metadata only: no image bytes in any artifact row.
    rows = sqlite3.connect(env.db).execute("select body from artifacts").fetchall()
    assert len(rows) == 2 and all(b"\x89PNG" not in body.encode() and len(body) < 20_000 for (body,) in rows)


async def test_invalid_images_never_become_assets(env) -> None:
    result = next(r for r in (await env.call("image.search", {"query": "football fans opinions"})).results
                  if r.provider_image_id == "commons-fans-opinions")
    fetched = await env.call("image.fetch", {"result": result})  # serves 1200x900 for an advertised 1600x900
    report = await env.call("image.validate", _validation(fetched.object,
                                                          attribution=result.attribution().model_dump(mode="json")))
    assert {e.code for e in report.errors} == {"dimension_mismatch", "aspect_ratio_mismatch"}
    # Even if a caller claims a valid report and forges the size, the asset tool re-validates the bytes.
    honest = await env.call("image.validate", _validation(fetched.object, declared_width=1200, expected_aspect_ratio=None,
                                                          attribution=result.attribution().model_dump(mode="json")))
    forged = _asset(fetched.object, result, honest).model_copy(update={"width": 1600})
    with pytest.raises(ImageAssetRejected) as caught:
        await env.call("image.create_asset", {"name": "image_v1", "asset": forged, "attribution_required": True,
                                              "expected_aspect_ratio": "16:9"})
    assert {e.code for e in caught.value.report.errors} == {"dimension_mismatch", "aspect_ratio_mismatch"}
    assert env.artifacts.list_for_task("task1") == []
    with pytest.raises(ValidationError, match="pass validation"):
        _asset(fetched.object, result, report)


async def test_assets_need_a_task(env) -> None:
    obj = env.artifacts.put_object(encode_png(1600, 900, [(3, 3, 3)]), "image/png")
    generated = GenerationAttribution(provider="mock", model="m", generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                                      prompt="d", prompt_hash=prompt_hash("d"), width=1600, height=900)
    report = await env.call("image.validate", _validation(obj, attribution=generated.model_dump(mode="json"),
                                                          origin="generated"))
    asset = ImageAsset(asset_id="img_x", visual_id="v", plan_id="p", lesson_section_id="s", visual_type="diagram",
                       purpose="p", description="d", origin="generated", object=obj, width=1600, height=900,
                       format="image/png", attribution=generated, validation=report)
    with pytest.raises(ToolError, match="inside a task"):
        await env.tools.call(env.caller, "image.create_asset", {"name": "x", "asset": asset,
                                                                "attribution_required": False}, scope(None)[0])


def test_generation_request_schema() -> None:
    req = ImageGenerationRequest(prompt="p", aspect_ratio="4:3", width=1280, height=960)
    assert req.seed is None and req.negative_prompt is None
    with pytest.raises(ValidationError):
        ImageGenerationRequest(prompt="", width=10, height=10)
