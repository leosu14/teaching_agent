"""Image generation tool: provider-independent generation with recorded provenance and usage."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.providers.image.base import ImageGenerationProvider, ImageGenerationProviderError, ProviderImageRequest
from app.schemas.common import RetryPolicy, utcnow
from app.schemas.visual import GenerationAttribution, ImageGenerationRequest, ImageGenerationResult
from app.tools.base import Tool, ToolError, ToolTransientError


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class ImageGenerationTool(Tool[ImageGenerationRequest, ImageGenerationResult]):
    name = "image.generate"
    description = "Generate an image with the configured provider and store it content-addressed; the result " \
                  "records provider, model, timestamp, prompt hash, parameters and usage."
    input_model = ImageGenerationRequest
    output_model = ImageGenerationResult
    permissions = frozenset({"media:generate", "artifact:write"})
    timeout_seconds = 120.0
    retry = RetryPolicy(max_attempts=2, backoff_seconds=0.1)

    def __init__(self, provider: ImageGenerationProvider, artifacts: ArtifactService,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self._provider = provider
        self._artifacts = artifacts
        self._clock = clock

    async def run(self, data: ImageGenerationRequest, scope: ExecutionScope) -> ImageGenerationResult:
        p = self._provider
        ignored = [name for name, given, supported in (
            ("negative_prompt", data.negative_prompt, p.supports_negative_prompt),
            ("seed", data.seed, p.supports_seed),
            ("style", data.style, p.supports_style),
        ) if given is not None and not supported]
        request = ProviderImageRequest(
            prompt=data.prompt, width=data.width, height=data.height,
            negative_prompt=None if "negative_prompt" in ignored else data.negative_prompt,
            style=None if "style" in ignored else data.style, seed=None if "seed" in ignored else data.seed,
        )
        try:
            image = await p.generate(request)
        except (ImageGenerationProviderError, ConnectionError, OSError) as exc:
            transient = getattr(exc, "transient", True)
            raise (ToolTransientError if transient else ToolError)(
                f"image generation provider '{p.name}' failed: {exc}") from exc
        if not image.content:
            raise ToolError(f"image generation provider '{p.name}' returned no content")
        obj = self._artifacts.put_object(image.content, image.media_type)
        metadata = GenerationAttribution(
            provider=p.name, model=image.model, generated_at=self._clock(), prompt=data.prompt,
            prompt_hash=prompt_hash(data.prompt), negative_prompt=request.negative_prompt, style=request.style,
            seed=image.seed if p.supports_seed else None, width=image.width, height=image.height,
            ignored_parameters=ignored, provider_metadata=image.metadata,
        )
        scope.usage.record_service(service=f"image_generation:{p.name}", results=image.usage.images,
                                   cost_usd=image.usage.cost_usd,
                                   units={"images": image.usage.images, "megapixels": image.usage.megapixels})
        return ImageGenerationResult(
            asset_id="img_" + obj.checksum[:16], provider=p.name, model=image.model, width=image.width,
            height=image.height, format=image.media_type, metadata=metadata, storage=obj, usage=image.usage,
        )
