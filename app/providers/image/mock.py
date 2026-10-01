"""Deterministic image generation: a PNG whose colour bands derive from the request. No network, no model."""

from __future__ import annotations

import hashlib
import json

from app.providers.image.base import GeneratedImage, ImageGenerationProvider, ProviderImageRequest
from app.schemas.visual import ImageUsage
from app.utils.images import colors_from, encode_png


class MockImageGenerationProvider(ImageGenerationProvider):
    """Same request, same bytes. Usage is one image with zero cost, since no paid API is called."""

    name = "mock"
    model = "mock-image-1"
    supports_negative_prompt = True
    supports_seed = True
    supports_style = True

    async def generate(self, request: ProviderImageRequest) -> GeneratedImage:
        seed = request.seed if request.seed is not None else 0
        key = json.dumps({**request.model_dump(), "seed": seed}, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(key).digest()
        content = encode_png(request.width, request.height, colors_from(digest, 4))
        megapixels = round(request.width * request.height / 1_000_000, 4)
        return GeneratedImage(
            content=content, media_type="image/png", width=request.width, height=request.height, model=self.model,
            seed=seed, usage=ImageUsage(requests=1, images=1, megapixels=megapixels, cost_usd=0.0),
            metadata={"steps": 20, "sampler": "mock-bands"},
        )
