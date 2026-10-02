"""Image generation adapter for the OpenAI Images API (POST /images/generations). Plain HTTP: no vendor SDK.

The API only generates a few fixed sizes, so the size with the closest aspect ratio is requested, then the image
is centre-cropped and resized to exactly the requested width and height; the native size and that post-processing
are recorded in the metadata. Every image is origin "generated". Nothing about the image is invented: metadata
holds only what the API returned plus what this adapter did.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
from typing import ClassVar

from PIL import Image

from app.providers.core.errors import ProviderResponseError
from app.providers.core.http import HttpClient
from app.providers.image.base import GeneratedImage, ImageGenerationProvider, ProviderImageRequest
from app.schemas.visual import ImageUsage

DEFAULT_MODEL = "gpt-image-1"
SIZES = {
    "gpt-image": ((1024, 1024), (1536, 1024), (1024, 1536)),
    "dall-e-3": ((1024, 1024), (1792, 1024), (1024, 1792)),
    "dall-e-2": ((256, 256), (512, 512), (1024, 1024)),
}


def sizes_for(model: str) -> tuple[tuple[int, int], ...]:
    return next((sizes for prefix, sizes in SIZES.items() if model.startswith(prefix)), SIZES["gpt-image"])


def closest_size(width: int, height: int, sizes: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    target = math.log(width / height)
    return min(sizes, key=lambda s: (abs(math.log(s[0] / s[1]) - target), -s[0] * s[1]))


def fit(content: bytes, width: int, height: int) -> tuple[bytes, tuple[int, int], bool]:
    """PNG of exactly width x height: centre-crop to the aspect ratio, then resize. Returns (png, native, changed)."""
    with Image.open(io.BytesIO(content)) as img:
        img.load()
        native = img.size
        if native == (width, height):
            return content, native, False
        target = width / height
        w, h = native
        if w / h > target:
            crop_w = round(h * target)
            box = ((w - crop_w) // 2, 0, (w - crop_w) // 2 + crop_w, h)
        else:
            crop_h = round(w / target)
            box = (0, (h - crop_h) // 2, w, (h - crop_h) // 2 + crop_h)
        out = img.convert("RGB").crop(box).resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        out.save(buffer, format="PNG", optimize=False)
        return buffer.getvalue(), native, True


class OpenAIImageGenerationProvider(ImageGenerationProvider):
    name = "openai"
    requires_network: ClassVar[bool] = True

    def __init__(self, http: HttpClient, *, model: str | None = None) -> None:
        self._http = http
        self.model = model or DEFAULT_MODEL

    def configuration(self) -> dict:
        return {"base_url": self._http.base_url, "model": self.model,
                "sizes": [f"{w}x{h}" for w, h in sizes_for(self.model)]}

    async def probe(self) -> str:
        await self._http.request("GET", f"/models/{self.model}")  # authenticated, nothing generated
        return "network"

    async def generate(self, request: ProviderImageRequest) -> GeneratedImage:
        native_w, native_h = closest_size(request.width, request.height, sizes_for(self.model))
        body = {"model": self.model, "prompt": request.prompt, "size": f"{native_w}x{native_h}", "n": 1}
        if self.model.startswith("dall-e"):
            body["response_format"] = "b64_json"  # GPT image models always return base64
        response = await self._http.request("POST", "/images/generations", json_body=body)
        data = response.json()
        try:
            item = data["data"][0]
            raw = base64.b64decode(item["b64_json"], validate=True)
        except (KeyError, IndexError, TypeError, binascii.Error) as exc:
            raise ProviderResponseError(f"{self.name}: response has no base64 image", provider=self.name) from exc
        try:
            content, native, resized = fit(raw, request.width, request.height)
        except OSError as exc:
            raise ProviderResponseError(f"{self.name}: returned bytes are not an image", provider=self.name) from exc
        metadata = {"vendor_request_id": response.vendor_request_id, "requested_size": body["size"],
                    "native_size": f"{native[0]}x{native[1]}",
                    "postprocess": "centre_crop_resize" if resized else None}
        if item.get("revised_prompt"):
            metadata["revised_prompt"] = item["revised_prompt"]
        if isinstance(data.get("usage"), dict):
            metadata["vendor_usage"] = data["usage"]
        return GeneratedImage(
            content=content, media_type="image/png", width=request.width, height=request.height, model=self.model,
            usage=ImageUsage(requests=1, images=1,
                             megapixels=round(request.width * request.height / 1_000_000, 4)),
            metadata=metadata,
        )
