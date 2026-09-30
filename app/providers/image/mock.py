"""Deterministic placeholder images rendered as SVG."""

from __future__ import annotations

import hashlib
from html import escape

from app.providers.image.base import GeneratedImage, ImageProvider


class MockImageProvider(ImageProvider):
    name = "mock"

    async def generate(self, prompt: str, width: int, height: int) -> GeneratedImage:
        hue = int(hashlib.sha256(prompt.encode()).hexdigest()[:2], 16) * 360 // 256
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="100%" height="100%" fill="hsl({hue},45%,85%)"/>'
            f'<text x="50%" y="50%" text-anchor="middle" font-family="sans-serif" font-size="20">'
            f"{escape(prompt[:60])}</text></svg>"
        )
        return GeneratedImage(content=svg.encode(), media_type="image/svg+xml", provider=self.name,
                              metadata={"prompt": prompt, "width": width, "height": height})
