"""Deterministic slide frames for the video compositor (Pillow).

A frame is the slide card (neutral background, deck title, slide title, a few content lines), the slide's
IMAGE_ASSET when it has one, and optionally a burned-in subtitle. Geometry is expressed as fractions of the frame
size, so 1920x1080, 1280x720 or any other resolution lays out the same. The same inputs give the same pixels: a
bundled font (or a configured font file), fixed resampling, no timestamps.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.providers.video.base import VideoCompositionError
from app.schemas.video import SlideCard, SubtitleStyle, VideoConfig

FontType = ImageFont.FreeTypeFont | ImageFont.ImageFont
# Fonts with full Latin coverage (accents, bullets), tried in order when no font file is configured. Pillow's
# bundled font is the last resort: it is always there but lacks many accented letters.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def resolve_font(configured: Path | None) -> str | None:
    """The configured font file, else the first available candidate, else None (Pillow's bundled font)."""
    if configured is not None:
        if not configured.is_file():
            raise VideoCompositionError(f"font file {configured} does not exist")
        return str(configured)
    return next((c for c in FONT_CANDIDATES if Path(c).is_file()), None)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


@lru_cache(maxsize=64)
def _font(path: str | None, size: int) -> FontType:
    size = max(8, size)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError as exc:
            raise VideoCompositionError(f"font file {path} cannot be loaded: {exc}") from exc
    return ImageFont.load_default(size=size)


def wrap(text: str, font: FontType, max_width: float) -> list[str]:
    """Greedy word wrap by rendered width; a single word wider than the line is kept whole."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def ellipsize(text: str, font: FontType, max_width: float) -> str:
    if font.getlength(text) <= max_width:
        return text
    while text and font.getlength(text + "\u2026") > max_width:
        text = text[:-1]
    return text.rstrip() + "\u2026"


class FrameRenderer:
    name = "pillow-frames/1"

    @property
    def font_name(self) -> str:
        return Path(self.font_path).name if self.font_path else "pillow-default"

    def __init__(self, config: VideoConfig, font_path: Path | None = None) -> None:
        self.config = config
        self.w, self.h = config.width, config.height
        self.font_path = resolve_font(font_path)
        self.background = _rgb(config.background)
        self.text = _rgb(config.text_color)
        self.accent = _rgb(config.accent_color)
        self.muted = _blend(self.text, self.background, 0.45)

    def font(self, fraction: float) -> FontType:
        return _font(self.font_path, round(self.h * fraction))

    def media_box(self, card: SlideCard) -> tuple[int, int, int, int]:
        """Where a card places its picture (an IMAGE_ASSET, or an inset generated clip): beside the text when the
        card has lines, else across the content area."""
        w, h = self.w, self.h
        margin = round(w * 0.06)
        title_lines = len(wrap(card.title, self.font(0.062), w - 2 * margin)[:2])
        top = max(round(h * 0.095) + title_lines * round(h * 0.075) + round(h * 0.03), round(h * 0.25))
        bottom = round(h * 0.78)
        if card.lines:
            return round(w * 0.48), top, w - margin, bottom
        return margin, top, w - margin, bottom

    def background_frame(self) -> Image.Image:
        """A plain frame in the background colour (behind a full-frame clip)."""
        return Image.new("RGB", (self.w, self.h), self.background)

    def subtitle_layer(self, text: str, style: SubtitleStyle) -> Image.Image:
        """The burned-in subtitle alone, on a transparent RGBA frame, for drawing over moving pictures. Same box and
        text placement as `with_subtitle`."""
        layer = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        if not text or not style.enabled:
            return layer
        font = self.font(style.font_size)
        lines = text.split("\n")[: style.max_lines]
        line_h = round(self.h * style.font_size * 1.3)
        pad = round(self.h * style.font_size * 0.4)
        widths = [font.getlength(line) for line in lines]
        box_w = min(self.w - 2 * pad, round(max(widths) + 2 * pad))
        box_h = line_h * len(lines) + 2 * pad - round(self.h * style.font_size * 0.3)
        x0 = (self.w - box_w) // 2
        y1 = self.h - round(self.h * style.bottom_margin)
        y0 = y1 - box_h
        draw = ImageDraw.Draw(layer)
        draw.rectangle((x0, y0, x0 + box_w, y1), fill=(*_rgb(style.box_color), round(255 * style.box_opacity)))
        text_layer = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        tdraw = ImageDraw.Draw(text_layer)
        y = y0 + pad - round(self.h * style.font_size * 0.15)
        for line, lw in zip(lines, widths):
            tdraw.text(((self.w - lw) / 2, y), line, font=font, fill=(*_rgb(style.text_color), 255))
            y += line_h
        return Image.alpha_composite(layer, text_layer)

    def card(self, card: SlideCard, image: bytes | None, position: tuple[int, int], *,
             reserve_media_box: bool = False) -> Image.Image:
        """The slide without subtitles. `reserve_media_box` lays the card out as if it had a picture and leaves the
        picture's box empty (an inset clip plays there)."""
        w, h = self.w, self.h
        frame = Image.new("RGB", (w, h), self.background)
        draw = ImageDraw.Draw(frame)
        margin = round(w * 0.06)
        draw.rectangle((0, 0, w, max(2, round(h * 0.012))), fill=self.accent)
        draw.text((margin, round(h * 0.04)), card.deck_title, font=self.font(0.028), fill=self.muted)

        title_font = self.font(0.062)
        y = round(h * 0.095)
        for line in wrap(card.title, title_font, w - 2 * margin)[:2]:
            draw.text((margin, y), line, font=title_font, fill=self.text)
            y += round(h * 0.075)
        top = max(y + round(h * 0.03), round(h * 0.25))
        bottom = round(h * 0.78)  # the band below is kept for subtitles

        lines = card.lines
        if image is not None:
            pic = self._decode(image)
            if lines:
                text_box = (margin, top, round(w * 0.44), bottom)
                image_box = (round(w * 0.48), top, w - margin, bottom)
            else:
                text_box, image_box = None, (margin, top, w - margin, bottom)
            self._place(frame, pic, image_box)
            if text_box:
                self._lines(draw, lines, text_box)
        elif reserve_media_box:
            if lines:
                self._lines(draw, lines, (margin, top, round(w * 0.44), bottom))
        elif lines:
            self._lines(draw, lines, (margin, top, w - margin, bottom))

        footer = f"{position[0]} / {position[1]}"
        font = self.font(0.022)
        draw.text((w - margin - font.getlength(footer), round(h * 0.95)), footer, font=font, fill=self.muted)
        if card.footer:
            room = w - 3 * margin - font.getlength(footer)
            draw.text((margin, round(h * 0.95)), ellipsize(card.footer, font, room), font=font, fill=self.muted)
        return frame

    def with_subtitle(self, base: Image.Image, text: str | None, style: SubtitleStyle) -> Image.Image:
        if not text or not style.enabled:
            return base
        frame = base.copy()
        font = self.font(style.font_size)
        lines = text.split("\n")[: style.max_lines]
        line_h = round(self.h * style.font_size * 1.3)
        pad = round(self.h * style.font_size * 0.4)
        widths = [font.getlength(line) for line in lines]
        box_w = min(self.w - 2 * pad, round(max(widths) + 2 * pad))
        box_h = line_h * len(lines) + 2 * pad - round(self.h * style.font_size * 0.3)
        x0 = (self.w - box_w) // 2
        y1 = self.h - round(self.h * style.bottom_margin)
        y0 = y1 - box_h
        overlay = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rectangle((x0, y0, x0 + box_w, y1), fill=(*_rgb(style.box_color), round(255 * style.box_opacity)))
        frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(frame)
        y = y0 + pad - round(self.h * style.font_size * 0.15)
        for line, lw in zip(lines, widths):
            draw.text(((self.w - lw) / 2, y), line, font=font, fill=_rgb(style.text_color))
            y += line_h
        return frame

    def _lines(self, draw: ImageDraw.ImageDraw, lines: list[str], box: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = box
        font = self.font(0.038)
        step = round(self.h * 0.052)
        y = y0
        for line in lines:
            for i, part in enumerate(wrap(line, font, x1 - x0 - round(self.w * 0.02))):
                if y + step > y1:
                    return
                prefix = "• " if i == 0 else "   "
                draw.text((x0, y), prefix + part, font=font, fill=self.text)
                y += step

    @staticmethod
    def _decode(data: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                return img.convert("RGB")
        except Exception as exc:  # Pillow raises several types for undecodable images
            raise VideoCompositionError(f"image cannot be decoded: {exc}") from exc

    @staticmethod
    def _place(frame: Image.Image, pic: Image.Image, box: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = box
        scale = min((x1 - x0) / pic.width, (y1 - y0) / pic.height)
        size = (max(1, round(pic.width * scale)), max(1, round(pic.height * scale)))
        fitted = pic.resize(size, Image.Resampling.LANCZOS)
        frame.paste(fitted, (x0 + (x1 - x0 - size[0]) // 2, y0 + (y1 - y0 - size[1]) // 2))


def png_bytes(frame: Image.Image) -> bytes:
    buf = io.BytesIO()
    frame.save(buf, "PNG", compress_level=6)
    return buf.getvalue()
