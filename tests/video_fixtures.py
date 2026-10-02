"""A small deterministic video scenario: three slides, one IMAGE_ASSET, three narration clips (two on the first
slide with a pause between them, none on the second), stored as real objects, with a consistent Presentation and
PresentationTimeline. Small enough (320x180 at 10 fps, 7 seconds) for real FFmpeg tests to stay fast."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from app.artifacts.service import ArtifactService
from app.schemas.audio import PresentationTimeline, SegmentTiming, SlideTiming
from app.schemas.presentation import (
    BulletsElement,
    ImageElement,
    Presentation,
    PresentationConfig,
    PresentationSlide,
    SlideLayout,
    SlideType,
    TextElement,
    TitleElement,
)
from app.schemas.video import AudioAssetInput, ImageAssetInput, VideoConfig, VideoPlanningRequest
from app.utils.audio import encode_wav
from app.utils.images import encode_png

RATE = 16000
SMALL = VideoConfig(width=320, height=180, fps=10)

# (segment id, slide id, start, end, frequency, text)
CLIPS = [
    ("s1_a1", "s1", 0.3, 1.3, 440.0, "Hola. Bienvenidos a la clase de fútbol."),
    ("s1_a2", "s1", 1.8, 2.6, 660.0, "Hoy hablamos de partidos."),
    ("s3_a1", "s3", 5.2, 6.7, 550.0, "Me encanta el fútbol porque es emocionante y muy divertido para todos."),
]
SLIDES = [("s1", 0.0, 3.0), ("s2", 3.0, 5.0), ("s3", 5.0, 7.0)]


def tone(seconds: float, freq: float, rate: int = RATE) -> bytes:
    n = round(seconds * rate)
    pcm = b"".join(struct.pack("<h", round(12000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n))
    return encode_wav(pcm, sample_rate=rate)


@dataclass
class Scenario:
    request: VideoPlanningRequest
    image: ImageAssetInput
    audio: list[AudioAssetInput]


def build_scenario(artifacts: ArtifactService, config: VideoConfig = SMALL) -> Scenario:
    png = artifacts.put_object(encode_png(160, 90, [(200, 40, 40), (40, 160, 60), (30, 60, 200)]), "image/png")
    image = ImageAssetInput(artifact_id="art_img1", asset_id="img_1", uri=png.uri, checksum=png.checksum,
                            media_type="image/png", width=160, height=90, alt_text="three coloured bands")
    audio = []
    for seg_id, slide_id, start, end, freq, text in CLIPS:
        obj = artifacts.put_object(tone(end - start, freq), "audio/wav")
        audio.append(AudioAssetInput(artifact_id=f"art_{seg_id}", segment_id=seg_id, slide_id=slide_id,
                                     uri=obj.uri, checksum=obj.checksum, media_type="audio/wav",
                                     duration=round(end - start, 3), text=text, language="es"))
    presentation = Presentation(
        presentation_id="pres_1", deck_id="deck_1", title="Fútbol (A2)", language="es",
        config=PresentationConfig(), builder="test",
        slides=[
            PresentationSlide(slide_id="s1", order=1, slide_type=SlideType.EXPLANATION, layout=SlideLayout.IMAGE_TEXT,
                              image_artifact_ids=["art_img1"], elements=[
                                  TitleElement(element_id="e1", region="title", text="El partido"),
                                  BulletsElement(element_id="e2", region="left", items=["Me gusta", "Me encanta"]),
                                  ImageElement(element_id="e3", region="image", artifact_id="art_img1",
                                               asset_id="img_1", object_uri=png.uri, checksum=png.checksum,
                                               media_type="image/png", width=160, height=90,
                                               alt_text="three coloured bands")]),
            PresentationSlide(slide_id="s2", order=2, slide_type=SlideType.EXERCISE, layout=SlideLayout.TITLE_CONTENT,
                              elements=[TitleElement(element_id="e4", region="title", text="Pregunta"),
                                        TextElement(element_id="e5", region="body", text="¿Quién ganó?")]),
            PresentationSlide(slide_id="s3", order=3, slide_type=SlideType.SUMMARY, layout=SlideLayout.SUMMARY,
                              elements=[TitleElement(element_id="e6", region="title", text="Resumen"),
                                        BulletsElement(element_id="e7", region="body", items=["Gustar", "Encantar"])]),
        ])
    timeline = PresentationTimeline(
        timeline_id="tl_test", presentation_artifact_id="art_pres", deck_id="deck_1", audio_plan_id="ap_test",
        audio_plan_artifact_id="art_ap", language="es", voice="mock-es-ES-1", duration=7.0,
        slides=[SlideTiming(slide_id=sid, order=i, start_time=a, end_time=b, duration=round(b - a, 3),
                            audio_segment_refs=[c[0] for c in CLIPS if c[1] == sid])
                for i, (sid, a, b) in enumerate(SLIDES, start=1)],
        segments=[SegmentTiming(segment_id=c[0], slide_id=c[1], order=i, audio_artifact_id=f"art_{c[0]}",
                                start_time=c[2], end_time=c[3], duration=round(c[3] - c[2], 3))
                  for i, c in enumerate(CLIPS, start=1)],
        resolver="test")
    request = VideoPlanningRequest(
        task_id="task_v", presentation=presentation, presentation_artifact_id="art_pres", timeline=timeline,
        timeline_artifact_id="art_tl", timeline_checksum="0" * 64, image_assets=[image], audio_assets=audio,
        config=config)
    return Scenario(request=request, image=image, audio=audio)
