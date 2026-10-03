"""VideoAgent: decides how an approved, narrated presentation becomes a video.

It turns the Presentation, its PresentationTimeline and the IMAGE_ASSET and AUDIO_ASSET references into a
VideoPlan with deterministic rules (see prompt.md); no model is called. It only makes semantic composition
decisions: which visual each slide shows, which narration plays when, the subtitle cues and the transitions. It
never runs FFmpeg, reads files, touches storage or calls a provider: the plan holds artifact references, and it is
checked through the ToolManager with the deterministic plan validator.
"""

from __future__ import annotations

import hashlib
import json

from app.agents.base import Agent, AgentContext, AgentSpec
from app.schemas.common import ModelTier
from app.schemas.events import EventType
from app.schemas.presentation import (
    BulletsElement,
    FooterElement,
    ImageElement,
    PresentationSlide,
    TableElement,
    TextElement,
    TitleElement,
)
from app.schemas.video import (
    IMAGE_MEDIA_TYPES,
    AudioTrack,
    GeneratedClip,
    GeneratedClipInput,
    ImageAssetInput,
    SlideCard,
    Subtitle,
    SubtitleTrack,
    Transition,
    TransitionPoint,
    TransitionType,
    VideoConfig,
    VideoPlan,
    VideoPlanningRequest,
    VideoPlanValidationReport,
    VideoSegment,
    VisualRef,
)

PLANNER = "video-planner/1"
MAX_CARD_LINES = 6


def video_plan_id_for(data: VideoPlanningRequest) -> str:
    """Same timeline, presentation, assets and configuration: same plan id (and the same video)."""
    body = json.dumps({
        "planner": PLANNER, "timeline": data.timeline_checksum, "timeline_ref": data.timeline_artifact_id,
        "presentation": data.presentation_artifact_id,
        "images": sorted((i.artifact_id, i.checksum) for i in data.image_assets),
        "audio": sorted((a.artifact_id, a.checksum) for a in data.audio_assets),
        "config": data.config.model_dump(mode="json"),
        # only plans with generated clips include them: every other plan id is unchanged
        **({"clips": sorted((c.slide_id, c.artifact_id, c.checksum, c.strategy.value, c.audio)
                            for c in data.generated_clips)} if data.generated_clips else {}),
    }, sort_keys=True)
    return "vp_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def subtitle_cues(track: AudioTrack, text: str, language: str, config: VideoConfig) -> list[Subtitle]:
    """The narration text as cues of at most max_lines x max_chars_per_line, timed across the track in proportion
    to their length. Times are whole milliseconds; the last cue ends with the narration."""
    style = config.subtitles
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= style.max_chars_per_line:
            lines[-1] += " " + word
        else:
            lines.append(word)
    cues = ["\n".join(lines[i:i + style.max_lines]) for i in range(0, len(lines), style.max_lines)]
    if not cues:
        return []
    weights = [len(c) for c in cues]
    start_ms, end_ms = round(track.start_time * 1000), round(track.end_time * 1000)
    span, total, done = end_ms - start_ms, sum(weights), 0
    out, cursor = [], start_ms
    for n, (cue, weight) in enumerate(zip(cues, weights), start=1):
        done += weight
        end = end_ms if n == len(cues) else start_ms + round(span * done / total)
        if end <= cursor:
            continue
        out.append(Subtitle(subtitle_id=f"{track.track_id}_c{n}", start_time=cursor / 1000, end_time=end / 1000,
                            text=cue, language=language, segment_ref=track.track_id))
        cursor = end
    return out


def card_for(slide: PresentationSlide | None, deck_title: str, slide_id: str) -> SlideCard:
    if slide is None:
        return SlideCard(deck_title=deck_title, title=slide_id)
    title, lines, footer = "", [], ""
    for el in slide.elements:
        match el:
            case TitleElement():
                title = title or el.text
            case TextElement() if el.role != "caption":
                lines.append(el.text)
            case BulletsElement():
                lines += el.items
            case TableElement():
                lines += [" | ".join(el.headers), *(" | ".join(r) for r in el.rows)]
            case FooterElement():
                footer = el.text
    return SlideCard(deck_title=deck_title, title=title or slide_id, lines=lines[:MAX_CARD_LINES], footer=footer)


class VideoAgent(Agent[VideoPlanningRequest, VideoPlan]):
    spec = AgentSpec(
        id="video",
        name="Video Planner",
        description="Plans the video of a narrated presentation: the visual, narration, subtitles and transition "
                    "of every slide, timed by the PresentationTimeline. Deterministic; no model call.",
        input_model=VideoPlanningRequest,
        output_model=VideoPlan,
        tier=ModelTier.CHEAP,
        tools=("video_plan.validate",),
    )

    async def run(self, data: VideoPlanningRequest, ctx: AgentContext) -> VideoPlan:
        scope = ctx.scope
        scope.emit(EventType.VIDEO_PLANNING_STARTED, presentation_artifact_id=data.presentation_artifact_id,
                   timeline_artifact_id=data.timeline_artifact_id, slides=len(data.timeline.slides),
                   images=len(data.image_assets), audio_assets=len(data.audio_assets),
                   width=data.config.width, height=data.config.height, fps=data.config.fps)
        try:
            plan = self.plan(data)
            report = await self.use_tool("video_plan.validate", data.validation_request(plan), ctx)
        except Exception as exc:
            scope.emit(EventType.VIDEO_FAILED, stage="planning", error=f"{type(exc).__name__}: {exc}"[:1000])
            raise
        assert isinstance(report, VideoPlanValidationReport)
        scope.emit(EventType.VIDEO_PLAN_CREATED, video_plan_id=plan.video_plan_id, segments=len(plan.slides),
                   audio_tracks=len(plan.audio_tracks), image_segments=len(plan.image_artifact_ids()),
                   subtitles=len(plan.subtitle_track.subtitles) if plan.subtitle_track else 0,
                   transition=data.config.transition.value, duration=plan.duration, valid=report.valid,
                   **({"generated_clips": len(plan.generated_clips())} if data.generated_clips else {}))
        # An invalid plan is returned as it is: the workflow's validation gate fails the task before composition.
        return plan

    def plan(self, data: VideoPlanningRequest) -> VideoPlan:
        p, timeline, config = data.presentation, data.timeline, data.config
        slides = {s.slide_id: s for s in p.slides}
        images = {i.artifact_id: i for i in data.image_assets}
        audio = {a.artifact_id: a for a in data.audio_assets}
        clips = {c.slide_id: c for c in data.generated_clips}
        segments: list[VideoSegment] = []
        tracks: list[AudioTrack] = []
        cues: list[Subtitle] = []
        skipped_images: list[str] = []
        for slide_t in timeline.slides:
            slide = slides.get(slide_t.slide_id)
            card = card_for(slide, p.title, slide_t.slide_id)
            image = self._image(slide, images)
            if image is not None and image.media_type not in IMAGE_MEDIA_TYPES:
                skipped_images.append(image.artifact_id)
                image = None
            seg_tracks = []
            for t in (x for x in timeline.segments if x.slide_id == slide_t.slide_id):
                asset = audio.get(t.audio_artifact_id)
                track = AudioTrack(
                    track_id=t.segment_id, slide_id=t.slide_id, artifact_id=t.audio_artifact_id,
                    uri=asset.uri if asset else "", checksum=asset.checksum if asset else "",
                    media_type=asset.media_type if asset else "", start_time=t.start_time, end_time=t.end_time,
                    duration=t.duration, asset_duration=asset.duration if asset else t.duration)
                seg_tracks.append(track)
                if asset is not None and config.subtitles.enabled:
                    cues += subtitle_cues(track, asset.text, asset.language or timeline.language, config)
            tracks += seg_tracks
            segments.append(VideoSegment(
                segment_id=f"v{slide_t.order:02d}_{slide_t.slide_id}"[:96], slide_id=slide_t.slide_id,
                order=slide_t.order, start_time=slide_t.start_time, end_time=slide_t.end_time,
                duration=slide_t.duration,
                visual_ref=VisualRef(kind="image_asset" if image else "slide_card", card=card, image=image),
                audio_refs=[t.track_id for t in seg_tracks],
                subtitle_refs=[c.subtitle_id for c in cues if c.segment_ref in {t.track_id for t in seg_tracks}],
                transition=Transition(),
                generated=self._clip(clips.get(slide_t.slide_id), slide_t.start_time, slide_t.end_time),
            ))
        segments = self._transitions(segments, config)
        transitions = [TransitionPoint(from_segment=a.segment_id, to_segment=b.segment_id, at=b.start_time,
                                       transition=b.transition) for a, b in zip(segments, segments[1:])]
        return VideoPlan(
            video_plan_id=video_plan_id_for(data), task_id=data.task_id,
            presentation_ref=data.presentation_artifact_id, timeline_ref=data.timeline_artifact_id,
            timeline_id=timeline.timeline_id, timeline_checksum=data.timeline_checksum, deck_id=timeline.deck_id,
            title=p.title, language=timeline.language, resolution=config.resolution, fps=config.fps,
            duration=timeline.duration, config=config, slides=segments, audio_tracks=tracks,
            subtitle_track=SubtitleTrack(track_id="subtitles", language=timeline.language,
                                         burned_in=config.subtitles.enabled, subtitles=cues)
            if config.subtitles.enabled else None,
            transitions=transitions, planner=PLANNER,
            metadata={"presentation_id": p.presentation_id, "silent_segments": [
                s.segment_id for s in segments if not s.audio_refs], "skipped_images": skipped_images,
                "missing_narration": timeline.missing_segments,
                **({"generated_clips": [c.segment_id for c in data.generated_clips],
                    "unplaced_clips": [c.segment_id for c in data.generated_clips
                                       if c.slide_id not in {t.slide_id for t in timeline.slides}]}
                   if data.generated_clips else {})},
        )

    @staticmethod
    def _image(slide: PresentationSlide | None, images: dict[str, ImageAssetInput]) -> ImageAssetInput | None:
        """The slide's first placed image, as the IMAGE_ASSET reference. An image the inputs do not know is still
        referenced (from the slide element) so that the plan validator rejects it rather than it vanishing."""
        if slide is None:
            return None
        element = next((e for e in slide.elements if isinstance(e, ImageElement)), None)
        if element is None:
            return None
        return images.get(element.artifact_id) or ImageAssetInput(
            artifact_id=element.artifact_id, asset_id=element.asset_id, uri=element.object_uri,
            checksum=element.checksum, media_type=element.media_type, width=element.width, height=element.height,
            alt_text=element.alt_text)

    @staticmethod
    def _clip(clip: GeneratedClipInput | None, start: float, end: float) -> GeneratedClip | None:
        """A generated clip plays from the start of its slide for its own length, cut at the slide's end: the
        narration (and so the timeline) stays authoritative, the clip never stretches a slide."""
        if clip is None:
            return None
        return GeneratedClip(
            segment_id=clip.segment_id, artifact_id=clip.artifact_id, uri=clip.uri, checksum=clip.checksum,
            media_type=clip.media_type, asset_duration=clip.duration, width=clip.width, height=clip.height,
            fps=clip.fps, has_audio=clip.has_audio, strategy=clip.strategy, audio=clip.audio,
            start_time=start, end_time=round(min(end, start + clip.duration), 3))

    @staticmethod
    def _transitions(segments: list[VideoSegment], config: VideoConfig) -> list[VideoSegment]:
        if config.transition == TransitionType.CUT:
            return segments
        out = [segments[0]]
        for prev, seg in zip(segments, segments[1:]):
            duration = round(min(config.fade_seconds, prev.duration, seg.duration), 3)
            out.append(seg.model_copy(update={"transition": Transition(type=TransitionType.FADE, duration=duration)}))
        return out

