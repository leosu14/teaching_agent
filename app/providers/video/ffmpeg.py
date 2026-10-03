"""FFmpeg-backed video composition and probing.

FFmpeg is an infrastructure dependency, used only here and only through `FFmpegAdapter`:

- Commands are argument lists built by typed code (`FFmpegCommand`) and run without a shell. No caller passes
  command-line text; file paths are separate arguments, always absolute, inside the workspace and prefixed with
  `file:` so FFmpeg never reads them as another protocol. Filter graphs contain only numbers, colours and labels
  generated here; no text from a lesson ever reaches FFmpeg (slide text and subtitles are drawn into PNG frames
  by `FrameRenderer`).
- Each run has a timeout; its arguments and FFmpeg's error output are written to the workspace, which the caller
  keeps when composition fails.

Composition: every slide segment is split where its subtitles change; each piece is one still frame held for a
whole number of frames (rounded on the global timeline, so pieces add up to exactly round(duration * fps)
frames). Pieces are concatenated per slide, faded where the plan says so, and concatenated again. Narration is
placed at its timeline position over a silent bed of exactly the timeline's length and mixed without
normalisation; nothing is concatenated blindly, and silence stays silence.

A segment with a generated clip is also split where the clip starts and ends. A piece inside the clip's window is
the clip's own frames (trimmed by frame number) over the slide background (full_frame_replace) or inside the
card's media box (inset), with the piece's subtitle drawn on top from a transparent layer. The clip's sound is
mixed in only when the plan asks for it. A plan without clips produces exactly the command it always did.

`FFmpegVideoNormalizer` converts a generated clip to the platform format (H.264 yuv420p in MP4 at the configured
resolution and frame rate, letterboxed, cut to its planned length, no audio stream unless its sound is kept).
"""

from __future__ import annotations

import json
import math
import resource
import subprocess
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path

from app.providers.video.base import (
    FrameStats,
    MediaReader,
    VideoComposer,
    VideoCompositionError,
    VideoNormalizationError,
    VideoNormalizer,
    VideoProbeError,
    VideoProber,
    read_media,
)
from app.providers.video.frames import FrameRenderer, png_bytes
from app.schemas.generative_video import InsertionStrategy
from app.schemas.video import (
    AudioCodec,
    AudioStreamProbe,
    AudioWindow,
    ClipNormalization,
    ComposedVideo,
    FrameSample,
    GeneratedClip,
    NormalizedClip,
    TransitionType,
    VideoCodec,
    VideoConfig,
    VideoContainer,
    VideoPlan,
    VideoProbe,
    VideoSegment,
    VideoStreamProbe,
)

VIDEO_ENCODERS = {VideoCodec.H264: "libx264"}
AUDIO_ENCODERS = {AudioCodec.AAC: "aac"}
MUXERS = {VideoContainer.MP4: "mp4"}
AUDIO_EXTENSIONS = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav", "audio/mpeg": ".mp3",
                    "audio/ogg": ".ogg", "audio/flac": ".flac"}
IMAGE_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg"}
LEVEL_RATE = 8000  # Hz: audio is decoded at this rate to measure levels
THUMB = (64, 36)


class FFmpegError(VideoCompositionError):
    pass


@dataclass(frozen=True)
class FFmpegCommand:
    """One FFmpeg invocation, as data. `argv()` is the only place arguments are assembled."""

    workspace: Path
    inputs: tuple[tuple[tuple[str, ...], Path], ...]  # (options before -i, file)
    output: Path
    filter_graph: str | None = None
    maps: tuple[str, ...] = ()
    output_options: tuple[str, ...] = ()
    pipe_output: bool = False  # write the output to stdout instead of a file

    def argv(self, binary: str) -> list[str]:
        args = [binary, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        for options, path in self.inputs:
            args += [*options, "-i", _file_arg(path, self.workspace)]
        if self.filter_graph is not None:
            args += ["-filter_complex", self.filter_graph]
        for label in self.maps:
            args += ["-map", label]
        args += list(self.output_options)
        args.append("pipe:1" if self.pipe_output else _file_arg(self.output, self.workspace))
        return args


def _file_arg(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    if workspace.resolve() not in resolved.parents:
        raise FFmpegError(f"refusing a path outside the workspace: {path}")
    return f"file:{resolved}"


@dataclass
class FFmpegAdapter:
    """Runs FFmpeg and ffprobe. The binaries are configured, never discovered from user input."""

    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    timeout_seconds: float = 600.0
    log: list[str] = field(default_factory=list)

    def run(self, command: FFmpegCommand, *, log_name: str = "ffmpeg") -> bytes:
        argv = command.argv(self.ffmpeg)
        (command.workspace / f"{log_name}.args.json").write_text(json.dumps(argv, indent=1), encoding="utf-8")
        try:
            proc = subprocess.run(argv, shell=False, stdin=subprocess.DEVNULL, capture_output=True,
                                  timeout=self.timeout_seconds, cwd=command.workspace, check=False)
        except FileNotFoundError as exc:
            raise FFmpegError(f"ffmpeg is not installed or not on PATH ({self.ffmpeg})") from exc
        except subprocess.TimeoutExpired as exc:
            raise FFmpegError(f"ffmpeg timed out after {self.timeout_seconds:.0f}s") from exc
        (command.workspace / f"{log_name}.stderr.log").write_bytes(proc.stderr)
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
            raise FFmpegError(f"ffmpeg exited with {proc.returncode}: {' | '.join(tail)[:800]}")
        return proc.stdout

    def probe_json(self, path: Path) -> dict:
        argv = [self.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                f"file:{path.resolve()}"]
        try:
            proc = subprocess.run(argv, shell=False, stdin=subprocess.DEVNULL, capture_output=True,
                                  timeout=120, check=False)
        except FileNotFoundError as exc:
            raise VideoProbeError(f"ffprobe is not installed or not on PATH ({self.ffprobe})") from exc
        except subprocess.TimeoutExpired as exc:
            raise VideoProbeError("ffprobe timed out") from exc
        if proc.returncode != 0:
            raise VideoProbeError(proc.stderr.decode("utf-8", "replace").strip()[:500] or "ffprobe failed")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise VideoProbeError(f"ffprobe returned unreadable output: {exc}") from exc

    def available(self) -> bool:
        try:
            return subprocess.run([self.ffmpeg, "-version"], shell=False, capture_output=True, timeout=30,
                                  check=False).returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


# --- Composition ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """A still frame held for `frames` frames, or (with `clip_offset`) that many frames of the segment's generated
    clip starting at its frame `clip_offset`."""

    segment: int
    frames: int
    subtitle: str | None
    clip_offset: int | None = None


def frame_at(t: float, fps: int) -> int:
    return round(t * fps + 1e-9)


def pieces_for(plan: VideoPlan) -> list[list[Piece]]:
    """Per segment, the stills (and clip stretches) that make it up: split wherever the burned-in subtitle changes
    and where a generated clip starts or ends."""
    fps = plan.fps
    out: list[list[Piece]] = []
    style = plan.config.subtitles
    for index, seg in enumerate(plan.slides):
        cues = plan.subtitles_for(seg) if style.enabled else []
        clip = seg.generated

        def clamp(t: float) -> float:
            return min(max(t, seg.start_time), seg.end_time)

        cuts = sorted({seg.start_time, seg.end_time, *(clamp(c.start_time) for c in cues),
                       *(clamp(c.end_time) for c in cues),
                       *((clamp(clip.start_time), clamp(clip.end_time)) if clip is not None else ())})
        pieces: list[Piece] = []
        for a, b in zip(cuts, cuts[1:]):
            frames = frame_at(b, fps) - frame_at(a, fps)
            if frames <= 0:
                continue
            mid = (a + b) / 2
            cue = next((c for c in cues if c.start_time <= mid < c.end_time), None)
            text = cue.text if cue else None
            offset = None
            if clip is not None and clip.start_time <= mid < clip.end_time:
                offset = frame_at(a, fps) - frame_at(clip.start_time, fps)
            prev = pieces[-1] if pieces else None
            if prev is not None and prev.subtitle == text and prev.clip_offset is None and offset is None:
                pieces[-1] = Piece(index, prev.frames + frames, text)  # merge stills that look the same
            elif (prev is not None and prev.subtitle == text and prev.clip_offset is not None and offset is not None
                  and prev.clip_offset + prev.frames == offset):
                pieces[-1] = Piece(index, prev.frames + frames, text, prev.clip_offset)  # one stretch of the clip
            else:
                pieces.append(Piece(index, frames, text, offset))
        if not pieces:
            raise FFmpegError(f"segment {seg.segment_id} is shorter than one frame")
        out.append(pieces)
    return out


def _fade_frames(seg: VideoSegment, nxt: VideoSegment | None, fps: int, total: int) -> tuple[int, int]:
    def half(transition) -> int:
        if transition.type != TransitionType.FADE:
            return 0
        return min(max(1, round(transition.duration / 2 * fps)), total // 2)

    return half(seg.transition), half(nxt.transition) if nxt is not None else 0


class FFmpegVideoComposer(VideoComposer):
    name = "ffmpeg-composer/1"

    def __init__(self, media: MediaReader, adapter: FFmpegAdapter | None = None, *, font_path: Path | None = None,
                 preset: str = "veryfast", crf: int = 20) -> None:
        self._media = media
        self._adapter = adapter or FFmpegAdapter()
        self._font_path = font_path
        self._preset = preset
        self._crf = crf

    def compose(self, plan: VideoPlan, workspace: Path) -> ComposedVideo:
        started, cpu_before = time.perf_counter(), _child_cpu()
        config = plan.config
        frames_dir = workspace / "frames"
        audio_dir = workspace / "audio"
        frames_dir.mkdir()
        audio_dir.mkdir()

        renderer = FrameRenderer(config, self._font_path)
        layout = pieces_for(plan)
        inputs: list[tuple[tuple[str, ...], Path]] = []
        graph: list[str] = []
        slide_labels = []
        stills: dict[tuple[int, str | None], Path] = {}
        clip_files: dict[int, Path] = {}
        layers: dict[str, Path] = {}
        for index, (seg, pieces) in enumerate(zip(plan.slides, layout)):
            clip = seg.generated
            image = self._image(seg)
            inset = clip is not None and clip.strategy == InsertionStrategy.INSET
            base = renderer.card(seg.visual_ref.card, image, (seg.order, len(plan.slides)),
                                 reserve_media_box=inset and image is None)
            labels = []
            for piece in pieces:
                if piece.clip_offset is not None:
                    assert clip is not None
                    labels.append(self._clip_piece(
                        piece, index, clip, base, renderer, plan, inputs, graph, clip_files, layers, frames_dir,
                        workspace))
                    continue
                key = (index, piece.subtitle)
                if key not in stills:
                    still = frames_dir / f"s{index:03d}_{len(stills):04d}.png"
                    still.write_bytes(png_bytes(renderer.with_subtitle(base, piece.subtitle, config.subtitles)))
                    stills[key] = still
                n = len(inputs)
                inputs.append((("-framerate", str(config.fps)), stills[key]))
                graph.append(f"[{n}:v]loop=loop={piece.frames - 1}:size=1:start=0,"
                             f"setpts=N/({config.fps}*TB),format={config.pixel_format}[p{n}]")
                labels.append(f"[p{n}]")
            total = sum(p.frames for p in pieces)
            nxt = plan.slides[index + 1] if index + 1 < len(plan.slides) else None
            fade_in, fade_out = _fade_frames(seg, nxt, config.fps, total)
            filters = [f"concat=n={len(labels)}:v=1:a=0"]
            color = f"0x{config.background}"
            if fade_in:
                filters.append(f"fade=t=in:s=0:n={fade_in}:c={color}")
            if fade_out:
                filters.append(f"fade=t=out:s={total - fade_out}:n={fade_out}:c={color}")
            graph.append("".join(labels) + ",".join(filters) + f"[v{index}]")
            slide_labels.append(f"[v{index}]")
        total_frames = sum(p.frames for pieces in layout for p in pieces)
        graph.append("".join(slide_labels) + f"concat=n={len(slide_labels)}:v=1:a=0,"
                     f"format={config.pixel_format}[vout]")

        rate, layout_name = config.audio_sample_rate, "stereo" if config.audio_channels == 2 else "mono"
        total_samples = round(total_frames / config.fps * rate)
        graph.append(f"anullsrc=r={rate}:cl={layout_name},atrim=end_sample={total_samples}[abed]")
        mix = ["[abed]"]
        for k, track in enumerate(plan.audio_tracks):
            data = read_media(self._media, track.uri, track.checksum, f"audio {track.artifact_id}")
            path = audio_dir / f"t{k:03d}{AUDIO_EXTENSIONS.get(track.media_type, '.bin')}"
            path.write_bytes(data)
            n = len(inputs)
            inputs.append(((), path))
            delay = round(track.start_time * 1000)
            graph.append(f"[{n}:a]atrim=duration={track.duration:.3f},asetpts=PTS-STARTPTS,aresample={rate},"
                         f"aformat=sample_fmts=fltp:channel_layouts={layout_name},"
                         f"adelay=delays={delay}:all=1[a{k}]")
            mix.append(f"[a{k}]")
        for index, seg in enumerate(plan.slides):
            clip = seg.generated
            if clip is None or clip.audio != "mixed" or not clip.has_audio:
                continue
            n = len(inputs)
            inputs.append(((), clip_files[index]))
            delay = round(clip.start_time * 1000)
            graph.append(f"[{n}:a]atrim=duration={clip.duration:.3f},asetpts=PTS-STARTPTS,aresample={rate},"
                         f"aformat=sample_fmts=fltp:channel_layouts={layout_name},"
                         f"adelay=delays={delay}:all=1[g{index}]")
            mix.append(f"[g{index}]")
        if len(mix) == 1:
            graph.append("[abed]anull[aout]")
        else:
            graph.append("".join(mix) + f"amix=inputs={len(mix)}:duration=first:dropout_transition=0:"
                         "normalize=0[aout]")

        output = workspace / f"video{_extension(config)}"
        command = FFmpegCommand(
            workspace=workspace, inputs=tuple(inputs), output=output, filter_graph=";".join(graph),
            maps=("[vout]", "[aout]"), output_options=self._encoding(config, total_frames))
        self._adapter.run(command, log_name="compose")
        if not output.exists() or output.stat().st_size == 0:
            raise FFmpegError("ffmpeg produced no output")
        cpu = _child_cpu() - cpu_before
        metadata_extra = {"generated_clips": len(clip_files)} if clip_files else {}
        return ComposedVideo(
            path=str(output), media_type=config.media_type, composer=self.name, frames=total_frames,
            duration=total_frames / config.fps, render_seconds=round(time.perf_counter() - started, 3),
            cpu_seconds=round(cpu, 3) if cpu >= 0 else None,
            subtitles_burned=len({p.subtitle for pieces in layout for p in pieces if p.subtitle}),
            metadata={"stills": len(stills), "pieces": sum(len(p) for p in layout),
                      "audio_tracks": len(plan.audio_tracks), "encoder": VIDEO_ENCODERS[config.codec],
                      "audio_encoder": AUDIO_ENCODERS[config.audio_codec], "preset": self._preset,
                      "rate_control": f"{config.bitrate_kbps}k" if config.bitrate_kbps else f"crf {self._crf}",
                      "frame_renderer": renderer.name, "font": renderer.font_name, **metadata_extra},
        )

    def _clip_piece(self, piece: Piece, index: int, clip: GeneratedClip, base, renderer: FrameRenderer,
                    plan: VideoPlan, inputs: list, graph: list[str], clip_files: dict[int, Path],
                    layers: dict[str, Path], frames_dir: Path, workspace: Path) -> str:
        """Filters for one stretch of a generated clip: the clip over the background (full frame) or in the card's
        media box (inset), then the subtitle layer on top. Returns the piece's output label."""
        config = plan.config
        fps, fmt = config.fps, config.pixel_format
        if index not in clip_files:
            data = read_media(self._media, clip.uri, clip.checksum, f"generated clip {clip.artifact_id}")
            clips_dir = workspace / "clips"
            clips_dir.mkdir(exist_ok=True)
            path = clips_dir / f"g{index:03d}.mp4"
            path.write_bytes(data)
            clip_files[index] = path
        full = clip.strategy == InsertionStrategy.FULL_FRAME_REPLACE
        under = frames_dir / f"c{index:03d}_{'bg' if full else 'card'}.png"
        if not under.exists():
            under.write_bytes(png_bytes(renderer.background_frame() if full else base))
        if full:
            x, y, w, h = 0, 0, config.width, config.height
        else:
            x0, y0, x1, y1 = renderer.media_box(plan.slides[index].visual_ref.card)
            scale = min((x1 - x0) / clip.width, (y1 - y0) / clip.height)
            w, h = max(2, int(clip.width * scale) // 2 * 2), max(2, int(clip.height * scale) // 2 * 2)
            x, y = x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2
        b = len(inputs)
        inputs.append((("-framerate", str(fps)), under))
        c = len(inputs)
        inputs.append(((), clip_files[index]))
        graph.append(f"[{b}:v]loop=loop={piece.frames - 1}:size=1:start=0,setpts=N/({fps}*TB),format={fmt}[cb{b}]")
        graph.append(f"[{c}:v]trim=start_frame={piece.clip_offset}:end_frame={piece.clip_offset + piece.frames},"
                     f"setpts=PTS-STARTPTS,scale={w}:{h},setsar=1,format={fmt}[cc{b}]")
        if not piece.subtitle or not config.subtitles.enabled:
            graph.append(f"[cb{b}][cc{b}]overlay=x={x}:y={y}:eof_action=repeat,format={fmt}[p{b}]")
            return f"[p{b}]"
        graph.append(f"[cb{b}][cc{b}]overlay=x={x}:y={y}:eof_action=repeat[co{b}]")
        if piece.subtitle not in layers:
            layer = frames_dir / f"sub{len(layers):04d}.png"
            layer.write_bytes(png_bytes(renderer.subtitle_layer(piece.subtitle, config.subtitles)))
            layers[piece.subtitle] = layer
        s = len(inputs)
        inputs.append((("-framerate", str(fps)), layers[piece.subtitle]))
        graph.append(f"[{s}:v]loop=loop={piece.frames - 1}:size=1:start=0,setpts=N/({fps}*TB),format=rgba[cs{b}]")
        graph.append(f"[co{b}][cs{b}]overlay=x=0:y=0:eof_action=repeat,format={fmt}[p{b}]")
        return f"[p{b}]"

    def _image(self, seg: VideoSegment) -> bytes | None:
        image = seg.visual_ref.image
        if image is None:
            return None
        return read_media(self._media, image.uri, image.checksum, f"image {image.artifact_id}")

    def _encoding(self, config: VideoConfig, total_frames: int) -> tuple[str, ...]:
        rate_control = (("-b:v", f"{config.bitrate_kbps}k", "-maxrate", f"{config.bitrate_kbps}k",
                         "-bufsize", f"{config.bitrate_kbps * 2}k") if config.bitrate_kbps
                        else ("-crf", str(self._crf)))
        return (
            "-c:v", VIDEO_ENCODERS[config.codec], "-preset", self._preset, "-tune", "stillimage", *rate_control,
            "-pix_fmt", config.pixel_format, "-r", str(config.fps), "-frames:v", str(total_frames),
            "-c:a", AUDIO_ENCODERS[config.audio_codec], "-b:a", f"{config.audio_bitrate_kbps}k",
            "-ar", str(config.audio_sample_rate), "-ac", str(config.audio_channels),
            "-movflags", "+faststart", "-map_metadata", "-1", "-fflags", "+bitexact",
            "-flags:v", "+bitexact", "-flags:a", "+bitexact", "-f", MUXERS[config.container],
        )


def _extension(config: VideoConfig) -> str:
    return "." + MUXERS[config.container]


def _child_cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


# --- Probing ----------------------------------------------------------------------------------


def _rate(value: str | None) -> float:
    if not value or value in {"0/0", "0"}:
        return 0.0
    num, _, den = value.partition("/")
    return float(num) / float(den or 1)


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class FFprobeVideoProber(VideoProber):
    name = "ffprobe/1"

    def __init__(self, adapter: FFmpegAdapter | None = None) -> None:
        self._adapter = adapter or FFmpegAdapter()

    def probe(self, path: Path) -> VideoProbe:
        data = self._adapter.probe_json(path)
        fmt = data.get("format") or {}
        streams = data.get("streams") or []
        if not fmt.get("format_name"):
            raise VideoProbeError("no container format reported")
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        return VideoProbe(
            container=fmt["format_name"], brand=(fmt.get("tags") or {}).get("major_brand"),
            duration=_float(fmt.get("duration")) or 0.0, size_bytes=int(fmt.get("size") or path.stat().st_size),
            video=None if video is None else VideoStreamProbe(
                codec=video.get("codec_name", ""), width=int(video.get("width") or 0),
                height=int(video.get("height") or 0),
                fps=_rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")),
                pixel_format=video.get("pix_fmt"),
                frames=int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None,
                duration=_float(video.get("duration"))),
            audio=[AudioStreamProbe(codec=a.get("codec_name", ""), sample_rate=int(a.get("sample_rate") or 0),
                                    channels=int(a.get("channels") or 0), duration=_float(a.get("duration")))
                   for a in audio],
            prober=self.name,
        )

    def audio_levels(self, path: Path, windows: list[AudioWindow]) -> dict[str, float]:
        """Decodes the first audio stream once (mono, 8 kHz, 16-bit) and measures the peak in each window."""
        if not windows:
            return {}
        workspace = path.parent
        pcm = self._adapter.run(FFmpegCommand(
            workspace=workspace, inputs=(((), path),), output=workspace / "levels.pcm",
            maps=("0:a:0",), output_options=("-ac", "1", "-ar", str(LEVEL_RATE), "-f", "s16le",
                                             "-c:a", "pcm_s16le"), pipe_output=True), log_name="levels")
        samples = array("h")
        samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
        levels = {}
        for w in windows:
            a = max(0, round(w.start_time * LEVEL_RATE))
            b = min(len(samples), round(w.end_time * LEVEL_RATE))
            peak = max((abs(s) for s in samples[a:b]), default=0)
            levels[w.label] = 20 * math.log10(peak / 32768) if peak else float("-inf")
        return levels

    def decode_errors(self, path: Path) -> list[str]:
        """Decodes every frame (and audio packet) once; FFmpeg's error output, if any, is the list of problems."""
        workspace = path.parent
        try:
            self._adapter.run(FFmpegCommand(workspace=workspace, inputs=(((), path),), output=workspace / "decode.null",
                                            output_options=("-f", "null"), pipe_output=True), log_name="decode")
        except FFmpegError as exc:
            return [str(exc)]
        log = workspace / "decode.stderr.log"
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
        return [line.strip() for line in lines if line.strip()][:20]

    def frame_stats(self, path: Path, samples: list[FrameSample]) -> dict[str, FrameStats]:
        """Decodes the video once and keeps a 64x36 greyscale thumbnail of each sampled frame."""
        if not samples:
            return {}
        probe = self.probe(path)
        fps = probe.video.fps if probe.video and probe.video.fps else 30.0
        numbered = sorted({max(0, math.floor(s.time * fps)) for s in samples})
        select = "+".join(f"eq(n,{n})" for n in numbered)
        workspace = path.parent
        raw = self._adapter.run(FFmpegCommand(
            workspace=workspace, inputs=(((), path),), output=workspace / "frames.raw", maps=("0:v:0",),
            output_options=("-vf", f"select='{select}',scale={THUMB[0]}:{THUMB[1]},format=gray",
                            "-fps_mode", "passthrough", "-f", "rawvideo"), pipe_output=True), log_name="frames")
        size = THUMB[0] * THUMB[1]
        thumbs = {n: raw[i * size:(i + 1) * size] for i, n in enumerate(numbered)}
        out = {}
        for s in samples:
            thumb = thumbs.get(max(0, math.floor(s.time * fps)), b"")
            if len(thumb) != size:
                raise VideoProbeError(f"frame at {s.time:.3f}s could not be decoded")
            mean = sum(thumb) / size
            stddev = math.sqrt(sum((v - mean) ** 2 for v in thumb) / size)
            out[s.label] = FrameStats(mean, stddev, thumb)
        return out


# --- Normalisation of generated clips ---------------------------------------------------------------------------


class FFmpegVideoNormalizer(VideoNormalizer):
    name = "ffmpeg-normalizer/1"

    def __init__(self, adapter: FFmpegAdapter | None = None, *, preset: str = "veryfast", crf: int = 20) -> None:
        self._adapter = adapter or FFmpegAdapter()
        self._preset = preset
        self._crf = crf

    def normalize(self, source: Path, probe: VideoProbe, target: ClipNormalization,
                  workspace: Path) -> NormalizedClip:
        """Scale into the target frame keeping the aspect ratio (letterboxed), resample to the target frame rate, cut
        to at most the planned length, encode H.264 yuv420p (and AAC when the sound is kept) in MP4."""
        started = time.perf_counter()
        frames = max(1, frame_at(min(target.duration, probe.duration), target.fps))
        w, h, fps = target.width, target.height, target.fps
        graph = (f"[0:v]fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease:flags=bicubic,"
                 f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format={target.pixel_format}[v]")
        keep_audio = target.keep_audio and bool(probe.audio)
        audio: tuple[str, ...] = (("-c:a", AUDIO_ENCODERS[target.audio_codec], "-b:a", "128k",
                                   "-ar", str(target.audio_sample_rate), "-ac", str(target.audio_channels),
                                   "-t", f"{frames / fps:.3f}") if keep_audio else ("-an",))
        output = workspace / f"normalized.{MUXERS[target.container]}"
        command = FFmpegCommand(
            workspace=workspace, inputs=(((), source),), output=output, filter_graph=graph,
            maps=("[v]", "0:a:0") if keep_audio else ("[v]",),
            output_options=("-c:v", VIDEO_ENCODERS[target.codec], "-preset", self._preset, "-crf", str(self._crf),
                            "-pix_fmt", target.pixel_format, "-r", str(fps), "-frames:v", str(frames), *audio,
                            "-movflags", "+faststart", "-map_metadata", "-1", "-fflags", "+bitexact",
                            "-flags:v", "+bitexact", "-flags:a", "+bitexact", "-f", MUXERS[target.container]))
        try:
            self._adapter.run(command, log_name="normalize")
        except FFmpegError as exc:
            raise VideoNormalizationError(str(exc)) from exc
        if not output.exists() or output.stat().st_size == 0:
            raise VideoNormalizationError("ffmpeg produced no normalised clip")
        return NormalizedClip(path=str(output), media_type=target.media_type, normalizer=self.name,
                              render_seconds=round(time.perf_counter() - started, 3),
                              metadata={"frames": frames, "audio": keep_audio, "encoder": VIDEO_ENCODERS[target.codec],
                                        "source_codec": probe.video.codec if probe.video else None,
                                        "source_size": f"{probe.video.width}x{probe.video.height}"
                                        if probe.video else None})
