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
    VideoProbeError,
    VideoProber,
    read_media,
)
from app.providers.video.frames import FrameRenderer, png_bytes
from app.schemas.video import (
    AudioCodec,
    AudioStreamProbe,
    AudioWindow,
    ComposedVideo,
    FrameSample,
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
    """A still frame held for `frames` frames."""

    segment: int
    frames: int
    subtitle: str | None


def frame_at(t: float, fps: int) -> int:
    return round(t * fps + 1e-9)


def pieces_for(plan: VideoPlan) -> list[list[Piece]]:
    """Per segment, the stills that make it up: split wherever the burned-in subtitle changes."""
    fps = plan.fps
    out: list[list[Piece]] = []
    style = plan.config.subtitles
    for index, seg in enumerate(plan.slides):
        cues = plan.subtitles_for(seg) if style.enabled else []
        cuts = sorted({seg.start_time, seg.end_time,
                       *(min(max(c.start_time, seg.start_time), seg.end_time) for c in cues),
                       *(min(max(c.end_time, seg.start_time), seg.end_time) for c in cues)})
        pieces = []
        for a, b in zip(cuts, cuts[1:]):
            frames = frame_at(b, fps) - frame_at(a, fps)
            if frames <= 0:
                continue
            mid = (a + b) / 2
            cue = next((c for c in cues if c.start_time <= mid < c.end_time), None)
            text = cue.text if cue else None
            if pieces and pieces[-1].subtitle == text:  # merge stills that look the same
                pieces[-1] = Piece(index, pieces[-1].frames + frames, text)
            else:
                pieces.append(Piece(index, frames, text))
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
        for index, (seg, pieces) in enumerate(zip(plan.slides, layout)):
            base = renderer.card(seg.visual_ref.card, self._image(seg), (seg.order, len(plan.slides)))
            labels = []
            for piece in pieces:
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
        return ComposedVideo(
            path=str(output), media_type=config.media_type, composer=self.name, frames=total_frames,
            duration=total_frames / config.fps, render_seconds=round(time.perf_counter() - started, 3),
            cpu_seconds=round(cpu, 3) if cpu >= 0 else None,
            subtitles_burned=len({p.subtitle for pieces in layout for p in pieces if p.subtitle}),
            metadata={"stills": len(stills), "pieces": sum(len(p) for p in layout),
                      "audio_tracks": len(plan.audio_tracks), "encoder": VIDEO_ENCODERS[config.codec],
                      "audio_encoder": AUDIO_ENCODERS[config.audio_codec], "preset": self._preset,
                      "rate_control": f"{config.bitrate_kbps}k" if config.bitrate_kbps else f"crf {self._crf}",
                      "frame_renderer": renderer.name, "font": renderer.font_name},
        )

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
