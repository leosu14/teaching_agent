# Configuration

Settings come from environment variables with the `TA_` prefix (or a `.env` file; see `.env.example`).
They are validated when the container is built; errors say what to fix.

| Variable | Default | Meaning |
|---|---|---|
| `TA_DATA_DIR` | `var` | Root for the SQLite database and object store |
| `TA_DATABASE_URL` | `sqlite:///$TA_DATA_DIR/teaching_agent.db` | Metadata database |
| `TA_OBJECT_STORE_DIR` | `$TA_DATA_DIR/objects` | Artifact blobs |
| `TA_ROUTING_FILE` | `config/routing.toml` | Model tiers, fallback chains, pricing, output limits |
| `TA_LLM_PROVIDERS` | `["mock"]` | Enabled LLM providers (only `mock` has an adapter now) |
| `TA_SEARCH_PROVIDER` / `TA_RETRIEVAL_PROVIDER` | `mock` / `local` | Web search and knowledge-base retrieval |
| `TA_IMAGE_PROVIDER` / `TA_TTS_PROVIDER` | `mock` | Media providers (`TA_IMAGE_PROVIDER` is image generation) |
| `TA_VIDEO_COMPOSER` | `ffmpeg` | `ffmpeg` composes a real MP4 (needs `ffmpeg` and `ffprobe`); `mock` writes a manifest-only file for tests |
| `TA_IMAGE_SEARCH_PROVIDER` | `mock` | Image search |
| `TA_CORPUS_DIR` | `fixtures/demo` | Corpus for the mock search, mock image search (`image_catalog.json`) and local knowledge base |
| `TA_MAX_REVISIONS` | `2` | Revision budget of the review loop |
| `TA_REVISION_EXHAUSTED_POLICY` | `fail` | `fail` or `accept_with_warnings` when the budget is used |
| `TA_DIAGNOSTIC_MAX_ROUNDS` | `2` | Adaptive diagnostic question rounds |
| `TA_DIAGNOSTIC_MEMORY_CONFIDENCE` | `0.6` | Confidence above which the diagnostic trusts memory instead of asking |
| `TA_RESEARCH_REQUIREMENT` | `mandatory` | `mandatory`: a failed research fails the lesson task; `optional`: continue with an empty bundle and a warning |
| `TA_RESEARCH_MAX_RESULTS` | `5` | Results requested per search query |
| `TA_RESEARCH_MAX_SOURCES` | `6` | Sources kept after ranking |
| `TA_RESEARCH_MIN_RELIABILITY` | `0.5` | Sources rated below this are rejected (the reason is kept in the bundle) |
| `TA_RESEARCH_CACHE` | `true` | Process-local cache of search results |
| `TA_VISUAL_FAILURE_POLICY` | `fail` | When a required visual cannot be produced: `fail` the lesson task, or `continue` with a warning. Optional visuals never stop a lesson |
| `TA_VISUAL_MAX_PER_LESSON` | `6` | Most visuals the visual planner may propose (`0` disables visuals) |
| `TA_VISUAL_MAX_CANDIDATES` | `3` | Searched candidates fetched and validated per visual before falling back or giving up |
| `TA_PRESENTATION_RENDERER` | `pptx` | `pptx` renders a real PowerPoint file locally with python-pptx; `mock` writes a deterministic JSON description (tests) |
| `TA_PRESENTATION_ASPECT_RATIO` | `16:9` | Slide size: `16:9` (13.33 x 7.5 in) or `4:3` (10 x 7.5 in) |
| `TA_PRESENTATION_MAX_SLIDES` | `20` | Most slides the slide planner may plan; a longer deck fails validation |
| `TA_AUDIO_FAILURE_POLICY` | `fail` | When a required narration segment cannot be voiced: `fail` the lesson task, or `continue` with a warning. Optional segments never stop a lesson |
| `TA_AUDIO_LANGUAGE` | lesson language | BCP 47 tag of the narration (`es-ES`, `en-US`, `zh-CN`, ...); by default the lesson's language of instruction |
| `TA_AUDIO_VOICE` | provider's first voice | A voice id from the TTS provider's catalog; it must speak the narration language |
| `TA_AUDIO_SPEAKING_RATE` | `1.0` | Speaking rate (0.5 to 2.0), where the provider supports it |
| `TA_AUDIO_FORMAT` | `wav` | Audio format requested from the provider (the mock produces WAV) |
| `TA_AUDIO_SAMPLE_RATE` | provider default | Requested sample rate in Hz; the validator checks the audio has it |
| `TA_AUDIO_MAX_WORDS_PER_SEGMENT` | `80` | Longest narration segment the planner may propose |
| `TA_AUDIO_SILENT_SLIDE_SECONDS` | `3.0` | How long the timeline shows a slide that has no narration |
| `TA_VIDEO_FAILURE_POLICY` | `fail` | Video is required by default: a composition or validation failure fails the task. `continue` makes it optional (the task completes with a warning and no VIDEO artifact) |
| `TA_VIDEO_WIDTH` / `TA_VIDEO_HEIGHT` / `TA_VIDEO_FPS` | `1920` / `1080` / `30` | Output resolution (even numbers) and frame rate |
| `TA_VIDEO_BITRATE_KBPS` | constant quality | Target video bitrate; unset uses constant-quality encoding |
| `TA_VIDEO_BACKGROUND` | `F4F6F8` | Slide card background colour (hex) |
| `TA_VIDEO_TRANSITION` / `TA_VIDEO_FADE_SECONDS` | `cut` / `0.5` | Slide transition (`cut` or `fade` through the background) |
| `TA_VIDEO_DURATION_TOLERANCE` | `0.1` | Allowed difference in seconds between the MP4 and the PresentationTimeline |
| `TA_VIDEO_SUBTITLES` / `TA_VIDEO_SUBTITLE_MAX_CHARS` | `true` / `42` | Burn in subtitles from the narration text; characters per line |
| `TA_VIDEO_FONT_PATH` | first of DejaVu Sans, Liberation Sans, Noto Sans, Arial | TrueType font for slide text and subtitles |
| `TA_FFMPEG_PATH` / `TA_FFPROBE_PATH` | `ffmpeg` / `ffprobe` | The FFmpeg binaries |
| `TA_VIDEO_TIMEOUT_SECONDS` | `1200` | Longest an FFmpeg run may take |
| `TA_VIDEO_WORK_DIR` / `TA_VIDEO_KEEP_FAILED_WORK` | `$TA_DATA_DIR/work` / `true` | Scratch space for composition; a failed job's directory is kept under `failed/` for diagnosis |

Defaults for the video format live only in `VideoConfig` (`app/schemas/video.py`); an unset `TA_VIDEO_*` variable keeps
the default. FFmpeg: `apt-get install ffmpeg` (Debian/Ubuntu; CI does this), `brew install ffmpeg` (macOS).
| `TA_LOG_LEVEL` / `TA_LOG_JSON` | `INFO` / `true` | Structured logging |

## Model routing (`config/routing.toml`)

```toml
[tiers]
reasoning = [{ provider = "mock", model = "mock-reasoning" }]   # ordered fallback chain
[agent_tiers]
teacher = "standard"                                             # optional per-agent override
[pricing.mock-reasoning]
input_per_mtok = 3.0
output_per_mtok = 15.0
```

Every model referenced by a tier needs a pricing entry, and every provider must be enabled. API keys,
model names and URLs never live in code.
