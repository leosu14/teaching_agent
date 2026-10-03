# Configuration

Settings come from environment variables with the `TA_` prefix (or a `.env` file; see `.env.example`).
They are validated when the container is built; errors say what to fix.

| Variable | Default | Meaning |
|---|---|---|
| `TA_DATA_DIR` | `var` | Root for the SQLite database and object store |
| `TA_DATABASE_URL` | `sqlite:///$TA_DATA_DIR/teaching_agent.db` | Metadata database |
| `TA_OBJECT_STORE_DIR` | `$TA_DATA_DIR/objects` | Artifact blobs |
| `TA_ROUTING_FILE` | `config/routing.toml` | Model tiers, fallback chains, pricing, output limits |
| `TA_RETRIEVAL_PROVIDER` | `local` | Knowledge-base retrieval |
| `TA_VIDEO_COMPOSER` | `ffmpeg` | `ffmpeg` composes a real MP4 (needs `ffmpeg` and `ffprobe`); `mock` writes a manifest-only file for tests |
| `TA_CORPUS_DIR` | `fixtures/demo` | Corpus for the mock search, mock image search (`image_catalog.json`) and local knowledge base |
| `TA_MAX_REVISIONS` | `2` | Revision budget of the review loop |
| `TA_REVISION_EXHAUSTED_POLICY` | `fail` | `fail` or `accept_with_warnings` when the budget is used |
| `TA_DIAGNOSTIC_MAX_ROUNDS` | `2` | Adaptive diagnostic question rounds |
| `TA_DIAGNOSTIC_MEMORY_CONFIDENCE` | `0.6` | Confidence above which the diagnostic trusts memory instead of asking |
| `TA_DIAGNOSTIC_MAX_QUESTIONS` | `12` | Adaptive questioning: most diagnostic questions in total |
| `TA_DIAGNOSTIC_MAX_FOLLOW_UPS` | `1` | Adaptive questioning: follow-up questions per missed concept |
| `TA_PEDAGOGY_BAND_GUIDED` | `0.3` | Difficulty bands: mastery from which a concept is `guided` (below: `foundational`) |
| `TA_PEDAGOGY_BAND_INDEPENDENT` | `0.6` | Mastery from which a concept is `independent` |
| `TA_PEDAGOGY_BAND_CONSOLIDATION` | `0.8` | Mastery from which a concept is `consolidation` |
| `TA_PEDAGOGY_MASTERY_TARGET` | `0.8` | Mastery at which a concept counts as mastered (no longer a gap; spaced review only) |
| `TA_PEDAGOGY_MAX_TARGET_CONCEPTS` | `2` | Most new target concepts per lesson |
| `TA_PEDAGOGY_LESSON_MINUTES` | unset | Time a lesson is planned for; unset uses the learner's `session_minutes` (else 30) |
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

| `TA_GENERATED_VIDEO_ENABLED` | `true` | Master switch for generated video segments. Only lessons that ask for the `video.generated_segments` capability get them either way; `false` drops the stage even for those |
| `TA_GENERATED_VIDEO_MIN_SECONDS` / `TA_GENERATED_VIDEO_MAX_SECONDS` | `3` / `10` | Length range of one generated clip; the clip follows its slide's narration inside it, snapped to a duration the provider offers |
| `TA_GENERATED_VIDEO_STRATEGY` | `full_frame_replace` | How a clip is shown: `full_frame_replace` (the clip fills the frame, subtitles on top) or `inset` (inside the slide card's media box). Pronunciation clips always use `inset` |
| `TA_GENERATED_VIDEO_REQUIRED` / `TA_GENERATED_VIDEO_FAILURE_POLICY` | `false` / `fail` | Planned clips are optional by default: a failure falls back to the slide's image or the slide. With `REQUIRED=true`, `fail` fails the task and `continue` falls back with a warning |
| `TA_GENERATED_VIDEO_POLL_INTERVAL_SECONDS` / `_POLL_TIMEOUT_SECONDS` / `_POLL_MAX_ATTEMPTS` | `5` / `600` / `120` | Generic job polling. After the timeout the task WAITs (kind `video_generation`) and resuming it polls the same jobs; a job that used every attempt fails |

Defaults for the video format live only in `VideoConfig` (`app/schemas/video.py`); an unset `TA_VIDEO_*` variable keeps
the default. FFmpeg: `apt-get install ffmpeg` (Debian/Ubuntu; CI does this), `brew install ffmpeg` (macOS).
| `TA_LOG_LEVEL` / `TA_LOG_JSON` | `INFO` / `true` | Structured logging |

## Providers

Provider settings have no `TA_` prefix (they use the standard vendor variable names). They are validated when the
container is built: a missing key, an unknown provider, a fallback equal to its primary or a real provider in
offline mode (the default; set `TEACHING_AGENT_MODE=production` to use real providers) stops startup with one message listing every problem by variable name, never by value. The old
`TA_LLM_PROVIDERS`, `TA_SEARCH_PROVIDER`, `TA_IMAGE_PROVIDER`, `TA_IMAGE_SEARCH_PROVIDER` and `TA_TTS_PROVIDER`
are no longer read (they only ever accepted the mocks).

| Variable | Default | Meaning |
|---|---|---|
| `TEACHING_AGENT_MODE` | `offline` | `offline`: only mock providers; any network provider is refused at startup and at request time. `production`: the configured providers run. Never inferred from the presence of a key |
| `TEACHING_AGENT_OFFLINE` | `false` | `true` forces offline mode whatever `TEACHING_AGENT_MODE` says (the test suite sets it); with `TEACHING_AGENT_MODE=production` it is reported as a contradiction |
| `LLM_PROVIDER` / `LLM_MODEL` | routing file (mock) | `mock`, `openai` (any OpenAI-compatible endpoint) or `anthropic`, and the model for every tier |
| `LLM_FALLBACK_PROVIDER` / `LLM_FALLBACK_MODEL` | none | Explicit fallback target, tried only after a transient failure (timeout, 429, 5xx, connection) |
| `LLM_<ROLE>_PROVIDER` / `LLM_<ROLE>_MODEL` | none | Per-role route without touching the agent. `<ROLE>` is an agent id (`TEACHER`, `SLIDE_PLANNER`, ...) or an alias (`REVIEWER` = content_reviewer, `PLANNER` = curriculum_planner, `EVALUATOR`, ...); `DEFAULT` is `LLM_PROVIDER`/`LLM_MODEL` |
| `LLM_TEMPERATURE` | provider default | Sent only when set |
| `LLM_INPUT_PRICE_PER_MTOK` / `LLM_OUTPUT_PRICE_PER_MTOK` | none | USD per million tokens for the env-configured models; without them the cost is reported as unknown (`null`), never guessed |
| `LLM_NATIVE_STRUCTURED_OUTPUT` | `true` | Use the vendor's JSON-schema mode when the schema fits it; output is validated either way |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_ORGANIZATION` | — / `https://api.openai.com/v1` / — | OpenAI or a compatible server (https required except on localhost) |
| `OPENAI_MAX_TOKENS_PARAM` | `max_completion_tokens` | `max_tokens` for older OpenAI-compatible servers |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` | — / `https://api.anthropic.com` | Anthropic Messages API |
| `TTS_PROVIDER` / `TTS_FALLBACK_PROVIDER` | `mock` / none | `mock` or `openai` (`/audio/speech`, returned as WAV) |
| `TTS_API_KEY` / `TTS_MODEL` / `TTS_BASE_URL` | `OPENAI_API_KEY` / `gpt-4o-mini-tts` / OpenAI | Speech credentials, model and endpoint |
| `TTS_LANGUAGES` | `en-US,en-GB,es-ES,es-MX,fr-FR,de-DE,zh-CN` | Languages the real TTS voices are offered in |
| `IMAGE_PROVIDER` / `IMAGE_FALLBACK_PROVIDER` | `mock` / none | `mock` or `openai` (`/images/generations`). Only generated visuals use it: the VisualAgent policy keeps photos and maps searched |
| `IMAGE_API_KEY` / `IMAGE_MODEL` / `IMAGE_BASE_URL` | `OPENAI_API_KEY` / `gpt-image-1` / OpenAI | Image generation credentials, model and endpoint |
| `IMAGE_SEARCH_PROVIDER` | `mock` | Image search (only the mock catalogue in this release) |
| `SEARCH_PROVIDER` / `SEARCH_FALLBACK_PROVIDER` | `mock` / none | `mock` or `tavily` |
| `SEARCH_API_KEY` / `SEARCH_BASE_URL` | `TAVILY_API_KEY` / `https://api.tavily.com` | Web search credentials and endpoint |
| `SEARCH_DEPTH` / `SEARCH_INCLUDE_RAW_CONTENT` | `basic` / `true` | Tavily search depth; fetch page text so evidence quotes can be checked against it |
| `SEARCH_INCLUDE_DOMAINS` / `SEARCH_EXCLUDE_DOMAINS` | none | Comma-separated website restrictions applied to every research query |
| `VIDEO_GENERATION_PROVIDER` / `VIDEO_GENERATION_FALLBACK_PROVIDER` | `mock` / none | `mock` or `minimax` (MiniMax Hailuo, plain HTTP). A key alone never selects MiniMax: it needs this variable and `TEACHING_AGENT_MODE=production`. The fallback applies to submission only |
| `VIDEO_GENERATION_API_KEY` / `VIDEO_GENERATION_MODEL` / `VIDEO_GENERATION_BASE_URL` | `MINIMAX_API_KEY` / `MiniMax-Hailuo-02` / `https://api.minimax.io/v1` | Video generation credentials, model and endpoint |
| `VIDEO_GENERATION_PRICE_PER_SECOND` | none | USD per generated second; without it the cost is unknown and the segment and seconds limits bound spending |
| `VIDEO_GENERATION_MAX_DOWNLOAD_BYTES` | 200 MB | Largest clip file accepted from the provider |
| `<CAP>_TIMEOUT_SECONDS` | LLM 120, TTS 90, IMAGE 110, SEARCH 15, IMAGE_SEARCH 15, VIDEO_GENERATION 60 (per request; jobs are polled) | Per-attempt timeout; every call has one |
| `<CAP>_MAX_ATTEMPTS` | 3 (IMAGE 2) | Attempts for transient failures only |
| `<CAP>_REQUESTS_PER_MINUTE` / `<CAP>_MAX_CONCURRENCY` | unlimited | Local, per-process rate limit; waiting emits `provider.rate_limited` |
| `PROVIDER_BACKOFF_SECONDS` / `_MULTIPLIER` / `PROVIDER_MAX_BACKOFF_SECONDS` | `0.5` / `2.0` / `8.0` | Bounded exponential backoff; a vendor `Retry-After` is honoured up to the cap |
| `PROVIDER_MAX_REQUEST_BYTES` / `PROVIDER_MAX_RESPONSE_BYTES` | 4 MB / 32 MB | Size limits on provider HTTP bodies |

The real adapters use plain HTTP through `httpx`, an optional extra: `pip install -e ".[providers]"`. No vendor SDK is
needed. `python scripts/run_provider_demo.py` shows the resolved configuration (keys only as set/missing);
`--smoke` sends one minimal request to each configured real provider. `RUN_PROVIDER_SMOKE_TESTS=true pytest
tests/smoke` does the same as tests; without credentials they are skipped.

## Production runs

`scripts/run_production_demo.py` runs a whole lesson on the real providers within a per-task budget
(`MAX_LLM_REQUESTS`, `MAX_LLM_TOKENS`, `MAX_SEARCH_REQUESTS`, `MAX_GENERATED_IMAGES`, `MAX_SEARCHED_IMAGES`,
`MAX_TTS_CHARACTERS`, `MAX_TTS_SECONDS`, `MAX_COST_USD`, `PRODUCTION_HEALTH_TIMEOUT_SECONDS`; no `TA_` prefix).
`MAX_GENERATED_VIDEO_SEGMENTS` (`2`), `MAX_GENERATED_VIDEO_SECONDS` (`20`) and `MAX_VIDEO_GENERATION_COST_USD` (none)
bound generated video segments in every lesson that asks for them: the video strategy plans within them, and
production tasks also enforce them before each submission. See
[production](production.md) for the variables, defaults, the dry run, confirmation, resume and the run report.

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

Every model referenced by a tier needs a pricing entry, except a model configured through `LLM_*` variables
without a price (its cost is reported as unknown). `LLM_PROVIDER`/`LLM_MODEL` replace every tier's chain
(with `LLM_FALLBACK_*` as its second target), and `LLM_<ROLE>_*` add per-agent routes; the agents never change. API keys,
model names and URLs never live in code.
