# Architecture

## Layers

Imports only go downward. `tests/unit/test_architecture.py` fails the build on any upward import.

```
api          FastAPI routes: validate input, call a service, return the result
services     application services + the composition root (container.py); the CLI uses the same services
runtime      orchestrator (task lifecycle), workflow engine + node types, task state machine, workflow templates
agents       decide WHAT to do; reach models only via ModelRouter and tools only via ToolManager
tools        do the HOW (search, retrieval, dedup + ranking, research cache, image search / fetch / generation /
             selection / validation / assets, slide plan validation, presentation build and render, TTS, audio
             plan and audio validation, audio assets, presentation timeline, video plan validation, video
             composition / validation / artifact (VideoService), learner memory, artifacts);
             no educational strategy
providers    replaceable adapters: LLM, search, retrieval, ranking, image generation, image search, presentation
             renderer (python-pptx), TTS, video composer + prober (FFmpeg adapter, mock); the provider core
             (registry, selector, invoker, typed errors, rate limiter, HTTP client) and the managed wrappers
learner      level frameworks, mastery rules, LearnerMemoryService (long-term memory)
artifacts    ArtifactService: versioning, content-hash dedup, dependency graph, content-addressed media objects
storage      SQLAlchemy/SQLite metadata repositories + filesystem object store (no business rules)
schemas / config / observability / utils   shared foundation
```

Extra rules enforced by the lint test: only `storage` imports SQLAlchemy; vendor SDKs (python-pptx included) only in
`providers`; audio libraries (`wave`, pydub, ffmpeg, ...) only in `providers` and `utils`; `subprocess`, Pillow and
video libraries only in `providers` (FFmpeg runs only through `app/providers/video/ffmpeg.py`);
agents may import only `providers.llm.base`/`router`/`structured` from providers; outside `providers`, only the
interface modules (`*.base`, the LLM router and StructuredLLM, `core.errors`/`registry`/`selector`) may be imported,
except by the composition root (`services/container.py`), which builds the concrete adapters; HTTP clients
(`httpx`, `requests`, ...) and vendor SDKs (`openai`, `anthropic`, ...) only in `providers`; the API imports only
services, schemas and a few exception types.

## Task lifecycle

`CREATED → PLANNING → RUNNING ⇄ REVIEWING / WAITING → COMPLETED`, plus `PAUSED`, `CANCELLED`, `FAILED`
(`app/runtime/tasks/state_machine.py`). Planning runs the Request Interpreter agent and picks a workflow
template by required capabilities. The orchestrator itself holds no educational logic.

## Workflow engine

Node types: `AgentNode`, `ToolNode`, `TransformNode`, `ConditionalNode`, `ParallelNode`, `ReviewNode`,
`HumanApprovalNode`, and the lesson workflow's `NarrationNode` (one tool sequence per audio segment). Nodes declare hard dependencies (`depends_on`, a skipped dependency skips the node) and
ordering-only dependencies (`after`). The engine applies per-node retry and timeout policies, persists a
checkpoint after every node (and inside the review loop), honours pause/cancel between nodes, and resumes
from the checkpoint: completed nodes never run again.

`ReviewNode` is the generate → review → revise mechanism: it records every round's verdict and issues,
passes the issues and the previous candidate to the generator, stops at `max_revisions`, then fails or
accepts with warnings according to `RevisionPolicy`.

`HumanApprovalNode` puts the task in `WAITING` with a prompt (for the diagnostic: questions without answer
keys) and continues when valid input is submitted. A node can also validate input against earlier outputs
(`validate_input`, e.g. every assessment question answered) and name events to emit when it starts waiting
and when input arrives. Answers may arrive in a later request or another process: the WAITING checkpoint is
all that is needed.

## Workflows

- `lesson_generation`: the lesson slice above. Research runs between the diagnostic and the planner
  (`research` → `research_policy` → `store_research` → `plan`); visuals run after review
  (`teach_review` → `visual_gate` → `visual` → `visual_policy`), then the lesson artifacts are stored and the
  presentation is made (`presentation_gate` → `slide_plan` → `validate_slide_plan` → `store_slide_plan` →
  `build_presentation` → `render_presentation`), then narrated (`audio_plan` → `validate_audio_plan` →
  `store_audio_plan` → `synthesize_audio` → `audio_policy` → `audio_timeline`) and turned into a video (`video_plan`
  → `validate_video_plan` → `store_video_plan` → `compose_video`) before `update_learner`. It stores `research_bundle` first,
  then `visual_plan` (parent: research_bundle) and one `image_<visual_id>` IMAGE_ASSET per visual (parent:
  visual_plan), then `lesson_plan` (parent: research_bundle), `lesson` (parents: lesson_plan, research_bundle and
  its image assets), `narration_script` and `review_report`, then `slide_plan` (parent: lesson) and `presentation`
  (parents: slide_plan, lesson and the image assets it places), then `audio_plan` (parents: presentation,
  slide_plan, lesson), one `audio_<segment_id>` AUDIO_ASSET per voiced segment (parent: audio_plan) and
  `presentation_timeline` (parents: presentation, audio_plan and the audio assets), then `video_plan` (parents:
  presentation_timeline, presentation and the image assets it shows), `subtitles` (WebVTT, parent: video_plan) and
  `video` (the MP4; parents: video_plan, presentation_timeline, presentation).
- `lesson_evaluation`: started for a completed lesson task (`TaskService.start_evaluation`), with the lesson
  task id in `plan.inputs`. It reads the lesson and lesson plan artifacts (`artifact.read`), takes a learner
  snapshot, asks `LearnerEvaluationAgent` for an assessment sized by objectives, taught concepts, level and
  mastery, WAITS for answers, asks the agent to grade them and recommend what's next, records the evidence
  through `learner.record_evaluation` (idempotent per task), and stores a `LEARNER_EVALUATION` artifact whose
  parent is the lesson. Events: `assessment.created`, `assessment.waiting`, `assessment.submitted`,
  `evaluation.started`, `evaluation.completed`, `recommendation.created`, `learner.mastery_updated`.

## Research

`ResearchAgent` (`app/agents/research/`) decides what the lesson needs researched (diagnosed gaps first, then
concepts not yet known, then known ones), plans one broad query plus one per concept, and calls three tools
through the ToolManager:

- `search.web` (`SearchTool`): provider-independent search. Input `SearchQuery` (text, language, subject,
  domain, max results); output `SearchResult`s, each with a stable `source_id` derived from the canonical URL
  and a `Source` carrying only what the provider supplied (publisher, author, date, language, type) plus the
  retrieval time. `SearchProvider` is the adapter interface (Tavily, Serper, Bing, Google, ... later);
  `MockSearchProvider` is deterministic over `fixtures/demo/web_corpus.json`. An optional `ResearchCache`
  (`get` / `set` / `invalidate`; `InMemoryResearchCache` now) stores complete results, so a cache hit keeps
  the original sources and retrieval times.
- `rag.retrieve`: knowledge-base passages as the same `SearchResult` shape.
- `research.rank`: deduplicates (same source id or canonical URL, identical content, same title and
  publisher; the queries that found a source are merged) and ranks with a pluggable `Ranker`.
  `HeuristicRanker` combines relevance, source quality (by provider-reported source type), freshness (only
  when a date exists), language match and publisher novelty, with configurable `RankingWeights`.

The agent selects sources above a reliability threshold, then asks the model for evidence: verbatim quotes
with character offsets into the source text, and key findings that point at that evidence. The agent's
`check` rejects any quote that is not literally at the stated location. Citations are built by code, one per
evidence item, never by the model. The resulting `ResearchBundle` holds the objective, queries, selected and
rejected sources, evidence, key findings, citations, status (`complete` / `partial` / `failed`), warnings and
errors; its validator enforces deduplicated sources and a complete Citation → Evidence → Source chain.

Search failures are recorded in the bundle, never papered over with invented sources. The workflow's
`research_policy` node (`app/runtime/workflows/research_policy.py`, `TA_RESEARCH_REQUIREMENT`) decides: when
research is `mandatory` a failed bundle fails the task; when `optional` the lesson continues with the empty
bundle and an explicit warning, which reaches `TaskResult.warnings`, and the reviewer marks uncited sections
as unverified. The planner and reviewer receive the full bundle and the teacher the part for the planned
concepts. Lesson sections cite citation ids, and the stored lesson carries the resolved `references`, so a
claim traces Lesson → ResearchBundle → Evidence → Source. Events: `research.started`,
`research.query_created`, `research.search_completed`, `research.source_selected`, `research.completed`,
`research.failed`. Search and retrieval usage is tracked per service in the task cost (`by_service`); a cost
is recorded only when the provider reports one.

## Visuals

`VisualAgent` (`app/agents/visual/`) runs only for a lesson the reviewer approved: the `visual_gate` node skips it
for a lesson accepted with warnings (with a warning saying so), and a rejected draft never reaches it, so no image
is made for a lesson that did not pass review. It asks the model for a `VisualPlan` from the lesson plan, the
approved lesson and the research bundle: one `VisualRequirement` per visual (visual id, purpose, lesson section,
concept, description, `VisualType`, preferred source, search query and/or generation prompt, aspect ratio, required,
attribution required). The schema rejects URLs anywhere in a requirement, a photo or map that is not searched, and
a generated visual that is not generated; the agent's `check` rejects unknown sections and too many visuals. The
plan id is a hash of the plan, so re-planning the same lesson the same way reuses the stored plan and images.
`VisualType` (photo, illustration, diagram, chart, map, icon, generated_visual) carries no subject logic.

For each requirement the agent tries the preferred source, then the other one when the visual allows it (a photo
or map is never replaced by a generated image). All of it goes through tools on the ToolManager:

- `image.search` (`ImageSearchTool`): `ImageSearchRequest` → `ImageSearchResult`s with a stable `image_id`, URLs,
  title, source page, publisher, creator, size, format, licence and credit line exactly as the provider reported
  them (unknown fields stay None; provider-only fields go to `metadata`). `ImageSearchProvider` is the adapter
  interface (Unsplash, Wikimedia Commons, Pexels, ... later); `MockImageSearchProvider` is deterministic over
  `fixtures/demo/image_catalog.json` and renders its downloads locally.
- `image.select` (`ImageSelector`): a deterministic score over relevance, aspect ratio, visual type, licence and
  credit availability, source quality and resolution; unlicensed candidates are ineligible when attribution is
  required. Candidates are passed through untouched, so the selected image keeps all its source metadata.
- `image.fetch`: downloads a candidate through its provider into the object store.
- `image.generate` (`ImageGenerationTool`): `ImageGenerationRequest` (prompt, negative prompt, aspect ratio, size,
  style, seed) → `ImageGenerationResult` (asset id, provider, model, generation metadata with timestamp and prompt
  hash, storage reference, usage). Parameters the provider does not support are recorded as ignored.
  `ImageGenerationProvider` is the adapter interface; `MockImageGenerationProvider` renders a deterministic PNG.
- `image.validate` (`ImageValidator`): reads format and size from the bytes, recomputes the checksum, and checks
  declared vs actual size and format, the expected aspect ratio, a non-empty source, licence and credit when
  attribution is required, and generation metadata (provider, model, timestamp, prompt hash). Failures are
  structured `ImageValidationError`s and an `image.validation_failed` event; the next candidate is tried.
- `image.create_asset` (`ImageAssetTool`): re-validates the bytes itself and only then stores an `IMAGE_ASSET`
  artifact whose metadata is the `ImageAsset`: visual id, plan id, section, type, origin, size, format, checksum,
  attribution, selection record (rank, score, signals, rejected candidates) and validation report.

Image bytes are content-addressed in the existing filesystem object store (`ArtifactService.put_object`,
`objects/sha256/<xx>/<sha256>.<ext>`): identical bytes are written once and every artifact using them points at
the same object; SQLite holds metadata only. Attribution is a separate chain from the lesson's textual citations:
lesson sections reference their image assets in `LessonSection.visuals` (attached by the workflow, never by a
model), while `citations` and `references` still trace Lesson → Citation → Evidence → Source. A generated image's
attribution has no source URL: it is never treated as evidence.

Failures are recorded per visual with every attempt (`VisualResult.failures`). An optional visual that fails
leaves a warning on the task result and the lesson artifact metadata. A required visual that fails follows
`TA_VISUAL_FAILURE_POLICY` in the `visual_policy` node: `fail` (default) fails the task with the visual and reason
named, `continue` completes with a warning. Events: `visual.started`, `visual.plan_created`,
`image.search_started`, `image.search_completed`, `image.selected`, `image.generation_started`,
`image.generation_completed`, `image.validation_failed`, `image.asset_created`, `visual.completed`,
`visual.failed`. Image search, fetch and generation usage is tracked per service in the task cost, with units
(requests, images, megapixels, bytes); a cost is recorded only when the provider reports one.

## Presentation

Planning, building and rendering are separate steps with separate representations:

```
Lesson + LessonPlan + ResearchBundle + IMAGE_ASSETs → SlidePlannerAgent → SlideDeckPlan (WHAT appears)
→ SlidePlanValidator → PresentationBuilder → Presentation (renderer-independent) → PresentationRenderer → .pptx
```

- `SlideDeckPlan` (`app/schemas/presentation.py`): deck id, title, language, level, topic, objective, metadata and
  `SlidePlan`s. Each slide has an id, order, `SlideType` (title, objectives, explanation, example, comparison,
  vocabulary, exercise, answer, summary, references), title, subtitle, structured content blocks (`TextBlock`,
  `BulletBlock`, `TableBlock`, `QuestionBlock`, `AnswerBlock`, `VocabularyBlock`, `CitationBlock`, `ImageBlock`;
  never raw HTML or one string), `visual_refs` (IMAGE_ASSET artifact ids), `citation_refs` (research citation
  ids), `section_refs` (lesson section ids), speaker notes, a semantic `SlideLayout` (title, title_content,
  two_column, image_text, full_image, exercise, summary) and a duration hint. No geometry or file details.
- `SlidePlannerAgent` (`app/agents/slides/`, id `slide_planner`) runs only for an approved lesson
  (`presentation_gate`). The model proposes the slides from the lesson, plan, the research citations and the image
  assets (ids and descriptions, never bytes or paths); code assigns a deterministic deck id and the lesson metadata,
  then calls `slide_plan.validate` through the ToolManager and sends any errors back to the model. It never builds or
  renders files, touches storage or calls image providers.
- `SlidePlanValidator` (`app/tools/presentation/validation.py`, tool `slide_plan.validate`) is deterministic: schema
  (slide types, block kinds, non-empty fields), slide count, unique ids, ordering 1..n, a title slide first,
  non-empty content slides, lesson sections, image artifacts (existing, declared, placed), citation ids, question
  ids (an answer only after its question), layout fit (an image layout has one image; text-only layouts none), the
  block a slide type needs, and density (blocks and words per slide). The workflow's `validate_slide_plan` node
  runs it with `enforce`: an invalid deck fails the task there, before any artifact of it exists.
- `PresentationBuilder` (`app/tools/presentation/builder.py`, tool `presentation.build`) turns the validated deck into
  a `Presentation` of `PresentationSlide`s and `PresentationElement`s (title, text, bullets, table, image, footer)
  placed in layout regions. It resolves citations Slide → Citation → Evidence → Source through the research bundle
  (numbered by first use, footers and a references slide) and images through their IMAGE_ASSET metadata (object,
  checksum, size, alt text, attribution credit), so the plan never duplicates image metadata. Visual attribution
  stays separate: Slide → IMAGE_ASSET → source or generation metadata.
- `PresentationRenderer` (`app/providers/presentation/base.py`): `render(presentation) -> RenderedPresentation`.
  `PptxPresentationRenderer` uses python-pptx locally: it maps each semantic layout to geometry from
  `PresentationConfig` (aspect ratio, width, height, language, theme, footer) and `PresentationTheme` (fonts,
  typography scale, spacing, colours, border/radius where supported), places the stored image bytes after checking
  their checksum (it never generates an image), adds footers, slide numbers and speaker notes, and writes
  byte-reproducible files (fixed zip and core-property timestamps). `MockPresentationRenderer` writes a canonical
  JSON description for tests.
- `presentation.render` stores the file content-addressed in the existing object store and creates the
  `PRESENTATION` artifact (MIME type, checksum, object key, renderer, slide/element counts, placed images, citations,
  layouts; parents: slide_plan, lesson, placed images). Its identity is the artifact id, not a path. Re-rendering the
  same presentation produces the same bytes, so a rerun reuses the artifact instead of adding a version.

Failures: an invalid plan fails the task at `validate_slide_plan`; a render failure fails it at
`render_presentation` with the error persisted and the built presentation still inspectable; a missing optional
visual follows the visual policy and is simply not placed. A lesson accepted with warnings gets no presentation (a
warning says so); a rejected lesson never reaches it. Resume continues from the last completed presentation node.
Events: `slide_planning.started`, `slide_plan.created`, `slide_plan.validated`, `presentation.build_started`,
`presentation.build_completed`, `presentation.render_started`, `presentation.render_completed`,
`presentation.artifact_created`, `presentation.failed`. The planner's tokens and cost are recorded like any agent's;
rendering is recorded per service (`presentation_render:<renderer>`, units: slides, images, bytes) with no cost.

## Audio

Planning, synthesis and timing are separate steps with separate representations:

```
Lesson + SlideDeckPlan (+ ResearchBundle) → AudioPlannerAgent → AudioPlan → AudioPlanValidator → TTSTool → TTSProvider
→ AudioValidator → AUDIO_ASSET → timing resolver → PresentationTimeline (→ PRESENTATION, slides, AUDIO_ASSETs)
```

- `AudioPlan` (`app/schemas/audio.py`): plan id, task id, deck id, language (BCP 47), default voice, metadata and
  ordered `AudioSegment`s. A segment names its slide, its source (`slide_title`, `slide_content`, `speaker_notes`,
  `exercise_instructions`, `answer_explanation`) and `source_ref`, the text, language, voice, speaking rate, pitch,
  pauses, a planning estimate of its duration and whether it is required. `start_time`/`end_time`/`duration` stay
  empty until the audio exists. No provider details are in the plan.
- `AudioPlannerAgent` (`audio_planner`, `app/agents/audio/`) runs only for a rendered presentation. The model sees the
  slides' speakable text (titles, text/bullet/vocabulary lines, speaker notes, questions, answers), never images,
  captions, tables or citations, and proposes concise segments. Code picks the voice from the provider's catalog
  (`tts.voices`), assigns deterministic segment ids (`<slide_id>_a<n>`) and plan id, and checks the plan with
  `audio_plan.validate` through the ToolManager, sending errors back to the model. It never synthesizes or stores.
- `AudioPlanValidator` (`app/tools/audio/validation.py`): unique segment ids, known slides in deck order, contiguous
  order, non-empty and concise text, no citation ids/source titles/URLs read aloud, language and voice against the
  provider's live catalog and, for a timed plan, no negative durations or overlaps. The workflow gate
  (`validate_audio_plan`) fails the task before any speech is made.
- `TTSProvider` (`app/providers/tts/base.py`): `voices()` and `synthesize(ProviderSpeechRequest)`, with declared
  formats and optional parameters (rate, pitch, sample rate) and reported usage (characters, tokens, seconds,
  estimated and actual cost). Real adapters (OpenAI TTS, ElevenLabs, Azure Speech, Google Cloud TTS) keep their SDKs
  in their own provider module. `MockTTSProvider` writes a real 16-bit PCM WAV, deterministic per request, whose
  length follows the text and rate and whose tone follows the voice and pitch.
- `TTSTool` (`tts.synthesize`): `TTSRequest` → `TTSResult` (stored object reference, declared duration, sample rate,
  channels, format, provider, model, usage, ignored parameters). Usage is recorded per service (`tts:<provider>`).
- `NarrationNode` (`synthesize_audio`) voices each segment: reuse an AUDIO_ASSET made from the same inputs whose bytes
  still match (`audio.find_asset`), otherwise synthesize and create the asset (`audio.create_asset`), which re-reads
  the bytes and runs `AudioValidator` (MIME type, readable container, non-empty, non-zero measured duration, declared
  vs measured duration, sample rate, channels, checksum; `app/utils/audio.py` has the container readers, WAV for now).
  Invalid audio never becomes an asset. Identical bytes are one object in the store.
- Failure policy (`audio_policy`): an optional segment that fails becomes a warning and its slide plays without it;
  a required one fails the task (`TA_AUDIO_FAILURE_POLICY=fail`, the default) or becomes a warning (`continue`).
  Generated assets are kept either way, and every failure names its segment, stage and reason.
- `audio.timeline` resolves timing deterministically in whole milliseconds from the measured durations: slides in
  deck order, each segment after its `pause_before` and followed by its `pause_after`, a slide without narration
  held for `TA_AUDIO_SILENT_SLIDE_SECONDS`. `PresentationTimeline` has one `SlideTiming` per slide (start, end,
  duration, segment refs) and one `SegmentTiming` per voiced segment (with its AUDIO_ASSET id), references the
  PRESENTATION artifact and slide ids (never the PPTX file, which is not modified) and is stored as
  PRESENTATION_TIMELINE. This is the timeline a video composer will consume.

A rejected lesson, a lesson accepted with warnings, or a failed presentation gets no audio. Resume continues from
the last completed audio node; a narration interrupted mid-plan reuses the assets already stored. Events:
`audio_planning.started`, `audio_plan.created`, `audio_plan.validated`, `tts.started`, `tts.completed`,
`audio.validation_failed`, `audio.asset_created`, `timeline.created`, `audio.completed`, `audio.failed`.

## Video

```
Presentation + PresentationTimeline + IMAGE_ASSETs + AUDIO_ASSETs → VideoAgent → VideoPlan → VideoPlanValidator
→ VideoNode: video.compose → VideoService → VideoComposer (FFmpegVideoComposer → FFmpegAdapter) → MP4 object
→ video.validate (VideoValidator + VideoProber/ffprobe) → video.create_artifact → VIDEO
```

- `VideoConfig` (`app/schemas/video.py`) holds every output default in one place: 1920x1080, 30 fps, H.264, AAC,
  MP4, yuv420p, background colour, cut transitions (or fade), ±0.1 s duration tolerance and the subtitle style
  (size and margin as fractions of the frame height, 42 characters x 2 lines, white on a translucent box).
  `TA_VIDEO_*` settings override single fields; 1280x720 or any even resolution and any frame rate work.
- `VideoPlan`: plan id (derived from the timeline checksum, every image and audio checksum and the configuration),
  presentation and timeline refs, resolution, fps, duration, one `VideoSegment` per slide (times from the timeline,
  `VisualRef` = the slide's IMAGE_ASSET on a slide card, or the slide card alone, its `AudioTrack` refs, subtitle
  refs and the incoming `Transition`), the `AudioTrack`s, the `SubtitleTrack` and the `TransitionPoint`s. Assets are
  referenced by artifact id, object URI and checksum. Nothing in the plan is FFmpeg-specific.
- `VideoAgent` (`video`, `app/agents/video/`) is deterministic (no model call) and makes only the semantic decisions:
  visuals, narration placement, subtitle cues (the AUDIO_ASSET narration text, split into cues timed in proportion
  to their length; no speech-to-text) and transitions. It validates its plan through `video_plan.validate`.
- `VideoPlanValidator` (`app/tools/video/validation.py`): unique ordered contiguous segments for known slides,
  durations (at least one frame), segment and audio times equal to the timeline, total duration, image and audio
  refs matching the given assets (and their checksums), audio inside its segment and not overlapping (unless
  `allow_audio_overlap`), subtitle timing, transitions and configuration. The `validate_video_plan` gate stops an
  invalid plan before any composition.
- `VideoService` (`app/tools/video/service.py`) drives the composer and prober against the object store: the
  composer writes into a `ScratchSpace` directory (`app/utils/workspace.py`; removed on success, kept under
  `work/failed/` on failure with the FFmpeg arguments and log), the MP4 is streamed into the content-addressed
  object store (`put_object_file`; never loaded whole, never in SQLite), validation measures a streamed scratch
  copy. A composition key (plan id + configuration + composer version) finds an equivalent VIDEO artifact to reuse.
- `FFmpegVideoComposer` (`app/providers/video/ffmpeg.py`): slide frames (card, image, burned-in subtitle) are drawn
  with Pillow (`frames.py`); each segment is split where its subtitle changes, every still is held for a whole
  number of frames on the global timeline (so the video has exactly round(duration x fps) frames), stills are
  concatenated per slide, faded through the background where the plan says so, and concatenated again. Narration
  is placed with `adelay` at its timeline position over a silent bed of the exact timeline length and mixed without
  normalisation, so silence (pauses, silent slides) is preserved. `FFmpegAdapter` runs typed argument lists without
  a shell, with `file:` paths inside the workspace only, a timeout, and no lesson text in any filter graph.
- `VideoValidator` with `FFprobeVideoProber`: `ftyp` magic plus the parser's container, checksum, non-zero duration
  within tolerance of the timeline, resolution, frame rate, codecs, an audio stream, narration audible in every
  narrated window and silence on silent slides (decoded PCM), and sampled frames (one per segment) that are not
  flat and change between slides.
- Failure policy (`VideoNode`, `app/runtime/workflows/video.py`): `TA_VIDEO_FAILURE_POLICY=fail` (default, video is
  required) fails the task with a persisted error and no VIDEO artifact, keeping every earlier artifact;
  `continue` completes with a warning and no VIDEO artifact. Progress is checkpointed after composition and after
  validation, so a resumed task does not compose again.

Video only follows a timeline (so an approved, rendered, narrated presentation). Events: `video_planning.started`,
`video_plan.created`, `video_plan.validated`, `video_composition.started`, `video_composition.completed`,
`video.validation_started`, `video.validation_completed`, `video.artifact_created`, `video.failed`. Usage is recorded
as `video_compose:<composer>` with render seconds, CPU seconds, frames and output bytes, and no cost.

## Providers

```
Agents ──► Tools / ModelRouter ──► provider interfaces (*.base) ──► Managed providers ──► concrete adapters
                                                                   (invoker: offline guard, request id,
                                                                    rate limit, timeout, retry, events,
                                                                    usage; explicit fallback chain)
```

- **Interfaces.** `LLMProvider`, `TTSProvider`, `ImageGenerationProvider`, `ImageSearchProvider`, `SearchProvider`
  share the `Provider` base: `provider_id`, `capabilities`, `configuration()` (never secrets), `health_check()`
  (a structured `HealthStatus`; local checks for mocks and Tavily, a cheap authenticated GET for LLMs), plus the
  capability's invocation method and usage. The mocks implement the same interfaces and stay the default.
- **Adapters** (plain HTTPS through `app/providers/core/http.py`, `httpx` as the optional `providers` extra, no
  SDKs): `llm/openai_compatible.py` (Chat Completions; any compatible server via `OPENAI_BASE_URL`),
  `llm/anthropic.py` (Messages API), `tts/openai.py` (`/audio/speech`, PCM wrapped as WAV so duration is measured),
  `image/openai.py` (`/images/generations`; the nearest supported size, centre-cropped to the exact request),
  `search/tavily.py` (`/search` with domain filters and page text). Image search stays on the mock catalogue.
- **Registry and selector.** `ProviderRegistry` holds every provider by capability, the default per capability
  and the last health status; it refuses network providers in offline mode. `ProviderSelector` resolves the
  provider (and model, for LLMs: tier chain or per-agent route) for a capability or agent from configuration;
  no agent names a vendor.
- **Invoker.** Every call runs through `ProviderInvoker`: a per-attempt timeout, retries only for timeouts, 429,
  5xx and connection failures (typed errors; never auth or invalid requests), bounded exponential backoff that
  honours `Retry-After` up to a cap, a local requests-per-minute and concurrency limit, a unique `preq_` request
  id (sent as a header where the vendor accepts one, and recorded with the vendor's own request id), a
  `ProviderUsage` record and the events `provider.request_started`, `provider.request_completed`,
  `provider.request_failed`, `provider.rate_limited` and `provider.fallback`. Events go to the task that made the
  call (the execution scope travels in a context variable set by ToolManager and the router).
- **Fallback** is explicit: only a configured `*_FALLBACK_PROVIDER` (or a later routing target) is tried, only
  after a transient failure, and always with a `provider.fallback` event.
- **Structured output.** `StructuredLLM` is the one place that turns a schema (Pydantic model or JSON Schema) into a
  validated object: it asks for the vendor's native JSON-schema mode when the schema fits its strict subset
  (otherwise JSON mode), parses fenced or embedded JSON, validates, and sends validation errors back for a
  correction. `Agent.generate` delegates to it.
- **Security.** Keys are `SecretStr`, registered with the redactor, and removed from logs, events, errors and
  provider error excerpts; https is required (except localhost), redirects are not followed, request and response
  bodies are size-limited. Usage and events never carry prompts, binary content or headers.
- **Run mode.** `TEACHING_AGENT_MODE=offline` (the default, or `TEACHING_AGENT_OFFLINE=true`) makes a real provider a
  startup error and makes the HTTP client refuse any request; the test suite runs this way, so CI needs no keys.
  Real providers run only with `TEACHING_AGENT_MODE=production`, never because a key is present.
- **Budgets.** The invoker asks the task's `UsageLedger` before every billable request (`admit`) and records every
  attempt as a `ProviderRequestRecord` (request id, node, agent, attempt, status, units, cost). A production task's
  budget lives in its metadata; a limit stops the task with `BudgetExceededError`, which is never retried or fallen
  back from. `TaskUsage` is summed from the records.

## Agents

Each agent declares id, name, description, input/output schemas, a system prompt (`prompt.md`), a tool
allow-list with permissions, a model tier, timeout and retry policy. Every model response goes through
`StructuredLLM`: parsed and validated against the output schema plus the agent's semantic `check`; failures go
back to the model with the error, up to `validation_retries`. The mock LLM returns text exactly like a real provider, so it goes
through the same path.

## Learner model

Subjects carry a pluggable `LevelFramework` (`cefr`, `mastery`, more can be registered). Concepts carry
mastery, confidence, evidence and exposure counts and a review date. The snapshot answers what the learner
knows, probably does not know, should learn next and should review. Recording a lesson or an evaluation is
idempotent per task, so a resumed task never applies the same evidence twice.

## Cost and observability

`ModelRouter` maps tiers (reasoning / standard / cheap) to fallback chains of provider models and prices every
call. Each task carries estimated cost (from the workflow template's expected calls) and actual cost and tokens,
broken down by agent and model. Every event (task, node, agent, tool, LLM call, review, artifact, learner) is
persisted and logged as JSON, so a task can be reconstructed from its event log.

A failed task records a failure category (`ConfigurationError`, `BudgetExceededError`, `ProviderError`, or the stage:
`ResearchError`, `VisualError`, `AudioError`, ...) classified from the typed errors in its cause chain
(`app/runtime/failures.py`).

## Production runs

`ProductionService` (`app/services/production.py`) plans, runs, resumes and reports a production lesson on top of
`TaskService`: it resolves the request (level framework, subject, language), checks readiness for the capabilities
the workflow needs, health-checks them, derives a deterministic run key for idempotency, verifies artifact checksums
on resume and invalidates only the nodes whose artifacts are missing or corrupt, and builds the run report (artifact
graph, per-node trace, usage). `scripts/run_production_demo.py` is a thin CLI over it; see
[production](production.md).

## Not in this release

Real image-search and AI video providers, MiniMax, AI avatars, web scraping, vector retrieval and embeddings, advanced slide
design, animations, cloud rendering, coding exercises and VS Code integration, UI, deployment. The provider and
tool interfaces they plug into already exist.
