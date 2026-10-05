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
learner      level frameworks, MasteryUpdater + review scheduler, evidence / learning-event / goal stores,
             learner-model builder, LearnerMemoryService (long-term memory)
pedagogy     the adaptive engine: ConceptGraph, KnowledgeBase interface, KnowledgeGapAnalyzer, PedagogicalPlanner,
             PedagogicalStrategy (GenericStrategy), NextLessonRecommender; pure, deterministic, no I/O
curriculum   long-term learning: CurriculumPlanner, validation, CurriculumEngine (versions, progress, completion,
             replanning), priority model, NextActionEngine, review policy, CurriculumTracker; reads the learner
             model and the concept graph, never providers, tools or storage
teaching     interactive sessions: TeachingPolicy, DifficultyController, grading, the session engine (pure
             transitions with a transactional outbox), summaries, the TeachingRepository interface; no providers,
             tools or storage
assessment   grading: normalisation, matching, rubrics (validation, aggregation), outcome classification, validation
             of a semantic grader's output, the grading pipeline (AssessmentEngine), the AssessmentRepository
             interface; pure, reaches a model only through the SemanticGrader interface its caller supplies
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

- `lesson_generation`: the lesson slice above. Before the diagnostic, `knowledge_graph` (`knowledge.concepts`) loads
  the domain's concept graph and `load_goal` (`learning_goal.resolve`) the learning goal the lesson is planned
  against. After it, `record_diagnostic` turns the graded answers into evidence, `learner_model`, `knowledge_gaps`
  and `pedagogical_plan` compute the adaptive state, and `store_pedagogy` stores `learning_evidence` → `learner_model`
  → `knowledge_gaps` → `pedagogical_plan` (each the parent of the next). Research runs between the diagnostic and the planner
  (`research` → `research_policy` → `store_research` → `plan`); visuals run after review
  (`teach_review` → `visual_gate` → `visual` → `visual_policy`), then the lesson artifacts are stored and the
  presentation is made (`presentation_gate` → `slide_plan` → `validate_slide_plan` → `store_slide_plan` →
  `build_presentation` → `render_presentation`), then narrated (`audio_plan` → `validate_audio_plan` →
  `store_audio_plan` → `synthesize_audio` → `audio_policy` → `audio_timeline`) and turned into a video (`video_plan`
  → `validate_video_plan` → `store_video_plan` → `compose_video`) before `update_learner`. It stores `research_bundle` first,
  then `visual_plan` (parent: research_bundle) and one `image_<visual_id>` IMAGE_ASSET per visual (parent:
  visual_plan), then `lesson_plan` (parents: research_bundle, pedagogical_plan), `lesson` (parents: lesson_plan, research_bundle and
  its image assets), `narration_script` and `review_report`, then `slide_plan` (parent: lesson) and `presentation`
  (parents: slide_plan, lesson and the image assets it places), then `audio_plan` (parents: presentation,
  slide_plan, lesson), one `audio_<segment_id>` AUDIO_ASSET per voiced segment (parent: audio_plan) and
  `presentation_timeline` (parents: presentation, audio_plan and the audio assets), then `video_plan` (parents:
  presentation_timeline, presentation and the image assets it shows), `subtitles` (WebVTT, parent: video_plan) and
  `video` (the MP4; parents: video_plan, presentation_timeline, presentation).
- `lesson_evaluation`: started for a completed lesson task (`TaskService.start_evaluation`), with the lesson
  task id in `plan.inputs`. It reads the lesson and lesson plan artifacts (`artifact.read`), takes a learner
  snapshot, asks `LearnerEvaluationAgent` for an assessment sized by objectives, taught concepts, level and
  mastery, WAITS for answers, grades each through the AssessmentService (`assessment.grade` tool node; see
  [semantic assessment](#semantic-assessment)), asks the agent to report on those grades and recommend what's next
  (an evaluation that disagrees with them is rejected), records the evidence
  through `learner.record_evaluation` (idempotent per task), rebuilds the learner model, asks the pedagogical
  engine for the next recommendation (`pedagogy.recommend`) and the feedback (`pedagogy.evaluation_feedback`), and
  stores a `LEARNER_EVALUATION` artifact whose parent is the lesson, then `learning_evidence` (parent: the
  evaluation), `learner_model` and `next_recommendation` (LEARNING_RECOMMENDATION). Events: `assessment.created`, `assessment.waiting`, `assessment.submitted`,
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

## Generated video segments

Optional short generated clips inside the lesson video, for the sections where motion teaches better than a still.
Only lessons whose request carries the capability `video.generated_segments` get the three extra nodes; every other
lesson runs exactly the nodes it ran before.

```
audio_policy → video_segment_plan (visual.video_strategy) → store_video_segment_plan (VIDEO_SEGMENT_PLAN)
→ generate_video_segments: video_generation.submit → generic poller → video_generation.status
  → video_generation.create_asset (download → validate bytes → normalise → validate → GENERATED_VIDEO_ASSET)
→ audio_timeline → video_plan (clips placed in their slides) → compose_video (FFmpeg) → VIDEO
```

- `VideoGenerationProvider` (`app/providers/video_generation/base.py`) is an async job interface: `submit`,
  `status`, `download`, `cancel`, `limits` (durations, aspect ratios, seed, cancel and audio support), with statuses
  SUBMITTED, PROCESSING, COMPLETED, FAILED and CANCELLED. `MockVideoGenerationProvider` is the default and renders a
  real, deterministic Motion-JPEG AVI per request. `MiniMaxVideoGenerationProvider` (Hailuo) maps MiniMax's
  task/query/file API onto it over plain HTTP; it lives only under `providers/` and is built only by the composition
  root, when `VIDEO_GENERATION_PROVIDER=minimax` in production mode. `ManagedVideoGenerationProvider` admits each
  submission against the task budget and records usage.
- `VideoStrategy` (`app/tools/visual/video_strategy.py`) extends visual planning with a deterministic pedagogical
  policy. Each lesson section gets a recorded decision: exercises, reviews, definitions, grammar and short factual
  text stay text; processes, movement, cause and effect, procedures, comparisons over time and pronunciation score
  from cue words (English plus the lesson language's own lexicon), visual hints and model suggestions (which only add
  weight). A section that already has an image needs a strong case. Durations follow the slide's narration within
  `TA_GENERATED_VIDEO_MIN/MAX_SECONDS`, snapped to what the provider offers. The budget takes the highest priority
  clips first (urgent knowledge gaps raise priority) within `MAX_GENERATED_VIDEO_SEGMENTS`,
  `MAX_GENERATED_VIDEO_SECONDS` and, when the price is known, `MAX_VIDEO_GENERATION_COST_USD`. Nothing is dropped
  silently: every skip has a reason and every budget cut a warning.
- `VideoPromptBuilder` writes every prompt in one fixed structure from cleaned lesson content (no URLs, emails,
  internal ids, markup or instructions). A request carries the prompt and technical parameters only; no learner
  data or id ever reaches a provider. Prompts are language-agnostic; only pronunciation clips name the language.
- The generation node (`app/runtime/workflows/generative_video.py`) owns the waiting, not the agents. It polls with
  the generic `poll_until` (`app/utils/polling.py`) for at most the poll timeout per run, then checkpoints the jobs
  and puts the task in WAITING (kind `video_generation`); resuming polls the same jobs. Cancelling the task cancels
  open jobs at the provider, or records them as cancelled locally when the provider cannot cancel, and starts none.
- Idempotency: a generation key (request hash + provider + model) indexes a ledger in the object store
  (`records/video-generations/<key>.json`) holding the job, the downloaded file and the normalised clip. A resume or
  a later task asking for the same clip reuses the job, the file and the normalised clip: nothing is submitted,
  downloaded or normalised twice.
- `ClipValidator` (`app/tools/video/clips.py`) checks a clip from its bytes, never from the provider's claims:
  checksum, container, video stream, decode errors, codec, duration, resolution, aspect ratio, frame rate and audio.
  The provider file is validated, normalised by the `VideoNormalizer` (FFmpeg: scaled and letterboxed to the lesson's
  resolution and frame rate, H.264 yuv420p MP4, muted unless kept) and validated again before it becomes a
  `GENERATED_VIDEO_ASSET` with the VIDEO_SEGMENT_PLAN as parent.
- Composition: the VideoAgent places a clip from the start of its slide for its own length, cut at the slide's end,
  so the narration timeline stays authoritative. TTS and subtitles are unchanged and drawn on top; clip audio is
  muted. `FULL_FRAME_REPLACE` fills the frame; `INSET` plays inside the slide card's media box. The VIDEO artifact's
  parents include the clip assets.
- Failure policy, per segment: an optional clip that fails at any stage (budget, submit, poll, download, validate,
  normalise) falls back to the slide's image or the slide, with a `generated_video.fallback` event and a warning; a
  required clip fails the task under `fail` and falls back under `continue`.

Events: `video_strategy.completed`, `video_generation.submitted`, `.polled`, `.waiting`, `.completed`, `.failed`,
`.cancelled`, `.reused`, `generated_video.validated`, `generated_video.asset_created`, `generated_video.fallback`.

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
mastery (0–1), confidence, evidence, correct / incorrect counts, the current incorrect streak, exposure counts and
a review date. The snapshot answers what the learner knows, probably does not know, should learn next and should
review. Recording a diagnostic, a lesson or an evaluation is idempotent per task, so a resumed task never applies
the same evidence twice.

History is stored, not just state: `LearningEvidence` (immutable, append-only: re-recording the same id is a no-op
and different content under the same id is refused), `LearningEvent`s (diagnostic and evaluation completed, lesson
completed, concept mastered or reviewed) and `LearningGoal`s, each in its own table. Any concept's state can be
rebuilt from its evidence and exposures alone (`LearnerMemoryService.rebuild`).

## Adaptive pedagogy

```
Learner → LearnerModel → Diagnostic → evidence → MasteryUpdater → Knowledge gaps → Pedagogical plan
→ Personalised lesson → Evaluation → evidence → MasteryUpdater → Next recommendation
```

What is decided by code and what by models:

| Deterministic (code, `app/learner` + `app/pedagogy`) | Model (agents) |
|---|---|
| Mastery from evidence (`MasteryUpdater`), confidence, review dates | A candidate grade for a free-text answer (validated; the outcome is classified by code) |
| Learner-model categories (mastered / developing / weak / unknown) | Diagnostic and assessment questions |
| Knowledge gaps and their priority, the recommended action | Wording of the lesson plan, explanations and examples |
| Target concepts, prerequisite review, introduce vs reinforce, activities, difficulty band, minutes | Exercises within the planned activity types |
| Next recommendation and evaluation feedback | |

- **Evidence.** Diagnostic and evaluation answers become `LearningEvidence` (source, correctness, score, difficulty,
  timestamp). Only evidence moves mastery; being taught a concept counts an exposure and schedules a review only.
  `MasteryUpdater` moves mastery toward the evidence score at a rate that grows with how informative the item was
  (a hard success or an easy failure) and shrinks as evidence accumulates; a `manual` placement is calibrated
  (`manual_weight`). Values stay within 0–1.
- **Spaced review.** `IntervalReviewScheduler` sets `next_review_at` from mastery (weak / medium / strong
  intervals), sooner after an error, longer after a correct answer that was retained over a gap. It is a protocol
  (`ReviewScheduler`), so a different algorithm can replace it.
- **Concept graph.** `ConceptGraph` validates prerequisites (known, no self-reference, no cycles) and keeps a stable
  topological order; `KnowledgeBase` is the interface the engine reads concepts from (`RetrieverKnowledgeBase`
  wraps the existing local knowledge base; no graph database).
- **Knowledge gaps.** `KnowledgeGapAnalyzer` scopes the goal's targets plus their prerequisites, treats concepts at
  or above `mastery_target` as mastered, and scores the rest:
  `priority = Σ weight × factor` over the deficit, prerequisite importance, goal relevance, recent errors, recency
  and repeated failure (weights in `PedagogyConfig.weights`). The action is `prerequisite_first` when a
  prerequisite is below `prerequisite_threshold`, `reteach` after repeated failure, `introduce` when unassessed or
  foundational, `reinforce` otherwise.
- **Planning.** `PedagogicalPlanner` picks targets by priority within the time available (the request, else the
  learner's session length): a target whose unmet prerequisites are at least `guided` is taught after reviewing
  them; otherwise it is deferred and its deepest blocking prerequisite is taught instead. Mastered concepts are
  only reviewed when due. Concepts taught before are reinforced, not re-introduced. The plan is validated (unique
  roles, objectives and activities on plan concepts, prerequisites first, positive minutes that add up within the
  time available) and its id is a hash of its content.
- **Strategies.** `PedagogicalStrategy` turns a concept treatment into objectives and activities;
  `GenericStrategy` (explain → practise at the learner's band → check) is the default for every domain, and a
  domain can name another one in `PedagogyConfig.strategies`. Activity types are open (`role_play`, `coding`, ...).
- **Difficulty bands.** foundational < 0.3 ≤ guided < 0.6 ≤ independent < 0.8 ≤ consolidation (configurable).
- **Adaptive questioning.** `AdaptiveQuestioningPolicy`: concepts memory already knows with confidence are not
  asked, a missed concept gets an easier follow-up, at most `max_follow_ups_per_concept` each, never more than
  `max_questions` in total; the diagnostic stops when nothing is left to follow up. The diagnostic agent's check
  enforces it.
- **Teacher.** The teacher receives the plan brief, the learner context and the gaps; its check enforces the plan's
  objectives, one objective per section of its own concept, a typed section purpose (`explanation`, `example`,
  `guided_practice`, `free_practice`, `review`, `assessment`) and no section on a concept outside the plan.
- **Privacy.** Providers get a pseudonymous learner context (level, preferences, the planned concepts' state), never
  the learner id, display name or full history; the learner id appears only in stored artifacts and events.
- **Failures.** Learner state changes only in `record_diagnostic` and `update_mastery`/`update_learner`, from
  validated, graded answers. A failed model call, review or evaluation leaves mastery as it was (the evaluation
  can be resumed); nothing is invented to fill a gap.

## Goals and curricula

```
Learner → LearningGoal → Curriculum (versioned) → objectives → mastery targets → next learning action
→ adaptive lesson for that objective → evaluation → mastery update → objective progress → curriculum progress
→ next learning action
```

The curriculum engine (`app/curriculum/`) extends the adaptive loop; it does not replace the learner model, the
mastery updater, the gap analysis, the pedagogical planner or the evaluation workflow. Goals are optional: a learner
without one keeps the adaptive loop exactly as before.

| Deterministic (code, `app/curriculum`) | Model (`learning_path_planner` agent) |
|---|---|
| A goal's scope: its targets (explicit, or the knowledge base's concepts up to a target level) plus every prerequisite from the knowledge base | Wording of each objective |
| Objective order (prerequisites first), role, mode (learn / maintain), target mastery, evidence required, priority | A suggested order (used only when it respects every prerequisite) |
| Validation, version ids and content hashes, replanning triggers | The explanation of the learning path |
| Objective status, curriculum progress, goal completion, the next action and its priority | |

- **Goal.** `LearningGoal` (title, description, domain, target level and concepts, optional target date, priority
  1–5, status ACTIVE / PAUSED / COMPLETED / CANCELLED). The same idempotency key (or, without one, the same request)
  never creates a second goal. COMPLETED is set only by the completion rule.
- **Planning.** A `curriculum_planning` task: concept graph → goal → learner model → `curriculum.draft` →
  `learning_path_planner` (sees a `CurriculumBrief`: concepts, prerequisites and a coarse state band; no learner,
  goal or task ids, evidence or numbers) → `curriculum.finalize` → a new version only when its content changed →
  LEARNING_GOAL → CURRICULUM_VERSION → LEARNING_OBJECTIVE artifacts → `curriculum.save`.
- **Validation** (`app/curriculum/validation.py`): concepts exist; prerequisites are exactly the knowledge base's,
  inside the curriculum and acyclic; every target is covered; target mastery is valid; prerequisites come first;
  objectives belong to the goal and the goal to the learner; version histories are consistent. An invalid
  curriculum fails the planning task; nothing is stored and no lesson starts from it. The model's proposal has no
  field for mastery, status, completion or prerequisites (unknown fields fail validation); unknown concepts are
  rejected as unresolved, concepts outside the scope are rejected.
- **Versions.** Immutable. The id derives from the content hash (goal definition, objective structure,
  configuration; not wording, progress or timestamps), so identical inputs give the same version and a resumed task
  never stores a second one. A goal change, a target-date change or a replanning trigger (repeated failure, a
  prerequisite that regressed, a concept mastered before its prerequisite) produces a new version when the path
  changes; older versions stay readable (`GET /goals/{id}/curriculum/versions`).
- **Objective status.** MASTERED (mastery ≥ target and enough evidence), BLOCKED (a prerequisite below
  `prerequisite_threshold`), IN_PROGRESS (some evidence), NOT_STARTED.
- **Next action** (`NextActionEngine`): per objective, REVIEW (mastered and due), EVALUATE (at target, evidence
  missing), LEARN (reteach after repeated failure, or below `practice_from`), PRACTICE (otherwise); BLOCKED
  objectives are never selected. COMPLETE when an active goal's completion rule holds; WAIT when nothing is
  eligible. The winner is the highest priority score
  (`score = prerequisite_ready × Σ weight × factor / Σ weight` over deficit, objective priority, goal priority, review
  urgency, target-date pressure, recent failure and recency; `app/curriculum/priority.py`), ties by goal priority,
  goal id, objective order. Completed goals contribute only reviews; paused and cancelled goals nothing.
- **Review.** `MasteryReviewPolicy` reads the learner model's review dates (the existing `IntervalReviewScheduler`):
  `next_review_at`, interval, review count, last review. Replaceable (`ReviewPolicy`).
- **Lessons.** `CurriculumService.start_lesson(action)` starts a normal lesson task with a `LessonFocus` (the
  objective and the action) in its metadata. The pedagogical planner then targets that objective's concept: LEARN
  teaches it, REVIEW retrieves it, PRACTICE adds an applied exercise, EVALUATE retrieves it and assesses it twice.
  The LEARNING_ACTION artifact and the lesson derive from the LEARNING_OBJECTIVE artifact.
- **Evaluation.** After the mastery update the evaluation runs `curriculum.track`: objective progress, transitions
  (`objective.started`, `objective.mastered`), completion (`goal.completed`), deterministic replanning, and the next
  action (a LEARNING_ACTION artifact). Nothing happens for a learner without curricula.
- **Target dates.** Feasibility (`sessions needed / sessions_per_week` against the days left) is reported as a
  structured warning (`deadline_infeasible`, `deadline_passed`); the plan is never compressed.

## Interactive teaching sessions

```
Lesson → Interactive Teaching Session → Teacher Turn → Learner Turn → Adaptive Response → Interaction Evidence
→ (completion) summary → LearningEvidence → mastery update → objective progress → next learning action
```

A session (`TeachingSession`, `app/teaching/`) is opt-in per lesson: `POST /lessons/{lesson_id}/teaching-session`
starts one on a generated lesson (the LESSON artifact, or the completed lesson task). It does not replace the
evaluation workflow, the learner model or the curriculum: it is one more source of evidence for them. Nothing about
it lives in `AgentContext`; its state is a serializable `SessionState` stored with the session.

| Deterministic (code, `app/teaching`) | Model (`teaching_session` agent) |
|---|---|
| The next action (EXPLAIN, ASK, HINT, FEEDBACK, RETEACH, PRACTICE, CHECK, SUMMARIZE, COMPLETE) | The wording of that one turn |
| Grading (normalised expected and accepted answers, multiple-choice labels) | The question's wording and its expected answer (validated) |
| Difficulty (`DifficultyController`), hint levels, misconception thresholds | Misconception candidates (validated, stored as evidence) |
| Completion (objective demonstrated, repeated failure, question limit, turn budget, learner stop) | Answers to learner questions, from the grounded material only |
| The summary's numbers, mastery (learner memory's updater), objective progress, the next action | The summary's narrative |

- **Lifecycle.** ACTIVE (the teacher owes turns) → WAITING_FOR_LEARNER → ... → COMPLETED; PAUSED, CANCELLED (keeps
  the history; cancelling again is a no-op) and FAILED. Every transition is a pure function in
  `app/teaching/engine.py` that returns a `SessionChange` (the new session, its new turns and evidence, and outbox
  items); `TeachingRepository.apply` writes it in one transaction.
- **Two phases.** A learner answer is first recorded and graded deterministically (the session stays ACTIVE with the
  teacher turns it owes planned in its state), then each owed teacher turn is generated and committed. If the model
  fails, nothing is lost: the session stays ACTIVE, and a resume, a read or the replayed request generates the turns.
- **Difficulty.** `increase_after_successes` consecutive correct answers raise it, `decrease_after_failures`
  consecutive misses lower it; a correct answer after a hint keeps it; an incorrect answer after a recorded
  misconception (`reteach_after_misconceptions`) reteaches. Every change is a DIFFICULTY change with its reason.
- **Hints.** Level 1 conceptual, 2 targeted, 3 a worked step; a hint never reveals the expected answer unless
  `reveal_answer_in_hints`. Hint use and the answer after a hint are evidence; a hinted correct answer counts as
  partial credit (`1 - hint_penalty × level`).
- **Learner questions** are answered only from the lesson's sections, its research and the knowledge base
  (`teaching.ground`). Citations must be refs of that material; with nothing relevant the answer is a structured
  limitation (`grounded: false`), never an invented source.
- **Validation of the model's output** (`TeachingSessionAgent.check`): the turn is exactly the requested action and
  concept at the requested difficulty; no COMPLETE; one question for question turns and none otherwise; a short-answer
  question does not contain its answer; hints do not reveal it; citations exist; misconceptions only for an incorrect
  answer about the session's concept. The output schema has no field for mastery, objective or session status
  (unknown fields fail validation). Invalid output is retried, then the turn fails as above.
- **Context window** (`TeachingTurnInput`): the objective, the lesson's sections, practice items, a compact state,
  the last `recent_turns` turns, this question and answer, and grounding passages. No learner, session, task or goal
  ids, no learner history, profile or other sessions.
- **Persistence and resume.** Sessions, turns (unique `(session_id, sequence)`), interaction evidence, request records
  and the outbox are SQL tables. A new process reads the session back and continues where it was; the outbox
  publishes events (stable ids, so never twice) and artifacts (deduplicated by content) that a crash left behind.
- **Idempotency and concurrency.** An answer carries a `client_turn_id`: the same id with the same answer replays the
  stored result, with a different answer is a 409. Starting a session is idempotent per `idempotency_key`. Every
  write is an optimistic-locking update (`version`); a concurrent answer gets a 409 conflict and is never silently
  dropped or applied twice.
- **Completion.** The summary (`TeachingSessionSummary`) is stored, then, once: each graded answer becomes
  `LearningEvidence` (`source_type: interaction`) recorded through `LearnerMemoryService.record_evidence` (practice
  sessions record none), the curriculum's objective progress is read back and `CurriculumService.next_action` selects
  the next action. TEACHING_SESSION_SUMMARY → LEARNING_EVIDENCE → LEARNER_MODEL → LEARNING_ACTION artifacts.
- **Artifacts.** LEARNING_GOAL → CURRICULUM_VERSION → LEARNING_OBJECTIVE → LESSON → TEACHING_SESSION →
  TEACHING_TURN → INTERACTION_EVIDENCE, all on the lesson task.
- **Events.** `teaching_session.started` / `paused` / `resumed` / `completed` / `cancelled` / `failed`,
  `teaching_turn.created`, `learner_answer.received`, `hint.given`, `misconception.detected`, `difficulty.changed`;
  stable ids, the concept, correctness, hint level and difficulty; no answer or turn text, no learner profile, no
  provider details.

## Semantic assessment

```
Lesson → AssessmentItem (+ AssessmentRubric) → attempt → normalisation → acceptable answers → deterministic rules
→ rubric grading → semantic grading (free text only) → validation → AssessmentGrade → LearningEvidence → mastery
→ objective progress → next learning action
```

`AssessmentService` (`app/services/assessment.py`) is the one place answers are graded. Interactive sessions, the
evaluation workflow (`POST /tasks/{id}/evaluation`, through the `grade_answers` tool node) and the assessment API all
call it; nothing else grades. The pipeline is `AssessmentEngine` (`app/assessment/engine.py`):

| Step | Decides | Model call |
|---|---|---|
| 1. Normalisation | Unicode form, case, punctuation, whitespace; accents only for `accent_insensitive_languages` (ñ is kept) | no |
| 2. Acceptable answers | The expected and acceptable answers; a choice or its label; true/false words per language | no |
| 3. Deterministic rules | An empty answer, an invalid choice, a known error (`known_errors`, with its misconception), any closed question that did not match | no |
| 4. Rubric grading | A `deterministic` rubric: each criterion is met when the answer contains one of its indicators | no |
| 5. Semantic grading | Free text (or short text with a rubric) that the steps above cannot decide | one validated call |

- **Items and rubrics.** `AssessmentItem` (prompt, expected answer or meaning, acceptable answers, response type
  SHORT_TEXT / FREE_TEXT / MULTIPLE_CHOICE / TRUE_FALSE, concept, objective, difficulty, language, known errors,
  misconception patterns) and `AssessmentRubric` (criteria with weights that must sum to 1, `required` criteria, a
  score scale, a passing threshold) are immutable: the same id with other content is a conflict. A free-text item
  without a rubric gets one required "meaning" criterion.
- **The model boundary.** The `semantic_grader` agent (standard tier) receives `SemanticGradingRequest`: the item,
  the rubric, the answer and at most `max_context_passages` lesson sections and research findings on the item's
  concept; no learner, task, goal or session ids, history or profile. It proposes a `SemanticGradeCandidate`;
  `app/assessment/validation.py` checks it in the agent (so invalid output is retried) and again in the engine (so no
  grader implementation can bypass it). Named rules: `malformed_json`, `state_mutation` (any mastery, objective, goal
  or curriculum field), `schema` (scores outside 0-1, unknown outcomes), `unknown_criterion`, `duplicate_criterion`,
  `missing_criterion`, `fabricated_citation`, `unknown_concept`, `contradictory_scores`, `contradictory_outcome`.
  Criterion scores are snapped to the scale; the overall score and the outcome are always recomputed.
- **Aggregation and outcome.** score = Σ criterion score × weight. `≥ correct_threshold` (the rubric's passing
  threshold) CORRECT, `≥ partial_threshold` PARTIAL, otherwise INCORRECT; an unmet required criterion caps CORRECT at
  PARTIAL. A semantic grade also needs confidence: `≥ accept_confidence` (0.85) stands; between `min_confidence`
  (0.60) and it, `mid_confidence: partial` caps CORRECT at PARTIAL and turns INCORRECT into UNCERTAIN; below
  `min_confidence` UNCERTAIN.
- **UNCERTAIN** is a first-class outcome, never converted to INCORRECT: the grader failed, was rejected, said the
  context is insufficient, or was not confident. The attempt and its grade are kept, no evidence is recorded, the
  feedback asks for another attempt, and in a session the same question stays open.
- **Misconceptions** come from known errors (rule, confidence 1.0) or the grader (semantic, at or above
  `misconception_min_confidence`, only on INCORRECT or PARTIAL grades, only about the item's concepts). They are
  recorded with the learning evidence and as `misconception.detected` events; mastery moves only by the score.
- **Attempts.** `AssessmentAttempt` (attempt id, number, answer, submitted_at, grade id, source) is stored with its
  grade in one write and never overwritten. The same attempt id with the same answer replays the stored grade
  (nothing graded or recorded again); with another answer it is a 409 conflict. A crash after grading is finished on
  the next submission without grading again.
- **Mastery.** An API attempt that is not UNCERTAIN becomes `LearningEvidence` (`source_type: assessment`; correct,
  partial with its score, or incorrect) recorded through `LearnerMemoryService.record_evidence`, then the objective's
  progress is read back and `CurriculumService.next_action` selects the next action. Sessions and evaluations record
  their graded answers through their own existing paths, with the grade's outcome and score.
- **Sessions.** `TeachingSessionService` grades every answer through the service (`source: teaching_session`) and
  hands the engine an `AnswerAssessment`; the engine never grades a model's output. PARTIAL closes the question with
  feedback and partial credit; UNCERTAIN keeps it open and the teacher asks again without revealing the answer.
- **Artifacts and events.** LESSON → ASSESSMENT_ITEM → ASSESSMENT_GRADE → ASSESSMENT_FEEDBACK, ASSESSMENT_RUBRIC →
  ASSESSMENT_GRADE, and ASSESSMENT_GRADE → LEARNING_EVIDENCE, on the lesson's (or the evaluation's) task.
  `assessment.started` / `graded` / `uncertain` / `completed` and `misconception.detected`, with stable ids: ids,
  outcome, scores, grader type and the grader's provider, model, tokens and estimated cost; never the answer, the
  prompt or a provider payload.
- **Cost.** Steps 1-4 call no model. A semantic grade records its calls, provider, model, tokens and estimated cost
  (`GraderUsage`, from the existing usage ledger); in the evaluation workflow they count in the task's cost.
- **Production.** The grader routes like every agent: mock offline (the default), the configured LLM provider only
  under `TEACHING_AGENT_MODE=production`. No new provider settings.

## Learning cycles

```
learner state → next learning action → cycle (RUNNING) → step: lesson workflow → step: interactive session or
evaluation → WAITING ⇄ learner response → evidence → mastery (learner memory's updater) → objective progress
→ next learning action → cycle COMPLETED
```

A learning cycle executes one curriculum action end to end. It is an orchestration layer, not an agent: it calls no
model and decides nothing the existing layers already decide. `LearningCycleService` (`app/services/learning_cycles.py`)
drives it; the state machine is pure (`app/curriculum/cycle.py`); the schemas are in `app/schemas/learning_cycle.py`
(a file of their own, since they reference curriculum, lesson and teaching schemas).

- **Dispatch.** The action comes from `CurriculumService.next_action` and is recorded on the cycle when it starts; the
  cycle never re-selects. Each action maps to existing workflows (`STEP_POLICY`):

  | Action | Steps |
  |---|---|
  | LEARN | a new lesson for the objective (lesson workflow, with its diagnostic) → interactive session (LEARN) |
  | REVIEW · PRACTICE | the objective's newest completed lesson (a new one if none) → interactive session (REVIEW · PRACTICE) |
  | EVALUATE | the objective's lesson (reused or new) → the evaluation workflow |
  | COMPLETE | no steps: the goal's completion rule is verified by code (`progress.goal_complete`), else FAILED |
  | WAIT | no steps: NOTHING_DUE, with the action (and `next_review_at`) as the outcome |

  Before the first step the action is checked against the current curriculum (`action_problem`): an objective that
  left the curriculum, a changed curriculum version, a blocked objective, or an already mastered one (except REVIEW)
  fail the cycle as VALIDATION without generating anything.
- **States.** RUNNING → WAITING (the learner is needed) → RUNNING → COMPLETED; BLOCKED (a retryable failure: resume
  retries), FAILED (permanent), CANCELLED. Steps: PENDING → STARTED (key recorded) → RUNNING (child attached) →
  WAITING → COMPLETED. One active (RUNNING, WAITING or BLOCKED) cycle per learner, enforced by a unique slot row.
- **Mastery and state.** Evidence comes from the session's or the evaluation's own grading (`AssessmentService`) and
  reaches mastery only through learner memory's updater, exactly as without a cycle; objective progress and goal
  completion are recomputed by the curriculum. The cycle copies their results into its outcome; models never write
  mastery, curriculum state, objective completion, task or cycle status.
- **Persistence.** `learning_cycles` (the cycle as a versioned JSON body), `learning_cycle_slots`,
  `learning_cycle_requests` (received responses). Every change is computed by a pure transition and applied against
  the version it was computed from (optimistic lock); a conflicting writer reloads.
- **Idempotency.** The cycle id is `stable_id(learner, idempotency_key)`: the same key returns, and continues, the
  same cycle. Before a child is created its step key is recorded; children carry the key (task metadata, the session's
  idempotency key), so a crash between creating a child and attaching it finds it again. A response is recorded with
  the transition that receives it (`client_response_id`, request hash); the same id and body replay, another body is
  a 409, and the child gets it with an idempotency key derived from the cycle and response id. Artifacts reuse
  identical content; events have ids derived from the cycle, type and version and the event store keeps one per id.
- **Resume.** WAITING returns to the caller; nothing polls. A response, a read (`GET`, which only observes) or
  `resume` reconciles the cycle with its child. `resume` retries a BLOCKED step through the child's own recovery
  (`TaskService.resume` from the workflow checkpoint, `TeachingSessionService.resume`); completed side effects are
  never repeated.
- **Failures.** VALIDATION (a stale or invalid action, an unmet completion rule, a configuration error: FAILED),
  PROVIDER (retryable: BLOCKED), WORKFLOW (a failed workflow stage: BLOCKED; an exhausted budget or a failed session:
  FAILED). Invalid learner input is refused with a 422 and changes nothing; the cycle keeps waiting.
- **Artifacts.** LEARNING_GOAL → CURRICULUM_VERSION → LEARNING_OBJECTIVE → LEARNING_CYCLE → LESSON (a cycle's lesson
  and its LEARNING_ACTION draft hang off the cycle artifact) → ASSESSMENT_GRADE / TEACHING_SESSION → LEARNING_EVIDENCE;
  the `learning_cycle_outcome` artifact links the cycle, the lesson and the resulting LEARNING_ACTION or evaluation.
- **Events.** `learning_cycle.started`, `action_selected`, `step_started`, `step_completed`, `waiting`,
  `response_received`, `action_completed`, `resumed`, `completed`, `failed`, `cancelled`, stored under the cycle id
  and published through the event bus from an outbox on the cycle (published after the state is saved, republished
  after a crash). Payloads carry ids, the action, kinds and counts; no learner id, answer, prompt, credential or
  provider payload.
- **Compatibility.** Cycles are opt-in. Lessons, sessions and evaluations started directly behave as before, and a
  learner without goals gets a WAIT (NOTHING_DUE) cycle.

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

Learned mastery models (BKT / IRT) and FSRS-style scheduling (the updater and scheduler are replaceable), a
curated multi-subject concept library, real image-search providers, video generation providers other than MiniMax, AI avatars, web scraping, vector retrieval and embeddings, advanced slide
design, animations, cloud rendering, coding exercises and VS Code integration, UI, deployment. The provider and
tool interfaces they plug into already exist.
