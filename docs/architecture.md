# Architecture

## Layers

Imports only go downward. `tests/unit/test_architecture.py` fails the build on any upward import.

```
api          FastAPI routes: validate input, call a service, return the result
services     application services + the composition root (container.py); the CLI uses the same services
runtime      orchestrator (task lifecycle), workflow engine + node types, task state machine, workflow templates
agents       decide WHAT to do; reach models only via ModelRouter and tools only via ToolManager
tools        do the HOW (search, retrieval, dedup + ranking, research cache, image search / fetch / generation /
             selection / validation / assets, learner memory, artifacts, media);
             no educational strategy
providers    replaceable adapters: LLM, search, retrieval, ranking, image generation, image search, TTS, video
             (mock/local in this release)
learner      level frameworks, mastery rules, LearnerMemoryService (long-term memory)
artifacts    ArtifactService: versioning, content-hash dedup, dependency graph, content-addressed media objects
storage      SQLAlchemy/SQLite metadata repositories + filesystem object store (no business rules)
schemas / config / observability / utils   shared foundation
```

Extra rules enforced by the lint test: only `storage` imports SQLAlchemy; vendor SDKs only in `providers`;
agents may import only `providers.llm.base`/`router` from providers; the API imports only services, schemas
and a few exception types.

## Task lifecycle

`CREATED → PLANNING → RUNNING ⇄ REVIEWING / WAITING → COMPLETED`, plus `PAUSED`, `CANCELLED`, `FAILED`
(`app/runtime/tasks/state_machine.py`). Planning runs the Request Interpreter agent and picks a workflow
template by required capabilities. The orchestrator itself holds no educational logic.

## Workflow engine

Node types: `AgentNode`, `ToolNode`, `TransformNode`, `ConditionalNode`, `ParallelNode`, `ReviewNode`,
`HumanApprovalNode`. Nodes declare hard dependencies (`depends_on`, a skipped dependency skips the node) and
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
  (`teach_review` → `visual_gate` → `visual` → `visual_policy` → `slides`). It stores `research_bundle` first,
  then `visual_plan` (parent: research_bundle) and one `image_<visual_id>` IMAGE_ASSET per visual (parent:
  visual_plan), then `lesson_plan` (parent: research_bundle), `lesson` (parents: lesson_plan, research_bundle and
  its image assets), `narration_script`, `slide_plan` and `review_report`.
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

## Agents

Each agent declares id, name, description, input/output schemas, a system prompt (`prompt.md`), a tool
allow-list with permissions, a model tier, timeout and retry policy. Every model response is parsed and
validated against the output schema plus the agent's semantic `check`; failures go back to the model with
the error, up to `validation_retries`. The mock LLM returns text exactly like a real provider, so it goes
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

## Not in this release

Real LLM/search/image/media providers, web scraping, vector retrieval and embeddings, slide rendering, PPTX, TTS
narration, video rendering, coding exercises and
VS Code integration, UI. The provider and tool interfaces they plug into already exist.
