# Architecture

## Layers

Imports only go downward. `tests/unit/test_architecture.py` fails the build on any upward import.

```
api          FastAPI routes: validate input, call a service, return the result
services     application services + the composition root (container.py); the CLI uses the same services
runtime      orchestrator (task lifecycle), workflow engine + node types, task state machine, workflow templates
agents       decide WHAT to do; reach models only via ModelRouter and tools only via ToolManager
tools        do the HOW (search, retrieval, dedup + ranking, research cache, learner memory, artifacts, media);
             no educational strategy
providers    replaceable adapters: LLM, search, retrieval, ranking, image, TTS, video (mock/local in this release)
learner      level frameworks, mastery rules, LearnerMemoryService (long-term memory)
artifacts    ArtifactService: versioning, content-hash dedup, dependency graph
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
  (`research` → `research_policy` → `store_research` → `plan`). It stores `research_bundle` first, then
  `lesson_plan` (parent: research_bundle), `lesson` (parents: lesson_plan, research_bundle),
  `narration_script`, `slide_plan` and `review_report`.
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

Real LLM/search/media providers, web scraping, vector retrieval and embeddings, PPTX rendering, video rendering, coding exercises and
VS Code integration, UI. The provider and tool interfaces they plug into already exist.
