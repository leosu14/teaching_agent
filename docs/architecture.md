# Architecture

## Layers

Imports only go downward. `tests/unit/test_architecture.py` fails the build on any upward import.

```
api          FastAPI routes: validate input, call a service, return the result
services     application services + the composition root (container.py); the CLI uses the same services
runtime      orchestrator (task lifecycle), workflow engine + node types, task state machine, workflow templates
agents       decide WHAT to do; reach models only via ModelRouter and tools only via ToolManager
tools        do the HOW (search, retrieval, learner memory, artifacts, media); no educational strategy
providers    replaceable adapters: LLM, search, retrieval, image, TTS, video (mock/local in this release)
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

- `lesson_generation`: the lesson slice above. It stores `sources`, `lesson_plan`, `lesson`,
  `narration_script`, `slide_plan` and `review_report` artifacts.
- `lesson_evaluation`: started for a completed lesson task (`TaskService.start_evaluation`), with the lesson
  task id in `plan.inputs`. It reads the lesson and lesson plan artifacts (`artifact.read`), takes a learner
  snapshot, asks `LearnerEvaluationAgent` for an assessment sized by objectives, taught concepts, level and
  mastery, WAITS for answers, asks the agent to grade them and recommend what's next, records the evidence
  through `learner.record_evaluation` (idempotent per task), and stores a `LEARNER_EVALUATION` artifact whose
  parent is the lesson. Events: `assessment.created`, `assessment.waiting`, `assessment.submitted`,
  `evaluation.started`, `evaluation.completed`, `recommendation.created`, `learner.mastery_updated`.

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

Real LLM/search/media providers, vector retrieval, PPTX rendering, video rendering, coding exercises and
VS Code integration, UI. The provider and tool interfaces they plug into already exist.
