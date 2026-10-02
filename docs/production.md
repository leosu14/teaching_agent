# Production runs

A production run takes one request (level, topic, language, learner) through the whole lesson pipeline on real
providers and produces a narrated MP4 lesson:

```
Task → Diagnostic → Research/RAG → Planning → Teacher ⇄ Reviewer → Visual planning → IMAGE_ASSET
→ Slide planning → PRESENTATION (.pptx) → Audio planning → TTS → AUDIO_ASSET → PresentationTimeline
→ VideoPlan → FFmpeg → video validation → VIDEO → (learner evaluation) → learner memory
```

It is the same workflow, agents and tools as the offline demos. Only the providers differ, and only because the
configuration says so: nothing switches to a real provider because an API key happens to be set.

## Installation

```bash
pip install -e ".[providers]"          # httpx for the real adapters (no vendor SDKs)
apt-get install ffmpeg fonts-dejavu-core   # or: brew install ffmpeg
```

Video is always composed locally with FFmpeg (`ffmpeg` and `ffprobe` on `PATH`, or `TA_FFMPEG_PATH` /
`TA_FFPROBE_PATH`).

## Modes

| `TEACHING_AGENT_MODE` | What runs |
|---|---|
| `offline` (default) | Mock providers only. Any real provider is refused at startup and again at request time. |
| `production` | The providers you configure. A production run additionally requires a real provider for every capability the workflow uses. |

`TEACHING_AGENT_OFFLINE=true` still forces offline mode (the test suite sets it); combined with
`TEACHING_AGENT_MODE=production` it is reported as a contradiction instead of silently picking one.

## Configuration

Put these in the environment or in `.env` (placeholders; never commit real keys):

```bash
TEACHING_AGENT_MODE=production

# LLM (every agent goes through StructuredLLM and the model router)
LLM_PROVIDER=openai                # openai (or any OpenAI-compatible endpoint) | anthropic
LLM_MODEL=<model-name>
OPENAI_API_KEY=<your-openai-key>
LLM_INPUT_PRICE_PER_MTOK=<usd>     # optional: without a price the cost is reported as unknown, never guessed
LLM_OUTPUT_PRICE_PER_MTOK=<usd>

# Web search (research)
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=<your-tavily-key>   # or SEARCH_API_KEY

# Image generation (visuals)
IMAGE_PROVIDER=openai              # IMAGE_API_KEY, or OPENAI_API_KEY

# Text to speech (narration)
TTS_PROVIDER=openai                # TTS_API_KEY, or OPENAI_API_KEY
TTS_LANGUAGES=es-ES,en-US          # languages the voices are offered in; the lesson language must be one of them

# Video: local FFmpeg (TA_VIDEO_COMPOSER=ffmpeg is the default)
```

Every other provider variable (fallbacks, per-role LLM routes, timeouts, retries, rate limits) is documented in
[configuration](configuration.md#providers).

Required capabilities are derived from the workflow and the budget: the lesson workflow uses `llm`, `search`, `image`
and `tts`. Image search is required only if `MAX_SEARCHED_IMAGES` is above 0, and image generation only if
`MAX_GENERATED_IMAGES` is above 0. A capability nobody uses is neither required nor health-checked.

Startup validation names each missing piece by variable, for example:

```
PROBLEM: production mode is not configured: set TEACHING_AGENT_MODE=production (the mode is 'offline', ...)
PROBLEM: the workflow needs a provider for search: set SEARCH_PROVIDER (one of ['tavily'])
PROBLEM: SEARCH_PROVIDER='tavily' needs a credential: set SEARCH_API_KEY or TAVILY_API_KEY
PROBLEM: TTS_PROVIDER='mock' is a mock: production mode needs a real tts provider (one of ['openai'])
```

Credentials are only ever reported as set or missing.

### Knowledge base

The diagnostic needs the concepts of the lesson's subject and topic from the knowledge base
(`TA_CORPUS_DIR/knowledge_base.json`). The preflight counts them and refuses the run if there are none. The bundled
corpus covers Spanish B1 "Climate change" (the documented example) and Spanish A2 "football". Add concept documents
for other subjects and topics; nothing in the pipeline is specific to Spanish.

## Budgets

Every production task carries a budget in its metadata (`task.metadata["budget"]`, visible with the task). The
provider layer checks it before every billable request and stops the task with `BudgetExceededError` instead of
exceeding it. A budget stop is never retried and never falls back to another provider.

| Variable | Default | Checked |
|---|---|---|
| `MAX_LLM_REQUESTS` | `80` | before each LLM request (failed attempts count) |
| `MAX_LLM_TOKENS` | `600000` | before each request (stops once used up) and after each response |
| `MAX_SEARCH_REQUESTS` | `20` | before each search request (failed attempts count) |
| `MAX_GENERATED_IMAGES` | `4` | in the visual plan (a plan over budget is sent back for revision) and before each generation |
| `MAX_SEARCHED_IMAGES` | `0` | in the visual plan (there is no real image search adapter yet; see limitations) |
| `MAX_TTS_CHARACTERS` | `30000` | before each synthesis, with that segment's characters |
| `MAX_TTS_SECONDS` | `1800` | before each synthesis (stops once used up) and after each response |
| `MAX_COST_USD` | none | before each request and after each response, against the *known* cost |
| `PRODUCTION_HEALTH_TIMEOUT_SECONDS` | `20` | per provider health check |

Limits that are only known from a response (tokens, audio seconds, cost) can be overrun by at most the one request
that crossed them. `MAX_COST_USD` only counts costs that are known: vendor-reported costs, or configured LLM pricing.
When a provider's price is unknown the report says so (`cost_complete: false`, `unpriced: [...]`) and the unit limits
are what bounds the spending. A budget-stopped task can be resumed after raising the limit; the new limits are applied
on resume.

## Running

The default example is a B1 Spanish lesson about climate change.

```bash
# 1. Dry run: configuration, providers, budget, knowledge base and the workflow's stages. No provider is called.
python scripts/run_production_demo.py --dry-run

# 2. Without confirmation nothing runs: the plan is shown and the command exits.
python scripts/run_production_demo.py --level B1 --topic "Climate change" --language es --learner-id demo-user

# 3. Confirmed: health checks of the required providers, then the run.
python scripts/run_production_demo.py --level B1 --topic "Climate change" --language es --learner-id demo-user \
    --output output --confirm --export
```

| Option | Meaning |
|---|---|
| `--level`, `--topic`, `--language`, `--learner-id` | The lesson (defaults `B1`, `Climate change`, `es`, `demo-user`) |
| `--subject` | Subject of the level framework; defaults to the language for CEFR levels (`es` → `spanish`) |
| `--output DIR` | Where `production_run.json` (and exported files) go; default `output` |
| `--dry-run` | Check everything, call nothing; exit 0 when offline (and say a production run is not possible) |
| `--confirm` / `--yes` | Required before any real provider is called |
| `--skip-evaluation` | Do not start the post-lesson evaluation |
| `--evaluate LESSON_TASK_ID --answers FILE` | Evaluation mode: grade the learner's answers (JSON: `{"question_id": "answer"}`) |
| `--diagnostic-answers FILE` | Diagnostic answers (`{"concept_or_question_id": "answer"}`); default: ask on a terminal, else unanswered |
| `--resume TASK_ID` | Continue a specific task |
| `--fresh` | Start a new task even if an identical run exists |
| `--export` | Copy `lesson.mp4`, `lesson.pptx` and `lesson.vtt` into `--output` |
| `--log-file FILE` | Structured production log (JSON lines); default stderr |
| `--mock` | Rehearse the identical path on the offline mock providers (no confirmation needed: nothing is paid) |

Exit codes: `0` success (or nothing to do), `1` the task failed, `2` configuration problem, `3` a required provider
is unhealthy (nothing was generated).

### Output

- `production_run.json`: task id, run key, level, topic, language, subject, learner, start and end time, status,
  providers and models, LLM routes, budget, health, the artifact graph (every artifact with its parents, the node
  that created it and that node's provider request ids), the graph check, a per-node trace, usage, estimated cost
  (and whether it is complete), warnings, errors (with their category and stage), the final video reference and
  per-stage counts. It never contains a credential.
- A human-readable summary on stdout (research, lesson, visuals, presentation, audio, video, usage, artifacts).
- With `--export`, copies of the video, presentation and subtitles. The canonical artifacts stay in the object store
  (`TA_DATA_DIR/objects`); nothing else is written to `--output`.

### Traceability

Every provider request has a request id (`preq_...`, sent to the vendor where the API accepts one, e.g.
`X-Client-Request-Id`), and is recorded with the task, workflow node and agent that made it, its attempt number,
status, latency, units and cost. The artifact graph links each artifact to its node, so a VIDEO traces back through
PresentationTimeline, AUDIO_ASSET, PRESENTATION, IMAGE_ASSET, LESSON and LESSON_PLAN to the RESEARCH_BUNDLE, and each
of those to the provider requests that produced it.

### Logs

Production logs are JSON lines with `ts`, `event`, `task_id`, `node`, `agent`, `provider`, `capability`, `model`,
`request_id`, `attempt`, `duration_ms`, `status` and, on failure, `error_type`, `category` and a shortened `error`.
Prompts, responses, binary data, headers and credentials are never logged; known secrets and credential-shaped
strings are redacted from every message.

## Resume and idempotency

A run is identified by a run key: a hash of the request, the learner, the providers and models, and the settings
that change the output. Running the same command again:

- reuses a completed identical run whose artifacts are all intact (no provider call; `reused: true`);
- continues an identical unfinished run (crashed, failed or paused) from its last checkpoint;
- starts a new run when anything relevant changed (a provider, model, voice or workflow setting), or with `--fresh`.
  Budgets are not part of the key, so raising a limit and running the same command again resumes the run.

On resume every artifact of the task is verified against its checksum. Only the nodes whose artifacts are missing or
corrupt (and the nodes depending on them) run again; research, the lesson, images and the presentation are not
regenerated when they are intact. A completed run with a corrupt artifact is not patched in place: a new run starts
and the report says why.

## Failures

Each failed task records a category and the stage it happened in:

| Category | Meaning |
|---|---|
| `ConfigurationError` | A setting is wrong or missing |
| `BudgetExceededError` | A budget limit stopped the task (raise it and resume) |
| `ProviderError` | A provider failed after retries and fallbacks (authentication, rate limits, outages, bad responses) |
| `PlanningError`, `ResearchError`, `TeachingError`, `ReviewError`, `VisualError`, `PresentationError`, `AudioError`, `VideoError`, `LearnerMemoryError`, `EvaluationError` | The stage that failed |

Research keeps its policy (`TA_RESEARCH_REQUIREMENT`): a `mandatory` research that produces nothing fails the task;
a web search outage that still leaves knowledge-base sources continues with the failure named in the warnings.

## Evaluation

By default the run starts the lesson's evaluation and leaves it waiting for the learner's answers. Grade it later:

```bash
python scripts/run_production_demo.py --evaluate <lesson_task_id> --answers answers.json
```

`--skip-evaluation` leaves the lesson without an evaluation. Grading updates learner memory (mastery, weak concepts,
reviews).

## Provider health and smoke tests

```bash
python scripts/run_provider_demo.py --smoke   # one minimal request per configured real provider, and readiness
RUN_PROVIDER_SMOKE_TESTS=true pytest tests/smoke
```

The production run checks the health of the required providers only, before generating anything.

## Live end-to-end test

CI never calls a real provider. The live test runs one small lesson (1 generated image, tight unit limits) on your
configured providers and aborts once the known cost exceeds `LIVE_E2E_MAX_COST` (USD, default `1.00`):

```bash
RUN_LIVE_E2E=true LIVE_E2E_MAX_COST=0.50 TEACHING_AGENT_MODE=production ... pytest tests/live -s
```

Without `RUN_LIVE_E2E=true`, or in offline mode, it is skipped.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `production mode is not configured` | `TEACHING_AGENT_MODE=production` |
| `contradicts TEACHING_AGENT_MODE=production` | Unset `TEACHING_AGENT_OFFLINE` |
| `the workflow needs a provider for X: set X_PROVIDER` | Configure that capability, or set its budget to 0 if it is optional (images) |
| `needs a credential: set ...` | Set the named key variable |
| `the knowledge base (...) has no concepts ...` | Add concept documents for the subject and topic to the knowledge base, or pass `--subject` |
| `UNHEALTHY: ...` (exit 3) | The provider's key, endpoint or network; nothing was generated |
| `budget exceeded: ...` | Raise the named limit and run the same command again (it resumes) |
| `voice ... does not speak ...` | Add the lesson language to `TTS_LANGUAGES`, or set `TA_AUDIO_VOICE` |
| FFmpeg errors | Install `ffmpeg`; the failed job's work directory is kept under `TA_DATA_DIR/work/failed/` |

## Security

- Keys are read from the environment only, registered for redaction, and reported only as set or missing.
- Authorization headers are built inside the HTTP client and never logged, stored, or put in events, task state,
  artifacts or reports.
- Provider error messages are sanitized before they reach task errors, events and reports.
- Network access exists only in the provider layer (enforced by the architecture tests); scripts go through services.

## Known limitations

- Image search has no real adapter yet: searched images come from the local catalog, so production plans use
  generated visuals only (`MAX_SEARCHED_IMAGES=0`).
- Concept maps come from the local knowledge base; topics it does not cover are refused at preflight.
- The cost limit covers known costs only; unpriced models are bounded by the unit limits.
- Trace linkage is per workflow node: a node's artifacts are linked to all of that node's provider requests.
- The real adapters are tested against faithful in-process fakes of the vendor APIs; the live test is opt-in.
