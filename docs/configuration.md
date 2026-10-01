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
| `TA_IMAGE_PROVIDER` / `TA_TTS_PROVIDER` / `TA_VIDEO_PROVIDER` | `mock` | Media providers (`TA_IMAGE_PROVIDER` is image generation) |
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
