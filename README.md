# Teaching Agent

An agentic, personalised education platform. It assesses what a learner knows, researches the topic
with traceable sources, plans a lesson for that learner, writes and reviews the lesson, turns it into a
PowerPoint presentation, narrates it with a timeline per slide, composes it into an MP4 video with subtitles,
stores versioned artifacts, and updates long-term learner memory, so the next lesson adapts.

This release is the **text-lesson vertical slice** with research, visuals, a presentation, its narration and the
lesson video, fully deterministic on mock providers (the video is composed locally with FFmpeg):

```
Request → Task → Request Interpreter → Learner Snapshot → Diagnostic (adaptive, can WAIT for answers)
→ Research (queries → search → dedup/rank → evidence → cited ResearchBundle) → Curriculum Plan → Teacher ⇄ Reviewer (revision loop)
→ Visual (approved lessons only: visual plan → image search / generation → selection → validation → IMAGE_ASSET) → Artifact Storage
→ Slide Planning → SlideDeckPlan validation → Presentation Build → Presentation Render (approved lessons only: PPTX → PRESENTATION)
→ Audio Planning → AudioPlan validation → TTS → audio validation → AUDIO_ASSET → PresentationTimeline (rendered presentations only)
→ Video Planning → VideoPlan validation → Video Composition (FFmpeg) → MP4 validation (ffprobe) → VIDEO
→ Learner Memory → COMPLETED
```

A completed lesson can then be evaluated (the post-lesson learning loop):

```
Completed lesson → Learner Evaluation (assessment) → WAITING for answers → Learner Evaluation (grading)
→ Learner Memory (mastery) → remaining gaps + next-learning recommendation → LEARNER_EVALUATION artifact → COMPLETED
```

## Run it

```bash
pip install -e ".[dev]"                # plus ffmpeg + ffprobe on PATH (apt-get install ffmpeg / brew install ffmpeg)
python scripts/run_demo.py          # one command that proves the slice works
python scripts/run_evaluation_demo.py  # lesson, then its evaluation, mastery before/after, recommendation
python scripts/run_research_demo.py    # the lesson's research: queries, results, sources, evidence, citations
python scripts/run_visual_demo.py      # the lesson's visuals: plan, search, selection, generation, validation, assets
python scripts/run_presentation_demo.py --out lesson.pptx  # slide plan, validation, build, real .pptx with the images
python scripts/run_audio_demo.py --out-dir narration/      # audio plan, mock TTS, real WAV assets, slide timings
python scripts/run_video_demo.py --out-dir video/          # video plan, FFmpeg composition, a real playable MP4
python scripts/run_provider_demo.py    # provider registry, selection, health, usage, retry, fallback (offline)
python scripts/run_production_demo.py --dry-run  # a production lesson's plan: providers, budget, stages
python -m pytest                    # unit, integration, e2e and architecture-lint tests
uvicorn app.api.main:app --reload   # the same services over HTTP
```

`run_demo.py` prints the task id, final status, every workflow step, the generated artifacts and their
parents, the learner's mastery changes, token usage, and estimated vs actual cost. `run_evaluation_demo.py`
runs that lesson, generates the assessment, shows the task WAITING, submits the fixture learner's answers and
prints the evaluation, mastery before and after, remaining gaps and the next recommendation.
`run_research_demo.py` runs the lesson and shows its research: the request, generated queries, mock search
results (with duplicates), selected and rejected sources, extracted evidence, citations, the stored
ResearchBundle, and how each lesson section's citations resolve to evidence and sources.
`run_visual_demo.py` runs the lesson and shows its visuals: the visual plan and requirements, each image search with
its selected and rejected candidates (one rejected by validation for misreporting its size), generated images, the
validation checks, attribution, the IMAGE_ASSET artifacts with checksums, and image usage.
`run_presentation_demo.py` runs the lesson and shows its presentation: the SlideDeckPlan slide by slide, its
validation, the built presentation with its numbered references, the render events, the stored PRESENTATION artifact
(MIME type, checksum, object key, parents) and the .pptx opened again with python-pptx, with the images on each slide.
`run_audio_demo.py` runs the lesson and shows its narration: the AudioPlan segment by segment, its validation, each
mock TTS call, the audio validation, the AUDIO_ASSET artifacts (each WAV opened with Python's `wave` module), the
PresentationTimeline with every slide's start and end time, the total audio duration and the artifact references.
`run_video_demo.py` runs the lesson through the video stages and shows the PRESENTATION, IMAGE_ASSET and AUDIO_ASSET
artifacts and the PresentationTimeline it starts from, the VideoPlan slide by slide (visual, narration or silence,
subtitles), its validation, the FFmpeg composition, the MP4 validation and the VIDEO artifact, then re-reads the MP4
from the object store and prints its duration (against the timeline), resolution, frame rate, audio stream,
subtitle status and checksum. `--width 1280 --height 720` and `--transition fade` change the output.

`run_provider_demo.py` forces offline mode and the mock providers, then prints the provider configuration
(credentials only as set/missing), every registered provider with its capabilities and health, the selected provider
per capability and per agent, the fallback configuration, one call per capability with its request id and usage, a
retry after rate limiting and an explicit fallback with their `provider.*` events, and a redaction check.

Real providers are opt-in: install the `providers` extra (`pip install -e ".[providers]"`), set
`TEACHING_AGENT_MODE=production` and e.g. `LLM_PROVIDER=anthropic LLM_MODEL=... ANTHROPIC_API_KEY=...`, `TTS_PROVIDER=openai`, `IMAGE_PROVIDER=openai`,
`SEARCH_PROVIDER=tavily` (see [configuration](docs/configuration.md#providers)). The same agents and workflows then
run on them; `run_provider_demo.py --smoke` sends each configured real provider one minimal request.

`run_production_demo.py` runs one complete lesson (B1 Spanish, "Climate change" by default; any level, topic and
language the knowledge base covers) on the real providers: preflight validation and a `--dry-run` that calls nothing,
explicit `--confirm`, provider health checks, a per-task budget enforced before every billable request, resume from
the last checkpoint without regenerating intact artifacts, and a `production_run.json` report with the artifact
graph, per-request traceability, usage and cost. `--mock` rehearses the same path offline. See
[docs/production.md](docs/production.md).

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/tasks` | Create and run a task (`{"request", "learner_id"}`); returns WAITING if the diagnostic needs answers |
| POST | `/tasks/{id}/answers` | Submit answers (`{"answers": [{"question_id", "answer"}]}`) for whatever a WAITING task is waiting on |
| POST | `/tasks/{id}/evaluation` | Start the post-lesson evaluation of a COMPLETED lesson task; returns the new task, WAITING for answers |
| GET | `/tasks/{id}` | Task state, plan, workflow checkpoint, cost, result |
| POST | `/tasks/{id}/pause` · `/resume` · `/cancel` | Task control; resume also recovers crashed or failed tasks |
| GET | `/tasks/{id}/artifacts` · `/events` | Artifacts with parent links; the persisted event log |
| PUT/GET | `/learners/{id}` | Learner profile (subjects with a level framework, preferences) |
| GET | `/learners/{id}/progress` | Mastery, weak concepts, due reviews |
| POST | `/learners/{id}/assessment` | Submit diagnostic answers for a WAITING task |
| GET | `/health` · `/agents` · `/tools` · `/providers` | Introspection |

See [docs/architecture.md](docs/architecture.md) for the design and [docs/configuration.md](docs/configuration.md)
for settings.
