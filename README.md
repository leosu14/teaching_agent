# Teaching Agent

An agentic, personalised education platform. It assesses what a learner knows, researches the topic
with traceable sources, plans a lesson for that learner, writes and reviews the lesson, turns it into a
PowerPoint presentation, narrates it with a timeline per slide, stores versioned artifacts, and updates long-term learner memory, so the next lesson adapts.

This release is the **text-lesson vertical slice** with research, visuals, a presentation and its narration, fully deterministic on mock providers:

```
Request → Task → Request Interpreter → Learner Snapshot → Diagnostic (adaptive, can WAIT for answers)
→ Research (queries → search → dedup/rank → evidence → cited ResearchBundle) → Curriculum Plan → Teacher ⇄ Reviewer (revision loop)
→ Visual (approved lessons only: visual plan → image search / generation → selection → validation → IMAGE_ASSET) → Artifact Storage
→ Slide Planning → SlideDeckPlan validation → Presentation Build → Presentation Render (approved lessons only: PPTX → PRESENTATION)
→ Audio Planning → AudioPlan validation → TTS → audio validation → AUDIO_ASSET → PresentationTimeline (rendered presentations only)
→ Learner Memory → COMPLETED
```

A completed lesson can then be evaluated (the post-lesson learning loop):

```
Completed lesson → Learner Evaluation (assessment) → WAITING for answers → Learner Evaluation (grading)
→ Learner Memory (mastery) → remaining gaps + next-learning recommendation → LEARNER_EVALUATION artifact → COMPLETED
```

## Run it

```bash
pip install -e ".[dev]"
python scripts/run_demo.py          # one command that proves the slice works
python scripts/run_evaluation_demo.py  # lesson, then its evaluation, mastery before/after, recommendation
python scripts/run_research_demo.py    # the lesson's research: queries, results, sources, evidence, citations
python scripts/run_visual_demo.py      # the lesson's visuals: plan, search, selection, generation, validation, assets
python scripts/run_presentation_demo.py --out lesson.pptx  # slide plan, validation, build, real .pptx with the images
python scripts/run_audio_demo.py --out-dir narration/      # audio plan, mock TTS, real WAV assets, slide timings
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
