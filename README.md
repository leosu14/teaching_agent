# Teaching Agent

An agentic, personalised education platform. It assesses what a learner knows, researches the topic
with traceable sources, plans a lesson for that learner, writes and reviews the lesson, plans slides,
stores versioned artifacts, and updates long-term learner memory, so the next lesson adapts.

This first release is the **text-lesson vertical slice**, fully deterministic on mock providers:

```
Request → Task → Request Interpreter → Learner Snapshot → Diagnostic (adaptive, can WAIT for answers)
→ Research → Curriculum Plan → Teacher ⇄ Reviewer (revision loop) → Slide Plan → Artifact Storage
→ Learner Memory → COMPLETED
```

## Run it

```bash
pip install -e ".[dev]"
python scripts/run_demo.py          # one command that proves the slice works
python -m pytest                    # unit, integration, e2e and architecture-lint tests
uvicorn app.api.main:app --reload   # the same services over HTTP
```

`run_demo.py` prints the task id, final status, every workflow step, the generated artifacts and their
parents, the learner's mastery changes, token usage, and estimated vs actual cost.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/tasks` | Create and run a task (`{"request", "learner_id"}`); returns WAITING if the diagnostic needs answers |
| GET | `/tasks/{id}` | Task state, plan, workflow checkpoint, cost, result |
| POST | `/tasks/{id}/pause` · `/resume` · `/cancel` | Task control; resume also recovers crashed or failed tasks |
| GET | `/tasks/{id}/artifacts` · `/events` | Artifacts with parent links; the persisted event log |
| PUT/GET | `/learners/{id}` | Learner profile (subjects with a level framework, preferences) |
| GET | `/learners/{id}/progress` | Mastery, weak concepts, due reviews |
| POST | `/learners/{id}/assessment` | Submit diagnostic answers for a WAITING task |
| GET | `/health` · `/agents` · `/tools` · `/providers` | Introspection |

See [docs/architecture.md](docs/architecture.md) for the design and [docs/configuration.md](docs/configuration.md)
for settings.
