# API security: authentication, learner ownership and learner-facing responses

```
External auth gateway (OIDC, SSO, sessions: whatever the deployment uses)
        |  signed identity headers
        v
API  ->  Authenticator  ->  Principal(user_id, learner_ids, roles, tenant_id)
        |
        v
Authorization boundary (app/api/authorization.py): owner of the resource, from stored data, in the principal's scope?
        |
        v
Application services  ->  runtime, agents, tools, providers (unchanged; they never see how the caller authenticated)
```

## Principal

Every protected route receives a `Principal` (`app/api/auth.py`):

| Field | Meaning |
|---|---|
| `user_id` | Who is calling. Recorded as the `user_id` of the tasks, curricula and learning cycles they start. Request bodies can no longer choose it (`user_id` in a body is accepted for compatibility and ignored). |
| `learner_ids` | The learner scope: the learners whose data this caller may read and change. A learner has their own id; a teacher or guardian may have several. |
| `roles` | Capabilities beyond the scope. `author`: may register assessment items (with their answer keys and rubrics) on lessons in scope. |
| `tenant_id` | Carried for the future tenant model; not enforced yet (see Deferred). |

## Authentication

Every route except `GET /health` requires an authenticated principal. The routers are mounted behind the
authentication dependency in `create_app`, so a new route is protected unless it is deliberately added to the public
router. There is **no anonymous mode**: a request without valid credentials gets `401` with
`{"error": "Unauthenticated", "detail": "authentication required"}` and `WWW-Authenticate: Bearer`, whatever the
reason (the reason is not disclosed).

`TA_AUTH_MODE` selects the authenticator:

| Mode | Use | Configuration |
|---|---|---|
| `gateway` (default) | Production | `TA_AUTH_GATEWAY_SECRET` (at least 32 characters, from the secret store), `TA_AUTH_GATEWAY_MAX_SKEW_SECONDS` (default 300) |
| `static` | Local development | `TA_AUTH_STATIC_TOKENS_FILE`: a JSON file of bearer tokens, never committed |

A configuration that cannot authenticate anyone stops the API from starting (`ConfigError` in `create_app`): the
default `gateway` mode without a secret, `static` without a tokens file, a short secret or token. Production cannot
fall back to anonymous access because no such fallback exists.

### Production: the auth gateway boundary

The API does not talk to an identity provider. A gateway in front of it (an API gateway, an ingress auth plugin or a
small backend-for-frontend) authenticates the user with the organisation's identity provider, maps the identity to a
learner scope, and forwards it on every request in these headers:

| Header | Value |
|---|---|
| `X-Auth-Subject` | The user id (e.g. the OIDC `sub`) |
| `X-Auth-Learners` | Comma-separated learner ids the user may act for |
| `X-Auth-Roles` | Comma-separated roles (`author`), or empty |
| `X-Auth-Tenant` | Optional tenant id |
| `X-Auth-Timestamp` | Unix seconds when the gateway signed the request |
| `X-Auth-Signature` | Hex HMAC-SHA256 with the shared secret over `v1\nMETHOD\nPATH\nTIMESTAMP\nSUBJECT\nTENANT\nLEARNERS\nROLES` |

`gateway_headers()` in `app/api/auth.py` is the reference implementation of the gateway's side. The signature binds the
identity to the method, the path and the time, so a captured set of headers cannot be reused on another route or after
the allowed skew. The gateway must strip any `X-Auth-*` headers a client sends, must be the only network path to the
API, and signs the path as the API receives it (after any prefix the gateway removes).

How the gateway maps identity to learner scope is a deployment decision; typical mappings:

- a learner signs in: `X-Auth-Learners` is their own learner id (e.g. a claim, or a lookup of `sub` -> learner id);
- a guardian or teacher: the learner ids they are linked to in the organisation's roster;
- an author (teacher creating assessments): the `author` role plus the learners whose lessons they write for.

Replacing the gateway with direct token verification (e.g. a JWT authenticator) means adding one `Authenticator`
class that returns the same `Principal`; nothing behind the boundary changes.

### Development

Create a tokens file outside the repository and point the API at it:

```bash
cat > ~/.teaching-agent-tokens.json <<'EOF'
{"<a random token of 16+ characters>": {"user_id": "dev", "learner_ids": ["demo-learner"], "roles": ["author"]}}
EOF
TA_AUTH_MODE=static TA_AUTH_STATIC_TOKENS_FILE=~/.teaching-agent-tokens.json uvicorn app.api.main:app --reload
curl -H "Authorization: Bearer <the token>" localhost:8000/learners/demo-learner
```

### Tests

Tests build the app with an injected `StaticTokenAuthenticator` (`tests/auth.py`): `client_for(container, *learner_ids,
roles=...)` returns a client that authenticates as a principal scoped to those learners, and `TestAuth.headers(...)`
registers further principals (deterministic tokens derived from the principal). Nothing in the test helpers is a
credential outside the test process. The demos call the services directly and are unaffected.

## Authorization

Learner-owned resources are checked at one backend boundary, `app/api/authorization.py`, before any service runs.
The owner is read from stored data by `OwnershipService` (`app/services/ownership.py`), never taken from the request:

| Resource | Owner |
|---|---|
| `/learners/{learner_id}/...` | the learner id itself |
| task (lesson, evaluation, curriculum planning) | `task.learner_id` |
| goal, its curriculum and versions | `goal.learner_id` |
| lesson (`/lessons/{id}`: LESSON artifact or lesson task) | the lesson task's learner |
| teaching session (and its turns, evidence, summary) | `session.learner_id` |
| assessment item | the learner whose lesson it was registered on |
| assessment attempt and grade | `attempt.learner_id` |
| learning cycle (and its events and artifacts) | `cycle.learner_id` |
| artifact (`/artifacts/{id}/content`) | the learner of the task, or learning cycle, that stored it |

Ids in request bodies are checked too: `learner_id` on `POST /tasks` and on attempts, `task_id` on
`POST /learners/{id}/assessment`, `lesson_id` and an existing item id on `POST /assessment-items`, and a client-chosen
`attempt_id` that already belongs to another learner. The services' own consistency checks (the item's lesson belongs
to the attempt's learner, a cycle's children belong to its learner) remain as a second line.

A resource outside the principal's scope gets the same response as one that does not exist: `404` with
`{"error": "NotFound", "detail": "not found"}`, so ids cannot be probed. A missing role is `403`. `/agents`, `/tools`
and `/providers` describe the system, not anyone's data: any authenticated principal may read them.

## Learner-facing responses

The internal records are unchanged: tasks keep their workflow state with every node's output (diagnostic items and
evaluation questions with `expected_answer` and `accepted_answers`), assessment items keep their answer keys, rubrics
and known errors, and the learner profile keeps the expected answer of every graded question. Grading uses them as
before, and the event log and `TaskError.message` keep the full diagnostic text for operators (the services'
`events()` and `get()`, and the logs). Responses are built from explicit DTOs instead:

| Route | Served as | Not served |
|---|---|---|
| every `/tasks` route, `POST /learners/{id}/assessment` | `TaskView` | `workflow` (answer keys, every node's output), `plan`, `metadata`, `control`, provider requests and per-model cost lines, artifact storage locations |
| `GET/PUT /learners/{id}` | `LearnerProfileView` | `expected` answers and the feedback that quotes them, in past assessments and mistakes (diagnostic questions recur, so a past key is a future key) |
| task and cycle `/artifacts` | `ArtifactView` | `uri` (a host path) and the content digest and byte length (`content_hash`, `size_bytes`): an artifact can hold an answer key (an open teaching question is stored as an `ASSESSMENT_ITEM` whose other fields the learner already sees), so its hash would check guessed keys offline and its size would give the key's length. Metadata is served by per-type allowlist (`PUBLIC_METADATA` in `app/schemas/artifact.py`), flat values only; a key or a type without an entry is not served. Never on it: free text a model or provider wrote (an `AUDIO_ASSET`'s narration `text`, which reads exercise answers aloud (issue #28); image titles, credits, descriptions, selection and rejection records), ids a model chooses (`visual_id`, section, slide and segment ids, `source_ref`), length signals of narration (clip `duration`, TTS `usage`), and storage keys, digests and byte lengths. The internal `Artifact` keeps all of it |
| `GET /tasks/{id}/events`, `GET /learning-cycles/{id}/events` | `LearnerEvent`: `event_id`, `type`, `status`, a fixed `message` chosen by the status, `task_id`, `node_id`, `at` | the event's `data` (raw exception and validation text, which can quote model output and so answer keys; storage URIs; provider, model, token and cost records), `agent_id`, `tool` |
| `TaskView.errors`, `CurriculumBuild.errors` | `TaskErrorView`: `category`, `stage`, `node_id`, `at` and a fixed message per failure category | the internal `TaskError.message` (raw exception text: a structured-output failure quotes the invalid output, answer keys included) |
| `POST /assessment-items/{id}/attempts`, `GET /assessment-attempts/{id}/grade`, `AnswerResult.assessment` of the teaching `/answers` route | `PublicGrade` | `uncertainty_reason` (the semantic grader's raw error text, which quotes its invalid output and so can quote the item's expected answer; an UNCERTAIN grade's fixed `feedback` asks for another attempt), `matched`, grader usage. Grader-written text it does serve (`feedback`, criterion `rationale`, misconception `type` and `description`) is checked first: see below |
| teaching sessions, attempts, cycles | already public views (`TeachingSessionView`, `AttemptView`, `LearnerPrompt`) | unchanged |

`POST /assessment-items` returns the registered item, answer key included, to its author (the `author` role is required).

A teaching session's own question item (`metadata.session_id`) is answered through the session only:
`POST /assessment-items/{id}/attempts` returns the uniform 404 for it, so the open question cannot be graded outside the
session's hints and penalties (`tests/e2e/test_artifact_answer_oracle.py`).

The semantic grader is given the item's expected and acceptable answers, and the text it writes is served in
`PublicGrade`. The `answer_revealed` validation rule (`app/assessment/validation.py`) applies the teacher's
`reveals()` check to every learner-visible grader text (`feedback.explanation`, `next_hint`, each `strengths` and
`errors` entry, each criterion `rationale`, each misconception `type` and `description`) against the expected and every
acceptable answer, except an answer the learner's own text already says. A leaking candidate is sent back to the grader
naming only the fields; if every attempt leaks, the grade is UNCERTAIN with the fixed feedback. The rule and fields stay
in the internal grade reason and `agent.validation_failed` events. Like the teacher's check, it matches whole words after
normalisation, so a paraphrase is not caught.

The teacher is given the open question's expected and acceptable answers too. While that question is open,
`TeachingSessionAgent.check` applies `reveals()` against all of them to everything of the teacher's served next to it:
the HINT response and each misconception label proposed with it, the UNCERTAIN feedback, the answer to a learner
question (`response` and `limitation`), and the text of the turn that asks a short-answer question. A leaking output
is sent back to the model without the answer; if every attempt leaks, the turn fails with the fixed
`TeacherUnavailable` text and the question stays open (`tests/e2e/test_teacher_answer_leak.py`).

`tests/unit/test_auth.py` walks every response schema in the OpenAPI document for answer-key and internal field
names; `tests/e2e/test_security_api.py` checks the actual responses of every route (including free-form metadata and
events) for those names and for a sentinel answer key. `tests/e2e/test_learner_event_sanitization.py` injects a
structured-output validation failure that quotes a canary answer key and checks that the canary reaches the internal
log and task record but none of the task, cycle or event responses.

## Artifact content

`GET /artifacts/{artifact_id}/content` streams an artifact's stored bytes to the learner who owns it. Every artifact
type can carry answer keys or material built from them (a deck can have answer slides; a lesson's exercises and check
questions hold the answers that grade practice; an open teaching question is an `ASSESSMENT_ITEM`), so the route is
**block-by-default** (`app/services/artifact_content.py`):

- Checks run in order: authentication (the usual `401`), ownership (`OwnershipService.artifact`), then the content
  policy. An unknown id, another learner's artifact and an artifact whose type is not servable all get the same
  `404 {"error": "NotFound", "detail": "not found"}`.
- `SERVABLE` maps a type to the media types it may be served as; anything else is blocked. `BLOCKED` records why
  each other type is not served. Every `ArtifactType` must be in exactly one of them
  (`tests/e2e/test_artifact_content.py`), so a new type is blocked until someone classifies it.
- **Only raster images are servable**: `IMAGE_ASSET` as `image/png`, `image/jpeg`, `image/webp` or `image/gif`, and
  only when its metadata records `planning: "answer-isolated/1"` (`PROVENANCE`). Every other type stays blocked, and
  so does `image/svg+xml` (an SVG is text: it could spell an answer out). Allowing a type is a reviewed change to
  `SERVABLE`, `BLOCKED`, its test and this section.
- Why images are answer-free (issue #26): an image comes from the search query or generation prompt the visual
  planner writes, and those are the only lesson-derived text that reaches image search or generation. The planner's
  model gets an explicit brief (`app/agents/visual/briefing.py`), not the lesson: title, subject, topic, level,
  objectives and the explanation, example and review sections (concept, heading, explanation, examples, analogy)
  behind opaque refs. No exercises or check questions, answers, answer explanations, lesson plan, research bundle,
  narration, introduction, summary or section ids. Any brief field that `reveals()` an exercise or check-question
  answer is dropped (a section whose concept name does is left out), and `VisualAgent.check` sends back a plan whose
  purpose, concept, description, search query, generation prompt or rationale states one, without quoting it.
  Images made before this change were planned from the whole lesson and carry no marker, so they stay blocked
  (`tests/integration/test_visual_answer_isolation.py`, `tests/e2e/test_image_content.py`).
- Residual risks: `reveals()` matches whole words, so a paraphrase or translation is not caught; a searched image is
  third-party and could show an answer by coincidence; a generated image can render text; image metadata (EXIF,
  PNG text chunks) is not stripped; answers authored after the lesson (evaluations, assessment items) can quote its
  teaching text, which the planner is meant to illustrate.
- Served content is streamed from `ObjectStore.open` in 64 KiB chunks, never read whole. Responses carry
  `Content-Disposition: attachment; filename="<artifact_id><ext>"`, `X-Content-Type-Options: nosniff`,
  `Cache-Control: no-store` and `Content-Security-Policy: default-src 'none'; sandbox`, and no `Content-Length`,
  `ETag` or `Last-Modified`: like `ArtifactView`, nothing reveals the object's location, digest or byte length.
- A servable artifact whose stored object is missing or unreadable (or whose URI is outside the object store) gets
  the fixed `500 {"error": "ContentUnavailable", "detail": "artifact content is unavailable"}`; the artifact id is
  logged for operators. Others still get the `404`, because ownership and the policy are checked first.
