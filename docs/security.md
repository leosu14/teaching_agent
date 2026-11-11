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
| `learner_ids` | The learner scope: the learners whose data this caller may read and change. A learner has their own id; a guardian may have several. A teacher does not need their class's learners here: rosters give read access (see Teachers, classes and rosters). |
| `roles` | Capabilities beyond the scope. `author`: may register assessment items (with their answer keys and rubrics) on lessons in scope. `teacher`: may create classes and read (never change) the data of the learners on their rosters. |
| `tenant_id` | Scopes classes: a class belongs to its creator's tenant, and rosters and join codes are matched within the caller's tenant only (no tenant matches only classes without one). Other data is not tenant-scoped yet. |

## Authentication

Every route except `GET /health` and the student web app's static files (`GET /app/`, below) requires an
authenticated principal. The routers are mounted behind the authentication dependency in `create_app`, so a new route
is protected unless it is deliberately added to the public router. There is **no anonymous mode**: a request without valid credentials gets `401` with
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
| `X-Auth-Roles` | Comma-separated roles (`author`, `teacher`), or empty |
| `X-Auth-Tenant` | Optional tenant id |
| `X-Auth-Timestamp` | Unix seconds when the gateway signed the request |
| `X-Auth-Signature` | Hex HMAC-SHA256 with the shared secret over `v1\nMETHOD\nPATH\nTIMESTAMP\nSUBJECT\nTENANT\nLEARNERS\nROLES` |

`gateway_headers()` in `app/api/auth.py` is the reference implementation of the gateway's side. The signature binds the
identity to the method, the path and the time, so a captured set of headers cannot be reused on another route or after
the allowed skew. The gateway must strip any `X-Auth-*` headers a client sends, must be the only network path to the
API, and signs the path as the API receives it (after any prefix the gateway removes).

How the gateway maps identity to learner scope is a deployment decision; typical mappings:

- a learner signs in: `X-Auth-Learners` is their own learner id (e.g. a claim, or a lookup of `sub` -> learner id);
- a guardian: the learner ids they are linked to;
- a teacher: the `teacher` role, usually with no learner ids: their classes' rosters, stored by the API, give them
  read access to their learners (below);
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

### The student web app

`/app/` serves the student web app (`app/api/web/`): an HTML page, a stylesheet and one script, with no data in them
(`/app/teacher.html` and `teacher.js`, the teacher dashboard, follow every rule below too).
They are public like `/health` (`app/api/routes/web.py` serves exactly the listed files, never a path from the
request), and everything the page shows comes from the authenticated JSON API, called from the page on the same
origin. `tests/unit/test_student_app.py` checks that these are the only routes outside the OpenAPI schema, so the
security matrix still covers every other route.

- **Sign-in.** Behind the gateway, the browser's session is the gateway's business (a cookie, typically): the page
  calls the API with `credentials: "same-origin"` and the gateway signs each request as for any other client. The
  page starts with `GET /me` (the caller's user id, learner ids and roles) and acts for the one learner in scope (a
  guardian picks one; an account without learners is told so). In `static` development mode the page asks for a
  development token and keeps it in `sessionStorage` (this tab only); a `401` returns to that screen.
- **Headers.** Every file is served with `Content-Security-Policy: default-src 'none'; script-src 'self';
  style-src 'self'; img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; base-uri 'none';
  form-action 'none'; frame-ancestors 'none'`, plus `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` and
  `Referrer-Policy: no-referrer`. No inline script or style, no third-party origin, no `eval`.
- **Rendering.** Model-written text (teacher turns, lesson text, questions) is only ever set with `textContent`; the
  script has no `innerHTML`-style sink (a unit test checks the source). Lesson images are fetched with the learner's
  credentials from `GET /artifacts/{id}/content` and shown from a `blob:` URL.

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

## Teachers, classes and rosters

MVP item 4. A principal with the `teacher` role creates classes; a learner joins one with its join code; the teacher
then **reads, never changes,** the data of every learner on their rosters, through the same learner routes and the
same learner-safe views the learner gets (no answer keys, no internal state).

| Route | Who | What |
|---|---|---|
| `POST /classes` | teacher | A new class of the caller (in the caller's tenant), with its join code. `201` |
| `GET /classes` | teacher | The caller's classes and rosters |
| `GET /classes/{class_id}` | the class's teacher | The class and its roster |
| `POST /classes/{class_id}/join-code` | the class's teacher | A new join code; the old one stops working at once, the roster stays |
| `DELETE /classes/{class_id}/learners/{learner_id}` | the class's teacher, or the learner (their own scope) | Ends the membership and the teacher's access through it. `204` |
| `POST /learners/{learner_id}/classes` | the learner (their own scope) | Join with `{"join_code": ...}`: `201`, or `200` with the same enrollment when already in |
| `GET /learners/{learner_id}/classes` | the learner (their own scope) | The classes they are in: class id and name, not the roster or the teacher |

The rule (`Access.check` in `app/api/authorization.py`), for the learner that owns a resource:

1. In the principal's own scope (`X-Auth-Learners`): read and change, as before.
2. Otherwise, when the principal has the `teacher` role and the learner is on the roster of a class whose
   `teacher_id` is the principal's user id and whose tenant is the principal's: a **read** (`GET` or `HEAD`) is
   allowed; anything else is `403 Forbidden`. The rule is by HTTP method, so a new non-GET route is closed to
   teachers by default. Inline checks of ids in request bodies (`POST /tasks`, attempts, `POST /assessment-items`)
   use the same rule, so they are changes too; the `author` role does not widen it.
3. Otherwise the uniform `404`. A class of another teacher or tenant, a wrong or retired join code, and a learner
   who left all get the same `404` as something that does not exist.

The rosters are looked up only for a learner outside the principal's own scope and only for a teacher, on every
request, so removing a learner, a learner leaving, or the gateway dropping the `teacher` role ends access at once.
A teacher's `403` discloses nothing, because they can already read the resource.

**Consent.** A learner joins with the code; a teacher cannot add a learner. That is why the join route is a change
in the learner's own scope and not a teacher route.

**Join codes** are 16 URL-safe characters (96 random bits from `secrets`), shown only in the `POST /classes` and
rotation responses, and stored only as a SHA-256 hash (`classes.join_code_hash`, unique), so neither the database
nor `GET /classes` reveals a usable code. A join takes the class row by its code hash under a row lock and inserts
the membership in the same transaction, so a join cannot land after the rotation that retired its code; the roster's
primary key makes a join idempotent under races (`tests/postgres/test_classes_postgres.py`).

**Residual points.** GET routes that reconcile state (`GET /learning-cycles/{id}`, next action, the teaching
session view: PR17 M1) do so for a teacher's read too; they never drive a child or call a model, and they store only what the learner's own next read would. `GET /learners/{id}/next-action` stores the curriculum's progress as of its `as_of` query parameter, so only the learner's own scope may pass it: from a teacher it is a `403`. Join codes are
bearer secrets: anyone with a learner's credentials and the code can join that learner, and a leaked code should
be rotated. There is no rate limit on joins at the API (gateway limits apply).

### The teacher dashboard

MVP item 7. Two read-only routes aggregate a class for its teacher; both take the class through `taught_class`, so a
class of another teacher or tenant, a principal without the `teacher` role and a class that does not exist all get the
uniform `404`, and only `GET` is served (any other method is `405`).

| Route | What |
|---|---|
| `GET /classes/{class_id}/dashboard` | Per learner on the roster: display name, mastery numbers and counts (`/progress`), goals with curriculum progress, learning-cycle state, last activity, open UNCERTAIN count |
| `GET /classes/{class_id}/uncertain-grades` | The roster's UNCERTAIN grades: the item's `prompt` and `choices` (what the learner was shown), the learner's own answer, the source, attempt number, time, grader type and confidence, and whether a later graded attempt settled it |

- **Roster only.** The roster is read on every request, like the access rule: a learner who left, or was removed,
  is not read at all. Nothing of a learner outside the roster is read.
- **The same learner-safe information.** The dashboard carries what the learner's own views carry. An UNCERTAIN grade
  is served without the answer key (`expected_answer`, acceptable answers, rubric, known errors) and without
  `uncertainty_reason`, the grader's raw text that can quote the expected answer (the same rule as `PublicGrade`).
  `tests/e2e/test_class_dashboard_api.py` grades an answer on an item whose key is the security matrix's sentinel and
  checks that no response carries it; both routes are in the security matrix.
- **Read-only.** Nothing is reconciled, recorded or graded: the service reads stored records only
  (`tests/integration/test_class_dashboard.py` compares every table before and after). The page itself sends only
  the teacher's own changes (creating a class, rotating its join code); `tests/unit/test_student_app.py` checks its
  script.
- **Residual points.** The learner page of the dashboard uses the learner routes a teacher may already read
  (above); it does not call `GET /learning-cycles/{id}` or the next action, the reads that reconcile or store. A
  teacher sees the learner's own answer text, which the learner wrote to be graded.

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

## Lesson reading

The lesson player of the student app reads `GET /lessons/{lesson_id}/reading` (`app/services/lesson_reading.py`), not
the lesson: the LESSON artifact, the deck, narration and video all stay blocked (above and below). The reading is
built by code, field by field, from the stored lesson: title, level, objective descriptions, introduction, each
section's heading, explanation, examples and analogy, the summary, answer-isolated images (only those the content
policy would serve) and references (http(s) links only). Never the exercises, check questions, narration, plan,
research bundle or review.

Text is checked sentence by sentence (examples, headings, analogies, image descriptions and reference fields as a
whole) with `reveals()` against every answer the learner is graded on with this lesson, and a sentence that states
one is left out:

- always: the lesson's exercise and check-question answers, and the expected and acceptable answers of every
  assessment item registered on the lesson (by its artifact or its task id; the session's graded questions are
  registered there too);
- while a question is open for the learner: the pending question of any of their teaching sessions, and the answer
  keys of any of their tasks waiting on a sheet (diagnostic or evaluation), read from the stored records on every
  request (so any replica sees them). The reading is never an open book on a question being answered.

`withheld` counts what was left out and `question_open` says whether an open question held back more than the
lesson's own keys, so the app can say "come back after answering". A lesson that is not complete is a `409`.
Ownership is the lesson's (a teacher reads their learner's, like every GET). Residual risks: as for images,
`reveals()` matches whole words, so a paraphrase or translation is not caught; a question opened after the learner
loaded the reading does not change a page already shown (the app reloads the reading each time it is opened).

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
  purpose, concept, description, search query, generation prompt or rationale states one. The model writes no
  identifiers: code assigns each visual id (`v<section>_<type>`, both system values; a `visual_id` the model writes
  anyway is dropped unread), so nothing model-written becomes an artifact name, and a rejection names a requirement
  by its position and a field by its name, never by a value the model wrote. The marker is written only when the
  finished plan passes the answer check again over every model-derived field, ids included, and then, per image,
  when its artifact name and every string of its full stored metadata (planned text, provider attribution and
  titles, selection and validation records), not only the allowlisted keys `ArtifactView` serves, pass it too: an
  image whose stored record states an answer is stored unmarked and never served.
  Images made before this change were planned from the whole lesson and carry no marker, so they stay blocked
  (`tests/integration/test_visual_answer_isolation.py`, `tests/e2e/test_image_content.py`).
- Residual risks: `reveals()` matches whole words, so a paraphrase or translation is not caught; a searched image is
  third-party and could show an answer by coincidence; a generated image can render text; image metadata (EXIF,
  PNG text chunks) is not stripped; answers authored after the lesson (evaluations, assessment items) can quote its
  teaching text, which the planner is meant to illustrate; a schema error sent back to the planner can replay the
  model's own rejected value (pydantic's `input_value`), which never reaches a provider or a learner. Images made
  before the fix keep their metadata in the artifact listing, and a re-run that reuses one returns it unmarked
  (blocked, so it fails closed).
- Served content is streamed from `ObjectStore.open` in 64 KiB chunks, never read whole. Responses carry
  `Content-Disposition: attachment; filename="<artifact_id><ext>"`, `X-Content-Type-Options: nosniff`,
  `Cache-Control: no-store` and `Content-Security-Policy: default-src 'none'; sandbox`, and no `Content-Length`,
  `ETag` or `Last-Modified`: like `ArtifactView`, nothing reveals the object's location, digest or byte length.
- A servable artifact whose stored object is missing or unreadable (or whose URI is outside the object store) gets
  the fixed `500 {"error": "ContentUnavailable", "detail": "artifact content is unavailable"}`; the artifact id is
  logged for operators. Others still get the `404`, because ownership and the policy are checked first.
