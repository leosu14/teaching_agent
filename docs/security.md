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
before. Responses are built from explicit DTOs instead:

| Route | Served as | Not served |
|---|---|---|
| every `/tasks` route, `POST /learners/{id}/assessment` | `TaskView` | `workflow` (answer keys, every node's output), `plan`, `metadata`, `control`, provider requests and per-model cost lines, artifact storage locations |
| `GET/PUT /learners/{id}` | `LearnerProfileView` | `expected` answers and the feedback that quotes them, in past assessments and mistakes (diagnostic questions recur, so a past key is a future key) |
| task and cycle `/artifacts` | `ArtifactView` | `uri` (a host path), in the artifact and in its metadata |
| teaching sessions, attempts, grades, cycles | already public views (`TeachingSessionView`, `AttemptView`, `PublicGrade`, `LearnerPrompt`) | unchanged |

`POST /assessment-items` returns the registered item, answer key included, to its author (the `author` role is required).

`tests/unit/test_auth.py` walks every response schema in the OpenAPI document for answer-key and internal field
names; `tests/e2e/test_security_api.py` checks the actual responses of every route (including free-form metadata and
events) for those names and for a sentinel answer key.
