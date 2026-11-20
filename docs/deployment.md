# Deployment

MVP item 8: how the single-teacher pilot runs. One image, run as four processes, on PostgreSQL and an S3-compatible
bucket, behind an identity gateway, with secrets from files and spending capped per day and at every vendor.

```
browser ──> login proxy (optional) ──> identity gateway ──> API ──┐
                                        (verifies the IdP's       ├──> PostgreSQL (schema under migrations)
                                         token, signs identity)   ├──> S3 bucket (artifacts)
                                      worker (lessons, cycles) ───┘
```

## The image

`Dockerfile` builds one image (Python 3.11, FFmpeg, DejaVu fonts, a non-root user). Its commands
(`deploy/entrypoint.sh`):

| Command | Runs |
|---|---|
| `api` | the HTTP API, port 8000 (`uvicorn app.api.main:app`) |
| `worker` | the task worker (`python -m app.services.worker`); one per pilot |
| `gateway` | the identity gateway, port 8080 (`uvicorn app.gateway.main:app`); the only published port |
| `migrate` | the schema to the newest revision, then exit (`python -m app.storage.migrate upgrade`) |
| `check` | the deployment check, then exit (`python -m app.services.deployment check --connect`) |
| `create-bucket` | create `TA_S3_BUCKET` if it does not exist, then exit (for a self-hosted store) |

`deploy/compose.yaml` is a complete pilot on one machine (PostgreSQL and an S3-compatible store, RustFS, beside the app). CI's `deploy` job
builds the image, starts that stack and runs `deploy/smoke.py` through the gateway. A hosted pilot uses the same
image with managed PostgreSQL and a managed bucket.

Release order: build the image, run `migrate`, run `check`, then replace `api`, `worker` and `gateway`.

## The production profile

`TA_DEPLOYMENT=production` makes the API and the worker refuse to start unless every deployment check passes; the
error names each problem. `check` prints the same list (and, with `--connect`, also reaches the database and the
bucket). The checks:

| Check | Setting |
|---|---|
| PostgreSQL, not SQLite | `TA_DATABASE_URL=postgresql+psycopg://...` |
| Schema under migrations | `TA_DATABASE_SCHEMA=migrations` (the default in production) |
| The S3 object store | `TA_OBJECT_STORE=s3` and `TA_S3_*` |
| The signed gateway identity | `TA_AUTH_MODE=gateway`, `TA_AUTH_GATEWAY_SECRET` of 32 characters or more |
| The worker is its own process | `TA_WORKER_EMBEDDED=false` |
| Daily caps on, per learner and overall | `TA_BUDGET_{LEARNER,GLOBAL}_DAILY_MAX_LLM_REQUESTS` and `..._MAX_LLM_TOKENS`, or `..._MAX_COST_USD` with LLM pricing configured |
| A hard cap at every real vendor in use | `TA_VENDOR_CAPS=openai=200,tavily=30` (USD per month) |

Vendor caps are set by the operator in each vendor's console (OpenAI, Anthropic, Tavily, MiniMax usage limits); the
application cannot set them. `TA_VENDOR_CAPS` records what was set, and a deployment whose configuration can call a
real vendor (any capability's provider or fallback, or a per-role LLM route) without a declared cap does not start.
On the offline mock providers there is no vendor and nothing to declare.

## Database and migrations

The schema is versioned with Alembic (`app/storage/migrations/`). Revision `0001_baseline` is the schema of MVP items
1 to 7; `0002_object_holds` adds the object holds and the object-deletion queue of the learner data deletion (MVP
item 9); `0003_teacher_review` adds `classes.lesson_approval` (off) and the `lesson_releases` and `grade_overrides`
tables of teacher review (Important item 2); `0004_class_assignments` adds the `class_assignments` and
`assignment_learners` tables of class assignments (Important item 3; nothing to backfill: existing goals are not
assigned); `0005_teaching_turn_lease` adds the two nullable teaching-turn lease columns to `teaching_sessions`
(issue #66 PR 1; nothing to backfill: no session holds a lease). `python -m app.storage.migrate`:

- `upgrade`: an empty database gets the whole schema; a database at an older revision is upgraded; a database that
  an earlier version made with `create_all` (no `alembic_version` table) is adopted: the tables and the columns of
  `ADDED_COLUMNS` it lacks are added, the result is compared with the baseline, and only an exact match is stamped
  (any other difference is listed and nothing is stamped).
- `check`: exit 1 unless the database is at the newest revision.
- `current`: the database's revision and the newest one.

In `migrations` mode the API and the worker never create or alter tables; on a database that is not at the newest
revision they refuse to start. Development keeps `create` mode (tables made on startup). A schema change ships as a
new revision (`alembic revision --autogenerate` against a database at head, reviewed); the unit and PostgreSQL tests
fail when the ORM and the migrated schema differ.

Backups are the managed database's (point-in-time recovery); restore into a new database and run `check`.

## Object storage

`TA_OBJECT_STORE=s3` stores artifact bytes in a bucket, through `app/providers/storage/s3.py` (AWS Signature
Version 4 over httpx, no SDK). Objects are recorded as store-relative URIs, `object:<key>`, so a row resolves on
any machine and the bucket and prefix can change by copying keys. Rows written before this release hold absolute
`file://` URIs; the filesystem store still reads them under its root.

| Variable | Meaning |
|---|---|
| `TA_S3_BUCKET` | the bucket (it must exist) |
| `TA_S3_REGION` | default `us-east-1` |
| `TA_S3_ENDPOINT_URL` | unset for AWS; the endpoint of MinIO, R2 and the like |
| `TA_S3_PREFIX` | an optional key prefix inside the bucket |
| `TA_S3_PATH_STYLE` | unset: path style for a custom endpoint, virtual-hosted for AWS |
| `TA_S3_ALLOW_HTTP` | plain http, only for a store on a private network |
| `TA_S3_ACCESS_KEY_ID`, `TA_S3_SECRET_ACCESS_KEY`, `TA_S3_SESSION_TOKEN` | the credentials (as secret files) |

New objects are written with `If-None-Match: *`, so an existing object is never replaced (the store needs
conditional writes: AWS S3, MinIO and RustFS have them). The bucket should block public access; the API is the only reader.
Instance-role credentials are not supported yet: use an access key scoped to the bucket.

## Identity

The API trusts only the gateway's HMAC-signed headers (`docs/security.md`). The gateway (`app/gateway/`) sits in
front of it and is the only thing published:

1. It takes the identity provider's token from `Authorization: Bearer` (API clients) or from the header a login
   proxy sets after the browser signs in (`X-Forwarded-Access-Token`, as oauth2-proxy sends it).
2. It verifies the token with the issuer's public keys (`TA_GATEWAY_JWKS_URL`, the issuer's discovery document, or a
   `TA_GATEWAY_JWKS_FILE`): an allowed asymmetric algorithm only (never `none` or HMAC), the issuer, the audience,
   and an unexpired `exp`. Without a valid token it answers 401 and the API is not called.
3. It drops what the client sent that is identity-shaped (`Authorization`, cookies, every `x-auth-*` header, the
   token headers, forwarding headers) and signs the verified identity over the method and the path.

| Variable | Default | Meaning |
|---|---|---|
| `TA_GATEWAY_UPSTREAM_URL` | `http://127.0.0.1:8000` | the API on the private network |
| `TA_GATEWAY_OIDC_ISSUER`, `TA_GATEWAY_OIDC_AUDIENCE` | | the IdP's issuer and the audience of its tokens |
| `TA_GATEWAY_ALGORITHMS` | `RS256,ES256` | allowed signature algorithms (asymmetric only) |
| `TA_GATEWAY_USER_CLAIM` | `sub` | the user id |
| `TA_GATEWAY_LEARNERS_CLAIM` | `ta_learners` | the learner scope (a guardian's children); absent: the user is their own learner (`TA_GATEWAY_SELF_LEARNER`) |
| `TA_GATEWAY_ROLES_CLAIM`, `TA_GATEWAY_ROLE_MAP` | `ta_roles`, `teacher=teacher,author=author` | IdP group or role values mapped to `teacher` and `author`; others are ignored |
| `TA_GATEWAY_TENANT_CLAIM` | unset | a tenant id |

For the pilot, create the teacher and the learners in the IdP (Google Workspace, Microsoft Entra ID, Auth0,
Keycloak, ...), put the teacher in a group mapped to `teacher`, and run a login proxy in front of the gateway for
the browser (for example oauth2-proxy with `--pass-access-token`, or `--set-authorization-header` with an ID token
whose audience is `TA_GATEWAY_OIDC_AUDIENCE`). The proxy must not forward a client-sent `X-Forwarded-Access-Token`;
the gateway verifies the token either way. Teachers' class access is decided by the API from the rosters; the IdP
only says who is a teacher.

## Secrets

Every secret can come from a file: `<NAME>_FILE=/run/secrets/<name>` sets `<NAME>` (`app/config/secrets.py`). This
is how Docker and Compose secrets, Kubernetes secrets and the cloud secret-store CSI drivers (or a Vault agent)
deliver them, so no secret is in the image, the compose file or the environment listing. The variables that may come
from files: `TA_AUTH_GATEWAY_SECRET`, `TA_DATABASE_URL`, `TA_S3_ACCESS_KEY_ID`, `TA_S3_SECRET_ACCESS_KEY`,
`TA_S3_SESSION_TOKEN` and the provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TTS_API_KEY`, `IMAGE_API_KEY`,
`SEARCH_API_KEY`, `TAVILY_API_KEY`, `VIDEO_GENERATION_API_KEY`, `MINIMAX_API_KEY`). Setting a variable both ways,
an empty or unreadable file, and a misspelt secret file variable stop the process.

`deploy/compose.yaml` reads them from `deploy/secrets/` (git-ignored): `database_url`, `postgres_password`,
`s3_access_key_id`, `s3_secret_access_key`, `gateway_secret` (the same file for the API and the gateway).
The gateway, the only published service, gets `gateway_secret` alone: it holds no database or object-store
credential. A static key set for it (`TA_GATEWAY_JWKS_FILE=/run/ta-config/jwks.json`) goes in `deploy/gateway/`
(git-ignored), mounted read-only; it is public key material, never a secret file.

## Learner data and privacy

Learners delete their own data from the student app (`DELETE /learners/{id}`): the rows go from PostgreSQL and the
objects from the bucket, while the spend ledger stays so the daily caps hold. The privacy notice
(`/app/privacy.html`) is part of the image; read it against the deployment and change it where it differs (who runs
the service and a contact, the vendors, backup retention) before going live. Database backups and bucket object
versions are not reached by a deletion. See [privacy](privacy.md).

## Live smoke run

The `live-smoke` workflow (`.github/workflows/live-smoke.yml`, MVP item 10) proves the deployed stack on the real
providers: one lesson queued through the gateway and run by the worker on OpenAI (LLM, TTS, images) and Tavily
(search) to an MP4 that plays (read from the bucket and probed with ffprobe; the API lists it but never serves a
VIDEO's bytes to a learner, since it is rendered from a deck that can hold answer slides, and the run checks that
refusal and that a servable image is served), one teaching-session answer graded, then the learner's data
deleted. It runs `deploy/live_smoke.py`, which reuses `deploy/smoke.py`'s identity provider and checks.

**When it runs.** Only when started by hand (Actions, `live-smoke`, Run workflow, branch `main`), with the input
`max_cost_usd` (default `1.00`, at most `5.00`). Its job runs only on `main` and inside the `staging` environment; it
has no push, pull-request or schedule trigger and is not a required check.

**The temporary stack.** The image is built from the dispatched commit and `deploy/compose.yaml` plus
`deploy/compose.live.yaml` start on the GitHub-hosted runner under the compose project `ta-live`, with
`TA_DEPLOYMENT=production`: PostgreSQL, the S3-compatible store, the migration, the API, the worker and the gateway.
Only the gateway is published, on the runner. Nothing is created outside the runner, and the runner is discarded
after the job.

**The ceiling.** `TA_BUDGET_TOTAL_MAX_COST_USD=max_cost_usd` (see [production](production.md#the-run-wide-cap)): every
paid call of the run is reserved against that one cap before it is sent, and refused when it would pass it; the task,
session, grading and daily USD budgets are set to the same amount, and the unit limits are small (one generated image,
no image search, no generated video). Afterwards the script reads the ledger in PostgreSQL directly and fails unless
the run's total is at or under the ceiling, nothing is left reserved, every paid attempt was charged to the cap and
none was unpriced. The report (`live-smoke-report.json`, a workflow artifact) gives the total and the cost by
capability and source.

**The keys.** They come only from the `staging` environment's secrets, are written to `deploy/secrets/` in a step
that never prints them, and reach the API and the worker as secret files (`OPENAI_API_KEY_FILE`,
`TAVILY_API_KEY_FILE`), never as environment variables; the gateway gets none. The run scans every service's logs, a
dump of the whole database, every object in the bucket, every API answer it received and its own report for each key
(as is, base64 and URL-encoded) before and after the deletion; a hit fails the run naming only the variable and the
place. Failure output is redacted (`***`), and a report holding a key is never written.

**Cleanup.** The learner's data is deleted through the API, then `docker compose down -v --remove-orphans` (with a
by-label fallback) and the key files and generated key set are removed, on success, failure, the script's own
30-minute deadline and SIGINT or SIGTERM (a cancelled run; a further signal during the cleanup does not stop it).
The step's 35-minute timeout is under the job's 45, and `if: always()` steps repeat the teardown
(`live_smoke.py teardown`) and fail the job if any container, volume or network of `ta-live`, or a key file, is left
(`live_smoke.py verify-clean`).

**The rehearsal.** The required `deploy` job runs the same script with `--mock`: the offline mock providers, fake
values in the key files (no real credential is reachable from that job), the mock TTS, images and search priced,
and the same cap, scans and cleanup; then once more with `--fail-after lesson`, which must fail and leave nothing.

**Setting up `staging` (once, by the repository owner):** Settings, Environments, New environment `staging`;
deployment branches: `main` only (a required reviewer is optional). Secrets: `OPENAI_API_KEY`, `TAVILY_API_KEY`.
Variables: the models and billing parameters, their prices (below), and `TA_VENDOR_CAPS` (for example
`openai=20,tavily=10`, the hard caps set in each vendor's console). A missing or invalid price, model, parameter or
vendor cap stops the run before anything starts (exit code 2), and a combination the prices cannot bound stops the
API and the worker at startup (`check` names it).

**Prices.** Every price is USD, copied from the vendor's own price list for exactly the model and parameters below
(OpenAI: https://openai.com/api/pricing, "Text tokens", "Audio" and "Image generation"; Tavily:
https://docs.tavily.com/documentation/api-credits), at or above it; the repository holds no rates.

| Variable | Unit | What it prices |
|---|---|---|
| `LLM_MODEL` | | The OpenAI chat model (base name, e.g. as listed; dated snapshots in replies are priced as it) |
| `LLM_INPUT_PRICE_PER_MTOK` / `LLM_OUTPUT_PRICE_PER_MTOK` | USD per 1M tokens | `LLM_MODEL`'s input (cached input is charged at this price too) and output tokens |
| `TTS_MODEL` | | `tts-1` or `tts-1-hd` (billed per character; a token-billed `gpt-4o*-tts` model is refused) |
| `TTS_PRICE_PER_MCHAR` | USD per 1M characters | `TTS_MODEL`'s speech price; speed and voice do not change it |
| `IMAGE_MODEL`, `IMAGE_QUALITY`, `IMAGE_SIZE` | | e.g. `gpt-image-1` at `low`/`medium`/`high` and `1024x1024`/`1536x1024`/`1024x1536`, or `dall-e-3` at `standard`/`hd` and its sizes; every image is generated at that quality and size |
| `IMAGE_PRICE_PER_IMAGE` | USD per image | The listed price of one image of `IMAGE_MODEL` at `IMAGE_QUALITY` and `IMAGE_SIZE` (for GPT image models, the image output tokens) |
| `IMAGE_INPUT_PRICE_PER_MTOK` | USD per 1M tokens | GPT image models only: the prompt's text input tokens |
| `SEARCH_PRICE_PER_REQUEST` | USD per request | The price of one Tavily credit (the run pins `SEARCH_DEPTH=basic`, one credit per search) |

Set `TTS_MODEL`, `IMAGE_MODEL`, `IMAGE_QUALITY` and `IMAGE_SIZE` before the prices, and price those exact values.

## Progress streams

`GET /tasks/{id}/stream` and `GET /learning-cycles/{id}/stream` are long-lived `text/event-stream` responses
(docs/architecture.md, Progress streams). The gateway streams them as they come (it proxies every response
unbuffered). A login proxy, load balancer or ingress in front of the gateway must not buffer `text/event-stream`
either, and its read timeout must exceed the heartbeat (`TA_STREAM_HEARTBEAT_SECONDS`, 15 s): nginx honours the
API's `X-Accel-Buffering: no`; elsewhere turn response buffering off for these paths. The gateway's own read timeout
(`TA_GATEWAY_TIMEOUT_SECONDS`, 300 s) is well above the heartbeat. A stream asks the client to reconnect after
`TA_STREAM_MAX_SECONDS` (10 minutes), so a token that expires is checked again by the gateway on the next
connection. Open streams are capped per principal (`TA_STREAM_MAX_PER_PRINCIPAL`) and in total
(`TA_STREAM_MAX_STREAMS`), with 429 above either (see "Stream caps" below). `deploy/smoke.py` reads a task's stream
through the gateway to its end while the worker runs the lesson.

### Stream caps

Both caps are counted in the memory of one API process (`StreamSlots` in `app/services/progress_stream.py`): the
per-principal cap and the "global" cap are per process. The pilot runs one API replica as one Uvicorn process
(`deploy/entrypoint.sh`), so today they hold for the whole deployment. Running several Uvicorn workers (`--workers`
or `WEB_CONCURRENCY`) or several API replicas would multiply both limits by the number of processes. Do not run more
than one API process until the caps are counted in a shared store; that belongs with the multi-replica readiness work
below.

## Time and leases

Lease expiry, fencing, the sweeper, spend-reservation expiry and the daily budget's UTC day are judged by the
database's clock on PostgreSQL (`clock_timestamp()`, read in each decision's own transaction; docs/architecture.md,
"Time"), not by the clocks of the API and worker machines. Requirements and failure behaviour:

- **Clocks.** Keep every host NTP-synchronised anyway (log timestamps, `created_at`), but a drifting API or worker
  clock no longer lets two processes hold one lease, abandons a live reservation, or charges a call to the wrong day.
  The database host's clock is the one that counts: it must be NTP-synchronised and must not jump backwards (a managed
  PostgreSQL is). A backwards jump of the database clock lengthens every live lease by that amount; a forward jump
  ends leases early, which the fencing token makes safe (a stale holder's writes are refused) at the cost of the
  repeated work.
- **Database unreachable during a renew.** A task worker's or job's heartbeat that cannot reach the database keeps
  trying on its next beat; if the lease runs out meanwhile, the next holder takes over after expiry and every later
  write of the old holder is refused (`LeaseLost`, nothing stored). A teaching-turn holder whose renew fails tries again
  on its next beat; if its lease runs out and another request takes over, its turns are refused (`TurnLeaseLost`)
  and the request answers 409 `TeachingTurnBusy`; otherwise its turns are stored as usual. A paid call made under a lease that
  was lost is still recorded in the spend ledger (settlement is never fenced).
- **Database unreachable when a decision is made.** No lease is claimed, no fenced write is made and no reservation
  (so no paid call) is made without reading the database's clock: those requests fail as any request does when the
  database is down.
- **SQLite** (development, one machine) keeps the process clock.

## Known limits

- One API replica (one Uvicorn process) and one worker: the remaining multi-replica items (a shared rate limiter,
  shared research cache, shared progress-stream caps, GET routes that do not write; issue #66 PRs 2 and 3) come next.
  Database-time leases and the teaching-turn lease are in place (PR 1).
- No metrics or alerting yet; `/health` and `check` are the readiness signals.
- The login proxy is the deployment's choice and is not part of the image.
