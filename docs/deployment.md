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

`deploy/compose.yaml` is a complete pilot on one machine (PostgreSQL and MinIO beside the app). CI's `deploy` job
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
1 to 7. `python -m app.storage.migrate`:

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
conditional writes: AWS S3 and MinIO have them). The bucket should block public access; the API is the only reader.
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

## Known limits

- One API replica and one worker: the multi-replica items (database-time leases, a shared rate limiter) come after
  the MVP.
- No metrics or alerting yet; `/health` and `check` are the readiness signals.
- The login proxy is the deployment's choice and is not part of the image.
