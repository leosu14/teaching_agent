"""Live end-to-end smoke run of the deployed stack on real providers, with one run-wide dollar ceiling (MVP item 10).

    python deploy/live_smoke.py --max-cost 1.00              # live: OpenAI and Tavily, keys in deploy/secrets/
    python deploy/live_smoke.py --mock --max-cost 2.00       # the rehearsal: mock providers, fake keys, nothing paid
    python deploy/live_smoke.py --mock --fail-after lesson   # the rehearsal of a failed run (must exit non-zero)
    python deploy/live_smoke.py teardown                     # remove whatever a run left (the workflow's always step)
    python deploy/live_smoke.py verify-clean                 # nothing of the run is left (containers, volumes, keys)

It starts a temporary stack (deploy/compose.yaml plus deploy/compose.live.yaml, compose project `ta-live`) built from
the checked-out commit, with TA_DEPLOYMENT=production: PostgreSQL, an S3-compatible store, the migration, the API,
the worker and the gateway. Everything goes through the published gateway, with tokens this script signs as the
identity provider (deploy/smoke.py's). A teacher creates a class that requires lesson approval and the learner joins
it; the learner's one lesson is queued, run by the worker and completed, and is then held: the learner cannot read
it, open a session on it or fetch its images (409). The class's teacher lists it, previews it, fetches its images
and approves it through the teacher routes (another teacher and the learner cannot); only then does the learner read
it. The same lesson is used throughout, and the review stage makes no paid call (checked in the ledger). Its MP4 is
read from the bucket and probed (the API refuses a VIDEO's bytes to a learner, which is checked); one
teaching-session answer is graded; the learner's progress and the teacher's dashboard show the lesson; then the
learner's data is deleted. Afterwards it checks the spend ledger in PostgreSQL directly (the run-wide total at or
under the ceiling, nothing left reserved, every paid attempt priced), scans the logs, the database, the bucket, the
API's answers, its own report and the review bundle for the provider keys, and tears the stack down.

The lesson outlives the stack for one human reviewer: the teacher's preview, the learner's reading, the release
decision and the images are sealed (deploy/review_bundle.py) to the reviewer's public key (REVIEW_PUBLIC_KEY) and
written to live-smoke-review.enc.json, which the workflow keeps as an artifact for 14 days. The report records each
stage's start, end and duration, and the lesson's cost from the ledger, split into what was measured from the
providers' usage and what was charged at the pre-send estimate.

The ceiling is enforced by the application before each paid call: TA_BUDGET_TOTAL_MAX_COST_USD=--max-cost (one
scope over all work that never resets), with a price for every paid capability (the API and the worker refuse to
start otherwise). Cleanup runs on success, failure, its own deadline (--deadline-minutes) and SIGINT/SIGTERM (a
cancelled GitHub Actions run); the workflow's `if: always()` steps repeat it and then run `verify-clean`.

The provider keys are read from deploy/secrets/openai_api_key and deploy/secrets/tavily_api_key (the workflow writes
them from the `staging` environment's secrets). They are never printed: failure output is redacted, and a key found
anywhere fails the run naming only the variable and the place. The rehearsal writes fake keys to the same files.

Exit codes: 0 passed, 1 the run failed, 2 configuration problem (nothing was started, nothing was paid).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import secrets
import shutil
import signal
import subprocess
import sys
import traceback
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, quote_plus

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_bundle  # noqa: E402 - deploy/review_bundle.py: the sealed lesson for the human reviewer
import smoke  # noqa: E402 - deploy/smoke.py: the identity provider, compose, SQL and bucket helpers

ROOT = smoke.ROOT
DEPLOY = smoke.DEPLOY
SECRETS = smoke.SECRETS
PROJECT = "ta-live"
LIVE_OVERRIDE = DEPLOY / "compose.live.yaml"
REPORT = ROOT / "live-smoke-report.json"
# The lesson for the human reviewer, sealed to REVIEW_PUBLIC_KEY (deploy/review_bundle.py). Kept after the run (the
# workflow uploads it); `teardown` and `verify-clean` leave it, since it holds no key and opens only for the reviewer.
REVIEW_BUNDLE = ROOT / "live-smoke-review.enc.json"
REVIEW_KEY_VAR = "REVIEW_PUBLIC_KEY"
LIVE_CLASS = "Live smoke class (lesson approval)"
IMAGE_SUFFIXES = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
MAX_CEILING_USD = 5.00  # a dispatch can never ask for more than this
KEY_FILES = {"OPENAI_API_KEY": "openai_api_key", "TAVILY_API_KEY": "tavily_api_key"}
# The live run's lesson: one unit of the pilot course (TA_COURSES=spanish-a1-b1).
LIVE_COURSE = "spanish-a1-b1"
LIVE_PROFILE = {
    "display_name": "Live smoke learner",
    "subjects": [{"subject": "spanish", "framework_id": "cefr", "target_level": "A1", "estimated_level": "A1",
                  "objectives": ["Greet people and introduce yourself"]}],
    "preferences": {"language_of_instruction": "en", "session_minutes": 15, "explanation_style": "examples-first"},
}
LIVE_REQUEST = "Create a short A1 Spanish lesson about greetings and introductions."
# The prices and caps a live run needs, from the `staging` environment's variables (not secrets).
LIVE_PRICE_VARS = ("LLM_INPUT_PRICE_PER_MTOK", "LLM_OUTPUT_PRICE_PER_MTOK", "TTS_PRICE_PER_MCHAR",
                   "IMAGE_PRICE_PER_IMAGE", "SEARCH_PRICE_PER_REQUEST")
# The billing parameters a live run pins rather than leaving to a vendor default (app/config/billing.py checks the
# combination at startup and refuses one the prices cannot bound).
LIVE_BILLING_VARS = ("TTS_MODEL", "IMAGE_MODEL", "IMAGE_QUALITY", "IMAGE_SIZE")
# The rehearsal prices its mock TTS, images and search too (the mock LLM models are priced in config/routing.toml),
# so every paid capability is charged to the run-wide cap exactly as in a live run. Arbitrary test values, not any
# vendor's rates: the live run's prices come from the `staging` environment (docs/deployment.md).
MOCK_PRICES = {"TTS_PRICE_PER_MCHAR": "15", "IMAGE_PRICE_PER_IMAGE": "0.04", "SEARCH_PRICE_PER_REQUEST": "0.008"}
# Field names that hold an answer key in the application's internal schemas: never in a learner's response.
ANSWER_KEY_FIELDS = {"expected_answer", "accepted_answers", "answer_key", "correct_answer"}
STAGES = ("stack", "lesson", "review", "grade")


class ConfigProblem(Exception):
    """The run cannot start; nothing was started and nothing was paid."""


class Interrupted(BaseException):
    """SIGINT, SIGTERM or SIGHUP (a cancelled workflow), or the run's own deadline."""


class Leak(AssertionError):
    """A provider key was found somewhere it must never be."""


# --- configuration -------------------------------------------------------------------------------------------------


def ceiling(value: str) -> float:
    try:
        amount = float(value)
    except ValueError:
        raise ConfigProblem(f"--max-cost {value!r} is not a number") from None
    if not math.isfinite(amount) or amount <= 0 or amount > MAX_CEILING_USD:
        raise ConfigProblem(f"--max-cost must be above 0 and at most {MAX_CEILING_USD:.2f} USD (got {value!r})")
    return round(amount, 6)


def positive_price(environ: dict[str, str], name: str) -> str | None:
    """None when the variable holds a finite, positive number; else what is wrong with it."""
    raw = environ.get(name, "").strip()
    if not raw:
        return f"{name} is not set"
    try:
        value = float(raw)
    except ValueError:
        return f"{name}={raw!r} is not a number"
    if not math.isfinite(value) or value <= 0:
        return f"{name}={raw!r} must be a positive price in USD"
    return None


def stack_environment(*, mock: bool, max_cost: float, environ: dict[str, str]) -> dict[str, str]:
    """The non-secret configuration of the stack. Fails closed (ConfigProblem) on a missing or invalid price, model
    or vendor cap: the application refuses an unpriced capability as well, this only fails before anything starts."""
    env = {"TA_BUDGET_TOTAL_MAX_COST_USD": f"{max_cost:g}",
           # daily unit caps stay on (the production profile requires them); the USD caps are the ceiling
           "TA_BUDGET_LEARNER_DAILY_MAX_LLM_REQUESTS": "300", "TA_BUDGET_LEARNER_DAILY_MAX_LLM_TOKENS": "1500000",
           "TA_BUDGET_GLOBAL_DAILY_MAX_LLM_REQUESTS": "5000", "TA_BUDGET_GLOBAL_DAILY_MAX_LLM_TOKENS": "20000000"}
    if mock:
        return {**env, "TEACHING_AGENT_MODE": "offline", "TA_COURSES": "", "TA_VENDOR_CAPS": "",
                "LLM_PROVIDER": "", "LLM_MODEL": "", "LLM_INPUT_PRICE_PER_MTOK": "", "LLM_OUTPUT_PRICE_PER_MTOK": "",
                "TTS_PROVIDER": "mock", "IMAGE_PROVIDER": "mock", "SEARCH_PROVIDER": "mock", "TTS_MODEL": "",
                "IMAGE_MODEL": "", "IMAGE_QUALITY": "", "IMAGE_SIZE": "", "IMAGE_INPUT_PRICE_PER_MTOK": "",
                **MOCK_PRICES}
    problems = [p for p in (positive_price(environ, v) for v in LIVE_PRICE_VARS) if p]
    if not environ.get("LLM_MODEL", "").strip():
        problems.append("LLM_MODEL is not set (the OpenAI model the prices are for)")
    for var in LIVE_BILLING_VARS:
        if not environ.get(var, "").strip():
            problems.append(f"{var} is not set (it fixes what a request is billed; the prices are for it)")
    image_input = None
    if environ.get("IMAGE_MODEL", "").strip().startswith("gpt-image"):
        image_input = positive_price(environ, "IMAGE_INPUT_PRICE_PER_MTOK")
        if image_input:
            problems.append(image_input + " (a GPT image model bills the prompt's input tokens too)")
    caps = environ.get("TA_VENDOR_CAPS", "")
    declared = {e.partition("=")[0].strip().lower() for e in caps.split(",") if "=" in e}
    for vendor in ("openai", "tavily"):
        if vendor not in declared:
            problems.append(f"TA_VENDOR_CAPS must declare the cap set in the {vendor} console (vendor=USD per month)")
    if problems:
        raise ConfigProblem("the live run is not configured:\n  - " + "\n  - ".join(problems))
    return {**env, "TEACHING_AGENT_MODE": "production", "TA_COURSES": LIVE_COURSE, "TA_VENDOR_CAPS": caps,
            "LLM_PROVIDER": "openai", "LLM_MODEL": environ["LLM_MODEL"].strip(),
            "TTS_PROVIDER": "openai", "IMAGE_PROVIDER": "openai", "SEARCH_PROVIDER": "tavily",
            "IMAGE_INPUT_PRICE_PER_MTOK": environ.get("IMAGE_INPUT_PRICE_PER_MTOK", "").strip(),
            **{v: environ[v].strip() for v in (*LIVE_PRICE_VARS, *LIVE_BILLING_VARS)}}


def review_recipient(*, mock: bool, environ: dict[str, str]):
    """(public key, private key or None) for the review bundle. Live: the reviewer's RSA public key from
    REVIEW_PUBLIC_KEY (a `staging` variable; a public key is not a secret), checked before anything starts, so a
    missing or weak key fails free (exit 2). The rehearsal makes a throwaway key pair, so it can prove the bundle it
    sealed opens; the private half lives only in this process."""
    if mock:
        from cryptography.hazmat.primitives.asymmetric import rsa

        private = rsa.generate_private_key(public_exponent=65537, key_size=review_bundle.MIN_KEY_BITS)
        return private.public_key(), private
    try:
        return review_bundle.load_public_key(environ.get(REVIEW_KEY_VAR, "")), None
    except review_bundle.BundleError as exc:
        raise ConfigProblem(f"the live run is not configured:\n  - {exc}") from None


# --- the provider keys: never printed, searched for everywhere --------------------------------------------------------


@dataclass
class Keys:
    values: dict[str, str] = field(default_factory=dict)  # variable -> key

    @classmethod
    def load(cls, *, mock: bool) -> Keys:
        """The live keys from deploy/secrets/ (written by the workflow), or fake ones the rehearsal writes there."""
        SECRETS.mkdir(parents=True, exist_ok=True)
        values = {}
        for var, name in KEY_FILES.items():
            path = SECRETS / name
            if mock:
                path.write_text(f"fake-{name}-{secrets.token_urlsafe(24)}", encoding="utf-8")
                path.chmod(0o644)  # read by the containers' non-root users
            if not path.is_file() or not path.read_text(encoding="utf-8").strip():
                raise ConfigProblem(f"{var}: deploy/secrets/{name} is missing or empty (a secret of the `staging` "
                                    "environment)")
            values[var] = path.read_text(encoding="utf-8").strip()
            if len(values[var]) < 8:
                raise ConfigProblem(f"{var}: deploy/secrets/{name} is too short to be a key")
        return cls(values)

    @staticmethod
    def forms(value: str) -> set[str]:
        """The ways a key could be written down: as is, base64 (standard and URL-safe, with or without padding) and
        URL-encoded."""
        raw = value.encode("utf-8")
        out = {value, quote(value, safe=""), quote_plus(value)}
        for encoded in (base64.b64encode(raw).decode(), base64.urlsafe_b64encode(raw).decode()):
            out |= {encoded, encoded.rstrip("=")}
        return {f for f in out if f}

    def scan(self, text: str) -> list[str]:
        """The variables whose key appears in `text`, in any of its forms."""
        return sorted(var for var, value in self.values.items() if any(f in text for f in self.forms(value)))

    def redact(self, text: str, extra: tuple[str, ...] = ()) -> str:
        forms = {f for v in (*self.values.values(), *extra) if v for f in self.forms(v)}
        for f in sorted(forms, key=len, reverse=True):
            text = text.replace(f, "***")
        return text


# --- the run's state and cleanup -------------------------------------------------------------------------------------


@dataclass
class Run:
    mock: bool
    max_cost: float
    fail_after: str | None = None
    build: bool = True  # build the image from the checkout (--no-build: use teaching-agent:local as it is)
    keys: Keys | None = None
    started: bool = False  # compose up was attempted: the stack must come down
    learner: str | None = None
    learner_headers: Mapping[str, str] | None = None
    deleted: bool = False
    cleaning: bool = False
    responses: list[str] = field(default_factory=list)  # every API answer body, scanned for the keys
    leaks: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    cleanup_errors: list[str] = field(default_factory=list)
    review_key: object = None  # the reviewer's RSA public key: the review bundle is sealed to it
    review_private_key: object = None  # the rehearsal's own throwaway key pair, to prove the bundle opens; never live
    stages: dict = field(default_factory=dict)  # stage -> {"started_at", "finished_at", "seconds"}
    bundle: str | None = None  # the review bundle's file name, once written


class Bearer(Mapping):
    """Request headers whose `Authorization` is a token signed afresh each time httpx reads it, so every request of
    the run carries an unexpired identity, however long the lesson takes: the IdP's tokens live 10 minutes
    (deploy/smoke.py, `token`), a real lesson can take longer, and polling, grading and the deletion (also cleanup's)
    must never get a 401 for it."""

    def __init__(self, key, subject: str, roles: list[str] | None = None) -> None:
        self._key, self._subject, self._roles = key, subject, roles

    def __getitem__(self, name: str) -> str:
        if name != "Authorization":
            raise KeyError(name)
        return f"Bearer {smoke.token(self._key, self._subject, roles=self._roles)}"

    def __iter__(self):
        return iter(("Authorization",))

    def __len__(self) -> int:
        return 1


def compose_command() -> list[str]:
    return ["docker", "compose", "-f", str(DEPLOY / "compose.yaml"), "-f", str(LIVE_OVERRIDE), "-p", PROJECT]


def on_signal(run: Run):
    def handler(signum, _frame) -> None:
        if run.cleaning:  # a second signal while cleaning up: finish the cleanup (the workflow's step repeats it)
            print(f"live smoke: signal {signum} during cleanup, continuing the cleanup", file=sys.stderr)
            return
        raise Interrupted(f"signal {signum}")
    return handler


def on_deadline(run: Run):
    def handler(_signum, _frame) -> None:
        if not run.cleaning:
            raise Interrupted("the run's deadline passed")
    return handler


def install_handlers(run: Run, deadline_minutes: float) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal(run))
    signal.signal(signal.SIGALRM, on_deadline(run))
    signal.alarm(max(1, int(deadline_minutes * 60)))


def cleanup(run: Run, *, keep_stack: bool = False) -> None:
    """Delete the learner's data (when the stack is up and they exist), take the stack and its volumes down, and
    remove the key files and generated configuration. Every step runs whatever the previous one did; failures are
    collected and reported, never raised."""
    run.cleaning = True
    signal.alarm(0)
    if run.started and run.learner and not run.deleted:
        try:
            with smoke.httpx.Client(base_url=f"http://127.0.0.1:{port()}", timeout=60) as c:
                r = c.delete(f"/learners/{run.learner}", headers=run.learner_headers)
                if r.status_code not in (200, 404):
                    run.cleanup_errors.append(f"deleting the learner's data answered {r.status_code}")
                run.deleted = True
        except Exception as exc:  # noqa: BLE001 - the stack and its volumes still go below
            run.cleanup_errors.append(f"deleting the learner's data failed ({type(exc).__name__})")
    if run.started and not keep_stack:
        try:
            done = smoke.compose("down", "-v", "--remove-orphans", "--timeout", "10", check=False, capture=True)
            if done.returncode != 0:
                run.cleanup_errors.append(f"docker compose down exited {done.returncode}")
        except Exception as exc:  # noqa: BLE001
            run.cleanup_errors.append(f"docker compose down failed ({type(exc).__name__})")
        remove_by_label(run.cleanup_errors)  # whatever `down` could not remove
    remove_generated_files(run.cleanup_errors)


def remove_by_label(errors: list[str]) -> None:
    """Remove every container, volume and network of the compose project by its label, without the compose files
    (which need the run's configuration to load): the fallback of `down`, and the workflow's `teardown` step."""
    label = f"label=com.docker.compose.project={PROJECT}"
    for kind, listing, removal in (("container", ["docker", "ps", "-aq", "--filter", label], ["docker", "rm", "-f"]),
                                   ("volume", ["docker", "volume", "ls", "-q", "--filter", label],
                                    ["docker", "volume", "rm", "-f"]),
                                   ("network", ["docker", "network", "ls", "-q", "--filter", label],
                                    ["docker", "network", "rm"])):
        try:
            ids = subprocess.run(listing, capture_output=True, text=True, check=False, timeout=60).stdout.split()
            if ids:
                done = subprocess.run([*removal, *ids], capture_output=True, text=True, check=False, timeout=120)
                if done.returncode != 0:
                    errors.append(f"removing the run's {kind}s exited {done.returncode}")
        except Exception as exc:  # noqa: BLE001 - the next kind is still removed
            errors.append(f"removing the run's {kind}s failed ({type(exc).__name__})")


def remove_generated_files(errors: list[str]) -> None:
    for path in (SECRETS, smoke.GATEWAY_CONFIG / "jwks.json"):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except OSError as exc:
            errors.append(f"removing {path.name} failed ({type(exc).__name__})")


def port() -> str:
    return os.environ.get("SMOKE_PORT", "8080")


# --- checks ------------------------------------------------------------------------------------------------------------


def answer_key_fields(value: object) -> set[str]:
    """Every answer-key field name anywhere in a JSON value."""
    found: set[str] = set()
    if isinstance(value, dict):
        found |= {k for k in value if k in ANSWER_KEY_FIELDS}
        for v in value.values():
            found |= answer_key_fields(v)
    elif isinstance(value, list):
        for v in value:
            found |= answer_key_fields(v)
    return found


def ledger(max_cost: float) -> dict:
    """The run-wide ledger from PostgreSQL: the total charged to the cap, what is still reserved, and the paid
    attempts by capability and cost source. Ids and numbers only."""
    counters = {m: (int(s), int(r)) for m, s, r in (
        line.split("|") for line in smoke.sql(
            "SELECT metric, settled, reserved FROM spend_counters WHERE scope_key = 'total:all';"))}
    settled, reserved = counters.get("cost_microusd", (0, 0))
    rows = [line.split("|") for line in smoke.sql(
        "SELECT capability, cost_source, count(*), coalesce(sum(final_cost_microusd), 0), "
        "coalesce(sum(estimated_cost_microusd), 0) FROM spend_entries "
        "WHERE state IN ('settled', 'unknown') GROUP BY 1, 2 ORDER BY 1, 2;")]
    left = int(smoke.sql("SELECT count(*) FROM spend_entries WHERE state = 'reserved';")[0])
    outside = int(smoke.sql("SELECT count(*) FROM spend_entries WHERE state IN ('settled', 'unknown') "
                            "AND scope_keys NOT LIKE '%\"total:all\"%';")[0])
    by_capability: dict[str, dict] = {}
    for capability, source, count, cost, reserved_cost in rows:
        entry = by_capability.setdefault(capability, {"attempts": 0, "cost_usd": 0.0, "reserved_usd": 0.0,
                                                      "reconciled_usd": 0.0, "sources": {}})
        entry["attempts"] += int(count)
        entry["cost_usd"] = round(entry["cost_usd"] + int(cost) / 1_000_000, 6)
        # reserved before dispatch (the upper bound) versus charged: from the usage the vendor reported ("priced",
        # "reported"), or the reservation itself ("estimate": no usable usage came back)
        entry["reserved_usd"] = round(entry["reserved_usd"] + int(reserved_cost) / 1_000_000, 6)
        if source in ("priced", "reported"):
            entry["reconciled_usd"] = round(entry["reconciled_usd"] + int(cost) / 1_000_000, 6)
        entry["sources"][source] = entry["sources"].get(source, 0) + int(count)
    return {"ceiling_usd": max_cost, "total_usd": round(settled / 1_000_000, 6),
            "reserved_usd": round(reserved / 1_000_000, 6), "entries_still_reserved": left,
            "attempts_outside_the_cap": outside, "by_capability": by_capability,
            "unpriced_attempts": sum(e["sources"].get("unpriced", 0) for e in by_capability.values())}


def assert_ledger(spend: dict, *, mock: bool) -> None:
    assert spend["entries_still_reserved"] == 0, spend
    assert spend["attempts_outside_the_cap"] == 0, spend
    assert spend["unpriced_attempts"] == 0, spend
    assert spend["total_usd"] <= spend["ceiling_usd"] + 1e-9, spend
    assert spend["total_usd"] > 0, spend  # paid work was charged to the cap
    expected = {"llm", "tts", "image", "search"}
    assert expected <= set(spend["by_capability"]), spend
    # LLM usage is reconciled from the reported tokens (priced as the configured model, also when the reply names a
    # dated snapshot of it): an attempt left at its upper-bound reservation is only one whose usage never came back.
    llm = spend["by_capability"]["llm"]["sources"]
    assert llm.get("priced", 0) > 0 and llm.get("priced", 0) >= llm.get("estimate", 0), spend
    if mock:  # the rehearsal's TTS, image and search are priced like a live run's
        for capability in ("tts", "image", "search"):
            assert spend["by_capability"][capability]["cost_usd"] > 0, spend


def bucket_leaks(keys: Keys) -> list[str]:
    """The bucket's objects scanned for the keys inside the API container (which holds the store's credentials;
    this script does not). Only object keys and variable names come back, never a key value."""
    forms = {var: sorted(keys.forms(value)) for var, value in keys.values.items()}
    script = smoke.BUCKET_KEYS.replace("    for key in root.iter(f\"{ns}Key\"):\n        print(key.text)\n",
                                       "    for key in root.iter(f\"{ns}Key\"):\n        found.append(key.text)\n")
    script = "found = []\n" + script + f"""
import json as _json
forms = _json.loads({json.dumps(json.dumps(forms))})
for name in found:
    body = store.get_key(name) or b""
    for var, fs in forms.items():
        if any(f.encode() in body for f in fs):
            print(f"LEAK {{var}} {{name}}")
print(f"SCANNED {{len(found)}}")
"""
    out = smoke.compose("exec", "-T", "api", "python", "-", stdin=script, capture=True).stdout
    assert "SCANNED" in out, "the bucket scan did not run"
    return [line.removeprefix("LEAK ") for line in out.splitlines() if line.startswith("LEAK ")]


def scan_everything(run: Run, when: str) -> None:
    """The keys must appear nowhere: every service's logs, the whole database, the bucket and the API's answers."""
    assert run.keys is not None
    logs = smoke.compose("logs", "--no-color", "--timestamps", check=False, capture=True)
    places = {"service logs": logs.stdout + logs.stderr,
              "database": smoke.compose("exec", "-T", "postgres", "pg_dump", "-U", "ta", "-d", "teaching_agent",
                                        "--data-only", capture=True).stdout,
              "API responses": "\n".join(run.responses)}
    for place, text in places.items():
        for var in run.keys.scan(text):
            run.leaks.append(f"{var} in the {place} ({when})")
    for hit in bucket_leaks(run.keys):
        var, _, name = hit.partition(" ")
        run.leaks.append(f"{var} in the bucket object {name} ({when})")


def record(run: Run, response) -> dict | list:
    run.responses.append(response.text)
    return response.json()


VIDEO_COPY = "/tmp/live-smoke-lesson.mp4"


def video_duration(artifact_id: str) -> float:
    """The lesson's MP4, streamed from the bucket inside the API container (which holds the store's credentials) and
    probed with ffprobe there: its duration in seconds, 0 if it does not play. The copy is removed afterwards."""
    uri = smoke.sql(f"SELECT body::json->>'uri' FROM artifacts WHERE artifact_id = {smoke.literal(artifact_id)};")
    assert uri and uri[0], f"no stored object for the VIDEO {artifact_id}"
    script = f"""
import shutil
from app.config.secrets import load_secret_files
from app.config.settings import Settings
from app.providers.storage.s3 import S3ObjectStore
load_secret_files()
store = S3ObjectStore(Settings().s3_config())
with store.open({uri[0]!r}) as src, open({VIDEO_COPY!r}, "wb") as dst:
    shutil.copyfileobj(src, dst)
print("COPIED")
"""
    try:
        out = smoke.compose("exec", "-T", "api", "python", "-", stdin=script, capture=True).stdout
        assert "COPIED" in out, "the VIDEO could not be read from the bucket"
        probed = smoke.compose("exec", "-T", "api", "ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "default=noprint_wrappers=1:nokey=1", VIDEO_COPY, capture=True).stdout.strip()
        return float(probed or 0)
    finally:
        smoke.compose("exec", "-T", "api", "rm", "-f", VIDEO_COPY, check=False)


def now_iso(at: float) -> str:
    return datetime.fromtimestamp(at, tz=timezone.utc).isoformat(timespec="seconds")


@contextmanager
def timed(run: Run, stage: str):
    """Record a stage's wall-clock start, end and duration in the report (also when it fails: then no end)."""
    started = smoke.time.time()
    run.stages[stage] = {"started_at": now_iso(started)}
    yield
    finished = smoke.time.time()
    run.stages[stage].update(finished_at=now_iso(finished), seconds=round(finished - started, 1))


def paid_attempts() -> int:
    """Every billable provider attempt the ledger holds (reserved, settled, unknown or failed): a stage that must make
    no paid call leaves this unchanged."""
    return int(smoke.sql("SELECT count(*) FROM spend_entries;")[0])


def cost_of(where: str) -> dict:
    """What the ledger charged for some attempts, by how the amount is known. `measured_usd`: from the usage the
    provider returned (its reported tokens, the characters, images or searches actually billed) at the configured
    prices, or a cost the provider reported itself; `estimate_usd`: no usable usage came back, so the pre-send
    reservation was charged. Neither is the vendor's invoice: compare with the vendor dashboards by hand."""
    rows = [line.split("|") for line in smoke.sql(
        "SELECT cost_source, count(*), coalesce(sum(final_cost_microusd), 0) FROM spend_entries "
        f"WHERE state IN ('settled', 'unknown') AND {where} GROUP BY 1 ORDER BY 1;")]
    failed = int(smoke.sql(f"SELECT count(*) FROM spend_entries WHERE state = 'failed' AND {where};")[0])
    out = {"attempts": 0, "charged_usd": 0.0, "measured_usd": 0.0, "estimate_usd": 0.0, "failed_attempts": failed,
           "sources": {}}
    for source, count, cost in rows:
        usd = int(cost) / 1_000_000
        out["attempts"] += int(count)
        out["charged_usd"] = round(out["charged_usd"] + usd, 6)
        if source in ("priced", "reported"):
            out["measured_usd"] = round(out["measured_usd"] + usd, 6)
        elif source == "estimate":
            out["estimate_usd"] = round(out["estimate_usd"] + usd, 6)
        out["sources"][source] = out["sources"].get(source, 0) + int(count)
    return out


def lesson_cost(task_id: str, session_id: str | None) -> dict:
    """The lesson's cost: its generation (the lesson task's attempts outside any session) and its teaching session
    (the session's attempts: the teacher's turns and the grading), and their sum."""
    generation = cost_of(f"task_id = {smoke.literal(task_id)} AND session_id IS NULL")
    session = cost_of(f"session_id = {smoke.literal(session_id)}") if session_id else cost_of("1 = 0")
    total = {k: round(generation[k] + session[k], 6) for k in ("charged_usd", "measured_usd", "estimate_usd")}
    return {"generation": generation, "teaching_session": session, **total,
            "basis": "ledger: measured = provider usage at the configured prices (or a provider-reported cost); "
                     "estimate = the pre-send reservation charged when no usage came back; not the vendor invoice"}


def write_review_bundle(run: Run, files: dict[str, bytes]) -> None:
    """Seal the lesson for the reviewer and write it. The plaintext is scanned for the provider keys first (a hit is a
    leak and nothing is written) and never written to disk. The rehearsal opens what it sealed and compares."""
    assert run.keys is not None and run.review_key is not None
    for name, body in files.items():
        for var in run.keys.scan(body.decode("latin-1")):
            run.leaks.append(f"{var} in the review bundle file {name} (not written)")
    if run.leaks:
        return
    envelope = review_bundle.seal(files, run.review_key)
    text = json.dumps(envelope)
    for var in run.keys.scan(text):  # cannot happen for ciphertext, but the file leaves the runner
        run.leaks.append(f"{var} in the sealed review bundle (not written)")
        return
    REVIEW_BUNDLE.write_text(text, encoding="utf-8")
    if run.review_private_key is not None:
        opened = review_bundle.unseal(json.loads(REVIEW_BUNDLE.read_text(encoding="utf-8")), run.review_private_key)
        assert opened == files, "the review bundle does not open to what was sealed"
    run.bundle = REVIEW_BUNDLE.name


def seal_lesson(run: Run, files: dict[str, bytes], preview: dict, release: dict) -> None:
    """(Re)write the review bundle with a REVIEW.md rendered from what the run knows so far."""
    run.report["stages"] = run.stages
    images = sorted(name for name in files if name.startswith("images/"))
    sheet = review_bundle.render(preview, release, run.report, images)
    write_review_bundle(run, {**files, "REVIEW.md": sheet.encode("utf-8")})
    assert run.bundle or run.leaks, "the review bundle was not written"


def stop_if(run: Run, stage: str) -> None:
    if run.fail_after == stage:
        raise RuntimeError(f"forced failure after the {stage} stage (--fail-after {stage})")


# --- the run -----------------------------------------------------------------------------------------------------------


def flow(run: Run, environment: dict[str, str]) -> None:
    os.environ.update(environment)  # compose interpolates compose.live.yaml from these (no secret among them)
    key = smoke.write_secrets()
    fixture = json.loads((smoke.FIXTURES / "learner.json").read_text(encoding="utf-8"))
    lid = fixture["learner_id"] if run.mock else f"live-{secrets.token_hex(4)}"
    me, other, teacher = Bearer(key, lid), Bearer(key, "other-learner"), Bearer(key, "teacher-1", roles=["teacher"])
    other_teacher = Bearer(key, "teacher-2", roles=["teacher"])

    run.started = True
    with timed(run, "stack"):
        smoke.compose("up", "-d", *(["--build"] if run.build else []))
    with smoke.httpx.Client(base_url=f"http://127.0.0.1:{port()}", timeout=120) as c:
        with timed(run, "stack_ready"):
            smoke.wait_for(c, "/gateway/health", timeout=600)
            checked = smoke.compose("run", "--rm", "api", "check", capture=True, check=False)
        lines = checked.stdout.splitlines()
        assert checked.returncode == 0 and lines and lines[-1] == "READY", run.keys.redact(checked.stdout)
        assert f"run-wide cap: max_cost_usd={run.max_cost:g}" in checked.stdout, run.keys.redact(checked.stdout)
        stop_if(run, "stack")

        # Identity: nothing gets through without a verified token.
        assert c.get(f"/learners/{lid}").status_code == 401
        assert c.get(f"/learners/{lid}", headers={"Authorization": "Bearer not-a-token"}).status_code == 401

        if run.mock:
            profile, request = fixture["profile"], fixture["request"]
        else:
            profile, request = LIVE_PROFILE, LIVE_REQUEST
        assert c.put(f"/learners/{lid}", json=profile, headers=me).status_code == 200
        run.learner, run.learner_headers = lid, me

        # The teacher's class requires lesson approval, and the learner joins it with its code (no paid call).
        with timed(run, "class_setup"):
            paid = paid_attempts()
            created = c.post("/classes", json={"name": LIVE_CLASS}, headers=teacher)
            assert created.status_code == 201, run.keys.redact(created.text)
            classroom = record(run, created)
            class_id = classroom["class_id"]
            on = c.patch(f"/classes/{class_id}", json={"lesson_approval": True}, headers=teacher)
            assert on.status_code == 200 and record(run, on)["lesson_approval"] is True, run.keys.redact(on.text)
            joined = c.post(f"/learners/{lid}/classes", json={"join_code": classroom["join_code"]}, headers=me)
            assert joined.status_code == 201, run.keys.redact(joined.text)
            record(run, joined)
            assert paid_attempts() == paid, "setting up the class made a paid provider call"

        # One lesson, queued through the gateway and run by the worker on the configured providers.
        with timed(run, "generation"):
            created = c.post("/tasks", json={"request": request, "learner_id": lid}, headers=me)
            assert created.status_code == 202, run.keys.redact(created.text)
            task = smoke.poll_task(c, record(run, created)["task_id"], me, timeout=1500)
            while task["status"] == "WAITING":
                answers = (smoke.diagnostic_answers(task) if run.mock else
                           [{"question_id": q["question_id"], "answer": "no lo sé"}
                            for q in task["waiting"]["prompt"]["questions"]])
                r = c.post(f"/tasks/{task['task_id']}/answers", json={"answers": answers}, headers=me)
                assert r.status_code == 202, run.keys.redact(r.text)
                record(run, r)
                task = smoke.poll_task(c, task["task_id"], me, timeout=1500)
        run.responses.append(json.dumps(task))
        assert task["status"] == "COMPLETED", run.keys.redact(json.dumps(task)[:2000])
        task_id = task["task_id"]
        run.report["generation_server_seconds"] = round(
            (datetime.fromisoformat(task["updated_at"]) - datetime.fromisoformat(task["created_at"])).total_seconds(), 1)
        artifacts = record(run, c.get(f"/tasks/{task_id}/artifacts", headers=me))
        assert answer_key_fields(artifacts) == set(), answer_key_fields(artifacts)
        videos = [a for a in artifacts if a["type"] == "VIDEO"]
        assert videos, {a["type"] for a in artifacts}
        video_id = videos[-1]["artifact_id"]
        images = [a["artifact_id"] for a in artifacts if a["type"] == "IMAGE_ASSET"]
        assert images, {a["type"] for a in artifacts}

        # Held for the teacher: the learner gets nothing of the lesson until it is approved.
        assert task.get("release") == "PENDING", task.get("release")
        assert c.get(f"/lessons/{task_id}/reading", headers=me).status_code == 409
        assert c.get(f"/artifacts/{images[0]}/content", headers=me).status_code == 409
        assert c.post(f"/lessons/{task_id}/teaching-session", json={}, headers=me).status_code == 409
        stop_if(run, "lesson")

        # The teacher's review of the same lesson, through the teacher routes: list, preview, images, approve.
        with timed(run, "review"):
            paid = paid_attempts()
            pending = record(run, c.get(f"/classes/{class_id}/lessons?status=PENDING", headers=teacher))
            assert [entry["task_id"] for entry in pending] == [task_id], pending
            preview_path = f"/classes/{class_id}/lessons/{task_id}/preview"
            previewed = c.get(preview_path, headers=teacher)
            assert previewed.status_code == 200, run.keys.redact(previewed.text)
            preview = record(run, previewed)
            assert answer_key_fields(preview) == set(), answer_key_fields(preview)
            assert preview["task_id"] == task_id and preview["sections"], "the preview holds no lesson"
            assert c.get(preview_path, headers=other_teacher).status_code == 404  # not that teacher's class
            assert c.get(preview_path, headers=me).status_code in (403, 404)  # not a teacher
            pictures: dict[str, bytes] = {}
            withheld_images = 0
            for artifact_id in images:  # #58: the preview page does not show them, but the teacher may fetch them
                fetched = c.get(f"/artifacts/{artifact_id}/content", headers=teacher)
                if fetched.status_code == 404:  # withheld by the content policy (e.g. its stored text names an
                    withheld_images += 1        # answer): refused to everyone, the teacher included, by design
                    continue
                assert fetched.status_code == 200, f"the teacher's image fetch answered {fetched.status_code}"
                kind = fetched.headers.get("content-type", "").split(";")[0].strip()
                pictures[f"images/{artifact_id}.{IMAGE_SUFFIXES.get(kind, 'bin')}"] = fetched.content
            assert pictures, "the teacher could fetch none of the lesson's images"
            approved = c.post(f"/classes/{class_id}/lessons/{task_id}/approve",
                              json={"note": "Approved by the live smoke run; human review follows"}, headers=teacher)
            assert approved.status_code == 200, run.keys.redact(approved.text)
            release = record(run, approved)
            assert release["status"] == "APPROVED", release
            assert c.post(f"/classes/{class_id}/lessons/{task_id}/reject", headers=teacher).status_code == 409
            assert record(run, c.get(f"/tasks/{task_id}", headers=me))["release"] == "APPROVED"
            read = c.get(f"/lessons/{task_id}/reading", headers=me)
            assert read.status_code == 200, run.keys.redact(read.text)
            reading = record(run, read)
            assert answer_key_fields(reading) == set(), answer_key_fields(reading)
            assert reading["lesson_id"] == preview["lesson_id"], "the learner reads another lesson than was approved"
            assert paid_attempts() == paid, "the teacher's review made a paid provider call"

        # The approved lesson is sealed for the human review now, so any later failure still leaves it.
        run.report["lesson"] = {"task_id": task_id, "lesson_id": preview["lesson_id"], "release": "APPROVED",
                                "images": len(images), "images_withheld": withheld_images}
        sealed_release = {k: release.get(k) for k in ("task_id", "lesson_id", "status", "requested_at",
                                                       "decided_at", "note", "topic", "subject", "task_status")}
        bundle_files = {"preview.json": json.dumps(preview, indent=2).encode(),
                        "reading.json": json.dumps(reading, indent=2).encode(),
                        "release.json": json.dumps(sealed_release, indent=2).encode(),
                        **pictures}
        seal_lesson(run, bundle_files, preview, sealed_release)

        # Released: the learner's surfaces as before (the VIDEO is never served, an image is).
        # The VIDEO is rendered from the deck, which can hold answer slides, so the API never serves its bytes to a
        # learner (app/services/artifact_content.py, BLOCKED): its content route answers 404 as for any blocked type.
        assert c.get(f"/artifacts/{video_id}/content", headers=me).status_code == 404
        served = [i for i in images if c.get(f"/artifacts/{i}/content", headers=me).status_code == 200]
        assert served, "no servable artifact content came back through the API"
        duration = video_duration(video_id)
        assert duration > 0, duration
        assert c.get(f"/tasks/{task_id}", headers=other).status_code == 404
        assert c.get(f"/artifacts/{served[0]}/content", headers=other).status_code == 404
        run.report["lesson"]["video_seconds"] = duration
        stop_if(run, "review")

        # One teaching-session answer, graded (the interactive path: teacher turns and grading in the API process).
        with timed(run, "grade"):
            started = c.post(f"/lessons/{task_id}/teaching-session", json={}, headers=me)
            assert started.status_code in (200, 201), run.keys.redact(started.text)
            session = record(run, started)
            assert answer_key_fields(session) == set(), answer_key_fields(session)
            question = session.get("waiting_question")
            assert question, "the teaching session asks no question"
            answer = question["choices"][0] if question.get("choices") else "Hola, me llamo Ana. Mucho gusto."
            graded = c.post(f"/teaching-sessions/{session['session_id']}/answers",
                            json={"answer": answer, "client_turn_id": "live-smoke-1"}, headers=me)
            assert graded.status_code == 200, run.keys.redact(graded.text)
            result = record(run, graded)
            assert result.get("answer_outcome") or result.get("correct") is not None, result
            assert answer_key_fields(result) == set(), answer_key_fields(result)
        stop_if(run, "grade")

        # The learning activity is recorded: the learner's progress and the teacher's dashboard show the lesson.
        progress = record(run, c.get(f"/learners/{lid}/progress", headers=me))
        assert progress["lessons_completed"] >= 1, progress
        board = record(run, c.get(f"/classes/{class_id}/dashboard", headers=teacher))
        assert [row["learner_id"] for row in board["learners"]] == [lid], "the dashboard does not list the learner"
        run.report["lesson_cost"] = lesson_cost(task_id, session["session_id"])
        seal_lesson(run, bundle_files, preview, sealed_release)  # again, now with the grade's timing and the cost

        # The gateway holds only its own secret: no provider key file, no database or object-store credential.
        seen = smoke.compose("exec", "-T", "gateway", "sh", "-c", "ls /run/secrets /run/ta-config", capture=True)
        assert set(seen.stdout.split()) - {"/run/secrets:", "/run/ta-config:"} == {"gateway_secret", "jwks.json"}

        scan_everything(run, "before the deletion")

        # The learner deletes their data (MVP item 9); the teacher cannot; nothing of theirs is left.
        assert c.delete(f"/learners/{lid}", headers=teacher).status_code in (403, 404)
        assert c.delete(f"/learners/{lid}", headers=other).status_code == 404
        tasks = set(smoke.sql(f"SELECT task_id FROM tasks WHERE learner_id = {smoke.literal(lid)};"))
        deleted = c.delete(f"/learners/{lid}", headers=me)
        assert deleted.status_code == 200, run.keys.redact(deleted.text)
        run.deleted = True
        record(run, deleted)
        assert deleted.json()["objects_pending"] == 0, deleted.text
        assert smoke.rows_naming({lid} | tasks) == {}, smoke.rows_naming({lid} | tasks)
        left = smoke.bucket_keys()
        assert not [k for k in left if any(t in k for t in tasks)], sorted(left)[:20]
        assert c.get(f"/learners/{lid}", headers=me).status_code == 404

    spend = ledger(run.max_cost)
    run.report.update({"mode": "mock" if run.mock else "live", "status": "PASSED", "lesson_video_seconds": duration,
                       "graded_outcome": result.get("answer_outcome"), "spend": spend, "stages": run.stages,
                       "review_bundle": run.bundle})
    assert_ledger(spend, mock=run.mock)
    scan_everything(run, "after the deletion")


def write_report(run: Run) -> None:
    """The report: ids, numbers and statuses only; scanned before it is written, and not written if a key is in it."""
    text = json.dumps(run.report, indent=2, sort_keys=True)
    if run.keys is not None and run.keys.scan(text):
        run.leaks.append("a key in the report (not written)")
        return
    REPORT.write_text(text, encoding="utf-8")


def show_failure(run: Run) -> None:
    """The stack's state and the last log lines, with every key replaced by *** (GitHub's masking is a second
    layer). Printed only on failure."""
    if not run.started or run.keys is None:
        return
    extra = tuple(p.read_text(encoding="utf-8").strip() for p in SECRETS.glob("*") if p.is_file())
    for args in (("ps", "-a"), ("logs", "--no-color", "--tail", "200")):
        try:
            done = smoke.compose(*args, check=False, capture=True)
            print(run.keys.redact(done.stdout + done.stderr, extra))
        except Exception as exc:  # noqa: BLE001
            print(f"live smoke: `docker compose {args[0]}` failed ({type(exc).__name__})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", nargs="?", default="run", choices=["run", "teardown", "verify-clean"])
    parser.add_argument("--mock", action="store_true", help="the rehearsal: mock providers, fake keys")
    parser.add_argument("--max-cost", default="1.00", help=f"the run-wide ceiling in USD (at most {MAX_CEILING_USD})")
    parser.add_argument("--fail-after", choices=STAGES, help="fail on purpose after a stage (cleanup rehearsal)")
    parser.add_argument("--deadline-minutes", type=float, default=30.0)
    parser.add_argument("--no-build", action="store_true", help="use the teaching-agent:local image as it is")
    args = parser.parse_args(argv)
    smoke.COMPOSE = compose_command()
    if args.action == "verify-clean":
        return verify_clean()
    if args.action == "teardown":
        return teardown()

    run = Run(mock=args.mock, max_cost=0.0, fail_after=args.fail_after, build=not args.no_build)
    install_handlers(run, args.deadline_minutes)
    code = 1
    try:
        try:
            run.max_cost = ceiling(args.max_cost)
            environment = stack_environment(mock=args.mock, max_cost=run.max_cost, environ=dict(os.environ))
            run.review_key, run.review_private_key = review_recipient(mock=args.mock, environ=dict(os.environ))
            run.keys = Keys.load(mock=args.mock)
        except ConfigProblem as exc:
            print(f"live smoke: {exc}", file=sys.stderr)
            code = 2
            return code
        REVIEW_BUNDLE.unlink(missing_ok=True)  # a bundle of an earlier run is never taken for this run's
        flow(run, environment)
        if run.leaks:
            raise Leak("; ".join(run.leaks))
        code = 0
    except (Exception, Interrupted) as exc:  # noqa: BLE001 - reported redacted below; cleanup follows
        run.cleaning = True  # from here on a further signal no longer interrupts: the cleanup must finish
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
        message = run.keys.redact(detail) if run.keys else type(exc).__name__
        print(f"live smoke: FAILED: {message[:6000]}", file=sys.stderr)
        run.report["status"] = "FAILED"
        run.report["failure"] = type(exc).__name__
        show_failure(run)
        code = 1
    finally:
        cleanup(run)
        if run.cleanup_errors:
            print("live smoke: cleanup problems: " + "; ".join(run.cleanup_errors), file=sys.stderr)
            code = code or 1
        if run.leaks:
            print("live smoke: KEY FOUND: " + "; ".join(run.leaks), file=sys.stderr)
            code = 1
        if run.report:
            run.report["stages"] = run.stages
            run.report.setdefault("review_bundle", run.bundle)
            run.report["cleanup_errors"] = run.cleanup_errors
            run.report["leaks"] = run.leaks  # places and variable names only
            write_report(run)
    print("live smoke: OK" if code == 0 else f"live smoke: exit {code}")
    return code


def teardown() -> int:
    """Remove the run's containers, volumes and networks, its key files and generated configuration, whatever state
    a run left them in (the workflow's `if: always()` step, after a failure, a timeout or a cancellation)."""
    errors: list[str] = []
    remove_by_label(errors)
    remove_generated_files(errors)
    (ROOT / "live-smoke-lesson.mp4").unlink(missing_ok=True)
    if errors:
        print("live smoke: teardown problems: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("live smoke: torn down")
    return 0


def verify_clean() -> int:
    """Nothing of a run is left: no container, volume or network of the compose project, and no key file or
    generated configuration. Exit 1 naming what is left."""
    label = f"label=com.docker.compose.project={PROJECT}"
    left = []
    for kind, cmd in (("container", ["docker", "ps", "-aq", "--filter", label]),
                      ("volume", ["docker", "volume", "ls", "-q", "--filter", label]),
                      ("network", ["docker", "network", "ls", "-q", "--filter", label])):
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            left.append(f"cannot list {kind}s ({done.returncode})")
        elif done.stdout.split():
            left.append(f"{len(done.stdout.split())} {kind}(s)")
    for path in (SECRETS, smoke.GATEWAY_CONFIG / "jwks.json", ROOT / "live-smoke-lesson.mp4"):
        if path.exists():
            left.append(path.name)
    if left:
        print("live smoke: left behind: " + ", ".join(left), file=sys.stderr)
        return 1
    print("live smoke: clean (no container, volume, network or key file of the run is left)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
