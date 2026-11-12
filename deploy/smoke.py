"""End-to-end smoke test of the deployed stack (deploy/compose.yaml), as CI's `deploy` job runs it.

    python deploy/smoke.py            # build, start, test, stop and remove everything (volumes included)
    python deploy/smoke.py --keep     # leave the stack running afterwards

It plays the identity provider: it makes an RSA key, writes its public half as the gateway's static key set
(TA_GATEWAY_JWKS_FILE, in deploy/gateway/) and signs the learners' and the teacher's tokens. Every secret is
generated into deploy/secrets/ for this run. Nothing reaches the network beyond the image registries; the app runs on
the offline mock providers, with the real FFmpeg, PostgreSQL and an S3-compatible store (RustFS).

What it checks, all through the published gateway only:
- the migration ran (the API and the worker start only at the newest schema revision) and `check` reports READY;
- no token, a bad token, or forged identity headers get 401 and never reach the API;
- a learner's lesson is queued (202), run by the worker process and completed, with its media in the bucket and
  served back through the API; another learner gets 404 for it;
- a teacher creates a class, the learner joins, and the teacher's dashboard lists the learner;
- the privacy notice is served; the learner deletes their data (the teacher and another learner cannot), after which
  their lesson, its content and their place on the dashboard are gone;
- the published gateway sees only its own secret (no database or object-store credential);
- running the migration again changes nothing.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
SECRETS = DEPLOY / "secrets"
GATEWAY_CONFIG = DEPLOY / "gateway"  # the gateway's public configuration (the static JWKS)
COMPOSE = ["docker", "compose", "-f", str(DEPLOY / "compose.yaml"),
           *(["-f", os.environ["SMOKE_COMPOSE_OVERRIDE"]] if os.environ.get("SMOKE_COMPOSE_OVERRIDE") else []),
           "-p", "ta-smoke"]
ISSUER, AUDIENCE, KID = "https://idp.smoke.test/", "teaching-agent", "smoke-key-1"
FIXTURES = ROOT / "fixtures" / "demo"
TERMINAL_JOB = {"SUCCEEDED", "FAILED", "CANCELLED"}


def b64(n: int) -> str:
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def write_secrets() -> rsa.RSAPrivateKey:
    SECRETS.mkdir(parents=True, exist_ok=True)
    password = secrets.token_urlsafe(24)
    values = {
        "postgres_password": password,
        "database_url": f"postgresql+psycopg://ta:{password}@postgres:5432/teaching_agent",
        "s3_access_key_id": "ta" + secrets.token_hex(8),
        "s3_secret_access_key": secrets.token_urlsafe(32),
        "gateway_secret": secrets.token_urlsafe(48),
    }
    for name, value in values.items():
        path = SECRETS / name
        path.write_text(value, encoding="utf-8")
        path.chmod(0o644)  # read by the containers' non-root users (a secret manager would set owners instead)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers = key.public_key().public_numbers()
    jwks = {"keys": [{"kty": "RSA", "kid": KID, "use": "sig", "alg": "RS256", "n": b64(numbers.n),
                      "e": b64(numbers.e)}]}
    GATEWAY_CONFIG.mkdir(parents=True, exist_ok=True)
    (GATEWAY_CONFIG / "jwks.json").write_text(json.dumps(jwks), encoding="utf-8")
    return key


def token(key, subject: str, *, roles: list[str] | None = None, audience: str = AUDIENCE) -> str:
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": audience, "sub": subject, "iat": now, "exp": now + 600}
    if roles:
        claims["ta_roles"] = roles
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": KID})


def compose(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    env = {**os.environ, "TA_GATEWAY_OIDC_ISSUER": ISSUER, "TA_GATEWAY_OIDC_AUDIENCE": AUDIENCE,
           "TA_GATEWAY_JWKS_FILE": "/run/ta-config/jwks.json", "TA_GATEWAY_PORT": os.environ.get("SMOKE_PORT", "8080")}
    return subprocess.run([*COMPOSE, *args], cwd=ROOT, env=env, check=check, text=True,
                          capture_output=capture)


def wait_for(client: httpx.Client, path: str, timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if client.get(path).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise AssertionError(f"{path} did not come up")


def poll_task(client: httpx.Client, task_id: str, headers: dict, timeout: float = 900) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/tasks/{task_id}", headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        if (body.get("job") or {}).get("status") in TERMINAL_JOB or body["status"] in {"WAITING", "COMPLETED",
                                                                                      "FAILED", "CANCELLED"}:
            if (body.get("job") or {}).get("status") not in {"QUEUED", "RUNNING"}:
                return body
        time.sleep(2)
    raise AssertionError(f"task {task_id} did not settle")


def diagnostic_answers(task: dict) -> list[dict]:
    answers = json.loads((FIXTURES / "answers.json").read_text(encoding="utf-8"))
    sheet = task["waiting"]["prompt"]
    key = answers["rounds"][sheet["round_number"] - 1]
    return [{"question_id": q["question_id"], "answer": key[q["concept_id"]]} for q in sheet["questions"]]


def run(port: str, build: bool = True) -> None:
    key = write_secrets()
    learner = json.loads((FIXTURES / "learner.json").read_text(encoding="utf-8"))
    lid = learner["learner_id"]
    me = {"Authorization": f"Bearer {token(key, lid)}"}
    other = {"Authorization": f"Bearer {token(key, 'other-learner')}"}
    teacher = {"Authorization": f"Bearer {token(key, 'teacher-1', roles=['teacher'])}"}

    compose("up", "-d", *(["--build"] if build else []))
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as c:
        wait_for(c, "/gateway/health", timeout=600)
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/app/").status_code == 200  # the web app's static files are public

        # Identity: nothing gets through without a verified token, and forged identity headers are dropped.
        assert c.get(f"/learners/{lid}").status_code == 401
        forged = {"x-auth-subject": lid, "x-auth-learners": lid, "x-auth-timestamp": str(int(time.time())),
                  "x-auth-signature": "0" * 64}
        assert c.get(f"/learners/{lid}", headers=forged).status_code == 401
        assert c.get(f"/learners/{lid}", headers={"Authorization": "Bearer not-a-token"}).status_code == 401
        wrong_aud = {"Authorization": f"Bearer {token(key, lid, audience='someone-else')}"}
        assert c.get(f"/learners/{lid}", headers=wrong_aud).status_code == 401

        # A learner's lesson, run by the worker process, with its media in the bucket.
        assert c.put(f"/learners/{lid}", json=learner["profile"], headers=me).status_code == 200
        created = c.post("/tasks", json={"request": learner["request"], "learner_id": lid}, headers=me)
        assert created.status_code == 202, created.text
        task = poll_task(c, created.json()["task_id"], me)
        while task["status"] == "WAITING":
            r = c.post(f"/tasks/{task['task_id']}/answers", json={"answers": diagnostic_answers(task)}, headers=me)
            assert r.status_code == 202, r.text
            task = poll_task(c, task["task_id"], me)
        assert task["status"] == "COMPLETED", json.dumps(task)[:2000]
        artifacts = c.get(f"/tasks/{task['task_id']}/artifacts", headers=me).json()
        types = {a["type"] for a in artifacts}
        assert {"LESSON", "PRESENTATION", "AUDIO_ASSET", "VIDEO"} <= types, types
        served = [a for a in artifacts
                  if c.get(f"/artifacts/{a['artifact_id']}/content", headers=me).status_code == 200]
        assert served, "no artifact content was served back from the bucket"
        assert c.get(f"/tasks/{task['task_id']}", headers=other).status_code == 404
        assert c.get(f"/artifacts/{served[0]['artifact_id']}/content", headers=other).status_code == 404
        assert c.get(f"/learners/{lid}", headers=other).status_code == 404

        # The teacher's class and dashboard.
        made = c.post("/classes", json={"name": "Smoke class"}, headers=teacher)
        assert made.status_code == 201, made.text
        assert c.post("/classes", json={"name": "x"}, headers=me).status_code in {403, 404}
        joined = c.post(f"/learners/{lid}/classes", json={"join_code": made.json()["join_code"]}, headers=me)
        assert joined.status_code == 201, joined.text
        dashboard = c.get(f"/classes/{made.json()['class_id']}/dashboard", headers=teacher)
        assert dashboard.status_code == 200 and lid in dashboard.text, dashboard.text
        assert c.get(f"/classes/{made.json()['class_id']}/dashboard", headers=other).status_code == 404

        # The learner deletes their data (MVP item 9): only their own scope may; the rows and the bucket's objects
        # go, and the teacher stops seeing them.
        assert c.get("/app/privacy.html").status_code == 200
        assert c.delete(f"/learners/{lid}", headers=teacher).status_code == 403
        assert c.delete(f"/learners/{lid}", headers=other).status_code == 404
        deleted = c.delete(f"/learners/{lid}", headers=me)
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["records"].get("tasks") and deleted.json()["objects_deleted"] > 0, deleted.text
        assert c.get(f"/learners/{lid}", headers=me).status_code == 404
        assert c.get(f"/tasks/{task['task_id']}", headers=me).status_code == 404
        assert c.get(f"/artifacts/{served[0]['artifact_id']}/content", headers=me).status_code == 404
        dashboard = c.get(f"/classes/{made.json()['class_id']}/dashboard", headers=teacher)
        assert dashboard.status_code == 200 and lid not in dashboard.text, dashboard.text

    # The published gateway holds only its own secret: no database URL or object-store key is visible in it.
    seen = compose("exec", "-T", "gateway", "sh", "-c", "ls /run/secrets /run/ta-config", capture=True).stdout
    names = set(seen.split()) - {"/run/secrets:", "/run/ta-config:"}
    assert names == {"gateway_secret", "jwks.json"}, seen

    again = compose("run", "--rm", "migrate", capture=True)
    assert "already at" in again.stdout, again.stdout
    checked = compose("run", "--rm", "api", "check", capture=True, check=False)
    print(checked.stdout)
    assert checked.returncode == 0 and "READY" in checked.stdout.splitlines()[-1], checked.stdout
    print("deploy smoke: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="leave the stack running")
    parser.add_argument("--no-build", action="store_true", help="use the teaching-agent:local image as it is")
    args = parser.parse_args()
    port = os.environ.get("SMOKE_PORT", "8080")
    try:
        run(port, build=not args.no_build)
        return 0
    except BaseException:
        compose("ps", "-a", check=False)
        compose("logs", "--no-color", "--tail", "200", check=False)
        raise
    finally:
        if not args.keep:
            compose("down", "-v", "--remove-orphans", check=False)


if __name__ == "__main__":
    sys.exit(main())
