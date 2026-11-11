"""The deployment check: `python -m app.services.deployment check [--connect]` (the image's `check` command).

It prints what a deployment is made of (database, schema mode, object store, identity, providers, daily caps,
declared vendor caps; never a secret) and every reason it is not a production deployment
(`Settings.deployment_problems`). With `--connect` it also opens the database and checks the schema is at the newest
migration, and writes, reads back and deletes a probe object in the object store. Run it after `migrate` and before
starting a new version; the API and the worker run the same configuration checks at startup when
`TA_DEPLOYMENT=production`.

Exit codes: 0 ready, 2 configuration problems, 3 a service is unreachable or not ready.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from app.config.routing import ConfigError
from app.config.settings import Settings


@dataclass
class DeploymentReport:
    summary: dict[str, str]
    problems: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 2 if self.problems else 3 if self.unreachable else 0

    def lines(self) -> list[str]:
        out = [f"{k}: {v}" for k, v in self.summary.items()]
        out += [f"PROBLEM: {p}" for p in self.problems]
        out += [f"UNREACHABLE: {u}" for u in self.unreachable]
        out.append("READY" if self.exit_code == 0 else "NOT READY")
        return out


def _caps(settings: Settings, scope: str) -> str:
    limits = {f: getattr(settings, f"budget_{scope}_{f}") for f in ("max_llm_requests", "max_llm_tokens",
                                                                     "max_cost_usd")}
    shown = [f"{k.removeprefix('max_')}={v}" for k, v in limits.items() if v is not None]
    return ", ".join(shown) or "off"


def summarize(settings: Settings) -> dict[str, str]:
    url = settings.resolved_database_url
    store = (f"s3 (bucket {settings.s3_bucket}, endpoint {settings.s3_endpoint_url or 'AWS ' + settings.s3_region})"
             if settings.object_store == "s3" else f"filesystem ({settings.resolved_object_store_dir})")
    try:
        caps = settings.declared_vendor_caps()
        declared = ", ".join(f"{k}=${v:g}/month" for k, v in sorted(caps.items())) or "none"
    except ConfigError:
        declared = "invalid"
    return {
        "deployment": settings.deployment,
        "database": url.split("://", 1)[0] + "://…",  # the URL may carry a password
        "schema": settings.resolved_database_schema,
        "object store": store,
        "identity": settings.auth_mode + (" (secret set)" if settings.auth_gateway_secret else " (no secret)")
        if settings.auth_mode == "gateway" else settings.auth_mode,
        "providers": settings.providers.mode + (": " + ", ".join(settings.real_vendors())
                                                if settings.real_vendors() else " (mock providers only)"),
        "daily caps per learner": _caps(settings, "learner_daily"),
        "daily caps overall": _caps(settings, "global_daily"),
        "vendor caps": declared,
    }


def check(settings: Settings, *, connect: bool = False) -> DeploymentReport:
    report = DeploymentReport(summary=summarize(settings))
    try:
        settings.validate_runtime()
    except ConfigError as exc:
        report.problems.append(str(exc))
    report.problems += settings.deployment_problems()
    if connect and not report.problems:
        report.unreachable += _connect(settings)
    return report


def _connect(settings: Settings) -> list[str]:
    from app.services.container import build_object_store
    from app.storage.db import create_db, dispose

    failures = []
    try:
        dispose(create_db(settings.resolved_database_url, settings.resolved_database_schema))
    except ConfigError as exc:
        failures.append(f"database: {exc}")
    except Exception as exc:  # noqa: BLE001 - reported by type only: messages can carry the URL
        failures.append(f"database: cannot connect ({type(exc).__name__})")
    store = None
    try:
        store = build_object_store(settings)
        store.check()
    except ConfigError as exc:
        failures.append(f"object store: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"object store: {exc if isinstance(exc, OSError) else type(exc).__name__}")
    finally:
        if store is not None and hasattr(store, "close"):
            store.close()
    return failures


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point (tested via `check`)
    from app.config.secrets import load_secret_files

    parser = argparse.ArgumentParser(description="Check that this configuration is a production deployment.")
    parser.add_argument("action", choices=["check", "create-bucket"],
                        help="check: the deployment checks; create-bucket: create TA_S3_BUCKET if it does not exist "
                             "(a self-hosted store such as the compose stack's)")
    parser.add_argument("--connect", action="store_true",
                        help="also check the database schema and the object store over the network")
    args = parser.parse_args(argv)
    try:
        load_secret_files()
        settings = Settings()
    except (ConfigError, ValueError) as exc:
        print(f"PROBLEM: {exc}")
        return 2
    if args.action == "create-bucket":
        from app.services.container import build_object_store

        store = build_object_store(settings)
        try:
            created = store.ensure_bucket()
        except (ConfigError, OSError) as exc:
            print(f"UNREACHABLE: object store: {exc}")
            return 3
        finally:
            store.close()
        print(f"bucket {settings.s3_bucket}: {'ready' if created else 'exists'}")
        return 0
    report = check(settings, connect=args.connect)
    print("\n".join(report.lines()))
    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
