"""The production deployment profile (MVP item 8): secrets from files, and the checks a deployment must pass before
the API or the worker starts (PostgreSQL under migrations, the S3 store, the signed gateway, daily caps on, a
declared hard cap at every real vendor)."""

from __future__ import annotations

import pytest

from app.config.routing import ConfigError
from app.config.secrets import load_secret_files
from app.config.settings import Settings
from app.services.container import build_container
from app.services.deployment import check

SECRET = "gateway-secret-" + "x" * 40
S3_SECRET = "s3-secret-access-key-value-0123456789"


def production(**overrides) -> Settings:
    values = dict(deployment="production", database_url="postgresql+psycopg://ta:pw-123456@db/ta",
                  object_store="s3", s3_bucket="lessons", s3_access_key_id="AKID", s3_secret_access_key=S3_SECRET,
                  auth_mode="gateway", auth_gateway_secret=SECRET,
                  budget_learner_daily_max_llm_requests=300, budget_learner_daily_max_llm_tokens=1_000_000,
                  budget_global_daily_max_llm_requests=5000, budget_global_daily_max_llm_tokens=20_000_000)
    return Settings(**(values | overrides))


# --- secrets from files ---------------------------------------------------------------------------------------------

def test_secret_files_set_their_variables(tmp_path) -> None:
    (tmp_path / "gw").write_text(SECRET + "\n")
    (tmp_path / "key").write_text("sk-test-key-value\n")
    env = {"TA_AUTH_GATEWAY_SECRET_FILE": str(tmp_path / "gw"), "OPENAI_API_KEY_FILE": str(tmp_path / "key")}
    env["TA_AUTH_STATIC_TOKENS_FILE"] = "/elsewhere/tokens.json"  # a setting that is a file, not a secret
    env["OTHER_TOOL_TOKEN_FILE"] = "/elsewhere/token"  # another program's variable: not ours to judge
    assert load_secret_files(env) == ["TA_AUTH_GATEWAY_SECRET", "OPENAI_API_KEY"]
    assert env["TA_AUTH_GATEWAY_SECRET"] == SECRET and env["OPENAI_API_KEY"] == "sk-test-key-value"


@pytest.mark.parametrize("case", ["both", "missing", "empty", "unknown"])
def test_secret_files_refuse_ambiguity_and_bad_files(tmp_path, case) -> None:
    (tmp_path / "empty").write_text("\n")
    env = {"both": {"TA_AUTH_GATEWAY_SECRET": "a", "TA_AUTH_GATEWAY_SECRET_FILE": str(tmp_path / "empty")},
           "missing": {"TA_S3_SECRET_ACCESS_KEY_FILE": str(tmp_path / "nope")},
           "empty": {"TA_DATABASE_URL_FILE": str(tmp_path / "empty")},
           "unknown": {"TA_S3_SECRET_KEY_FILE": str(tmp_path / "empty")}}[case]
    with pytest.raises(ConfigError) as refused:
        load_secret_files(env)
    assert str(tmp_path) not in str(refused.value) or case == "unknown"


# --- the deployment checks ------------------------------------------------------------------------------------------

def test_a_complete_deployment_has_no_problems() -> None:
    assert production().deployment_problems() == []
    assert production().resolved_database_schema == "migrations"
    assert Settings().resolved_database_schema == "create"  # development is unchanged


@pytest.mark.parametrize("overrides, expected", [
    ({"database_url": None}, "PostgreSQL"),
    ({"database_schema": "create"}, "TA_DATABASE_SCHEMA"),
    ({"object_store": "filesystem"}, "TA_OBJECT_STORE"),
    ({"s3_bucket": None}, "TA_S3_BUCKET"),
    ({"s3_endpoint_url": "http://minio:9000"}, "TA_S3_ALLOW_HTTP"),
    ({"auth_mode": "static"}, "TA_AUTH_MODE"),
    ({"auth_gateway_secret": "short"}, "TA_AUTH_GATEWAY_SECRET"),
    ({"worker_embedded": True}, "TA_WORKER_EMBEDDED"),
    ({"budget_learner_daily_max_llm_requests": None}, "TA_BUDGET_LEARNER_DAILY_"),
    ({"budget_global_daily_max_llm_tokens": None}, "TA_BUDGET_GLOBAL_DAILY_"),
    ({"vendor_caps": "openai=lots"}, "TA_VENDOR_CAPS"),
])
def test_each_missing_piece_is_named(overrides, expected) -> None:
    problems = production(**overrides).deployment_problems()
    assert len(problems) == 1 and expected in problems[0], problems


def test_daily_caps_by_cost_alone_need_llm_pricing(monkeypatch) -> None:
    monkeypatch.setenv("TEACHING_AGENT_MODE", "production")
    monkeypatch.delenv("TEACHING_AGENT_OFFLINE", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cost_only = dict(budget_learner_daily_max_llm_requests=None, budget_learner_daily_max_llm_tokens=None,
                     budget_learner_daily_max_cost_usd=2.0, vendor_caps="openai=100")
    problems = production(**cost_only).deployment_problems()
    assert len(problems) == 1 and "does not bind" in problems[0]
    monkeypatch.setenv("LLM_INPUT_PRICE_PER_MTOK", "5")
    monkeypatch.setenv("LLM_OUTPUT_PRICE_PER_MTOK", "25")
    assert production(**cost_only).deployment_problems() == []


def test_every_real_vendor_in_use_needs_a_declared_cap(monkeypatch) -> None:
    monkeypatch.setenv("TEACHING_AGENT_MODE", "production")
    monkeypatch.delenv("TEACHING_AGENT_OFFLINE", raising=False)
    for name, value in {"LLM_PROVIDER": "anthropic", "LLM_MODEL": "m", "ANTHROPIC_API_KEY": "k",
                        "LLM_FALLBACK_PROVIDER": "openai", "LLM_FALLBACK_MODEL": "m2", "OPENAI_API_KEY": "k2",
                        "SEARCH_PROVIDER": "tavily", "TAVILY_API_KEY": "k3"}.items():
        monkeypatch.setenv(name, value)
    settings = production(vendor_caps="anthropic=150")
    assert settings.real_vendors() == ["anthropic", "openai", "tavily"]
    problems = settings.deployment_problems()
    assert len(problems) == 1 and "openai, tavily" in problems[0]
    assert production(vendor_caps="anthropic=150, openai=50,tavily=20").deployment_problems() == []
    assert production(vendor_caps="anthropic=150,openai=50,tavily=20").declared_vendor_caps()["tavily"] == 20


def test_offline_mock_providers_need_no_vendor_caps() -> None:
    assert production().real_vendors() == [] and production().deployment_problems() == []


def test_a_production_deployment_that_fails_its_checks_does_not_start(tmp_path) -> None:
    settings = production(data_dir=tmp_path, database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
                          object_store="filesystem")
    with pytest.raises(ConfigError, match="not a production deployment.*PostgreSQL.*TA_OBJECT_STORE"):
        build_container(settings)
    assert not (tmp_path / "db.sqlite").exists()  # refused before anything was created


def test_the_deployment_report_names_everything_and_shows_no_secret() -> None:
    report = check(production())
    text = "\n".join(report.lines())
    assert report.exit_code == 0 and text.endswith("READY")
    for secret in (SECRET, S3_SECRET, "pw-123456"):
        assert secret not in text
    assert "database: postgresql+psycopg://…" in text and "daily caps overall: llm_requests=5000" in text
    bad = check(production(object_store="filesystem", auth_mode="static"))
    assert bad.exit_code == 2 and bad.lines()[-1] == "NOT READY"
    assert sum(line.startswith("PROBLEM:") for line in bad.lines()) == 2
