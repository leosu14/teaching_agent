"""Import-lint: layers may only import downward (api -> services -> runtime -> agents -> tools -> providers)."""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"

FOUNDATION = {"schemas", "config", "observability", "utils"}
ALLOWED: dict[str, set[str]] = {
    "schemas": {"schemas"},
    "config": {"config", "schemas"},
    "utils": {"utils", "schemas"},
    "observability": {"observability", "schemas", "config"},
    "providers": FOUNDATION | {"providers"},
    "storage": FOUNDATION | {"storage"},
    "learner": FOUNDATION | {"learner"},
    "artifacts": FOUNDATION | {"artifacts"},
    "tools": FOUNDATION | {"tools", "providers", "learner", "artifacts"},
    "agents": FOUNDATION | {"agents", "tools", "providers"},
    "runtime": FOUNDATION | {"runtime", "agents", "tools", "providers"},
    "services": FOUNDATION | {"services", "runtime", "agents", "tools", "providers", "learner", "artifacts", "storage"},
    "api": FOUNDATION | {"api", "services"},
}
# Finer rules on top of the layer order.
AGENT_PROVIDER_MODULES = {"app.providers.llm.base", "app.providers.llm.router"}
SQL_PACKAGES = {"sqlalchemy"}
VENDOR_SDKS = {"anthropic", "openai", "google", "requests", "httpx", "boto3", "elevenlabs", "minimax", "pptx", "azure"}
# Audio processing libraries: only providers (and the stdlib-based probe in utils) may touch audio bytes.
AUDIO_LIBS = {"wave", "pydub", "ffmpeg", "soundfile", "audioop", "pyaudio"}
API_EXCEPTION_MODULES = {  # the API may import exception types from lower layers, nothing else
    "app.runtime.orchestrator.orchestrator": {"InvalidInput"},
    "app.runtime.tasks.state_machine": {"InvalidTransition"},
    "app.learner.frameworks": {"UnknownFramework"},
    "app.learner.memory": {"UnknownLearner"},
    "app.storage.repositories": {"NotFound"},
}


def imports(path: Path) -> list[tuple[str, list[str], int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, [], node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.module, [a.name for a in node.names], node.lineno))
    return found


def violations_for(layer: str, module: str, names: list[str], where: str) -> list[str]:
    problems = []
    root = module.split(".")[0]
    if root == "app":
        parts = module.split(".")
        if len(parts) < 2:
            return [f"{where}: bare 'app' import"]
        target = parts[1]
        allowed = ALLOWED[layer]
        if layer == "api" and module in API_EXCEPTION_MODULES:
            if not set(names) <= API_EXCEPTION_MODULES[module]:
                problems.append(f"{where}: api may only import exception types from {module}")
        elif target not in allowed:
            problems.append(f"{where}: layer '{layer}' must not import '{module}'")
        if layer == "agents" and target == "providers" and module not in AGENT_PROVIDER_MODULES:
            problems.append(f"{where}: agents reach models only through app.providers.llm.router, not {module}")
    else:
        if root in SQL_PACKAGES and layer != "storage":
            problems.append(f"{where}: only the storage layer may use {root}")
        if root in VENDOR_SDKS and layer != "providers":
            problems.append(f"{where}: vendor SDK '{root}' outside the providers layer")
        if root in AUDIO_LIBS and layer not in {"providers", "utils"}:
            problems.append(f"{where}: audio library '{root}' outside providers and utils")
        if root == "fastapi" and layer != "api":
            problems.append(f"{where}: fastapi outside the api layer")
    return problems


def scan() -> list[str]:
    problems = []
    for path in sorted(APP.rglob("*.py")):
        rel = path.relative_to(APP)
        if len(rel.parts) < 2:
            continue
        layer = rel.parts[0]
        assert layer in ALLOWED, f"new top-level package app/{layer} needs a layer rule in this test"
        for module, names, line in imports(path):
            problems += violations_for(layer, module, names, f"app/{rel}:{line}")
    return problems


def test_no_layer_violations() -> None:
    assert scan() == []


def test_checker_flags_upward_imports() -> None:
    assert violations_for("tools", "app.agents.base", [], "x")
    assert violations_for("agents", "app.storage.repositories", [], "x")
    assert violations_for("agents", "app.providers.llm.mock", [], "x")
    assert violations_for("agents", "sqlalchemy.orm", [], "x")
    assert violations_for("runtime", "app.services.container", [], "x")
    assert violations_for("providers", "app.tools.base", [], "x")
    assert violations_for("api", "app.runtime.workflow.engine", ["WorkflowEngine"], "x")
    assert violations_for("tools", "openai", [], "x")
    assert violations_for("agents", "pptx", [], "x") and violations_for("tools", "pptx.util", [], "x")
    assert not violations_for("providers", "pptx", [], "x")
    assert violations_for("agents", "pydub", [], "x") and violations_for("agents", "wave", [], "x")
    assert violations_for("tools", "ffmpeg", [], "x") and violations_for("runtime", "elevenlabs", [], "x")
    assert violations_for("agents", "app.providers.tts.mock", [], "x")
    assert not violations_for("providers", "wave", [], "x") and not violations_for("utils", "wave", [], "x")
    assert not violations_for("agents", "app.providers.llm.router", [], "x")
    assert not violations_for("services", "app.storage.repositories", [], "x")
