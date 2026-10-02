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
# Agents reach models only through the router and StructuredLLM: never a provider, a provider interface or a vendor.
AGENT_PROVIDER_MODULES = {"app.providers.llm.router", "app.providers.llm.structured"}
# Everything else above the provider layer sees only provider interfaces (the capability `base` modules, the typed
# errors and the provider-layer schemas). Concrete providers are known to the composition root alone.
PROVIDER_INTERFACE_MODULES = AGENT_PROVIDER_MODULES | {"app.providers.core.errors", "app.providers.core.registry",
                                                     "app.providers.core.selector"}
COMPOSITION_ROOT = "services/container.py"
SQL_PACKAGES = {"sqlalchemy"}
VENDOR_SDKS = {"anthropic", "openai", "google", "requests", "httpx", "httpx2", "aiohttp", "urllib3", "boto3",
               "botocore", "elevenlabs", "minimax", "pptx", "azure", "tavily", "cohere", "mistralai", "replicate",
               "stability_sdk", "dashscope", "vertexai"}
# Raw network access from the standard library: also providers only (the HTTP client lives there).
NETWORK_MODULES = {"urllib.request", "http.client", "socket", "ssl", "ftplib", "smtplib"}
# Audio processing libraries: only providers (and the stdlib-based probe in utils) may touch audio bytes.
AUDIO_LIBS = {"wave", "pydub", "ffmpeg", "soundfile", "audioop", "pyaudio"}
# Video and imaging infrastructure (FFmpeg runs through subprocess; frames are drawn with Pillow): providers only.
VIDEO_LIBS = {"subprocess", "PIL", "imageio_ffmpeg", "moviepy", "cv2", "av"}
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


def is_provider_interface(module: str) -> bool:
    return module in PROVIDER_INTERFACE_MODULES or (module.startswith("app.providers.") and module.endswith(".base")
                                                    and not module.startswith("app.providers.core."))


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
            problems.append(f"{where}: agents reach models only through app.providers.llm.router and "
                            f"app.providers.llm.structured, not {module}")
        elif (target == "providers" and layer not in {"providers", "agents"} and not is_provider_interface(module)
              and COMPOSITION_ROOT not in where):
            problems.append(f"{where}: only provider interfaces may be imported outside the provider layer, "
                            f"not {module}")
    else:
        if root in SQL_PACKAGES and layer != "storage":
            problems.append(f"{where}: only the storage layer may use {root}")
        if root in VENDOR_SDKS and layer != "providers":
            problems.append(f"{where}: vendor SDK '{root}' outside the providers layer")
        if any(module == m or module.startswith(m + ".") for m in NETWORK_MODULES) and layer != "providers":
            problems.append(f"{where}: network module '{module}' outside the providers layer")
        if root in AUDIO_LIBS and layer not in {"providers", "utils"}:
            problems.append(f"{where}: audio library '{root}' outside providers and utils")
        if root in VIDEO_LIBS and layer != "providers":
            problems.append(f"{where}: video/process library '{root}' outside the providers layer")
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
    assert violations_for("agents", "subprocess", [], "x") and violations_for("tools", "PIL.Image", [], "x")
    assert violations_for("runtime", "subprocess", [], "x") and violations_for("utils", "subprocess", [], "x")
    assert violations_for("agents", "app.providers.video.ffmpeg", [], "x")
    assert not violations_for("providers", "subprocess", [], "x") and not violations_for("providers", "PIL", [], "x")
    assert not violations_for("agents", "app.providers.llm.router", [], "x")
    assert not violations_for("services", "app.storage.repositories", [], "x")


def test_checker_flags_vendor_code_outside_the_provider_layer() -> None:
    # Agents never import providers directly: no concrete provider, no provider interface, no provider core.
    for module in ("app.providers.llm.mock", "app.providers.llm.base", "app.providers.llm.anthropic",
                   "app.providers.llm.openai_compatible", "app.providers.tts.base", "app.providers.core.http",
                   "app.providers.managed", "app.providers.search.tavily"):
        assert violations_for("agents", module, [], "x"), module
    assert not violations_for("agents", "app.providers.llm.structured", [], "x")
    # Agents, tools, services, runtime and the API never import a vendor SDK or a raw network module.
    for layer in ("agents", "tools", "services", "runtime", "api", "learner", "artifacts", "utils", "config"):
        for sdk in ("openai", "anthropic", "google.genai", "elevenlabs", "minimax", "httpx", "requests", "aiohttp",
                    "tavily", "boto3", "urllib.request", "http.client", "socket"):
            assert violations_for(layer, sdk, [], "x"), (layer, sdk)
    for sdk in ("httpx", "openai", "urllib.request", "PIL"):
        assert not violations_for("providers", sdk, [], "x")
    # Tools and runtime see provider interfaces only; the composition root is the one place that knows concretes.
    for layer in ("tools", "runtime", "services"):
        for module in ("app.providers.tts.openai", "app.providers.search.mock", "app.providers.core.http",
                       "app.providers.core.invoker", "app.providers.managed"):
            assert violations_for(layer, module, [], f"app/{layer}/x.py:1"), (layer, module)
        for module in ("app.providers.tts.base", "app.providers.core.errors", "app.providers.llm.router"):
            assert not violations_for(layer, module, [], f"app/{layer}/x.py:1"), (layer, module)
    assert not violations_for("services", "app.providers.tts.openai", [], "app/services/container.py:1")
    assert not violations_for("utils", "urllib.parse", [], "x")
