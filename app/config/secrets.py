"""Secrets from files: `<NAME>_FILE=/run/secrets/<name>` sets `<NAME>` to the file's content.

This is how a secret manager hands secrets to the containers (Docker and Compose secrets, Kubernetes secrets, the
AWS / GCP / Azure secret-store CSI drivers, a Vault agent): mounted as files, readable by the app's user only, never
in the image, the compose file or the process listing. Every process entry point (API, worker, migrations, the
identity gateway, the deployment check) calls `load_secret_files()` before it reads its settings.

Only the variables in `SECRET_VARIABLES` may come from a file, so a stray `*_FILE` variable cannot set an arbitrary
setting. A variable set both directly and by a file is a configuration error rather than a silent choice.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

from app.config.routing import ConfigError

SECRET_VARIABLES = (
    # the API's side of the gateway contract, and the gateway's
    "TA_AUTH_GATEWAY_SECRET",
    # the database URL carries the database password
    "TA_DATABASE_URL",
    # the object store
    "TA_S3_ACCESS_KEY_ID", "TA_S3_SECRET_ACCESS_KEY", "TA_S3_SESSION_TOKEN",
    # provider keys (app/config/providers.py, KNOWN_PROVIDERS)
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TTS_API_KEY", "IMAGE_API_KEY", "SEARCH_API_KEY", "TAVILY_API_KEY",
    "VIDEO_GENERATION_API_KEY", "MINIMAX_API_KEY",
)
MAX_SECRET_BYTES = 64 * 1024
# Settings that name a file themselves (they are not secrets read into a variable).
NOT_SECRET_FILES = {"TA_AUTH_STATIC_TOKENS_FILE", "TA_GATEWAY_JWKS_FILE", "TA_ROUTING_FILE"}


def load_secret_files(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Read every `<NAME>_FILE` of SECRET_VARIABLES into `<NAME>`. Returns the names set (never the values)."""
    env = os.environ if environ is None else environ
    loaded = []
    for name in SECRET_VARIABLES:
        path = env.get(f"{name}_FILE", "").strip()
        if not path:
            continue
        if env.get(name):
            raise ConfigError(f"{name} and {name}_FILE are both set; set only one")
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise ConfigError(f"{name}_FILE: cannot read the secret file ({type(exc).__name__})") from None
        if len(raw) > MAX_SECRET_BYTES:
            raise ConfigError(f"{name}_FILE: the secret file is larger than {MAX_SECRET_BYTES} bytes")
        value = raw.decode("utf-8", "strict").strip()
        if not value:
            raise ConfigError(f"{name}_FILE: the secret file is empty")
        env[name] = value
        loaded.append(name)
    unknown = sorted(k for k in env if k.endswith("_FILE") and k not in NOT_SECRET_FILES
                     and k.removesuffix("_FILE") not in SECRET_VARIABLES
                     and any(w in k for w in ("SECRET", "KEY", "TOKEN", "PASSWORD")))
    if unknown:  # a misspelt secret would otherwise leave its setting unset
        raise ConfigError(f"{', '.join(unknown)}: not a secret that can be read from a file "
                          f"(one of {', '.join(SECRET_VARIABLES)})")
    return loaded
