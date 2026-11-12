"""ASGI entry point: `uvicorn app.api.main:app`."""

from app.api.app import create_app
from app.config.secrets import load_secret_files
from app.config.settings import Settings
from app.observability.logging import configure_logging

load_secret_files()  # <NAME>_FILE secrets (a secret manager's mounted files), before any setting is read
_settings = Settings()
configure_logging(_settings.log_level, _settings.log_json)
app = create_app()
