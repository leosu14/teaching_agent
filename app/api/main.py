"""ASGI entry point: `uvicorn app.api.main:app`."""

from app.api.app import create_app
from app.config.settings import Settings
from app.observability.logging import configure_logging

_settings = Settings()
configure_logging(_settings.log_level, _settings.log_json)
app = create_app()
