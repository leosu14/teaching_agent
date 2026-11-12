"""ASGI entry point of the identity gateway: `uvicorn app.gateway.main:app` (the image's `gateway` command)."""

from app.config.secrets import load_secret_files
from app.gateway.app import create_gateway
from app.observability.logging import configure_logging

load_secret_files()
configure_logging("INFO", True)
app = create_gateway()
