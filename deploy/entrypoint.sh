#!/bin/sh
# The image's commands. Secrets come from <NAME>_FILE variables (mounted secret files), read by each process.
set -eu
command="${1:-api}"
shift || true
case "$command" in
  api)
    exec uvicorn app.api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips '*' "$@" ;;
  worker)
    exec python -m app.services.worker "$@" ;;
  gateway)
    exec uvicorn app.gateway.main:app --host 0.0.0.0 --port "${PORT:-8080}" "$@" ;;
  migrate)
    exec python -m app.storage.migrate upgrade "$@" ;;
  create-bucket)
    exec python -m app.services.deployment create-bucket "$@" ;;
  check)
    exec python -m app.services.deployment check --connect "$@" ;;
  *)
    exec "$command" "$@" ;;
esac
