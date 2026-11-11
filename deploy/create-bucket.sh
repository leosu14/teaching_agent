#!/bin/sh
# Create the artifact bucket in the compose stack's MinIO (idempotent).
set -eu
mc alias set pilot http://minio:9000 "$(cat /run/secrets/s3_access_key_id)" "$(cat /run/secrets/s3_secret_access_key)" >/dev/null
mc mb --ignore-existing pilot/teaching-agent
