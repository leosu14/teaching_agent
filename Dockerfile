# The Teaching Agent image (MVP item 8): one image, four commands (see deploy/entrypoint.sh and docs/deployment.md).
#   api      the HTTP API (uvicorn app.api.main:app), port 8000
#   worker   the task worker (python -m app.services.worker)
#   gateway  the identity gateway in front of the API (uvicorn app.gateway.main:app), port 8080
#   migrate  bring the database schema to the newest revision, then exit
#   check    the deployment check (python -m app.services.deployment check --connect), then exit
FROM python:3.11-slim

# FFmpeg composes the lesson video; DejaVu fonts draw slide text and subtitles.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TA_DATA_DIR=/var/lib/teaching-agent \
    TA_VIDEO_FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf

WORKDIR /opt/teaching-agent
# Dependencies first (cached while only the code changes), then the code. The package is installed in place: the
# bundled courses, fixtures and routing configuration are read from the source tree.
COPY pyproject.toml README.md ./
RUN mkdir -p app && touch app/__init__.py \
    && pip install ".[providers,postgres,gateway]" \
    && pip uninstall -y teaching-agent && rm -rf app build *.egg-info
COPY app ./app
COPY config ./config
COPY courses ./courses
COPY fixtures ./fixtures
COPY deploy/entrypoint.sh /usr/local/bin/ta-entrypoint
RUN pip install --no-deps -e . \
    && chmod 0755 /usr/local/bin/ta-entrypoint \
    && useradd --system --uid 10001 --home-dir /var/lib/teaching-agent --create-home teaching-agent \
    && python -c "import app.api.app, app.gateway.app, app.storage.migrate, app.providers.storage.s3"

USER teaching-agent
EXPOSE 8000 8080
ENTRYPOINT ["ta-entrypoint"]
CMD ["api"]
