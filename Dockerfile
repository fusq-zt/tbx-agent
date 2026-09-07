# syntax=docker/dockerfile:1.7

# The default image is a portable CPU/control-plane image. A verified CUDA base
# image must be supplied explicitly for rank03 GPU inference; see
# docs/deployment/docker.md. Model weights and D-FINE are mounted at runtime and
# are never copied into this image.
ARG TBX_BASE_IMAGE=python:3.12-slim
FROM ${TBX_BASE_IMAGE} AS runtime

ARG TBX_INSTALL_EXTRAS=ui
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp/tbx-home \
    XDG_CACHE_HOME=/tmp/tbx-cache \
    TBX_AGENT_PROJECT_ROOT=/app \
    TBX_AGENT_CONFIG_DIR=/app/configs \
    TBX_AGENT_KNOWLEDGE_DIR=/app/knowledge \
    TBX_AGENT_DATA_ROOT=/data \
    TBX_AGENT_DB_PATH=/data/tbx_agent.sqlite3 \
    TBX_ARTIFACT_ROOT=/data/artifacts

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY knowledge ./knowledge
COPY evaluation ./evaluation
COPY retrieval ./retrieval
COPY ui ./ui
COPY .streamlit ./.streamlit

RUN python -m pip install ".[${TBX_INSTALL_EXTRAS}]" \
    && if command -v addgroup >/dev/null 2>&1 && command -v adduser >/dev/null 2>&1; then \
         addgroup --system --gid 10001 tbx; \
         adduser --system --uid 10001 --ingroup tbx --home /nonexistent --no-create-home tbx; \
       elif command -v groupadd >/dev/null 2>&1 && command -v useradd >/dev/null 2>&1; then \
         groupadd --system --gid 10001 tbx; \
         useradd --system --uid 10001 --gid tbx --home-dir /nonexistent \
           --shell /usr/sbin/nologin tbx; \
       else \
         echo "Base image must provide adduser/addgroup or useradd/groupadd" >&2; exit 2; \
       fi \
    && mkdir -p /data/artifacts \
    && chown -R tbx:tbx /data

USER tbx
EXPOSE 8000 8501
STOPSIGNAL SIGTERM

FROM runtime AS api
CMD ["python", "-m", "uvicorn", "tbx_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS ui
CMD ["python", "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
