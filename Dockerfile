# syntax=docker/dockerfile:1.7
# -------------------------------------------------------------------
# RAG Retriever (MCP) — Docker image
# -------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (Optional) small runtime helpers
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl tini gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 10001 appuser

WORKDIR /app

# Copy project metadata and sources
# Expecting repository layout:
#   pyproject.toml
#   ragretriever/...
COPY pyproject.toml ./
COPY . ./ragretriever

# Install the package and its dependencies
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir .

# Default network port (can be changed with RS__SERVER__PORT)
EXPOSE 8080

# Sensible defaults; override via env or YAML
ENV RS__SERVER__HOST=0.0.0.0 \
    RS__SERVER__PORT=8080

USER appuser

# Use tini for proper signal handling in containers
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "ragretriever.main"]
