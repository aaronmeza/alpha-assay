# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
#
# Streamlit dashboard image for browsing backtest run results.
#
# Build context: repo root (docker compose handles this)
# Exposes: 8501 (Streamlit default)
# Mount: /runs (bind from RUNS_DIR on the host)

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install the framework package + dashboard optional deps first so
# Docker layer caching works across app-code changes. README/LICENSE/
# NOTICE are required by pyproject.toml for hatchling to build the
# package metadata.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[dashboard]"

# Copy the dashboard app
COPY dashboard/ ./dashboard/

# Streamlit config: disable the browser auto-open + telemetry.
# PYTHONPATH=/app makes `dashboard` importable as a top-level package
# (streamlit puts the script's directory on sys.path, not its parent).
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    PYTHONPATH=/app \
    RUNS_DIR=/runs

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=15s \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.headless", "true", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0"]
