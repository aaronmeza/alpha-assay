# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
#
# ES-bars recorder image.
#
# Long-running service that subscribes to a single ES futures contract
# (1-min bars, RTH-only) via IBKRAdapter and writes rolling daily
# parquet shards to /data/es_bars in the canonical OHLCV schema accepted
# by alpha_assay.data.databento_adapter.load_parquet. Mirrors the Phase
# U breadth-recorder image so the the deployment host factory host pulls a single
# Python 3.11-slim base layer for both recorders + paper-trader.
#
# Build context is the repo root (see docker-compose.yml build.context).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    TZ=America/Chicago

WORKDIR /app

# System deps: tzdata for America/Chicago session math, curl for healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime

# Copy package metadata first for build-cache reuse on pure code changes.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY infra ./infra

RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install pyarrow tzdata

EXPOSE 8002

ENTRYPOINT ["python", "-u", "/app/infra/recorders/ibkr_es_bars/run.py"]
