# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
#
# breadth-recorder image.
#
# Long-running service that subscribes to IBKR TICK-NYSE and AD-NYSE, aggregates
# ticks into 1-min bars, and writes rolling daily parquet shards to
# /data/breadth. Matches the paper-trader base image so the the deployment host factory host
# only needs to pull one Python 3.11-slim layer.
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

EXPOSE 8001

ENTRYPOINT ["python", "-u", "/app/infra/recorders/ibkr_breadth/run.py"]
