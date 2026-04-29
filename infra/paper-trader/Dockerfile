# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
#
# paper-trader image.
#
# Thin Python 3.11-slim base (matches pyproject.toml requires-python and the
# rest of the repo). The entrypoint runs scripts/paper_trader_stub.py, which is
# a heartbeat-only placeholder kept alive solely so Prometheus has a live
# scrape target while replaces
# the ENTRYPOINT with the real `alpha_assay paper` command.
#
# Build context is the repo root (see docker-compose.yml build.context).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=America/Chicago

WORKDIR /app

# System deps: tzdata for America/Chicago, curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime

# Copy package metadata first for build-cache reuse on pure code changes.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --upgrade pip \
    && pip install -e .

EXPOSE 8000

ENTRYPOINT ["python", "-u", "scripts/paper_trader_stub.py"]
