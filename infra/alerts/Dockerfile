# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
#
# Lightweight alert poller. Stdlib-only (no pip dependencies).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=America/Chicago

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime

COPY infra/alerts/main.py /app/main.py

CMD ["python", "-u", "/app/main.py"]
