FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    TZ=America/Chicago

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY infra/feed ./infra/feed
COPY configs ./configs

RUN pip install --upgrade pip && pip install -e . && pip install tzdata

# Run as root inside container to match the recorder/dashboard pattern;
# bind-mounts from the host are owned by the host user (ameza) and the
# container needs write access to the WAL directory.
RUN mkdir -p /var/lib/alphaassay/wal

EXPOSE 8003

CMD ["python", "-u", "-m", "infra.feed.run"]
