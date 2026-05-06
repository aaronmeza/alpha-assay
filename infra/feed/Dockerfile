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

RUN pip install --upgrade pip && pip install -e .

RUN useradd -u 10001 -m feed && \
    mkdir -p /var/lib/alphaassay/wal && \
    chown -R feed:feed /var/lib/alphaassay

USER feed

EXPOSE 8003

CMD ["python", "-u", "-m", "infra.feed.run"]
