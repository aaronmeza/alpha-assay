FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY src /app/src
COPY infra/feed /app/infra/feed
COPY configs /app/configs

RUN pip install --no-cache-dir -e .

RUN useradd -u 10001 -m feed && \
    mkdir -p /var/lib/alphaassay/wal && \
    chown -R feed:feed /var/lib/alphaassay

USER feed

EXPOSE 8003

CMD ["python", "-u", "-m", "infra.feed.run"]
