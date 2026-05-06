FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    TZ=America/Chicago

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY infra/bot ./infra/bot

RUN pip install --upgrade pip && pip install -e . && pip install docker

RUN mkdir -p /var/lib/alphaassay/bot

CMD ["python", "-u", "-m", "infra.bot.main"]
