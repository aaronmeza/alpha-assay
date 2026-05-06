FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY src /app/src
COPY infra/bot /app/infra/bot

RUN pip install --no-cache-dir -e . docker

RUN mkdir -p /var/lib/alphaassay/bot

CMD ["python", "-u", "-m", "infra.bot.main"]
