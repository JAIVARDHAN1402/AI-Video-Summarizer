# Image for both the API (server.main) and the RQ workers (server.worker).
# Same code, different command — see docker-compose.yml.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    VS_CACHE_DIR=/data/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-server.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -r requirements-server.txt

COPY . .

EXPOSE 8000
CMD ["python", "-m", "server.main"]
