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
# torch's default Linux wheel bundles the CUDA runtime — roughly 2.5 GB of it —
# which is dead weight on a CPU-only box and slows every cold start and deploy.
# Install the CPU build first so the requirements files resolve against it.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 torchvision==0.18.1 && \
    pip install -r requirements.txt -r requirements-server.txt

COPY . .

EXPOSE 8000
CMD ["python", "-m", "server.main"]
