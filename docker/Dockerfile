FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies — cached layer (only invalidates if pyproject.toml changes) ──
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .      # sin -e: install real, no symlinks

# ── Source code ───────────────────────────────────────────────────────────────
COPY src/       ./src/
COPY pipelines/ ./pipelines/

# ── Runtime directories ───────────────────────────────────────────────────────
# data/raw y data/processed se montan via volume en docker-compose
RUN mkdir -p data/raw data/processed mlruns

# ── Non-root user — security best practice ───────────────────────────────────
RUN useradd --create-home --no-log-init appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Default command: serve API ────────────────────────────────────────────────
# Override via 'command:' in docker-compose to run training:
#   command: ["python", "pipelines/train_pipeline.py"]
EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
