FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first — layer cache friendly
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Source code
COPY src/       ./src/
COPY pipelines/ ./pipelines/

# Runtime dirs — data/raw se monta via volume en docker-compose
RUN mkdir -p data/raw data/processed mlruns

# Default: serve the API
# Override with --entrypoint or 'command:' in docker-compose for training
EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]