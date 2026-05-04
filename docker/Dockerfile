FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .
COPY src/ ./src/
COPY pipelines/ ./pipelines/
RUN mkdir -p data/raw data/processed
EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
