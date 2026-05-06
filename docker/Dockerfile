FROM python:3.12-slim

# ROOT: sistema + usuario
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential curl git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

# ROOT: directorios con permisos
RUN mkdir -p /app/data/raw /app/data/processed /app/mlruns \
    && chown -R appuser:appuser /app

# appuser desde aquí
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"
WORKDIR /app

# deps Python
COPY --chown=appuser:appuser pyproject.toml README.md ./
RUN pip install --no-cache-dir ".[dev]"

# código
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pipelines/ ./pipelines/

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]