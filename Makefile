.PHONY: install train serve up down test lint

install:
	pip install -e ".[dev]"

train:
	python pipelines/train_pipeline.py

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

up:
	docker compose -f docker/docker-compose.yml up --build -d
	@echo ""
	@echo "MLflow UI → http://localhost:5000"
	@echo "API       → http://localhost:8000"
	@echo "API docs  → http://localhost:8000/docs"

down:
	docker compose -f docker/docker-compose.yml down

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ pipelines/
