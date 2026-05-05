.PHONY: install train serve up down test lint \
        mlflow train-local demo demo-down logs check format

# ── Development ───────────────────────────────────────────────────────────────

install:       ## Install package + dev dependencies
	pip install -e ".[dev]"

train:         ## Run training pipeline locally (MLflow must be running)
	python pipelines/train_pipeline.py

serve:         ## Run API locally with hot-reload (model must be in Registry)
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# ── Docker ────────────────────────────────────────────────────────────────────

up:            ## Build and start all services (MLflow + train + API)
	docker compose -f docker/docker-compose.yml up --build -d
	@echo ""
	@echo "  MLflow UI → http://localhost:5000"
	@echo "  API       → http://localhost:8000"
	@echo "  API docs  → http://localhost:8000/docs"
	@echo ""
	@echo "  Tip: run 'make logs' to follow container output"

down:          ## Stop and remove all containers
	docker compose -f docker/docker-compose.yml down

demo-down:     ## Stop containers AND delete volumes (clean slate)
	docker compose -f docker/docker-compose.yml down -v

logs:          ## Follow logs from all containers
	docker compose -f docker/docker-compose.yml logs -f

logs-train:    ## Follow only the training job logs
	docker compose -f docker/docker-compose.yml logs -f train

logs-api:      ## Follow only the API logs
	docker compose -f docker/docker-compose.yml logs -f api

# ── Local dev shortcuts ───────────────────────────────────────────────────────

mlflow:        ## Start MLflow tracking server locally
	mlflow server \
		--host 0.0.0.0 \
		--port 5000 \
		--backend-store-uri sqlite:///mlruns/mlflow.db \
		--default-artifact-root mlruns/artifacts

train-local:   ## MLflow + train in one shot (local, no Docker)
	@echo "Starting MLflow in background..."
	mlflow server \
		--host 0.0.0.0 --port 5000 \
		--backend-store-uri sqlite:///mlruns/mlflow.db \
		--default-artifact-root mlruns/artifacts & \
	sleep 3 && \
	python pipelines/train_pipeline.py

rollback:
	python scripts/rollback.py

registry-list:
	python scripts/registry_list.py

# ── Quality ───────────────────────────────────────────────────────────────────

test:          ## Run test suite
	pytest tests/ -v --tb=short

test-cov:      ## Run tests with coverage report
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:          ## Check code style with ruff
	ruff check src/ tests/ pipelines/

format:        ## Auto-fix style issues
	ruff format src/ tests/ pipelines/

check: lint test  ## lint + test in one command (use before commit)
	@echo "✅ All checks passed"

# ── Help ──────────────────────────────────────────────────────────────────────

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help