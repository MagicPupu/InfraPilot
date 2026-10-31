.PHONY: install up down ingest ask ui test lint typecheck eval pull-models

LLM_MODEL ?= $(shell sed -n 's/^LLM_MODEL=//p' .env.example)
EMBED_MODEL ?= $(shell sed -n 's/^EMBED_MODEL=//p' .env.example)
Q ?= What is the naming convention for containers?

install:
	uv sync

up:
	docker compose up -d --wait db langfuse-web langfuse-worker

down:
	docker compose down

ingest:
	uv run infrapilot ingest docs/runbooks

ask:
	uv run infrapilot ask "$(Q)"

ui:
	uv run --extra ui streamlit run src/infrapilot/ui.py

test:
	uv run pytest -m "not ollama"

lint:
	uv run ruff check src tests evals && uv run ruff format --check src tests evals

typecheck:
	uv run mypy

pull-models:
	ollama pull $(LLM_MODEL)
	ollama pull $(EMBED_MODEL)

# Local-only (needs Ollama): never run in CI.
eval:
	uv run python evals/run_evals.py $(if $(MODELS),--models $(MODELS),)
