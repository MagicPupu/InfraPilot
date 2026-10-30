.PHONY: install test lint pull-models eval

LLM_MODEL ?= $(shell sed -n 's/^LLM_MODEL=//p' .env.example)
EMBED_MODEL ?= $(shell sed -n 's/^EMBED_MODEL=//p' .env.example)

install:
	uv sync

test:
	uv run pytest -m "not ollama"

lint:
	uv run ruff check src tests

pull-models:
	ollama pull $(LLM_MODEL)
	ollama pull $(EMBED_MODEL)

# Local-only: never run in CI.
eval:
	uv run python -m infrapilot.evals.run_evals
