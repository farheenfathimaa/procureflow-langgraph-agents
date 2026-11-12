SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
UV := uv

.PHONY: help install up down logs ps restart test lint format check ingest eval mcp clean

help:
	@echo "ProcureFlow targets"
	@echo "  make install   Create .venv and install app + dev/eval dependencies"
	@echo "  make up        Build and start Postgres, Redis, app and MCP with Docker Compose"
	@echo "  make down      Stop Compose services (volumes are kept)"
	@echo "  make logs      Follow application logs"
	@echo "  make test      Run pytest with MOCK_LLM=true (no external API calls)"
	@echo "  make lint      Run ruff checks"
	@echo "  make format    Format Python with ruff"
	@echo "  make ingest    Re-embed the 20-document corpus inside the app container"
	@echo "  make eval      Run the RAGAS evaluation and update docs/eval_results.md"
	@echo "  make mcp       Run the MCP supplier server over stdio"

install:
	$(UV) python install 3.11
	$(UV) venv --python 3.11 .venv
	$(UV) pip install --python $(PYTHON) -e ".[dev,embeddings,eval]"

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f app

ps:
	docker compose ps

restart:
	docker compose restart app

test:
	MOCK_LLM=true $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

check: lint test

ingest:
	docker compose exec app python -m rag.ingest --force

eval:
	$(PYTHON) -m evals.run_eval

mcp:
	$(PYTHON) -m mcp_server.server

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
