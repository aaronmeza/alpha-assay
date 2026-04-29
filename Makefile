.PHONY: help setup lint test sample backtest paper report kill reset

PYTHON ?= python3
VENV ?= .venv

.DEFAULT_GOAL := help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create venv and install package with all extras
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip wheel
	$(VENV)/bin/pip install -e ".[nautilus,databento,ibkr,observability,dev]"

lint: ## Run ruff + black --check
	$(VENV)/bin/ruff check .
	$(VENV)/bin/black --check .

test: ## Run pytest
	$(VENV)/bin/pytest -q

sample: ## Generate synthetic 2-day sample fixture
	$(VENV)/bin/python tests/fixtures/make_sample.py --out tests/fixtures/sample_2d.csv --days 2

backtest: ## Run backtest via alpha_assay CLI (pass ARGS=...)
	$(VENV)/bin/alpha_assay backtest $(ARGS)

paper: ## Run paper trading session (pass ARGS=...)
	$(VENV)/bin/alpha_assay paper $(ARGS)

report: ## Generate reports (pass ARGS=...)
	$(VENV)/bin/alpha_assay report $(ARGS)

kill: ## Kill any running alpha_assay processes
	$(VENV)/bin/alpha_assay kill

reset: ## Reset alpha_assay state (pass ARGS=...)
	$(VENV)/bin/alpha_assay reset $(ARGS)
