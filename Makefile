.PHONY: help setup lint test sample backtest paper report kill reset

PYTHON ?= python3
VENV ?= .venv

# Resolve interpreter and tool paths: prefer the project venv when it
# exists, otherwise fall back to whatever is on PATH. This lets fresh
# clones run targets like `make sample` without first running
# `make setup`, and lets CI invoke targets without provisioning a venv.
PY := $(shell test -x $(VENV)/bin/python && echo $(VENV)/bin/python || echo $(PYTHON))
RUFF := $(shell test -x $(VENV)/bin/ruff && echo $(VENV)/bin/ruff || echo ruff)
BLACK := $(shell test -x $(VENV)/bin/black && echo $(VENV)/bin/black || echo black)
PYTEST := $(shell test -x $(VENV)/bin/pytest && echo $(VENV)/bin/pytest || echo pytest)
ALPHA_ASSAY := $(shell test -x $(VENV)/bin/alpha_assay && echo $(VENV)/bin/alpha_assay || echo alpha_assay)

.DEFAULT_GOAL := help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create venv and install package with all extras
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip wheel
	$(VENV)/bin/pip install -e ".[nautilus,databento,ibkr,observability,dev]"

lint: ## Run ruff + black --check
	$(RUFF) check .
	$(BLACK) --check .

test: ## Run pytest
	$(PYTEST) -q

sample: ## Generate synthetic 2-day sample fixture
	$(PY) tests/fixtures/make_sample.py --out tests/fixtures/sample_2d.csv --days 2

backtest: ## Run backtest via alpha_assay CLI (pass ARGS=...)
	$(ALPHA_ASSAY) backtest $(ARGS)

paper: ## Run paper trading session (pass ARGS=...)
	$(ALPHA_ASSAY) paper $(ARGS)

report: ## Generate reports (pass ARGS=...)
	$(ALPHA_ASSAY) report $(ARGS)

kill: ## Kill any running alpha_assay processes
	$(ALPHA_ASSAY) kill

reset: ## Reset alpha_assay state (pass ARGS=...)
	$(ALPHA_ASSAY) reset $(ARGS)
