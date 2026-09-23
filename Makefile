.PHONY: help setup test lint clean ingest benchmark-embed serve

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Distributed Wikipedia Semantic Search & RAG Engine"
	@echo "Available commands:"
	@echo "  make setup            Create venv and install dependencies"
	@echo "  make test             Run unit and integration tests"
	@echo "  make test-unit        Run fast unit tests"
	@echo "  make sample-data      Generate synthetic 100k-doc Wikipedia Parquet dataset"
	@echo "  make run-ingest       Run the Map-Reduce ingestion pipeline"
	@echo "  make benchmark-embed  Run GPU embedding throughput benchmark"
	@echo "  make serve            Start FastAPI serving layer"

setup:
	@test -d $(VENV) || /usr/bin/python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@mkdir -p data/raw data/chunks data/indexes data/cache

test:
	$(PYTHON) -m pytest tests/ -v

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

sample-data:
	$(PYTHON) scripts/generate_sample_data.py --output data/raw/wikipedia_sample_100k.parquet --count 100000

run-ingest:
	$(PYTHON) scripts/run_ingest.py --config configs/ingestion.yaml

benchmark-embed:
	$(PYTHON) scripts/benchmark_embed.py --config configs/embedding.yaml

serve:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

