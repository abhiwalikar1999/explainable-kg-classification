# Shortcuts for Linux/macOS/Docker. On Windows run the python commands.
PY ?= python

.PHONY: all lint test quality pipeline gnn demo docker-build

all: lint test quality pipeline

lint:
	$(PY) -m ruff check src tests app

test:
	$(PY) -m pytest -q -m "not quality"

quality:  ## model-quality gate: results reproduce, accuracy floors hold
	$(PY) -m pytest -q -m quality

pipeline:  ## every stage except the GNN (~8 min)
	$(PY) -m src.pipeline

gnn:  ## R-GCN baselines, needs requirements-gnn.txt (~45 min CPU)
	$(PY) -m src.pipeline --stages gnn figures

demo:
	$(PY) -m streamlit run app/demo.py

docker-build:
	docker build -t explainable-kg-classification .
