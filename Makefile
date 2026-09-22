.PHONY: install install-pii data train train-minilm train-bilstm evaluate api test docker-up docker-down clean

PY := .venv/bin/python
PIP := .venv/bin/pip

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-pii: install
	$(PIP) install -e ".[pii]"
	$(PY) -m spacy download en_core_web_sm

data:
	$(PY) scripts/generate_synthetic_data.py --records 2400 --seed 42

train-minilm:
	$(PY) -m src.training.train --model-type minilm --epochs 30 --output-dir artifacts/current

train-bilstm:
	$(PY) -m src.training.train --model-type bilstm --epochs 20 --output-dir artifacts/current

train: train-minilm

evaluate:
	$(PY) -m src.training.evaluate --artifact artifacts/current

api:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf artifacts/current data/processed/*.csv data/processed/*.npz data/*.db .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
