#!/bin/sh
set -e

if command -v alembic >/dev/null 2>&1 && [ "${UKMC_SKIP_MIGRATIONS:-false}" != "true" ]; then
  alembic upgrade head || echo "Alembic migration skipped (database not reachable yet)"
fi

if [ ! -f "${UKMC_MODEL_DIR:-artifacts/current}/model.pt" ]; then
  echo "No model artifact found at ${UKMC_MODEL_DIR:-artifacts/current}; training MiniLM on synthetic/sandbox data..."
  python -m src.training.train --model-type minilm --epochs 20 --output-dir "${UKMC_MODEL_DIR:-artifacts/current}"
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000
