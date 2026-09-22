#!/bin/sh
set -e

if command -v alembic >/dev/null 2>&1 && [ "${UKMC_SKIP_MIGRATIONS:-false}" != "true" ]; then
  alembic upgrade head || echo "Alembic migration skipped (database not reachable yet)"
fi

if [ ! -f "${UKMC_MODEL_DIR:-artifacts/current}/model.pt" ]; then
  echo "Warning: no model artifact at ${UKMC_MODEL_DIR:-artifacts/current}; serving bootstrap keyword model" >&2
fi

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
