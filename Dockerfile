FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal build dependencies (torch/presidio ship wheels; libgomp helps torch).
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the project first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY alembic.ini ./
COPY alembic ./alembic
COPY data ./data

RUN pip install --no-cache-dir -e ".[pii]"

# Download the spaCy model used by the vendored UK-PII-Detector-Redactor and the
# MiniLM sentence encoder at build time so container startup is fast.
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
