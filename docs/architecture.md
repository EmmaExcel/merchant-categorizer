# Architecture

## Overview

The service is a local, privacy-first pipeline:

```
raw transaction description
  -> PII redaction (adapter over UK-PII-Detector-Redactor + UK banking regex)
  -> preprocessing (uppercase, Unicode/whitespace, banking-code tagging,
     numeric-noise removal, marker preservation)
  -> metadata features (amount bucket, direction, payment rail, MCC, PII flag)
  -> PyTorch classifier (MiniLM default or BiLSTM-attention baseline)
  -> FastAPI response (category, confidence, Top-3, review flag)
  -> persistence (redacted/cleaned text only)
```

## Components

### Ingestion (`src/ingestion`)

- `providers/base.py` — `BankProvider` ABC with `fetch_accounts()`,
  `fetch_transactions(account_id, from_date, to_date)` and
  `normalise_transaction(raw_transaction)`.
- `providers/truelayer_sandbox.py` — TrueLayer Data API Sandbox provider.
  Uses `POST /connect/token`, `GET /data/v1/accounts`, and
  `GET /data/v1/accounts/{id}/transactions`. Defaults to local fixture mode
  (`data/raw/truelayer_sandbox_example.json`) when credentials are absent.
- `fixtures.py` — loaders for the committed synthetic CSV and sandbox JSON.

### Privacy (`src/privacy`)

- `redactor_adapter.py` — the only place raw text is redacted. Wraps the
  vendored `uk_pii_redactor` package (upstream UK-PII-Detector-Redactor,
  Presidio + spaCy) and a deterministic UK banking regex backend for sort codes
  and account numbers. Canonical placeholders: `[PERSON]`, `[SORT_CODE]`,
  `[ACCOUNT_NUMBER]`, `[EMAIL]`, `[PHONE]`, `[ADDRESS]`.
- `vendor/uk_pii_redactor/` — vendored upstream core/recognisers/services with
  attribution (`ATTRIBUTION.md`).

### Preprocessing (`src/preprocessing`)

- `transaction_cleaner.py` — implements the exact documented pipeline:

  1. PII redaction
  2. uppercase normalisation
  3. Unicode (NFKC) and whitespace normalisation
  4. banking-code detection/tagging (`TFR` → `TRANSFER`,
     `ATM WITHDRAWAL` → `CASH WITHDRAWAL`)
  5. removal of transaction IDs, dates, terminal IDs, long numeric references
  6. preservation of merchant markers (`PAYPAL`, `SQ`, `AMZN`, `TFL`, …)
  7. cleaned description
  8. tokenizer

- `features.py` — the 21 labels, amount buckets
  (`micro/small/medium/large/very_large`), direction, payment-rail detection,
  MCC bucketing, PII type vector, and tensor encoding.

### Models (`src/models`)

- `base.py` — shared `BaseCategoriser` interface, `ModelConfig`,
  `MetaEmbedder`, device selection, seed configuration, warmup/cosine schedule,
  and full-artifact save/load.
- `minilm_classifier.py` — default model. Sentence-transformers
  `all-MiniLM-L6-v2` pooled embedding + learned categorical metadata embeddings
  + dropout MLP head.
- `bilstm_attention.py` — educational baseline. WordPiece tokenizer trained
  only on the synthetic training corpus, learned embedding, bidirectional LSTM,
  additive attention pooling implemented in PyTorch, metadata embeddings, and a
  linear head.
- `bootstrap.py` — deterministic keyword fallback used only when no trained
  artifact exists (labelled `bootstrap` in API responses).

### Training (`src/training`)

- `config.py` — `TrainingConfig` dataclass.
- `data.py` — redaction, cleaning, feature preparation, and the grouped
  70/15/15 split by merchant signature.
- `train.py` — training loop: class weights from the train split only, AdamW,
  linear warmup + cosine decay, gradient clipping, early stopping on validation
  macro F1, best-checkpoint persistence, model-card generation, and file-based
  run logging.
- `evaluate.py` — macro/weighted F1, Top-1/Top-3 accuracy, per-class report,
  confusion matrix, confidence distribution, and an incorrect-predictions CSV
  that excludes raw text.
- `metrics.py` — numpy-only metric implementations.
- `experiment_logger.py` — lightweight JSONL run tracker under `artifacts/runs`.

### API (`src/api`)

- `main.py` — FastAPI app with `/health`, `/model`, `/predict`,
  `/predict/batch`, `/feedback`, `/data/{subject_id}` (deletion), and an
  admin reload endpoint.
- `schemas.py` — Pydantic request/response models.
- `dependencies.py` — lazy singleton model loading with bootstrap fallback,
  redactor and cleaner dependencies.

### Database (`src/database`)

- `models.py` — `redacted_transactions`, `predictions`, `feedback_labels`,
  `model_versions`, `training_runs`.
- `session.py` — SQLAlchemy engine/session factory; SQLite locally, PostgreSQL
  in production via `UKMC_DATABASE_URL`.
- `deletion.py` — `delete_subject_data()` and `delete_transaction_data()`.

## Data flow invariants

1. Raw descriptions are redacted **before** preprocessing or persistence.
2. The processed training dataset has no `raw_description` column.
3. Feedback stores redacted text only.
4. Logs never contain raw PII after the redaction stage.
