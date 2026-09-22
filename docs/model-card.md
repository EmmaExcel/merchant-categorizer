# Model card (repository template)

The definitive, auto-generated model card for each trained artifact is written
to `artifacts/<run>/model-card.md` by the training script. This file describes
the shared context for all models in this repository.

## Model overview

| | Default model | Educational baseline |
|---|---|---|
| Name | `minilm` | `bilstm` |
| Text encoder | `sentence-transformers/all-MiniLM-L6-v2` pooled embedding | WordPiece tokenizer trained only on the synthetic corpus |
| Sequence model | none; the encoder is not fine-tuned | bidirectional LSTM + additive attention pooling |
| Metadata | amount bucket, direction, payment rail, MCC, PII type vector | same |
| Head | dropout + MLP | linear |
| Framework | PyTorch only | PyTorch only |

## Data

- `data/raw/synthetic_uk_transactions.csv` — 2,400 generated, labelled, noisy
  UK transaction descriptions (seeded, deterministic).
- `data/raw/truelayer_sandbox_example.json` — TrueLayer Data API sandbox
  example fixture projected from the same synthetic records.
- No real personal bank transaction data is used.

## Intended use

Local, privacy-first merchant categorisation of UK bank transaction
descriptions for personal finance tooling and research prototypes.

## Out-of-scope / non-goals

- Production categorisation of real consumer banking data.
- Any financial, credit, or fraud decisioning.
- Any external LLM inference.

## Limitations

- Synthetic merchants and descriptions do not capture the long tail of real UK
  transaction data.
- The merchant-grouped split means the test set contains merchants unseen in
  training; real-world generalisation is likely lower than the reported scores.
- PII redaction is heuristic and may miss novel personal-data formats.
- All reported metrics are **synthetic/sandbox evaluation results**, not
  real-world production performance.

## Metrics (reported on the held-out synthetic/sandbox test split)

The training script targets, and the evaluation script reports:

- **Macro F1** — primary metric
- **Weighted F1**
- **Top-1 accuracy**
- **Top-3 accuracy** — the correct label appears anywhere in the three
  highest-probability predictions
- Per-class precision, recall, F1 and support
- Confusion matrix and confidence distribution (images)
- Incorrect-prediction CSV with raw descriptions excluded

## Feedback and retraining

The `/feedback` endpoint stores corrected labels against **redacted** text
only. A future iteration can retrain on those consented, redacted records.
