# Vendored code attribution

This directory contains a vendored subset of
[UK-PII-Detector-Redactor](https://github.com/EmmaExcel/UK-PII-Detector-Redactor).

## What is vendored

- `uk_pii_redactor/core/config.py` — upstream application settings
- `uk_pii_redactor/services/pii_engine.py` — upstream `PiiEngine` detection/redaction service
- `uk_pii_redactor/recognizers/` — upstream UK NHS number, NINO, postcode and phone recognisers

## Modifications made for this project

1. The upstream `app` package has been renamed to `uk_pii_redactor` so it does
   not collide with this repository's own package naming, and imports were
   updated accordingly.
2. The module-level `pii_engine = PiiEngine()` singleton is now constructed
   lazily through `get_pii_engine()` so importing the package never fails when
   the optional presidio/spaCy dependencies are not installed.
3. No upstream detection logic has been changed.

## Dependency requirement

The vendored code requires `presidio-analyzer`, `presidio-anonymizer` and the
spaCy `en_core_web_sm` model, exactly as the upstream project does. Install
them with:

```bash
pip install -e ".[pii]"
python -m spacy download en_core_web_sm
```

If they are unavailable at runtime, `src/privacy/redactor_adapter.py` falls back
to a deterministic UK banking regex redactor so the pipeline still fails safe.
