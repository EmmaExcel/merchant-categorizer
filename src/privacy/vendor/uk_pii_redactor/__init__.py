"""Vendored subset of the UK-PII-Detector-Redactor project.

Source: https://github.com/EmmaExcel/UK-PII-Detector-Redactor

Only the core configuration, the recognition engine service, and the four UK
recognisers are vendored. The ``app`` package from upstream has been renamed to
``uk_pii_redactor`` to avoid clashing with this repository's own ``app``-style
naming, and the module-level engine is constructed lazily so that importing this
package never raises when optional dependencies (presidio-analyzer,
presidio-anonymizer, spaCy model) are unavailable.

See ATTRIBUTION.md next to this directory for licence and upstream details.
"""

from uk_pii_redactor.services.pii_engine import PiiEngine, get_pii_engine, pii_engine

__all__ = ["PiiEngine", "get_pii_engine", "pii_engine"]
