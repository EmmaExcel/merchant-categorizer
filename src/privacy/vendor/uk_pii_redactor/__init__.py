"""Vendored subset of the UK-PII-Detector-Redactor project.

Source: https://github.com/EmmaExcel/UK-PII-Detector-Redactor
See ATTRIBUTION.md in this directory for licence and upstream details.
"""

from uk_pii_redactor.services.pii_engine import PiiEngine, get_pii_engine, pii_engine

__all__ = ["PiiEngine", "get_pii_engine", "pii_engine"]
