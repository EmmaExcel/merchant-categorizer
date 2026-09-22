
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from privacy.redactor_adapter import RedactorAdapter, get_redactor

_MONTH_NAMES = (
    r"\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|"
    r"OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|JUN|JUL|AUG|SEPT|SEP|OCT|NOV|DEC)\b"
)






_TRANSFER_TAGGING = [
    (re.compile(r"\bTRANSFER\s+(?:TO|FROM|INTO)\b", re.IGNORECASE), "TRANSFER "),
    (re.compile(r"\bTFR\s+(?:TO|FROM|INTO)\b", re.IGNORECASE), "TRANSFER "),
    (re.compile(r"\bTFR\b", re.IGNORECASE), "TRANSFER "),
    (re.compile(r"\bBANK\s+TRANSFER\s+(?:TO|FROM|INTO)\b", re.IGNORECASE), "TRANSFER "),
    (re.compile(r"\bFASTER\s+PAYMENTS?\s+(?:TO|FROM|INTO)\b", re.IGNORECASE), "TRANSFER "),
    (re.compile(r"\bFPS\s+(?:TO|FROM|INTO)\b", re.IGNORECASE), "TRANSFER "),
    (re.compile(r"\bCASH\s+WITHDRAWAL\b", re.IGNORECASE), "CASH WITHDRAWAL "),
    (re.compile(r"\bATM\s+WITHDRAWAL\b", re.IGNORECASE), "CASH WITHDRAWAL "),
]


_RAIL_REMOVAL = [
    re.compile(r"\bCARD\s+(?:PAYMENT|PURCHASE|TRANSACTION|SALE)\b", re.IGNORECASE),
    re.compile(r"\bCARD\b", re.IGNORECASE),
    re.compile(r"\bPOS\b", re.IGNORECASE),
    re.compile(r"\bBACS\b", re.IGNORECASE),
    re.compile(r"\bDIRECT\s+DEBIT\b", re.IGNORECASE),
    re.compile(r"\bDD\b", re.IGNORECASE),
    re.compile(r"\bSTANDING\s+ORDER\b", re.IGNORECASE),
    re.compile(r"\bSO\b", re.IGNORECASE),
    re.compile(r"\bCHAPS\b", re.IGNORECASE),
    re.compile(r"\bFASTER\s+PAYMENTS?\b", re.IGNORECASE),
    re.compile(r"\bFPS\b", re.IGNORECASE),
    re.compile(r"\bAPPLE\s+PAY\b", re.IGNORECASE),
    re.compile(r"\bGOOGLE\s+PAY\b|\bGPAY\b", re.IGNORECASE),
    re.compile(r"\bSTRIPE\b", re.IGNORECASE),
    re.compile(r"\bVISA\b|\bMASTERCARD\b|\bMAESTRO\b", re.IGNORECASE),
]


_MARKER_NORMALISATION = [
    (re.compile(r"\bPAYPAL\s*\*", re.IGNORECASE), "PAYPAL "),
    (re.compile(r"\bSQ\s*\*", re.IGNORECASE), "SQ "),
    (re.compile(r"\bSQUARE\s*(?:CASH|UP)?\s*\*", re.IGNORECASE), "SQ "),
]

_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\s+\d{2,4}\b"),
    re.compile(r"\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\s+\d{2,4}\b"),

    re.compile(r"\b\d{1,2}(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\b"),
    re.compile(r"\b(?:MON|TUE|WED|THU|FRI|SAT|SUN)[A-Z]*\b"),
]

_NUMERIC_NOISE = [

    re.compile(r"\b\d{6,}\b"),

    re.compile(r"\b\d{3,5}\b"),

    re.compile(r"\b(?=[A-Z0-9]{8,}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]+\b"),
]

_PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_]+\]")

_STOPWORD_TOKENS = {"TO", "FROM", "PAYMENT", "PAYMENTS", "REF", "REFERENCE"}


@dataclass
class CleanedTransaction:
    redacted_description: str
    cleaned_description: str
    tokens: List[str] = field(default_factory=list)
    pii_entity_types: List[str] = field(default_factory=list)


class TransactionCleaner:
    def __init__(self, redactor: Optional[RedactorAdapter] = None) -> None:
        self.redactor = redactor or get_redactor()




    def clean(self, raw_description: str, redact: bool = True) -> CleanedTransaction:
        redaction = self.redactor.redact(raw_description) if redact else None
        redacted = redaction.redacted_text if redaction else raw_description
        entity_types = redaction.entity_types if redaction else []

        cleaned = self.clean_redacted(redacted)
        return CleanedTransaction(
            redacted_description=redacted.strip(),
            cleaned_description=cleaned,
            tokens=self.tokenize(cleaned),
            pii_entity_types=entity_types,
        )

    def clean_redacted(self, redacted_text: str) -> str:
        text = redacted_text or ""

        text = text.upper()
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()



        for pattern, replacement in _TRANSFER_TAGGING:
            text = pattern.sub(replacement, text)
        for pattern, replacement in _MARKER_NORMALISATION:
            text = pattern.sub(replacement, text)
        for pattern in _RAIL_REMOVAL:
            text = pattern.sub(" ", text)


        text = _PLACEHOLDER_PATTERN.sub(" ", text)



        for pattern in _DATE_PATTERNS:
            text = pattern.sub(" ", text)
        text = re.sub(_MONTH_NAMES, " ", text)
        for pattern in _NUMERIC_NOISE:
            text = pattern.sub(" ", text)

        text = " ".join(
            token for token in text.split() if token not in _STOPWORD_TOKENS
        )



        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(?<!\w)-(?!\w)", " ", text)
        text = re.sub(r"[^\w\s&'\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def tokenize(cleaned_text: str) -> List[str]:
        if not cleaned_text:
            return []

        text = re.sub(r"[^A-Z0-9&'\-]+", " ", cleaned_text.upper())
        return [token for token in text.split() if token]




    @staticmethod
    def merchant_signature(cleaned_description: str, merchant_name: Optional[str]) -> str:
        if merchant_name and merchant_name.strip():
            name = unicodedata.normalize("NFKC", merchant_name.strip().upper())
            name = re.sub(r"[^A-Z0-9&']+", " ", name).strip()
            tokens = name.split()
            if tokens:



                return tokens[0]
        cleaned = cleaned_description or ""
        tokens = cleaned.split()
        return " ".join(tokens[:3]) if tokens else "unknown"
