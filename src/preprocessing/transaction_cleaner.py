"""UK transaction-description preprocessing.

Exact pipeline implemented here::

    raw transaction description
      -> PII redaction
      -> uppercase normalisation
      -> Unicode and whitespace normalisation
      -> banking-code detection/tagging
      -> removal of transaction IDs, dates, terminal IDs, and long numeric refs
      -> preservation of useful merchant markers (PAYPAL, SQ, AMZN, TFL, ...)
      -> cleaned description
      -> tokenizer
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from privacy.redactor_adapter import RedactorAdapter, get_redactor

# ---------------------------------------------------------------------------
# Regex rules (order matters)
# ---------------------------------------------------------------------------

_MONTH_NAMES = (
    r"\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|"
    r"OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|JUN|JUL|AUG|SEPT|SEP|OCT|NOV|DEC)\b"
)

# Transfer rails are *preserved* as the TRANSFER marker because they are the
# category signal for Transfers. Replacements carry a trailing space so tokens
# never concatenate (e.g. "TRANSFERSALARY"). Standalone FPS / FASTER PAYMENT
# are deliberately NOT tagged: they are removed by _RAIL_REMOVAL, which keeps
# salary rows like "FPS SALARY ACME LTD" clean.
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

# Rails that carry no category signal are removed.
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

# Markers to preserve, with their noisy prefixes normalised away.
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
    # Compact date tokens such as "22JAN" or "22SEPTEMBER".
    re.compile(r"\b\d{1,2}(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\b"),
    re.compile(r"\b(?:MON|TUE|WED|THU|FRI|SAT|SUN)[A-Z]*\b"),
]

_NUMERIC_NOISE = [
    # Six-digit date-like numbers (YYMMDD) and longer pure-digit references.
    re.compile(r"\b\d{6,}\b"),
    # Terminal IDs and short numeric tails (3-5 digits) that follow merchants.
    re.compile(r"\b\d{3,5}\b"),
    # Alphanumeric transaction IDs: letters + digits, length >= 8.
    re.compile(r"\b(?=[A-Z0-9]{8,}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]+\b"),
]

_PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_]+\]")

_STOPWORD_TOKENS = {"TO", "FROM", "PAYMENT", "PAYMENTS", "REF", "REFERENCE"}


@dataclass
class CleanedTransaction:
    """Output of :meth:`TransactionCleaner.clean`."""

    redacted_description: str
    cleaned_description: str
    tokens: List[str] = field(default_factory=list)
    pii_entity_types: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "redacted_description": self.redacted_description,
            "cleaned_description": self.cleaned_description,
            "tokens": self.tokens,
            "pii_entity_types": self.pii_entity_types,
        }


class TransactionCleaner:
    """Cleans one transaction description according to the documented flow."""

    def __init__(self, redactor: Optional[RedactorAdapter] = None) -> None:
        self.redactor = redactor or get_redactor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clean(self, raw_description: str, redact: bool = True) -> CleanedTransaction:
        """Run the full pipeline on a raw description.

        When ``redact`` is False the input is assumed to be already redacted
        (the caller keeps the redacted text from a previous step).
        """
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
        """Run the non-redaction cleaning stages on already-redacted text."""
        text = redacted_text or ""

        # 1. Uppercase.
        text = text.upper()

        # 2. Unicode + whitespace normalisation.
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()

        # 3. Banking-code detection/tagging (transfer rails become a marker).
        for pattern, replacement in _TRANSFER_TAGGING:
            text = pattern.sub(replacement, text)

        # 4. Marker preservation before generic rail removal.
        for pattern, replacement in _MARKER_NORMALISATION:
            text = pattern.sub(replacement, text)

        # 5. Remove non-signal rails.
        for pattern in _RAIL_REMOVAL:
            text = pattern.sub(" ", text)

        # 6. Remove PII placeholders (they carry no category signal).
        text = _PLACEHOLDER_PATTERN.sub(" ", text)

        # 7. Remove dates, weekdays and month names.
        for pattern in _DATE_PATTERNS:
            text = pattern.sub(" ", text)
        text = re.sub(_MONTH_NAMES, " ", text)

        # 8. Remove transaction IDs, terminal IDs and long numeric references.
        for pattern in _NUMERIC_NOISE:
            text = pattern.sub(" ", text)

        # 9. Drop residual stopword tokens.
        text = " ".join(
            token for token in text.split() if token not in _STOPWORD_TOKENS
        )

        # 10. Final whitespace collapse and tidy punctuation.
        text = re.sub(r"\s+", " ", text).strip()
        # Keep intra-word hyphens (CO-OP) but remove other punctuation.
        text = re.sub(r"(?<!\w)-(?!\w)", " ", text)
        text = re.sub(r"[^\w\s&'\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def tokenize(cleaned_text: str) -> List[str]:
        """Tokenize a cleaned description into uppercase word tokens."""
        if not cleaned_text:
            return []
        # Keep alphanumerics, ampersands, apostrophes and intra-word hyphens.
        text = re.sub(r"[^A-Z0-9&'\-]+", " ", cleaned_text.upper())
        return [token for token in text.split() if token]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @staticmethod
    def merchant_signature(cleaned_description: str, merchant_name: Optional[str]) -> str:
        """A normalised merchant signature used for train/test group splits.

        Uses the structured merchant name when available; otherwise falls back
        to the cleaned description. This keeps near-duplicate descriptions in
        the same split group.
        """
        if merchant_name and merchant_name.strip():
            name = unicodedata.normalize("NFKC", merchant_name.strip().upper())
            name = re.sub(r"[^A-Z0-9&']+", " ", name).strip()
            tokens = name.split()
            if tokens:
                # First normalised token groups merchant variants such as
                # "TFL" / "TFL TRAVEL CHARGE" / "TRANSPORT FOR LONDON" into one
                # merchant family so near-duplicates stay in one split.
                return tokens[0]
        cleaned = cleaned_description or ""
        tokens = cleaned.split()
        return " ".join(tokens[:3]) if tokens else "unknown"
