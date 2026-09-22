"""PII redaction adapter for UK bank transaction descriptions.

This module is the single choke point through which every raw transaction
description passes before preprocessing or persistence.

It wraps the vendored `UK-PII-Detector-Redactor`_ implementation (Microsoft
Presidio + spaCy plus UK recognisers) and layers a deterministic UK banking
regex backend on top for identifiers the upstream project does not cover, such
as sort codes and account numbers.

.. _UK-PII-Detector-Redactor: https://github.com/EmmaExcel/UK-PII-Detector-Redactor

Design rules
------------
* Accept raw transaction description text.
* Call the upstream ``PiiEngine.redact_text`` interface unchanged when
  available.
* Return redacted text plus a structured list of detected entity types.
* Fail safely: raw PII is never written to logs or exception messages.
* Easy to replace: implement :class:`RedactorBackend` and pass it to
  :class:`RedactorAdapter` (or set ``UKMC_REDACTOR_BACKEND`` later).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical placeholders required by the project
# ---------------------------------------------------------------------------

PLACEHOLDERS: Dict[str, str] = {
    "PERSON": "[PERSON]",
    "SORT_CODE": "[SORT_CODE]",
    "ACCOUNT_NUMBER": "[ACCOUNT_NUMBER]",
    "EMAIL": "[EMAIL]",
    "PHONE": "[PHONE]",
    "ADDRESS": "[ADDRESS]",
    "CARD_NUMBER": "[CARD_NUMBER]",
    "IBAN": "[IBAN]",
    "OTHER": "[REDACTED]",
}

# Upstream entity types we request from the UK-PII-Detector-Redactor. We
# deliberately exclude ORGANIZATION and LOCATION for merchant descriptions:
# "TESCO STORES LONDON" must survive redaction so the classifier can use it.
UPSTREAM_TARGET_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "UK_PHONE_NUMBER",
    "PHONE_NUMBER",
    "UK_POSTCODE",
    "CREDIT_CARD",
    "IBAN_CODE",
]

# Upstream entity type -> canonical entity type used across this project.
UPSTREAM_TO_CANONICAL = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "UK_PHONE_NUMBER": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "UK_POSTCODE": "ADDRESS",
    "CREDIT_CARD": "CARD_NUMBER",
    "IBAN_CODE": "IBAN",
    "UK_NHS_NUMBER": "OTHER",
    "UK_NINO": "OTHER",
    "IP_ADDRESS": "OTHER",
    "DATE_TIME": "OTHER",
}


@dataclass(frozen=True)
class RedactionEntity:
    """A single redacted entity with its canonical type and placeholder."""

    entity_type: str
    placeholder: str
    start: int
    end: int
    score: float = 1.0
    source: str = "uk-banking-regex"


@dataclass
class RedactionResult:
    """Result of redacting one description.

    ``redacted_text`` never contains the raw matched PII values. On failure,
    ``ok`` is False and ``redacted_text`` is a safe controlled marker.
    """

    redacted_text: str
    entities: List[RedactionEntity] = field(default_factory=list)
    ok: bool = True
    error: Optional[str] = None
    backend: str = "none"

    @property
    def entity_types(self) -> List[str]:
        """Ordered list of canonical entity types that were redacted."""
        return [e.entity_type for e in self.entities]


class RedactionError(Exception):
    """Controlled redaction failure. Never carries raw PII in the message."""

    def __init__(self, message: str = "Redaction failed; raw text was not logged."):
        super().__init__(message)


class RedactorBackend(Protocol):
    """Minimal interface for a pluggable redaction backend."""

    name: str

    def detect(self, text: str) -> List[RedactionEntity]:
        """Return redaction entities for ``text`` without mutating it."""
        ...


# ---------------------------------------------------------------------------
# Backend 1: vendored UK-PII-Detector-Redactor (presidio + spaCy)
# ---------------------------------------------------------------------------

class PresidioUkBackend:
    """Calls the vendored upstream ``PiiEngine.redact_text`` interface.

    The upstream API used here is:

    ``PiiEngine.redact_text(text, entities=None, mode="placeholder",
    mask_char="*", score_threshold=0.5) -> Dict``

    with keys ``original_text``, ``redacted_text``, ``entities`` and
    ``total_entities_found``.
    """

    name = "uk-pii-detector-redactor"

    def __init__(self, score_threshold: float = 0.45) -> None:
        self.score_threshold = score_threshold
        self._engine = None
        self._init_error: Optional[str] = None

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        if self._init_error is not None:
            raise RedactionError(self._init_error)
        try:
            from uk_pii_redactor.services.pii_engine import get_pii_engine

            self._engine = get_pii_engine()
            logger.info("uk-pii-detector-redactor engine loaded (presidio + spaCy)")
        except Exception as exc:  # noqa: BLE001 - fail safe, never log raw text
            self._init_error = (
                "uk-pii-detector-redactor unavailable; falling back to regex backend"
            )
            logger.warning("%s (%s)", self._init_error, type(exc).__name__)
            raise RedactionError(self._init_error) from exc
        return self._engine

    def detect(self, text: str) -> List[RedactionEntity]:
        engine = self._ensure_engine()
        result = engine.redact_text(
            text=text,
            entities=UPSTREAM_TARGET_ENTITIES,
            mode="placeholder",
            score_threshold=self.score_threshold,
        )
        entities: List[RedactionEntity] = []
        for item in result.get("entities", []):
            upstream_type = item.get("entity_type", "")
            canonical = UPSTREAM_TO_CANONICAL.get(upstream_type, "OTHER")
            entities.append(
                RedactionEntity(
                    entity_type=canonical,
                    placeholder=PLACEHOLDERS[canonical],
                    start=int(item.get("start", 0)),
                    end=int(item.get("end", 0)),
                    score=float(item.get("score", 1.0)),
                    source=self.name,
                )
            )
        return entities


# ---------------------------------------------------------------------------
# Backend 2: deterministic UK banking regex (always available, no network)
# ---------------------------------------------------------------------------

class UkBankingRegexBackend:
    """Deterministic regex redaction for UK banking identifiers.

    Covers the entity types required by the project that the upstream redactor
    does not detect (sort codes, account numbers) plus safety-net patterns for
    email, UK phone numbers, postcodes/addresses and card numbers.
    """

    name = "uk-banking-regex"

    # Order matters: longer / more specific patterns first.
    _PATTERNS: List[tuple[str, re.Pattern[str], float]] = [
        # UK sort code: 20-45-67
        ("SORT_CODE", re.compile(r"\b\d{2}-\d{2}-\d{2}\b"), 1.0),
        # Email address
        ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), 1.0),
        # UK phone numbers: mobile 07xxx, +44 7xxx, geographic 01/02/03
        (
            "PHONE",
            re.compile(
                r"(?:\+44\s?7\d{3}\s?\d{3}\s?\d{3}|\b07\d{3}\s?\d{3}\s?\d{3}\b|"
                r"\b(?:01\d{2,4}|02\d|03\d{2})\s?\d{3,4}\s?\d{3,4}\b)"
            ),
            1.0,
        ),
        # IBAN (GB-prefixed, common spacing)
        (
            "IBAN",
            re.compile(r"\bGB\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b"),
            1.0,
        ),
        # Card number: 13-19 digits, optionally space/grouped
        (
            "CARD_NUMBER",
            re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
            0.9,
        ),
        # UK postcode -> ADDRESS
        (
            "ADDRESS",
            re.compile(
                r"\b(?:GIR\s?0AA|(?:[A-PR-UWYZ][0-9][0-9A-HJKPSTUW]?|[A-PR-UWYZ]"
                r"[A-HK-Y][0-9][0-9ABEHMNPRVWXY]?)\s?[0-9][ABD-HJLNP-UW-Z]{2})\b"
            ),
            1.0,
        ),
        # Street address heuristics: number + street keyword
        (
            "ADDRESS",
            re.compile(
                r"\b\d{1,4}\s+[A-Z][A-Za-z]+\s+(?:ROAD|STREET|AVENUE|LANE|CLOSE|"
                r"DRIVE|WAY|GARDENS|GROVE|TERRACE|COURT|PLACE|WALK|CRESCENT)\b"
            ),
            0.9,
        ),
        # Account number: 8 digits directly after a sort code
        (
            "ACCOUNT_NUMBER",
            re.compile(r"(?<=\d{2}-\d{2}-\d{2})\s+(\d{8})\b"),
            1.0,
        ),
        # Account/reference numbers with explicit labels
        (
            "ACCOUNT_NUMBER",
            re.compile(r"\b(?:ACCOUNT|ACCT|REFERENCE|REF)\s*[:#]?\s*(\d{6,12})\b"),
            0.9,
        ),
        # Person name in a payment context: "BACS JOHN SMITH 20-45-67"
        (
            "PERSON",
            re.compile(
                r"\b(?:BACS|FPS|FASTER PAYMENT|CHAPS|PAYMENT (?:TO|FROM))\s+"
                r"([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){0,1})\s+(?=\d{2}-\d{2}-\d{2})"
            ),
            0.85,
        ),
        # Person name before an account number label
        (
            "PERSON",
            re.compile(
                r"\b(?:BACS|FPS|FASTER PAYMENT|CHAPS|PAYMENT (?:TO|FROM))\s+"
                r"([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){0,1})\s+(?=(?:ACCOUNT|ACCT|REF))"
            ),
            0.8,
        ),
    ]

    _COMPANY_TOKENS = {
        "LTD", "LIMITED", "PLC", "LLP", "CORP", "INC", "GROUP", "HOLDINGS",
        "SERVICES", "SOLUTIONS", "PAYROLL", "SALARY", "WAGES", "PROPERTIES",
    }

    def detect(self, text: str) -> List[RedactionEntity]:
        # Normalise case for matching but keep offsets against the original.
        upper = text.upper()
        candidates: List[RedactionEntity] = []
        for entity_type, pattern, score in self._PATTERNS:
            for match in pattern.finditer(upper):
                start, end = match.span()
                if entity_type in {"SORT_CODE", "EMAIL", "PHONE", "IBAN", "CARD_NUMBER", "ADDRESS"}:
                    group_start, group_end = start, end
                else:
                    # PERSON / ACCOUNT_NUMBER patterns capture the identifier in group 1.
                    if match.lastindex and match.group(1) is not None:
                        group_start, group_end = match.span(1)
                    else:
                        group_start, group_end = start, end
                if group_end <= group_start:
                    continue
                matched_text = upper[group_start:group_end]
                if entity_type == "PERSON" and self._looks_like_company(matched_text):
                    continue
                candidates.append(
                    RedactionEntity(
                        entity_type=entity_type,
                        placeholder=PLACEHOLDERS[entity_type],
                        start=group_start,
                        end=group_end,
                        score=score,
                        source=self.name,
                    )
                )
        return self._resolve_overlaps(candidates)

    @classmethod
    def _looks_like_company(cls, matched_text: str) -> bool:
        """Avoid redacting company names as PERSON (e.g. 'ACME LTD')."""
        tokens = matched_text.split()
        return any(token in cls._COMPANY_TOKENS for token in tokens)

    @staticmethod
    def _resolve_overlaps(candidates: List[RedactionEntity]) -> List[RedactionEntity]:
        """Keep the best entity when spans overlap (priority then length)."""
        priority = {
            "SORT_CODE": 9,
            "ACCOUNT_NUMBER": 8,
            "EMAIL": 7,
            "PHONE": 7,
            "CARD_NUMBER": 7,
            "IBAN": 7,
            "ADDRESS": 6,
            "PERSON": 5,
            "OTHER": 1,
        }
        candidates.sort(key=lambda e: (e.start, e.end))
        resolved: List[RedactionEntity] = []
        for candidate in candidates:
            overlaps = any(
                not (candidate.end <= r.start or candidate.start >= r.end)
                for r in resolved
            )
            if overlaps:
                continue
            resolved.append(candidate)
        resolved.sort(
            key=lambda e: (-priority.get(e.entity_type, 0), e.start, -(e.end - e.start))
        )
        # Re-resolve after priority sort.
        final: List[RedactionEntity] = []
        for candidate in resolved:
            overlaps = any(
                not (candidate.end <= r.start or candidate.start >= r.end)
                for r in final
            )
            if not overlaps:
                final.append(candidate)
        final.sort(key=lambda e: e.start)
        return final


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class RedactorAdapter:
    """Redacts raw transaction descriptions through pluggable backends.

    Usage::

        adapter = RedactorAdapter()
        result = adapter.redact("BACS JOHN SMITH 20-45-67 12345678")
        result.redacted_text  # "BACS [PERSON] [SORT_CODE] [ACCOUNT_NUMBER]"
        result.entity_types   # ["PERSON", "SORT_CODE", "ACCOUNT_NUMBER"]

    To replace the redactor later, implement :class:`RedactorBackend` and pass
    ``backends=[MyBackend()]``.
    """

    def __init__(
        self,
        use_presidio: bool = True,
        score_threshold: float = 0.45,
        backends: Optional[List[RedactorBackend]] = None,
    ) -> None:
        if backends is not None:
            self.backends = backends
        else:
            self.backends = []
            if use_presidio:
                self.backends.append(PresidioUkBackend(score_threshold=score_threshold))
            self.backends.append(UkBankingRegexBackend())

    def redact(self, text: str) -> RedactionResult:
        """Redact ``text`` and return a structured result.

        This method never raises for redaction failures: it returns a
        :class:`RedactionResult` with ``ok=False`` and a controlled error that
        contains no raw text.
        """
        if not text or not text.strip():
            return RedactionResult(
                redacted_text=text or "", entities=[], ok=True, backend="empty"
            )

        collected: List[RedactionEntity] = []
        used_backends: List[str] = []
        successful_backends = 0
        warned_backends = set()
        for backend in self.backends:
            try:
                detected = backend.detect(text)
                successful_backends += 1
                if detected:
                    collected.extend(detected)
                    used_backends.append(backend.name)
            except Exception as exc:  # noqa: BLE001 - fail safe per backend
                backend_name = getattr(backend, "name", type(backend).__name__)
                # Log each failing backend once per adapter instance, and never
                # include raw text in log output.
                if backend_name not in warned_backends:
                    warned_backends.add(backend_name)
                    logger.warning(
                        "Redaction backend '%s' failed (%s); continuing with remaining backends",
                        backend_name,
                        type(exc).__name__,
                    )

        if not collected:
            if successful_backends == 0:
                # Controlled failure: never return raw text when redaction
                # could not run at all.
                return RedactionResult(
                    redacted_text="[REDACTION_ERROR]",
                    entities=[],
                    ok=False,
                    error="Redaction failed; raw text was not logged.",
                    backend=",".join(used_backends) or "none",
                )
            return RedactionResult(
                redacted_text=text, entities=[], ok=True, backend=",".join(used_backends) or "none"
            )

        merged = self._merge_entities(collected)
        redacted = self._apply_replacements(text, merged)
        return RedactionResult(
            redacted_text=redacted,
            entities=merged,
            ok=True,
            backend=",".join(used_backends),
        )

    @staticmethod
    def _merge_entities(entities: List[RedactionEntity]) -> List[RedactionEntity]:
        """Merge entities from several backends, dropping overlaps by priority."""
        priority = {
            "SORT_CODE": 9,
            "ACCOUNT_NUMBER": 8,
            "EMAIL": 7,
            "PHONE": 7,
            "CARD_NUMBER": 7,
            "IBAN": 7,
            "ADDRESS": 6,
            "PERSON": 5,
            "OTHER": 1,
        }
        ordered = sorted(entities, key=lambda e: e.start)
        merged: List[RedactionEntity] = []
        for entity in ordered:
            overlaps = any(
                not (entity.end <= existing.start or entity.start >= existing.end)
                for existing in merged
            )
            if not overlaps:
                merged.append(entity)
                continue
            # Replace lower-priority overlaps only if strictly better.
            for i, existing in enumerate(merged):
                if not (entity.end <= existing.start or entity.start >= existing.end):
                    if priority.get(entity.entity_type, 0) > priority.get(
                        existing.entity_type, 0
                    ):
                        merged[i] = entity
                    break
        merged.sort(key=lambda e: e.start)
        return merged

    @staticmethod
    def _apply_replacements(text: str, entities: List[RedactionEntity]) -> str:
        chunks: List[str] = []
        cursor = 0
        for entity in entities:
            start = max(entity.start, cursor)
            end = min(entity.end, len(text))
            if start >= end:
                continue
            chunks.append(text[cursor:start])
            chunks.append(entity.placeholder)
            cursor = end
        chunks.append(text[cursor:])
        return "".join(chunks)


#: Process-wide default adapter. Constructed lazily by :func:`get_redactor`.
_default_adapter: Optional[RedactorAdapter] = None


def get_redactor() -> RedactorAdapter:
    """Return the process-wide redactor adapter (lazy singleton)."""
    global _default_adapter
    if _default_adapter is None:
        from config import settings

        _default_adapter = RedactorAdapter(
            use_presidio=settings.PII_USE_PRESIDIO,
            score_threshold=settings.PII_SCORE_THRESHOLD,
        )
    return _default_adapter
