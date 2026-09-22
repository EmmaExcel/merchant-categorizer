
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)





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




UPSTREAM_TARGET_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "UK_PHONE_NUMBER",
    "PHONE_NUMBER",
    "UK_POSTCODE",
    "CREDIT_CARD",
    "IBAN_CODE",
]


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


_ENTITY_PRIORITY: Dict[str, int] = {
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


@dataclass(frozen=True)
class RedactionEntity:
    entity_type: str
    placeholder: str
    start: int
    end: int
    score: float = 1.0
    source: str = "uk-banking-regex"


@dataclass
class RedactionResult:

    redacted_text: str
    entities: List[RedactionEntity] = field(default_factory=list)
    ok: bool = True
    error: Optional[str] = None
    backend: str = "none"

    @property
    def entity_types(self) -> List[str]:
        return [e.entity_type for e in self.entities]


class RedactionError(Exception):

    def __init__(self, message: str = "Redaction failed; raw text was not logged."):
        super().__init__(message)


class RedactorBackend(Protocol):
    name: str

    def detect(self, text: str) -> List[RedactionEntity]:
        ...






class PresidioUkBackend:

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
        except Exception as exc:  # noqa: BLE001
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






class UkBankingRegexBackend:

    name = "uk-banking-regex"


    _PATTERNS: List[tuple[str, re.Pattern[str], float]] = [

        ("SORT_CODE", re.compile(r"\b\d{2}-\d{2}-\d{2}\b"), 1.0),

        ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), 1.0),

        (
            "PHONE",
            re.compile(
                r"(?:\+44\s?7\d{3}\s?\d{3}\s?\d{3}|\b07\d{3}\s?\d{3}\s?\d{3}\b|"
                r"\b(?:01\d{2,4}|02\d|03\d{2})\s?\d{3,4}\s?\d{3,4}\b)"
            ),
            1.0,
        ),

        (
            "IBAN",
            re.compile(r"\bGB\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b"),
            1.0,
        ),

        (
            "CARD_NUMBER",
            re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
            0.9,
        ),

        (
            "ADDRESS",
            re.compile(
                r"\b(?:GIR\s?0AA|(?:[A-PR-UWYZ][0-9][0-9A-HJKPSTUW]?|[A-PR-UWYZ]"
                r"[A-HK-Y][0-9][0-9ABEHMNPRVWXY]?)\s?[0-9][ABD-HJLNP-UW-Z]{2})\b"
            ),
            1.0,
        ),

        (
            "ADDRESS",
            re.compile(
                r"\b\d{1,4}\s+[A-Z][A-Za-z]+\s+(?:ROAD|STREET|AVENUE|LANE|CLOSE|"
                r"DRIVE|WAY|GARDENS|GROVE|TERRACE|COURT|PLACE|WALK|CRESCENT)\b"
            ),
            0.9,
        ),

        (
            "ACCOUNT_NUMBER",
            re.compile(r"(?<=\d{2}-\d{2}-\d{2})\s+(\d{8})\b"),
            1.0,
        ),

        (
            "ACCOUNT_NUMBER",
            re.compile(r"\b(?:ACCOUNT|ACCT|REFERENCE|REF)\s*[:#]?\s*(\d{6,12})\b"),
            0.9,
        ),

        (
            "PERSON",
            re.compile(
                r"\b(?:BACS|FPS|FASTER PAYMENT|CHAPS|PAYMENT (?:TO|FROM))\s+"
                r"([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){0,1})\s+(?=\d{2}-\d{2}-\d{2})"
            ),
            0.85,
        ),

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

        upper = text.upper()
        candidates: List[RedactionEntity] = []
        for entity_type, pattern, score in self._PATTERNS:
            for match in pattern.finditer(upper):
                start, end = match.span()
                if entity_type in {"SORT_CODE", "EMAIL", "PHONE", "IBAN", "CARD_NUMBER", "ADDRESS"}:
                    group_start, group_end = start, end
                else:

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
        tokens = matched_text.split()
        return any(token in cls._COMPANY_TOKENS for token in tokens)

    @staticmethod
    def _resolve_overlaps(candidates: List[RedactionEntity]) -> List[RedactionEntity]:
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
            key=lambda e: (
                -_ENTITY_PRIORITY.get(e.entity_type, 0),
                e.start,
                -(e.end - e.start),
            )
        )

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






class RedactorAdapter:

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
            except Exception as exc:  # noqa: BLE001
                backend_name = getattr(backend, "name", type(backend).__name__)


                if backend_name not in warned_backends:
                    warned_backends.add(backend_name)
                    logger.warning(
                        "Redaction backend '%s' failed (%s); continuing with remaining backends",
                        backend_name,
                        type(exc).__name__,
                    )

        if not collected:
            if successful_backends == 0:


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

            for i, existing in enumerate(merged):
                if not (entity.end <= existing.start or entity.start >= existing.end):
                    if _ENTITY_PRIORITY.get(entity.entity_type, 0) > _ENTITY_PRIORITY.get(
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



_default_adapter: Optional[RedactorAdapter] = None


def get_redactor() -> RedactorAdapter:
    global _default_adapter
    if _default_adapter is None:
        from config import settings

        _default_adapter = RedactorAdapter(
            use_presidio=settings.PII_USE_PRESIDIO,
            score_threshold=settings.PII_SCORE_THRESHOLD,
        )
    return _default_adapter
