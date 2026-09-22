from typing import List, Dict, Any, Optional
import re
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from uk_pii_redactor.core.config import settings
from uk_pii_redactor.recognizers import (
    NhsNumberRecognizer,
    UkNinoRecognizer,
    UkPostcodeRecognizer,
    UkPhoneRecognizer,
)

DEFAULT_PLACEHOLDERS = {
    "UK_NHS_NUMBER": "[REDACTED_NHS_NUMBER]",
    "UK_NINO": "[REDACTED_UK_NINO]",
    "UK_POSTCODE": "[REDACTED_UK_POSTCODE]",
    "UK_PHONE_NUMBER": "[REDACTED_PHONE_NUMBER]",
    "PERSON": "[REDACTED_PERSON]",
    "EMAIL_ADDRESS": "[REDACTED_EMAIL_ADDRESS]",
    "LOCATION": "[REDACTED_LOCATION]",
    "ORGANIZATION": "[REDACTED_ORGANIZATION]",
    "CREDIT_CARD": "[REDACTED_CREDIT_CARD]",
    "IBAN_CODE": "[REDACTED_IBAN_CODE]",
    "IP_ADDRESS": "[REDACTED_IP_ADDRESS]",
    "DATE_TIME": "[REDACTED_DATE_TIME]",
    "PHONE_NUMBER": "[REDACTED_PHONE_NUMBER]",
}

UK_SPECIFIC_ENTITIES = [
    "UK_NHS_NUMBER",
    "UK_NINO",
    "UK_POSTCODE",
    "UK_PHONE_NUMBER",
]

UK_ALGORITHMIC_ENTITIES = {
    "UK_NHS_NUMBER",
    "UK_NINO",
    "UK_POSTCODE",
    "UK_PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
}

SPACY_STATISTICAL_ENTITIES = {
    "ORGANIZATION",
    "LOCATION",
    "PERSON",
    "DATE_TIME",
    "PHONE_NUMBER",
    "UK_NHS",
}

ALL_SUPPORTED_ENTITIES = [
    "UK_NHS_NUMBER",
    "UK_NINO",
    "UK_POSTCODE",
    "UK_PHONE_NUMBER",
    "PERSON",
    "EMAIL_ADDRESS",
    "LOCATION",
    "ORGANIZATION",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "DATE_TIME",
]

IGNORED_PHRASES = {
    "OCCUPATIONAL HEALTH ASSESSMENT",
    "OCCUPATIONAL HEALTH",
    "FINAL SUMMARY",
    "GMC REF",
    "GMC",
    "GMC NUMBER",
    "GMC NO",
    "DIRECT TELEPHONE",
    "DIRECT LINE",
    "REFERRING PRACTICE",
    "REFERRING GP",
    "REFERRING DOCTOR",
    "SECURE EMAIL",
    "EMAIL",
    "EMAIL ADDRESS",
    "CONTACT TELEPHONE",
    "CONTACT",
    "TELEPHONE",
    "PHONE",
    "TEL",
    "MOB",
    "MOBILE",
    "NHS",
    "NHS NUMBER",
    "NHS NO",
    "NHS IDENTIFIER",
    "NINO",
    "NATIONAL INSURANCE",
    "NI NUMBER",
    "DOB",
    "DATE OF BIRTH",
    "DATE",
    "PATIENT",
    "PATIENT NAME",
    "ADDRESS",
    "BILLING ADDRESS",
    "HOME ADDRESS",
    "CLINICAL DETAILS",
    "SUMMARY",
    "DIAGNOSIS",
    "SYMPTOMS",
    "REFERRAL",
    "CONSULTANT",
    "SURGERY",
    "HOSPITAL",
    "CLINIC",
    "HEALTH CENTRE",
    "DYSPHAGIA",
    "DYSPNOEA",
    "ERYTHEMA",
    "HYPERTENSION",
    "TACHYCARDIA",
    "BRADYCARDIA",
    "DIABETES",
    "SCIATICA",
    "LUMBAGO",
    "MRI",
    "ECG",
    "CT SCAN",
    "X-RAY",
    "ULTRASOUND",
}

class PiiEngine:
    def __init__(self, spacy_model: Optional[str] = None):
        model_name = spacy_model or settings.SPACY_MODEL
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }
        try:
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        except Exception:
            fallback_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            provider = NlpEngineProvider(nlp_configuration=fallback_config)
            nlp_engine = provider.create_engine()
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

        self.anonymizer = AnonymizerEngine()
        self._register_uk_recognizers()

    def _register_uk_recognizers(self) -> None:
        self.analyzer.registry.add_recognizer(NhsNumberRecognizer())
        self.analyzer.registry.add_recognizer(UkNinoRecognizer())
        self.analyzer.registry.add_recognizer(UkPostcodeRecognizer())
        self.analyzer.registry.add_recognizer(UkPhoneRecognizer())

    def _is_suppressed_phrase(self, raw_slice: str) -> bool:
        trimmed = raw_slice.strip()
        if not trimmed:
            return True
        upper_raw = trimmed.upper()
        if upper_raw in IGNORED_PHRASES:
            return True
        normalized = re.sub(r"^[\W_]+|[\W_]+$", "", upper_raw).strip()
        if normalized in IGNORED_PHRASES:
            return True
        colon_stripped = re.sub(r"[:\s]+$", "", upper_raw).strip()
        if colon_stripped in IGNORED_PHRASES:
            return True
        for part in re.split(r"[\r\n]+", upper_raw):
            clean_part = re.sub(r"^[\W_]+|[\W_]+$", "", part).strip()
            if clean_part in IGNORED_PHRASES:
                return True
        return False

    def _filter_and_resolve_overlaps(
        self,
        text: str,
        results: List[RecognizerResult],
        selected_entities: Optional[List[str]] = None,
        score_threshold: float = 0.5,
    ) -> List[RecognizerResult]:
        valid_candidates: List[RecognizerResult] = []
        for r in results:
            if r.score < score_threshold:
                continue
            if selected_entities is not None and r.entity_type not in selected_entities:
                continue

            start, end = r.start, r.end
            while start < end and text[start] in " \t\r\n":
                start += 1
            while end > start and text[end - 1] in " \t\r\n":
                end -= 1
            if start >= end:
                continue

            r.start = start
            r.end = end

            if r.entity_type in SPACY_STATISTICAL_ENTITIES and "\n" in text[r.start:r.end]:
                continue

            raw_slice = text[r.start:r.end]
            if self._is_suppressed_phrase(raw_slice):
                continue
            valid_candidates.append(r)

        algorithmic_candidates = [
            r for r in valid_candidates
            if r.entity_type in UK_ALGORITHMIC_ENTITIES
        ]

        retained_candidates: List[RecognizerResult] = []
        for r in valid_candidates:
            if r.entity_type in SPACY_STATISTICAL_ENTITIES:
                overlaps_algo = any(
                    not (r.end <= a.start or r.start >= a.end)
                    for a in algorithmic_candidates
                )
                if overlaps_algo:
                    continue
            retained_candidates.append(r)

        priority_map = {
            "UK_NHS_NUMBER": 1000,
            "UK_NINO": 1000,
            "UK_POSTCODE": 1000,
            "UK_PHONE_NUMBER": 1000,
            "EMAIL_ADDRESS": 500,
            "CREDIT_CARD": 500,
            "IBAN_CODE": 500,
            "IP_ADDRESS": 400,
            "PERSON": 200,
            "LOCATION": 150,
            "ORGANIZATION": 100,
            "DATE_TIME": 50,
            "PHONE_NUMBER": 20,
            "UK_NHS": 10,
        }

        retained_candidates.sort(
            key=lambda r: (
                -priority_map.get(r.entity_type, 0),
                -r.score,
                -(r.end - r.start),
                r.start,
            )
        )

        resolved: List[RecognizerResult] = []
        for candidate in retained_candidates:
            overlaps = any(
                not (candidate.end <= existing.start or candidate.start >= existing.end)
                for existing in resolved
            )
            if not overlaps:
                resolved.append(candidate)

        resolved.sort(key=lambda r: r.start)
        return resolved

    def analyze_text(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        score_threshold: float = 0.5,
    ) -> List[RecognizerResult]:
        target_entities = entities if entities is not None else ALL_SUPPORTED_ENTITIES
        raw_results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=target_entities,
            score_threshold=0.1,
        )
        return self._filter_and_resolve_overlaps(
            text=text,
            results=raw_results,
            selected_entities=entities,
            score_threshold=score_threshold,
        )

    def redact_text(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        mode: str = "placeholder",
        mask_char: str = "*",
        score_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        results = self.analyze_text(
            text=text,
            entities=entities,
            score_threshold=score_threshold,
        )

        entity_items: List[Dict[str, Any]] = []
        chunks: List[str] = []
        cursor = 0

        for r in results:
            if r.start > cursor:
                chunks.append(text[cursor:r.start])

            matched_text = text[r.start:r.end]
            if mode == "mask":
                replacement = mask_char * len(matched_text)
            elif mode == "blackout":
                replacement = "█" * len(matched_text)
            elif mode == "redact_label":
                replacement = f"[REDACTED_{r.entity_type}]"
            else:
                replacement = DEFAULT_PLACEHOLDERS.get(
                    r.entity_type, f"[REDACTED_{r.entity_type}]"
                )

            chunks.append(replacement)
            entity_items.append({
                "entity_type": r.entity_type,
                "text": matched_text,
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 4),
                "replacement": replacement,
            })
            cursor = r.end

        if cursor < len(text):
            chunks.append(text[cursor:])

        redacted_text = "".join(chunks)

        return {
            "original_text": text,
            "redacted_text": redacted_text,
            "entities": entity_items,
            "total_entities_found": len(entity_items),
        }

    def redact_batch(
        self,
        texts: List[str],
        entities: Optional[List[str]] = None,
        mode: str = "placeholder",
        mask_char: str = "*",
        score_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        return [
            self.redact_text(
                text=t,
                entities=entities,
                mode=mode,
                mask_char=mask_char,
                score_threshold=score_threshold,
            )
            for t in texts
        ]

# NOTE (vendor adaptation): the upstream module instantiates a global engine at
# import time. We keep the same public attribute but build it lazily so that
# importing this module never fails when presidio or the spaCy model is missing.
pii_engine = None


def get_pii_engine() -> "PiiEngine":
    global pii_engine
    if pii_engine is None:
        pii_engine = PiiEngine()
    return pii_engine
