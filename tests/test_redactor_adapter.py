"""Tests for the PII redactor adapter."""

from __future__ import annotations

import logging

import pytest

from privacy.redactor_adapter import RedactorAdapter, RedactionEntity, RedactorBackend


class _ExplodingBackend:
    name = "exploding"

    def detect(self, text: str):
        raise RuntimeError("boom")


def test_redacts_person_sort_code_and_account_number(redactor):
    result = redactor.redact("BACS JOHN SMITH 20-45-67 12345678 RENT SEPTEMBER")
    assert result.ok is True
    assert "[PERSON]" in result.redacted_text
    assert "[SORT_CODE]" in result.redacted_text
    assert "[ACCOUNT_NUMBER]" in result.redacted_text
    assert "JOHN SMITH" not in result.redacted_text
    assert "20-45-67" not in result.redacted_text
    assert "12345678" not in result.redacted_text
    assert set(result.entity_types) >= {"PERSON", "SORT_CODE", "ACCOUNT_NUMBER"}


def test_redacts_email_and_phone(redactor):
    text = "PAYMENT TO JOE BLOGGS CONTACT 07700 900123 OR JOE@EXAMPLE.COM"
    result = redactor.redact(text)
    assert result.ok is True
    assert "[EMAIL]" in result.redacted_text
    assert "[PHONE]" in result.redacted_text
    assert "07700 900123" not in result.redacted_text
    assert "JOE@EXAMPLE.COM" not in result.redacted_text


def test_redacts_uk_postcode_as_address(redactor):
    text = "PAYMENT REF 42 ABBEY ROAD LONDON NW8 9AY"
    result = redactor.redact(text)
    assert result.ok is True
    assert "[ADDRESS]" in result.redacted_text
    assert "NW8 9AY" not in result.redacted_text


def test_redacts_account_reference_label(redactor):
    text = "BACS ACME LTD PAYROLL REF 12345678"
    result = redactor.redact(text)
    assert result.ok is True
    assert "[ACCOUNT_NUMBER]" in result.redacted_text
    assert "12345678" not in result.redacted_text


def test_no_pii_returns_original(redactor):
    text = "CARD PAYMENT TESCO STORES 3402 LONDON"
    result = redactor.redact(text)
    assert result.ok is True
    assert result.redacted_text == text
    assert result.entity_types == []


def test_fail_safe_returns_controlled_error_and_no_raw_logs(caplog):
    adapter = RedactorAdapter(backends=[_ExplodingBackend()])
    raw = "BACS JOHN SMITH 20-45-67 12345678"
    with caplog.at_level(logging.WARNING):
        result = adapter.redact(raw)
    assert result.ok is False
    assert result.redacted_text == "[REDACTION_ERROR]"
    assert "JOHN SMITH" not in caplog.text
    assert "20-45-67" not in caplog.text
    assert "12345678" not in caplog.text


def test_adapter_accepts_replaceable_backends(redactor):
    class _CustomBackend:
        name = "custom"

        def detect(self, text: str):
            if "SECRET" in text:
                return [RedactionEntity("OTHER", "[REDACTED]", 0, 6, source="custom")]
            return []

    adapter = RedactorAdapter(backends=[_CustomBackend()])
    result = adapter.redact("SECRET STUFF")
    assert result.ok is True
    assert "[REDACTED]" in result.redacted_text
