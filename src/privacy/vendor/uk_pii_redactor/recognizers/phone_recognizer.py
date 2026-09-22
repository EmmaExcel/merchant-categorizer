import re
from presidio_analyzer import Pattern, PatternRecognizer

def validate_uk_phone(raw_value: str) -> bool:
    digits = re.sub(r"\D", "", raw_value)
    if raw_value.strip().startswith("+44"):
        if len(digits) == 12 and digits.startswith("44"):
            inner = digits[2:]
            return inner[0] in ("1", "2", "3", "7")
        return False
    if len(digits) == 11 and digits.startswith("0"):
        return digits[1] in ("1", "2", "3", "7")
    return False

class UkPhoneRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [
            Pattern(
                "UK_PHONE_MOBILE_INTL",
                r"(?:\+44\s?7\d{3}|\+44\s?\(0\)\s?7\d{3})\s?\d{3}\s?\d{3}\b",
                0.85,
            ),
            Pattern(
                "UK_PHONE_LANDLINE_INTL",
                r"(?:\+44\s?(?:1\d{1,4}|2\d{1,3}|3\d{2})|\+44\s?\(0\)\s?(?:1\d{1,4}|2\d{1,3}|3\d{2}))\s?\d{3,4}\s?\d{3,4}\b",
                0.8,
            ),
            Pattern(
                "UK_PHONE_MOBILE_LOCAL",
                r"\b07\d{3}\s?\d{3}\s?\d{3}\b",
                0.8,
            ),
            Pattern(
                "UK_PHONE_LANDLINE_LOCAL",
                r"\b(?:01\d{2,4}|02\d|03\d{2})\s?\d{3,4}\s?\d{3,4}\b",
                0.75,
            ),
            Pattern(
                "UK_PHONE_BRACKETED",
                r"\((?:01\d{2,4}|02\d|03\d{2}|07\d{3})\)\s?\d{3,4}\s?\d{3,4}\b",
                0.8,
            ),
        ]
        context = [
            "phone",
            "mobile",
            "tel",
            "telephone",
            "call",
            "contact",
            "cell",
            "dial",
            "fax",
            "landline",
            "sms",
            "text",
        ]
        super().__init__(
            supported_entity="UK_PHONE_NUMBER",
            patterns=patterns,
            context=context,
            supported_language="en",
        )

    def validate_result(self, pattern_text: str) -> bool:
        return validate_uk_phone(pattern_text)
