import re
from typing import List, Optional
from presidio_analyzer import Pattern, PatternRecognizer

def validate_nhs_modulus11(raw_value: str) -> bool:
    digits = [c for c in raw_value if c.isdigit()]
    if len(digits) != 10:
        return False
    weights = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    total = sum(int(digits[i]) * weights[i] for i in range(9))
    remainder = total % 11
    check_digit = 11 - remainder
    if check_digit == 11:
        check_digit = 0
    elif check_digit == 10:
        return False
    return check_digit == int(digits[9])

class NhsNumberRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [
            Pattern("NHS_UNSPACED", r"\b\d{10}\b", 0.6),
            Pattern("NHS_SPACED_3_3_4", r"\b\d{3}\s\d{3}\s\d{4}\b", 0.8),
            Pattern("NHS_HYPHEN_3_3_4", r"\b\d{3}-\d{3}-\d{4}\b", 0.8),
        ]
        context = [
            "nhs",
            "patient",
            "hospital",
            "clinic",
            "medical",
            "gp",
            "doctor",
            "dr",
            "surgery",
            "referral",
            "consultant",
        ]
        super().__init__(
            supported_entity="UK_NHS_NUMBER",
            patterns=patterns,
            context=context,
            supported_language="en",
        )

    def validate_result(self, pattern_text: str) -> bool:
        return validate_nhs_modulus11(pattern_text)
