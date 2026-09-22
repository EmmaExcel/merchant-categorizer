import re
from presidio_analyzer import Pattern, PatternRecognizer

NINO_EXCLUDED_PREFIXES = {"BG", "GB", "KN", "NK", "NT", "TN", "ZZ"}
NINO_INVALID_FIRST_LETTERS = set("DFIQUV")
NINO_INVALID_SECOND_LETTERS = set("DFIOQUV")

def validate_uk_nino(raw_value: str) -> bool:
    cleaned = re.sub(r"\s+", "", raw_value).upper()
    if len(cleaned) not in (8, 9):
        return False
    prefix = cleaned[:2]
    digits = cleaned[2:8]
    suffix = cleaned[8:] if len(cleaned) == 9 else ""
    if not digits.isdigit():
        return False
    if prefix == "QQ":
        return suffix in ("", "A", "B", "C", "D")
    if prefix in NINO_EXCLUDED_PREFIXES:
        return False
    if prefix[0] in NINO_INVALID_FIRST_LETTERS:
        return False
    if prefix[1] in NINO_INVALID_SECOND_LETTERS:
        return False
    if not prefix.isalpha():
        return False
    if suffix and suffix not in ("A", "B", "C", "D"):
        return False
    return True

class UkNinoRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [
            Pattern(
                "NINO_SPACED_PAIRS",
                r"\b[A-Za-z]{2}\s\d{2}\s\d{2}\s\d{2}\s[A-Da-d ]\b",
                0.8,
            ),
            Pattern(
                "NINO_SPACED_HALVES",
                r"\b[A-Za-z]{2}\s\d{6}\s[A-Da-d ]\b",
                0.75,
            ),
            Pattern(
                "NINO_UNSPACED",
                r"\b[A-Za-z]{2}\d{6}[A-Da-d]\b",
                0.7,
            ),
        ]
        context = [
            "nino",
            "national insurance",
            "ni number",
            "hmrc",
            "dwp",
            "payroll",
            "p45",
            "p60",
            "tax",
            "salary",
            "benefits",
        ]
        super().__init__(
            supported_entity="UK_NINO",
            patterns=patterns,
            context=context,
            supported_language="en",
        )

    def validate_result(self, pattern_text: str) -> bool:
        return validate_uk_nino(pattern_text)
