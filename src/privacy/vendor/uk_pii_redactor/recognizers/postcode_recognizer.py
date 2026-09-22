import re
from presidio_analyzer import Pattern, PatternRecognizer

UK_POSTCODE_REGEX = re.compile(
    r"^(?:GIR\s?0AA|(?:[A-PR-UWYZ][0-9][0-9A-HJKPSTUW]?|[A-PR-UWYZ][A-HK-Y][0-9][0-9ABEHMNPRVWXY]?)\s?[0-9][ABD-HJLNP-UW-Z]{2})$",
    re.IGNORECASE,
)

def validate_uk_postcode(raw_value: str) -> bool:
    cleaned = raw_value.strip().upper()
    return bool(UK_POSTCODE_REGEX.match(cleaned))

class UkPostcodeRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [
            Pattern(
                "UK_POSTCODE_FULL",
                r"\b(?:GIR\s?0AA|(?:[A-PR-UWYZ][0-9][0-9A-HJKPSTUW]?|[A-PR-UWYZ][A-HK-Y][0-9][0-9ABEHMNPRVWXY]?)\s?[0-9][ABD-HJLNP-UW-Z]{2})\b",
                0.75,
            ),
        ]
        context = [
            "postcode",
            "postal code",
            "address",
            "street",
            "road",
            "lane",
            "close",
            "avenue",
            "drive",
            "way",
            "flat",
            "apartment",
            "city",
            "county",
            "residence",
            "location",
        ]
        super().__init__(
            supported_entity="UK_POSTCODE",
            patterns=patterns,
            context=context,
            supported_language="en",
        )

    def validate_result(self, pattern_text: str) -> bool:
        return validate_uk_postcode(pattern_text)
