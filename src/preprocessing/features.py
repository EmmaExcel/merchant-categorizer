
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

LABELS: List[str] = [
    "Groceries",
    "Dining",
    "Coffee Shops",
    "Transport",
    "Fuel",
    "Shopping",
    "Entertainment",
    "Travel",
    "Utilities",
    "Rent/Mortgage",
    "Insurance",
    "Healthcare",
    "Education",
    "Salary",
    "Transfers",
    "Cash Withdrawal",
    "Subscriptions",
    "Charity",
    "Fees",
    "Taxes",
    "Other",
]

LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABELS)}





AMOUNT_BUCKETS: List[str] = ["micro", "small", "medium", "large", "very_large"]
AMOUNT_BUCKET2ID: Dict[str, int] = {b: i for i, b in enumerate(AMOUNT_BUCKETS)}


def amount_bucket(amount: Optional[float]) -> str:
    if amount is None:
        return "micro"
    value = abs(float(amount))
    if value < 10.0:
        return "micro"
    if value < 50.0:
        return "small"
    if value < 200.0:
        return "medium"
    if value < 1000.0:
        return "large"
    return "very_large"


def amount_bucket_id(amount: Optional[float]) -> int:
    return AMOUNT_BUCKET2ID[amount_bucket(amount)]






DIRECTION2ID: Dict[str, int] = {"debit": 0, "credit": 1}


def normalise_direction(direction: Optional[str]) -> str:
    if direction is None:
        return "debit"
    value = str(direction).strip().lower()
    if value in {"debit", "out", "outgoing", "payment", "withdrawal", "money_out", "card_payment"}:
        return "debit"
    if value in {"credit", "in", "incoming", "deposit", "money_in", "refund", "salary"}:
        return "credit"
    return "debit"


def direction_id(direction: Optional[str]) -> int:
    return DIRECTION2ID[normalise_direction(direction)]






PAYMENT_RAILS: List[str] = [
    "CARD",
    "POS",
    "BACS",
    "FPS",
    "CHAPS",
    "DD",
    "SO",
    "ATM",
    "TFR",
    "PAYPAL",
    "STRIPE",
    "SQUARE",
    "APPLE_PAY",
    "GOOGLE_PAY",
    "OTHER",
]
RAIL2ID: Dict[str, int] = {r: i for i, r in enumerate(PAYMENT_RAILS)}

_RAIL_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    ("APPLE_PAY", re.compile(r"\bAPPLE\s*PAY\b", re.IGNORECASE)),
    ("GOOGLE_PAY", re.compile(r"\bGOOGLE\s*PAY\b|\bGPAY\b", re.IGNORECASE)),
    ("PAYPAL", re.compile(r"\bPAYPAL\b", re.IGNORECASE)),
    ("STRIPE", re.compile(r"\bSTRIPE\b", re.IGNORECASE)),
    ("SQUARE", re.compile(r"\bSQUARE\b|\bSQ\b", re.IGNORECASE)),
    ("POS", re.compile(r"\bPOS\b|\bPOINT\s*OF\s*SALE\b", re.IGNORECASE)),
    ("CARD", re.compile(r"\bCARD\s*(PAYMENT|PURCHASE|TRANSACTION)?\b|\bVISA\b|\bMASTERCARD\b|\bMAESTRO\b", re.IGNORECASE)),
    ("BACS", re.compile(r"\bBACS\b", re.IGNORECASE)),
    ("FPS", re.compile(r"\bFPS\b|\bFASTER\s*PAYMENTS?\b", re.IGNORECASE)),
    ("CHAPS", re.compile(r"\bCHAPS\b", re.IGNORECASE)),
    ("DD", re.compile(r"\bDD\b|\bDIRECT\s*DEBIT\b", re.IGNORECASE)),
    ("SO", re.compile(r"\bSO\b|\bSTANDING\s*ORDER\b", re.IGNORECASE)),
    ("ATM", re.compile(r"\bATM\b|\bCASH\s*(WITHDRAWAL|MACHINE)\b|\bLINK\s+CASH\b", re.IGNORECASE)),
    ("TFR", re.compile(r"\bTFR\b|\bTRANSFER\b|\bBANK\s+GIRO\s+CREDIT\b", re.IGNORECASE)),
]


def detect_payment_rail(transaction_type: Optional[str], raw_description: Optional[str]) -> str:
    if transaction_type:
        value = str(transaction_type).strip().upper()
        aliases = {
            "CARD": "CARD",
            "CARD PAYMENT": "CARD",
            "CARD_PAYMENT": "CARD",
            "POS": "POS",
            "BACS": "BACS",
            "FPS": "FPS",
            "FASTER PAYMENTS": "FPS",
            "FASTER_PAYMENT": "FPS",
            "CHAPS": "CHAPS",
            "DD": "DD",
            "DIRECT DEBIT": "DD",
            "DIRECT_DEBIT": "DD",
            "SO": "SO",
            "STANDING ORDER": "SO",
            "STANDING_ORDER": "SO",
            "ATM": "ATM",
            "CASH WITHDRAWAL": "ATM",
            "TFR": "TFR",
            "TRANSFER": "TFR",
            "PAYPAL": "PAYPAL",
            "STRIPE": "STRIPE",
            "SQUARE": "SQUARE",
            "SQ": "SQUARE",
            "APPLE PAY": "APPLE_PAY",
            "APPLE_PAY": "APPLE_PAY",
            "GOOGLE PAY": "GOOGLE_PAY",
            "GOOGLE_PAY": "GOOGLE_PAY",
        }
        if value in aliases:
            return aliases[value]

    text = (raw_description or "").upper()
    for rail, pattern in _RAIL_PATTERNS:
        if pattern.search(text):
            return rail
    return "OTHER"


def rail_id(transaction_type: Optional[str], raw_description: Optional[str]) -> int:
    return RAIL2ID[detect_payment_rail(transaction_type, raw_description)]








MCC_BUCKET_COUNT = 24

_CURATED_MCC_BUCKETS: Dict[str, int] = {

    "5411": 0,
    "5499": 0,
    "5300": 0,

    "5812": 1,
    "5814": 1,
    "5811": 1,

    "4111": 2,
    "4121": 2,
    "4131": 2,
    "4789": 2,

    "5541": 3,
    "5542": 3,
    "5172": 3,

    "5310": 4,
    "5399": 4,
    "5691": 4,
    "5311": 4,

    "7832": 5,
    "7999": 5,
    "7922": 5,

    "4511": 6,
    "4722": 6,
    "4411": 6,

    "4900": 7,
    "4814": 7,
    "4812": 7,
    "4899": 7,

    "6300": 8,
    "5960": 8,

    "5912": 9,
    "8011": 9,
    "8021": 9,
    "8049": 9,

    "8220": 10,
    "8299": 10,

    "6011": 11,
    "6010": 11,

    "5968": 12,
    "5734": 12,

    "8398": 13,

    "9399": 14,
    "9211": 14,

    "6012": 15,
    "6051": 15,
}


def mcc_bucket(mcc: Optional[str | int]) -> int:
    if mcc is None or str(mcc).strip() == "":
        return MCC_BUCKET_COUNT - 1
    code = re.sub(r"\D", "", str(mcc))
    if not code:
        return MCC_BUCKET_COUNT - 1
    if code in _CURATED_MCC_BUCKETS:
        return _CURATED_MCC_BUCKETS[code]
    fallback = sum(ord(ch) for ch in code)
    return 16 + (fallback % (MCC_BUCKET_COUNT - 17))






CANONICAL_PII_TYPES: List[str] = [
    "PERSON",
    "SORT_CODE",
    "ACCOUNT_NUMBER",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "CARD_NUMBER",
    "IBAN",
    "OTHER",
]


def pii_detected(entity_types: Optional[List[str]]) -> bool:
    return bool(entity_types)


def pii_type_vector(entity_types: Optional[List[str]]) -> List[int]:
    found = {str(t).upper() for t in (entity_types or [])}
    return [1 if t in found else 0 for t in CANONICAL_PII_TYPES]






def build_metadata_features(
    amount: Optional[float],
    direction: Optional[str],
    transaction_type: Optional[str],
    mcc: Optional[str | int],
    pii_entities: Optional[List[str]],
    raw_description: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "amount_bucket": amount_bucket(amount),
        "amount_bucket_id": amount_bucket_id(amount),
        "direction": normalise_direction(direction),
        "direction_id": direction_id(direction),
        "payment_rail": detect_payment_rail(transaction_type, raw_description or ""),
        "payment_rail_id": rail_id(transaction_type, raw_description or ""),
        "mcc_bucket": mcc_bucket(mcc),
        "mcc_present": bool(mcc is not None and str(mcc).strip() != ""),
        "pii_detected": pii_detected(pii_entities),
        "pii_types": [str(t) for t in (pii_entities or [])],
    }


def encode_metadata_tensor(batch_features: List[Dict[str, Any]]) -> Dict[str, Any]:
    import torch

    return {
        "amount_bucket": torch.tensor(
            [f["amount_bucket_id"] for f in batch_features], dtype=torch.long
        ),
        "direction": torch.tensor(
            [f["direction_id"] for f in batch_features], dtype=torch.long
        ),
        "payment_rail": torch.tensor(
            [f["payment_rail_id"] for f in batch_features], dtype=torch.long
        ),
        "mcc": torch.tensor(
            [f["mcc_bucket"] for f in batch_features], dtype=torch.long
        ),
        "mcc_present": torch.tensor(
            [1.0 if f["mcc_present"] else 0.0 for f in batch_features], dtype=torch.float
        ),
        "pii_types": torch.tensor(
            [pii_type_vector(f["pii_types"]) for f in batch_features], dtype=torch.float
        ),
    }
