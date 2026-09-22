"""Generate the committed synthetic UK transaction dataset and sandbox fixture.

Deterministic (seeded) generation of >= 2,000 labelled, noisy UK bank
transaction descriptions. All names, sort codes, account numbers, emails and
phone numbers are fictional; no real personal data is used.

Usage:
    python scripts/generate_synthetic_data.py [--records 2400] [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Category catalogue
CATALOG: Dict[str, Dict[str, Any]] = {
    "Groceries": {
        "merchants": [
            "TESCO", "TESCO EXPRESS", "TESCO STORES", "SAINSBURY'S", "SAINSBURYS",
            "ALDI", "LIDL", "ASDA", "WAITROSE", "MARKS & SPENCER", "M&S",
            "MORRISONS", "CO-OP", "COOP", "OCADO", "ICELAND",
        ],
        "rails": ["CARD PAYMENT", "POS", "CARD", "APPLE PAY", "GOOGLE PAY", "VISA"],
        "amount": (3.0, 120.0),
        "mcc": ["5411", "5499", "5300"],
        "direction": "debit",
        "locations": ["LONDON", "MANCHESTER", "BIRMINGHAM", "LEEDS", "BRISTOL", ""],
    },
    "Dining": {
        "merchants": [
            "NANDO'S", "NANDOS", "WAGAMAMA", "PIZZA EXPRESS", "ZIZZI", "PREZZO",
            "FIVE GUYS", "MCDONALD'S", "MCDONALDS", "KFC", "BURGER KING", "SUBWAY",
            "WETHERSPOONS", "GREENE KING", "PIZZA HUT", "DOMINO'S", "HARVESTER",
        ],
        "rails": ["CARD PAYMENT", "POS", "CARD", "APPLE PAY", "VISA", "MASTERCARD"],
        "amount": (8.0, 90.0),
        "mcc": ["5812", "5814"],
        "direction": "debit",
        "locations": ["LONDON", "MANCHESTER", "GLASGOW", "CARDIFF", "EDINBURGH", ""],
    },
    "Coffee Shops": {
        "merchants": [
            "COSTA", "COSTA COFFEE", "PRET A MANGER", "PRET", "GREGGS", "STARBUCKS",
            "CAFFE NERO", "NERO", "TIM HORTONS", "CAFE LOCAL", "BLACK SHEEP COFFEE",
        ],
        "rails": ["CARD PAYMENT", "POS", "CARD", "APPLE PAY", "SQ", "VISA"],
        "amount": (2.5, 16.0),
        "mcc": ["5814", "5812"],
        "direction": "debit",
        "locations": ["LONDON", "MANCHESTER", "OXFORD", "CAMBRIDGE", ""],
    },
    "Transport": {
        "merchants": [
            "TFL", "TFL TRAVEL CHARGE", "TRANSPORT FOR LONDON", "UBER", "UBER TRIP",
            "BOLT", "TRAINLINE", "NATIONAL RAIL", "STAGECOACH", "FIRST BUS",
            "ARRIVA", "BLACK CAB", "GETT", "SANTANDER CYCLES", "LIME",
        ],
        "rails": ["CARD PAYMENT", "POS", "CARD", "APPLE PAY", "PAYPAL"],
        "amount": (2.4, 85.0),
        "mcc": ["4111", "4121", "4131", "4789"],
        "direction": "debit",
        "locations": ["LONDON", "MANCHESTER", "LEEDS", "BIRMINGHAM", ""],
    },
    "Fuel": {
        "merchants": [
            "BP", "SHELL", "ESSO", "TEXACO", "GULF", "JET", "MURCO",
            "SAINSBURYS PETROL", "TESCO PETROL", "ASDA PETROL", "MORRISONS PETROL",
        ],
        "rails": ["CARD PAYMENT", "POS", "CARD", "VISA", "MASTERCARD"],
        "amount": (20.0, 130.0),
        "mcc": ["5541", "5542", "5172"],
        "direction": "debit",
        "locations": ["LONDON", "BIRMINGHAM", "MANCHESTER", ""],
    },
    "Shopping": {
        "merchants": [
            "AMAZON", "AMZN", "AMAZON.CO.UK", "EBAY", "ETSY", "ARGOS", "JOHN LEWIS",
            "JL", "NEXT", "PRIMARK", "H&M", "ZARA", "IKEA", "CURRY'S", "CURRYS",
            "JD SPORTS", "SPORTS DIRECT",
        ],
        "rails": ["CARD PAYMENT", "POS", "CARD", "PAYPAL", "STRIPE", "VISA"],
        "amount": (8.0, 260.0),
        "mcc": ["5310", "5399", "5691", "5311"],
        "direction": "debit",
        "locations": ["LONDON", "MANCHESTER", ""],
    },
    "Entertainment": {
        "merchants": [
            "CINEWORLD", "ODEON", "VUE", "VUE CINEMA", "TICKETMASTER", "SEE TICKETS",
            "STEAM GAMES", "STEAM", "PLAYSTATION NETWORK", "XBOX", "NINTENDO ESHOP",
            "SKY STORE", "BETFRED",
        ],
        "rails": ["CARD PAYMENT", "POS", "CARD", "PAYPAL", "VISA"],
        "amount": (6.0, 140.0),
        "mcc": ["7832", "7999", "7922"],
        "direction": "debit",
        "locations": ["LONDON", "MANCHESTER", ""],
    },
    "Travel": {
        "merchants": [
            "BOOKING.COM", "EXPEDIA", "AIRBNB", "BRITISH AIRWAYS", "BA", "EASYJET",
            "RYANAIR", "JET2", "TUI", "PREMIER INN", "TRAVELODGE", "EUROSTAR",
            "LASTMINUTE.COM",
        ],
        "rails": ["CARD PAYMENT", "CARD", "PAYPAL", "VISA", "MASTERCARD"],
        "amount": (30.0, 900.0),
        "mcc": ["4511", "4722", "4411"],
        "direction": "debit",
        "locations": ["LONDON", "GATWICK", "HEATHROW", ""],
    },
    "Utilities": {
        "merchants": [
            "BRITISH GAS", "OCTOPUS ENERGY", "EDF ENERGY", "E.ON", "EON",
            "SCOTTISH POWER", "SSE", "THAMES WATER", "SEVERN TRENT",
            "UNITED UTILITIES", "BT", "BT GROUP", "VIRGIN MEDIA", "SKY BROADBAND",
            "TALKTALK", "VODAFONE", "O2", "EE", "THREE",
        ],
        "rails": ["DD", "DIRECT DEBIT", "SO", "STANDING ORDER", "BACS"],
        "amount": (15.0, 300.0),
        "mcc": ["4900", "4814", "4812", "4899"],
        "direction": "debit",
        "locations": [""],
    },
    "Rent/Mortgage": {
        "merchants": [
            "RENT", "MONTHLY RENT", "LANDLORD", "HOUSING ASSOCIATION", "L&Q",
            "CLARION HOUSING", "PEABODY", "SANTANDER MORTGAGE", "HALIFAX MORTGAGE",
            "NATIONWIDE MORTGAGE", "BARCLAYS MORTGAGE", "MORTGAGE PAYMENT",
        ],
        "rails": ["BACS", "DD", "DIRECT DEBIT", "SO", "STANDING ORDER", "FPS"],
        "amount": (400.0, 2500.0),
        "mcc": ["6513", "9399"],
        "direction": "debit",
        "locations": [""],
    },
    "Insurance": {
        "merchants": [
            "AVIVA", "ADMIRAL", "DIRECT LINE", "CHURCHILL", "LV=", "LV", "AXA",
            "MORE THAN", "HASTINGS", "PRUDENTIAL", "LEGAL & GENERAL", "VITALITY",
        ],
        "rails": ["DD", "DIRECT DEBIT", "SO", "STANDING ORDER", "BACS"],
        "amount": (20.0, 400.0),
        "mcc": ["6300", "5960"],
        "direction": "debit",
        "locations": [""],
    },
    "Healthcare": {
        "merchants": [
            "BOOTS", "BOOTS PHARMACY", "SUPERDRUG", "NHS", "NHS PRESCRIPTION",
            "DENTIST", "DENTAL", "OPTICIANS", "SPECSAVERS", "VISION EXPRESS", "BUPA",
        ],
        "rails": ["CARD PAYMENT", "POS", "CARD", "DD", "BACS"],
        "amount": (4.0, 150.0),
        "mcc": ["5912", "8011", "8021", "8049"],
        "direction": "debit",
        "locations": ["LONDON", "MANCHESTER", ""],
    },
    "Education": {
        "merchants": [
            "UNIVERSITY", "UCL", "KINGS COLLEGE LONDON", "OPEN UNIVERSITY", "COURSERA",
            "UDEMY", "CITY LIT", "NEWCASTLE COLLEGE", "SCHOOL FEES", "NURSERY FEES",
        ],
        "rails": ["BACS", "DD", "CARD PAYMENT", "PAYPAL"],
        "amount": (20.0, 500.0),
        "mcc": ["8220", "8299"],
        "direction": "debit",
        "locations": ["LONDON", "NEWCASTLE", ""],
    },
    "Salary": {
        "merchants": [
            "ACME LIMITED", "ACME LTD", "GLOBEX CORP", "INNOTECH LTD",
            "BLUEFIN SOLUTIONS", "NORTHWIND SERVICES", "HALCYON GROUP", "PRIME HIRE LTD",
        ],
        "rails": ["BACS", "FPS", "FASTER PAYMENT"],
        "amount": (1200.0, 6000.0),
        "mcc": [""],
        "direction": "credit",
        "locations": [""],
    },
    "Transfers": {
        "merchants": [""],
        "rails": ["TFR", "TRANSFER", "FPS", "FASTER PAYMENT", "BANK TRANSFER"],
        "amount": (10.0, 5000.0),
        "mcc": [""],
        "direction": "debit",
        "locations": [""],
    },
    "Cash Withdrawal": {
        "merchants": ["LINK ATM", "ATM", "CASH MACHINE"],
        "rails": ["ATM", "CASH WITHDRAWAL", "LINK ATM"],
        "amount": (10.0, 500.0),
        "mcc": ["6011", "6010"],
        "direction": "debit",
        "locations": ["LONDON", "MANCHESTER", "LEEDS", ""],
    },
    "Subscriptions": {
        "merchants": [
            "NETFLIX", "NETFLIX.COM", "SPOTIFY", "SPOTIFY PREMIUM", "AMAZON PRIME",
            "PRIME VIDEO", "DISNEY+", "DISNEY PLUS", "NOW TV", "NOWTV",
            "APPLE MUSIC", "APPLE.COM/BILL", "GOOGLE STORAGE", "GOOGLE ONE", "ADOBE",
            "MICROSOFT 365", "DROPBOX",
        ],
        "rails": ["CARD PAYMENT", "CARD", "PAYPAL", "APPLE PAY", "VISA"],
        "amount": (3.99, 35.0),
        "mcc": ["4899", "5968", "5734"],
        "direction": "debit",
        "locations": [""],
    },
    "Charity": {
        "merchants": [
            "JUSTGIVING", "OXFAM", "BRITISH HEART FOUNDATION", "BHF", "CANCER RESEARCH UK",
            "RED CROSS", "SAVE THE CHILDREN", "RNLI", "MACMILLAN", "WWF",
            "GREENPEACE", "NSPCC", "AGE UK",
        ],
        "rails": ["CARD PAYMENT", "CARD", "DD", "BACS", "PAYPAL"],
        "amount": (2.0, 100.0),
        "mcc": ["8398"],
        "direction": "debit",
        "locations": ["LONDON", ""],
    },
    "Fees": {
        "merchants": ["HALIFAX", "BARCLAYS", "NATIONWIDE", "LLOYDS", "HSBC", ""],
        "rails": ["DD", "SO", "BACS"],
        "amount": (1.0, 50.0),
        "mcc": ["6012", "6051"],
        "direction": "debit",
        "locations": [""],
    },
    "Taxes": {
        "merchants": ["HMRC", "HM REVENUE & CUSTOMS", "COUNCIL TAX", "DVLA", ""],
        "rails": ["DD", "BACS", "FPS", "CARD PAYMENT"],
        "amount": (30.0, 500.0),
        "mcc": ["9399", "9211"],
        "direction": "debit",
        "locations": [""],
    },
    "Other": {
        "merchants": [
            "POST OFFICE", "GOV.UK", "PASSPORT OFFICE", "ROYAL MAIL",
            "DRY CLEANERS", "HAIRDRESSERS",
        ],
        "rails": ["CARD PAYMENT", "CARD", "POS", "BACS"],
        "amount": (5.0, 200.0),
        "mcc": ["5999", "7299"],
        "direction": "debit",
        "locations": ["LONDON", ""],
    },
}

# Transfer templates (no merchant names).
_TRANSFER_TEMPLATES = [
    "TFR TO SAVINGS",
    "TFR TO SAVINGS ACCOUNT",
    "TRANSFER TO SAVINGS",
    "TRANSFER TO ISA",
    "FASTER PAYMENT TO SAVINGS",
    "FPS TO SAVINGS",
    "BANK TRANSFER BETWEEN ACCOUNTS",
    "TFR TO JOINT ACCOUNT",
    "TRANSFER TO EASY ACCESS SAVER",
    "FASTER PAYMENT FROM CURRENT ACCOUNT",
]


_SALARY_TEMPLATES = [
    "{rail} {merchant} PAYROLL",
    "{rail} {merchant} SALARY",
    "{rail} SALARY {merchant}",
    "{rail} {merchant} PAYROLL REF {ref}",
    "{rail} MONTHLY SALARY {merchant}",
]



_RENT_PII_TEMPLATES = [
    "BACS JOHN SMITH 20-45-67 12345678 RENT SEPTEMBER",
    "BACS JANE DOE 12-34-56 87654321 RENT AUGUST",
    "FPS ALEX TAYLOR 55-66-77 11223344 RENT JULY",
    "BACS CHRIS EVANS 33-44-55 99887766 RENT OCTOBER",
    "BACS SAM WILSON 11-22-33 44556677 RENT JUNE",
]

_FEE_TEMPLATES = [
    "MONTHLY ACCOUNT FEE",
    "PACKAGED ACCOUNT FEE",
    "OVERDRAFT FEE",
    "OVERDRAFT INTEREST CHARGE",
    "FOREIGN TRANSACTION FEE",
    "CASH WITHDRAWAL FEE",
    "LATE PAYMENT FEE",
    "ACCOUNT FEE {merchant}",
]

_TAX_TEMPLATES = [
    "{rail} HMRC PAYMENT",
    "{rail} COUNCIL TAX",
    "{rail} HM REVENUE & CUSTOMS",
    "{rail} SELF ASSESSMENT TAX",
    "{rail} INCOME TAX PAYMENT",
    "{rail} DVLA VEHICLE TAX",
]

_UTILITY_TEMPLATES = [
    "{rail} {merchant}",
    "{rail} {merchant} REF {ref}",
    "{rail} {merchant} PAYMENT",
]

_DEFAULT_TEMPLATES = [
    "{rail} {merchant}",
    "{rail} {merchant} {location}",
    "{rail} {merchant} {terminal}",
    "{rail} {merchant} {location} {date}",
    "{rail} {merchant} {terminal} {date}",
    "{rail} {merchant} REF {ref}",
    "{rail} {merchant} {location} {terminal}",
]

_LOCATIONS_POOL = ["LONDON", "MANCHESTER", "BIRMINGHAM", "LEEDS", "BRISTOL",
                   "GLASGOW", "CARDIFF", "EDINBURGH", "OXFORD", "CAMBRIDGE"]


def _month() -> str:
    return random.choice(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    )


def _date_token() -> str:
    return random.choice(
        [
            f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/20{random.randint(23, 26)}",
            f"20{random.randint(23, 26)}{random.randint(1, 12):02d}{random.randint(1, 28):02d}",
            f"{random.randint(1, 28):02d} {_month()}",
            f"{random.randint(1, 28):02d}{_month()}",
        ]
    )


def _ref() -> str:
    return str(random.randint(10000000, 99999999))


def _terminal() -> str:
    return str(random.randint(100, 99999))


def _amount(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)


def _rail_type(rail: str) -> str:
    mapping = {
        "CARD PAYMENT": "CARD",
        "POS": "POS",
        "CARD": "CARD",
        "APPLE PAY": "APPLE_PAY",
        "GOOGLE PAY": "GOOGLE_PAY",
        "VISA": "CARD",
        "MASTERCARD": "CARD",
        "DD": "DD",
        "DIRECT DEBIT": "DD",
        "SO": "SO",
        "STANDING ORDER": "SO",
        "BACS": "BACS",
        "FPS": "FPS",
        "FASTER PAYMENT": "FPS",
        "CHAPS": "CHAPS",
        "ATM": "ATM",
        "CASH WITHDRAWAL": "ATM",
        "LINK ATM": "ATM",
        "TFR": "TFR",
        "TRANSFER": "TFR",
        "BANK TRANSFER": "TFR",
        "FASTER": "FPS",
        "BANK": "TFR",
        "PAYPAL": "PAYPAL",
        "STRIPE": "STRIPE",
        "SQ": "SQUARE",
    }
    return mapping.get(rail, "OTHER")


def _render_description(label: str, merchant: str, rail: str, location: str) -> Tuple[str, str, Optional[str]]:
    merchant_name: Optional[str] = merchant or None
    ref = _ref()
    terminal = _terminal()
    date = _date_token()

    if label == "Salary":
        template = random.choice(_SALARY_TEMPLATES)
        raw = template.format(rail=rail, merchant=merchant, ref=ref)
        if random.random() < 0.2:
            raw += f" {date}"
        return raw, _rail_type(rail), merchant_name

    if label == "Transfers":
        raw = random.choice(_TRANSFER_TEMPLATES)
        if random.random() < 0.4:
            raw += f" {date}"
        if random.random() < 0.3:
            raw += f" REF {ref}"
        return raw, _rail_type(raw.split(" ")[0]), None

    if label == "Rent/Mortgage" and random.random() < 0.2:
        raw = random.choice(_RENT_PII_TEMPLATES)
        return raw, "BACS", merchant_name

    if label == "Fees":
        template = random.choice(_FEE_TEMPLATES)
        raw = template.format(merchant=merchant)
        return raw, "DD" if random.random() < 0.5 else "SO", merchant_name

    if label == "Taxes":
        template = random.choice(_TAX_TEMPLATES)
        raw = template.format(rail=rail)
        return raw, _rail_type(rail), merchant_name

    if label == "Utilities":
        template = random.choice(_UTILITY_TEMPLATES)
        raw = template.format(rail=rail, merchant=merchant, ref=ref)
        return raw, _rail_type(rail), merchant_name

    template = random.choice(_DEFAULT_TEMPLATES)
    raw = template.format(rail=rail, merchant=merchant, location=location,
                          terminal=terminal, date=date, ref=ref)

    if random.random() < 0.15:
        raw = raw.replace("  ", " ")
        raw = raw.strip() + random.choice([" //", " *", "."])
    return raw, _rail_type(rail), merchant_name


def generate_records(total: int, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    labels = list(CATALOG.keys())
    records: List[Dict[str, Any]] = []

    per_label = max(1, total // len(labels))
    remainder = total - per_label * len(labels)

    for i, label in enumerate(labels):
        count = per_label + (1 if i < remainder else 0)
        meta = CATALOG[label]
        merchants = meta["merchants"]
        rails = meta["rails"]
        amount_lo, amount_hi = meta["amount"]
        mccs = meta["mcc"]
        direction = meta["direction"]
        locations = meta["locations"]

        for _ in range(count):
            merchant = random.choice(merchants)
            rail = random.choice(rails)
            location = random.choice(locations or [""])
            description, transaction_type, merchant_name = _render_description(
                label, merchant, rail, location
            )

            if label == "Salary":
                direction = "credit"
            elif label == "Transfers" and random.random() < 0.25:
                direction = "credit"

            mcc = random.choice(mccs) if mccs else ""
            records.append(
                {
                    "raw_description": description,
                    "amount": f"{_amount(amount_lo, amount_hi):.2f}",
                    "currency": "GBP",
                    "direction": direction,
                    "transaction_type": transaction_type,
                    "merchant_name": merchant_name or "",
                    "mcc": mcc,
                    "label": label,
                    "source": "synthetic",
                }
            )

    random.shuffle(records)
    return records


def generate_sandbox_fixture(
    records: List[Dict[str, Any]], n: int = 40, seed: int = 7
) -> Tuple[Dict[str, Any], List[int]]:
    random.seed(seed)
    indices = random.sample(range(len(records)), min(n, len(records)))
    transactions = []
    for i, idx in enumerate(indices, start=1):
        record = records[idx]
        raw_type = record["transaction_type"].lower()
        truelayer_type = {
            "CARD": "card_payment",
            "POS": "pos",
            "BACS": "bacs",
            "FPS": "faster_payment",
            "CHAPS": "chaps",
            "DD": "direct_debit",
            "SO": "standing_order",
            "ATM": "atm",
            "TFR": "transfer",
            "PAYPAL": "card_payment",
            "STRIPE": "card_payment",
            "SQUARE": "card_payment",
            "APPLE_PAY": "card_payment",
            "GOOGLE_PAY": "card_payment",
        }.get(record["transaction_type"], raw_type)
        transactions.append(
            {
                "transaction_id": f"tx-sandbox-{i:04d}",
                "account_id": "acc-ukmc-sandbox-001",
                "description": record["raw_description"],
                "amount": record["amount"],
                "currency": "GBP",
                "transaction_type": truelayer_type,
                "merchant_name": record["merchant_name"] or None,
                "mcc": record["mcc"] or None,
                "direction": record["direction"],
                "timestamp": f"2026-09-{random.randint(1, 28):02d}T{random.randint(8, 20):02d}:{random.randint(0, 59):02d}:00Z",
            }
        )
    return {
        "provider": "truelayer-sandbox",
        "accounts": [
            {"account_id": "acc-ukmc-sandbox-001", "account_type": "current", "currency": "GBP"}
        ],
        "transactions": transactions,
    }, indices


def write_outputs(records: List[Dict[str, Any]], sandbox: Dict[str, Any]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / "synthetic_uk_transactions.csv"
    json_path = RAW_DIR / "truelayer_sandbox_example.json"

    fieldnames = [
        "raw_description", "amount", "currency", "direction", "transaction_type",
        "merchant_name", "mcc", "label", "source",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(sandbox, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {csv_path}")
    print(f"Wrote {len(sandbox['transactions'])} sandbox transactions to {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sandbox", type=int, default=40)
    args = parser.parse_args()

    records = generate_records(args.records, args.seed)
    sandbox, sandbox_indices = generate_sandbox_fixture(records, args.sandbox)
    sandbox_ids = set(sandbox_indices)
    for i, record in enumerate(records):
        if i in sandbox_ids:
            record["source"] = "sandbox"
    write_outputs(records, sandbox)


if __name__ == "__main__":
    main()
