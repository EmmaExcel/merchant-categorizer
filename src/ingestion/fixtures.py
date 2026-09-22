from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from config import settings


def load_sandbox_fixture(path: Path | None = None) -> Dict[str, Any]:
    target = path or settings.FIXTURE_ACCOUNTS_PATH
    with target.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_synthetic_transactions(path: Path | None = None) -> List[Dict[str, Any]]:
    target = path or settings.FIXTURE_TRANSACTIONS_PATH
    records: List[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(row)
    return records
