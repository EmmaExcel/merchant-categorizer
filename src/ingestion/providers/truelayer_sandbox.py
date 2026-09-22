
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config import settings
from ingestion.providers.base import Account, BankProvider, NormalisedTransaction

logger = logging.getLogger(__name__)

TRUELAYER_TYPE_MAP = {
    "card_payment": "CARD",
    "card": "CARD",
    "pos": "POS",
    "bacs": "BACS",
    "faster_payment": "FPS",
    "fps": "FPS",
    "chaps": "CHAPS",
    "direct_debit": "DD",
    "standing_order": "SO",
    "atm": "ATM",
    "transfer": "TFR",
    "paypal": "PAYPAL",
    "stripe": "STRIPE",
    "square": "SQUARE",
    "apple_pay": "APPLE_PAY",
    "google_pay": "GOOGLE_PAY",
}


class TrueLayerSandboxProvider(BankProvider):
    name = "truelayer-sandbox"

    def __init__(
        self,
        fixture_mode: Optional[bool] = None,
        fixture_path: Optional[Path] = None,
        base_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        self.fixture_mode = settings.FIXTURE_MODE if fixture_mode is None else fixture_mode
        self.fixture_path = fixture_path or settings.FIXTURE_ACCOUNTS_PATH
        self.base_url = base_url or settings.TRUELAYER_BANK_BASE_URL
        self.client_id = client_id or settings.TRUELAYER_CLIENT_ID
        self.client_secret = client_secret or settings.TRUELAYER_CLIENT_SECRET
        self._fixture: Optional[Dict[str, Any]] = None
        self._access_token: Optional[str] = None




    def _load_fixture(self) -> Dict[str, Any]:
        if self._fixture is None:
            if not self.fixture_path.exists():
                raise FileNotFoundError(
                    f"Fixture file not found at {self.fixture_path}. "
                    "Re-run `python scripts/generate_synthetic_data.py`."
                )
            with self.fixture_path.open("r", encoding="utf-8") as fh:
                self._fixture = json.load(fh)
        return self._fixture

    def _use_fixture(self) -> bool:
        if self.fixture_mode:
            return True
        if not self.client_id or not self.client_secret:
            logger.info(
                "TrueLayer credentials not set; falling back to local fixture mode."
            )
            return True
        return False




    def fetch_accounts(self) -> List[Account]:
        if self._use_fixture():
            fixture = self._load_fixture()
            return [
                Account(
                    account_id=item["account_id"],
                    account_type=item.get("account_type", "current"),
                    currency=item.get("currency", "GBP"),
                    provider=self.name,
                    metadata={"source": "fixture"},
                )
                for item in fixture.get("accounts", [])
            ]

        headers = {"Authorization": f"Bearer {self._get_access_token()}"}
        response = httpx.get(f"{self.base_url}/data/v1/accounts", headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return [
            Account(
                account_id=item["account_id"],
                account_type=item.get("account_type", "current"),
                currency=item.get("currency", "GBP"),
                provider=self.name,
                metadata={"source": "truelayer-sandbox"},
            )
            for item in payload.get("results", [])
        ]

    def fetch_transactions(
        self,
        account_id: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[NormalisedTransaction]:
        if self._use_fixture():
            fixture = self._load_fixture()
            transactions = fixture.get("transactions", [])
            if account_id:
                transactions = [
                    t for t in transactions if t.get("account_id") == account_id
                ]
            normalised = [self.normalise_transaction(t) for t in transactions]
            return self._filter_dates(normalised, from_date, to_date)

        headers = {"Authorization": f"Bearer {self._get_access_token()}"}
        params: Dict[str, str] = {}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()
        response = httpx.get(
            f"{self.base_url}/data/v1/accounts/{account_id}/transactions",
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            self.normalise_transaction(item) for item in payload.get("results", [])
        ]

    def normalise_transaction(self, raw_transaction: Dict[str, Any]) -> NormalisedTransaction:
        account_id = str(raw_transaction.get("account_id", ""))
        transaction_id = str(
            raw_transaction.get("transaction_id")
            or raw_transaction.get("id")
            or self._fixture_transaction_id(raw_transaction)
        )
        raw_description = str(
            raw_transaction.get("description")
            or raw_transaction.get("merchant_name")
            or raw_transaction.get("remittance_information")
            or ""
        )
        amount = self._parse_amount(raw_transaction.get("amount"))
        currency = str(raw_transaction.get("currency", "GBP"))
        direction = self._parse_direction(raw_transaction, amount)
        transaction_type = self._parse_type(raw_transaction.get("transaction_type"))
        merchant_name = raw_transaction.get("merchant_name") or raw_transaction.get(
            "provider_name"
        )
        mcc = raw_transaction.get("mcc") or raw_transaction.get("merchant_category")
        timestamp = self._parse_timestamp(raw_transaction.get("timestamp"))

        return NormalisedTransaction(
            transaction_id=transaction_id,
            account_id=account_id,
            raw_description=raw_description,
            amount=amount,
            currency=currency,
            direction=direction,
            transaction_type=transaction_type,
            merchant_name=str(merchant_name) if merchant_name else None,
            mcc=str(mcc) if mcc is not None else None,
            timestamp=timestamp,
            provider=self.name,
            metadata={"source": "fixture" if self._use_fixture() else "truelayer-sandbox"},
        )




    def _get_access_token(self) -> str:
        if self._access_token:
            return self._access_token
        if not self.client_id or not self.client_secret:
            raise RuntimeError("TrueLayer client credentials are not configured.")
        auth = httpx.BasicAuth(self.client_id, self.client_secret)
        data = {
            "grant_type": "client_credentials",
            "scope": "accounts transactions",
        }
        response = httpx.post(
            f"{self.base_url}/connect/token", auth=auth, data=data, timeout=30
        )
        response.raise_for_status()
        self._access_token = str(response.json()["access_token"])
        return self._access_token

    @staticmethod
    def _fixture_transaction_id(raw_transaction: Dict[str, Any]) -> str:
        payload = json.dumps(raw_transaction, sort_keys=True, default=str)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return f"fixture-{digest}"

    @staticmethod
    def _parse_amount(raw_amount: Any) -> float:
        if raw_amount is None:
            return 0.0
        if isinstance(raw_amount, (int, float)):
            return float(raw_amount)
        return float(str(raw_amount).replace(",", "").replace("£", ""))

    @staticmethod
    def _parse_direction(raw: Dict[str, Any], amount: float) -> str:
        direction = raw.get("direction") or raw.get("transaction_category")
        if direction:
            value = str(direction).strip().lower()
            if value in {"debit", "out", "outgoing", "payment"}:
                return "debit"
            if value in {"credit", "in", "incoming", "deposit"}:
                return "credit"
        return "credit" if amount < 0 else "debit"

    @staticmethod
    def _parse_type(raw_type: Any) -> str:
        if not raw_type:
            return "OTHER"
        return TRUELAYER_TYPE_MAP.get(str(raw_type).strip().lower(), str(raw_type).upper())

    @staticmethod
    def _parse_timestamp(raw_timestamp: Any) -> Optional[datetime]:
        if not raw_timestamp:
            return None
        try:
            return datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _filter_dates(
        transactions: List[NormalisedTransaction],
        from_date: Optional[date],
        to_date: Optional[date],
    ) -> List[NormalisedTransaction]:
        filtered = []
        for transaction in transactions:
            if transaction.timestamp is None:
                filtered.append(transaction)
                continue
            day = transaction.timestamp.date()
            if from_date and day < from_date:
                continue
            if to_date and day > to_date:
                continue
            filtered.append(transaction)
        return filtered


def get_provider() -> TrueLayerSandboxProvider:
    return TrueLayerSandboxProvider()
