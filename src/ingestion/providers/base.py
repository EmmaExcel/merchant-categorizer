from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional


@dataclass
class Account:
    account_id: str
    account_type: str
    currency: str = "GBP"
    provider: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalisedTransaction:
    transaction_id: str
    account_id: str
    raw_description: str
    amount: float
    currency: str = "GBP"
    direction: str = "debit"
    transaction_type: str = "OTHER"
    merchant_name: Optional[str] = None
    mcc: Optional[str] = None
    timestamp: Optional[datetime] = None
    provider: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BankProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_accounts(self) -> List[Account]:
        ...

    @abstractmethod
    def fetch_transactions(
        self,
        account_id: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[NormalisedTransaction]:
        ...

    @abstractmethod
    def normalise_transaction(self, raw_transaction: Dict[str, Any]) -> NormalisedTransaction:
        ...
