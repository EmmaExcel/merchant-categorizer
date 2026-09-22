"""Provider abstraction for Open Banking ingestion."""

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
    """A provider-independent transaction record."""

    transaction_id: str
    account_id: str
    raw_description: str
    amount: float
    currency: str = "GBP"
    direction: str = "debit"  # debit | credit
    transaction_type: str = "OTHER"
    merchant_name: Optional[str] = None
    mcc: Optional[str] = None
    timestamp: Optional[datetime] = None
    provider: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "account_id": self.account_id,
            "raw_description": self.raw_description,
            "amount": self.amount,
            "currency": self.currency,
            "direction": self.direction,
            "transaction_type": self.transaction_type,
            "merchant_name": self.merchant_name,
            "mcc": self.mcc,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "provider": self.provider,
        }


class BankProvider(ABC):
    """Interface every ingestion provider must implement."""

    name: str = "base"

    @abstractmethod
    def fetch_accounts(self) -> List[Account]:
        """Return the accounts visible to the authorised user."""

    @abstractmethod
    def fetch_transactions(
        self,
        account_id: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[NormalisedTransaction]:
        """Return normalised transactions for an account in a date window."""

    @abstractmethod
    def normalise_transaction(self, raw_transaction: Dict[str, Any]) -> NormalisedTransaction:
        """Map a provider-specific raw transaction to :class:`NormalisedTransaction`."""
