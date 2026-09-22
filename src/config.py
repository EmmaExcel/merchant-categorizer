"""Central application settings.

All secrets are read from environment variables (optionally from a local
``.env`` file). The application never ships with populated credentials and the
fixture mode works with no credentials at all.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    return float(value)


class Settings:
    """Environment-backed settings for the whole service."""

    def __init__(self) -> None:
        self.APP_NAME = "uk-merchant-categoriser"
        self.APP_VERSION = "1.0.0"
        self.ENV = os.getenv("UKMC_ENV", "dev")
        self.LOG_LEVEL = os.getenv("UKMC_LOG_LEVEL", "INFO")

        # Database -----------------------------------------------------------
        self.DATABASE_URL = os.getenv(
            "UKMC_DATABASE_URL", "sqlite:///./data/ukmc.db"
        )

        # Ingestion -----------------------------------------------------------
        self.FIXTURE_MODE = _as_bool(os.getenv("UKMC_FIXTURE_MODE"), True)
        self.TRUELAYER_CLIENT_ID = os.getenv("UKMC_TRUELAYER_CLIENT_ID", "")
        self.TRUELAYER_CLIENT_SECRET = os.getenv("UKMC_TRUELAYER_CLIENT_SECRET", "")
        self.TRUELAYER_REDIRECT_URI = os.getenv("UKMC_TRUELAYER_REDIRECT_URI", "")
        self.TRUELAYER_BANK_BASE_URL = os.getenv(
            "UKMC_TRUELAYER_BANK_BASE_URL", "https://api.truelayer-sandbox.com"
        )
        self.FIXTURE_ACCOUNTS_PATH = PROJECT_ROOT / "data" / "raw" / "truelayer_sandbox_example.json"
        self.FIXTURE_TRANSACTIONS_PATH = (
            PROJECT_ROOT / "data" / "raw" / "synthetic_uk_transactions.csv"
        )

        # Model serving -------------------------------------------------------
        self.MODEL_TYPE = os.getenv("UKMC_MODEL_TYPE", "minilm").strip().lower()
        self.MODEL_DIR = Path(os.getenv("UKMC_MODEL_DIR", str(PROJECT_ROOT / "artifacts" / "current")))
        self.DEVICE = os.getenv("UKMC_DEVICE", "auto").strip().lower()
        self.SEED = int(os.getenv("UKMC_SEED", "42"))
        self.CONFIDENCE_THRESHOLD = _as_float(
            os.getenv("UKMC_CONFIDENCE_THRESHOLD"), 0.75
        )

        # PII redaction -------------------------------------------------------
        self.PII_USE_PRESIDIO = _as_bool(os.getenv("UKMC_PII_USE_PRESIDIO"), True)
        self.PII_SCORE_THRESHOLD = _as_float(
            os.getenv("UKMC_PII_SCORE_THRESHOLD"), 0.45
        )

        self.ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
        self.RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
        self.DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
        self.DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
