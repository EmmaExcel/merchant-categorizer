from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def _env_sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "FIXTURE_MODE", True)
    monkeypatch.setattr(settings, "PII_USE_PRESIDIO", False)
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(settings, "MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setattr(settings, "DEVICE", "cpu")
    monkeypatch.setattr(settings, "SEED", 42)
    monkeypatch.setattr(settings, "CONFIDENCE_THRESHOLD", 0.75)


    from api import dependencies as deps
    from database import session as db_session
    from privacy.redactor_adapter import get_redactor as _gr

    db_session._engine = None
    db_session._SessionLocal = None
    _gr._default_adapter = None
    deps._model_cache.clear()
    deps._cleaner_cache = None
    yield


@pytest.fixture
def redactor():
    from privacy.redactor_adapter import RedactorAdapter

    return RedactorAdapter(use_presidio=False)


@pytest.fixture
def cleaner(redactor):
    from preprocessing.transaction_cleaner import TransactionCleaner

    return TransactionCleaner(redactor=redactor)


@pytest.fixture
def db_session(tmp_path):
    from database import session as db_session
    from database.models import Base

    db_session._engine = None
    db_session._SessionLocal = None
    engine = db_session.get_engine()
    Base.metadata.create_all(engine)
    factory = db_session.get_session_factory()
    session = factory()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
def stub_artifact(tmp_path):
    from models.base import ModelConfig
    from models.bilstm_attention import BiLSTMAttentionClassifier

    corpus = [
        "TESCO STORES LONDON", "ALDI MANCHESTER", "COSTA COFFEE", "TFL TRAVEL CHARGE",
        "BRITISH GAS", "NETFLIX SUBSCRIPTION", "TRANSFER SAVINGS", "SALARY PAYROLL",
        "RENT", "ATM CASH WITHDRAWAL", "AMAZON SHOPPING", "HMRC TAX", "BP FUEL",
    ]
    config = ModelConfig(
        model_type="bilstm",
        vocab_size=64,
        embedding_dim=16,
        lstm_hidden=16,
        lstm_layers=1,
        hidden=32,
        max_length=24,
        meta_embed_dim=8,
    )
    model = BiLSTMAttentionClassifier(config)
    model.tokenizer = model.train_tokenizer(corpus, vocab_size=64, max_length=24)
    artifact_dir = tmp_path / "model"
    model.save_artifact(
        artifact_dir,
        metrics={
            "macro_f1": 0.82,
            "weighted_f1": 0.84,
            "top1_accuracy": 0.81,
            "top3_accuracy": 0.94,
        },
        training_info={"training_timestamp": "2026-09-22T00:00:00+00:00"},
    )
    return artifact_dir


@pytest.fixture
def api_client(stub_artifact, monkeypatch):
    from fastapi.testclient import TestClient

    from api import dependencies as deps
    from api.main import app

    monkeypatch.setattr(settings, "MODEL_DIR", str(stub_artifact))
    deps._model_cache.clear()
    with TestClient(app) as client:
        yield client
