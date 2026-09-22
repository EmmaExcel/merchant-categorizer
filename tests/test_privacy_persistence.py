from __future__ import annotations

import logging

from sqlalchemy import select

RAW_PII = "BACS JOHN SMITH 20-45-67 12345678 RENT SEPTEMBER"


def test_raw_description_not_persisted(api_client, db_session):
    from database.models import RedactedTransaction

    response = api_client.post(
        "/predict",
        json={
            "description": RAW_PII,
            "amount": 1200.0,
            "direction": "debit",
            "transaction_type": "BACS",
            "subject_id": "privacy-subject",
        },
    )
    assert response.status_code == 200

    rows = db_session.execute(select(RedactedTransaction)).scalars().all()
    assert len(rows) >= 1
    for row in rows:
        assert "JOHN SMITH" not in row.redacted_description
        assert "20-45-67" not in row.redacted_description
        assert "12345678" not in row.redacted_description
        assert row.redacted_description != RAW_PII


def test_raw_description_not_written_to_logs(api_client, caplog):
    with caplog.at_level(logging.DEBUG):
        api_client.post(
            "/predict",
            json={
                "description": RAW_PII,
                "amount": 1200.0,
                "direction": "debit",
                "transaction_type": "BACS",
            },
        )
    assert "JOHN SMITH" not in caplog.text
    assert "20-45-67" not in caplog.text
    assert "12345678" not in caplog.text


def test_feedback_stores_only_redacted_text(api_client, db_session):
    from database.models import FeedbackLabel

    api_client.post(
        "/feedback",
        json={
            "redacted_description": "BACS [PERSON] [SORT_CODE] [ACCOUNT_NUMBER] RENT SEPTEMBER",
            "original_prediction": "Transfers",
            "corrected_category": "Rent/Mortgage",
            "note": "landlord payment",
        },
    )
    rows = db_session.execute(select(FeedbackLabel)).scalars().all()
    assert len(rows) == 1
    assert rows[0].redacted_description == "BACS [PERSON] [SORT_CODE] [ACCOUNT_NUMBER] RENT SEPTEMBER"
    assert "JOHN SMITH" not in rows[0].redacted_description
    assert "20-45-67" not in rows[0].redacted_description


def test_prepared_dataset_never_persists_raw(tmp_path, monkeypatch):
    from training.config import TrainingConfig
    from training.data import prepare_dataset

    config = TrainingConfig()
    monkeypatch.setattr("config.settings.DATA_PROCESSED_DIR", tmp_path)
    prepare_dataset(config, save_processed=True)

    processed = tmp_path / "prepared_dataset.csv"
    lines = processed.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert "raw_description" not in header


    for line in lines[1:]:
        assert "JOHN SMITH" not in line
        assert "20-45-67" not in line
