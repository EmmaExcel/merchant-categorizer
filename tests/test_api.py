from __future__ import annotations

from preprocessing.features import LABELS


def test_health(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "1.0.0"
    assert data["fixture_mode"] is True


def test_model_info(api_client):
    response = api_client.get("/model")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "bilstm"
    assert data["version"] == "1.0.0"
    assert data["supported_labels"] == LABELS
    assert data["metrics"]["macro_f1"] == 0.82


def test_predict_returns_expected_schema(api_client):
    payload = {
        "description": "CARD PAYMENT TESCO STORES 3402 LONDON",
        "amount": 42.65,
        "currency": "GBP",
        "direction": "debit",
        "transaction_type": "CARD",
        "mcc": None,
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "predicted_category", "confidence", "top_3_predictions",
        "redacted_description", "cleaned_description",
        "pii_entities_redacted", "requires_review", "model_version",
    }
    assert data["predicted_category"] in LABELS
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_3_predictions"]) == 3
    assert data["cleaned_description"] == "TESCO STORES LONDON"
    assert data["pii_entities_redacted"] == []
    assert isinstance(data["requires_review"], bool)


def test_predict_redacts_pii_before_cleaning(api_client):
    payload = {
        "description": "BACS JOHN SMITH 20-45-67 12345678 RENT SEPTEMBER",
        "amount": 1200.0,
        "direction": "debit",
        "transaction_type": "BACS",
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "[PERSON]" in data["redacted_description"]
    assert "[SORT_CODE]" in data["redacted_description"]
    assert "[ACCOUNT_NUMBER]" in data["redacted_description"]
    assert "JOHN SMITH" not in data["redacted_description"]
    assert "20-45-67" not in data["redacted_description"]
    assert set(data["pii_entities_redacted"]) >= {"PERSON", "SORT_CODE", "ACCOUNT_NUMBER"}
    assert data["cleaned_description"] == "RENT"


def test_predict_requires_review_below_threshold(api_client):
    payload = {"description": "XYZ UNKNOWN DESCRIPTION", "amount": 5.0, "direction": "debit"}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # The stub model is untrained, so confidence may be anything; only the
    # flag/threshold consistency is asserted here.
    assert data["requires_review"] is (data["confidence"] < 0.75)


def test_predict_batch(api_client):
    payload = {
        "transactions": [
            {"description": "TESCO STORES LONDON", "amount": 10.0, "direction": "debit"},
            {"description": "TFL TRAVEL CHARGE", "amount": 2.8, "direction": "debit"},
        ]
    }
    response = api_client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert all("predicted_category" in r for r in data["results"])


def test_predict_validation_errors(api_client):
    response = api_client.post("/predict", json={"description": ""})
    assert response.status_code == 422
    response = api_client.post(
        "/predict", json={"description": "TESCO", "direction": "sideways"}
    )
    assert response.status_code == 422


def test_feedback_accepts_corrected_category(api_client, db_session):
    payload = {
        "redacted_description": "CARD PAYMENT TESCO STORES [REDACTED] LONDON",
        "original_prediction": "Shopping",
        "corrected_category": "Groceries",
        "note": "supermarket",
    }
    response = api_client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["stored"] == "redacted"
    assert data["id"] > 0


def test_feedback_rejects_invalid_category(api_client):
    payload = {
        "redacted_description": "TESCO STORES LONDON",
        "original_prediction": "Shopping",
        "corrected_category": "NotACategory",
    }
    response = api_client.post("/feedback", json=payload)
    assert response.status_code == 422


def test_deletion_endpoint(api_client, db_session):
    # Insert a transaction with a subject id via /predict, then delete it.
    api_client.post(
        "/predict",
        json={
            "description": "TESCO STORES LONDON",
            "subject_id": "subject-42",
            "amount": 10.0,
            "direction": "debit",
        },
    )
    response = api_client.delete("/data/subject-42")
    assert response.status_code == 200
    data = response.json()
    assert data["subject_id"] == "subject-42"
    assert data["deleted"]["redacted_transactions"] >= 1
