"""Tests for the browser demo page at ``/``."""

from __future__ import annotations

PRIVACY_WARNING = "Use sample data only. Do not enter real financial or personal information."


def test_demo_page_serves_html(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert PRIVACY_WARNING in response.text


def test_demo_page_has_inputs_and_result_targets(api_client):
    html = api_client.get("/").text
    for element_id in ("description", "amount", "direction", "transaction_type", "result"):
        assert f'id="{element_id}"' in html
    assert "/predict" in html


def test_demo_page_has_example_buttons(api_client):
    html = api_client.get("/").text
    for key in ("tesco", "tfl", "netflix", "salary", "redaction"):
        assert f'data-example="{key}"' in html


def test_demo_predict_payload_contract(api_client):
    """The payload the demo JS sends must produce every field the page renders."""
    response = api_client.post(
        "/predict",
        json={
            "description": "CARD PAYMENT TESCO STORES 3402 LONDON",
            "amount": 42.65,
            "direction": "debit",
            "transaction_type": "CARD",
        },
    )
    assert response.status_code == 200
    data = response.json()
    for field in (
        "predicted_category",
        "confidence",
        "requires_review",
        "cleaned_description",
        "redacted_description",
        "pii_entities_redacted",
        "top_3_predictions",
        "model_version",
    ):
        assert field in data
    assert len(data["top_3_predictions"]) == 3
    assert data["cleaned_description"] == "TESCO STORES LONDON"
