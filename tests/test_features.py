from __future__ import annotations

from preprocessing.features import (
    LABELS,
    amount_bucket,
    build_metadata_features,
    detect_payment_rail,
    mcc_bucket,
    normalise_direction,
    pii_type_vector,
)


def test_labels_have_21_categories():
    assert len(LABELS) == 21
    assert LABELS[0] == "Groceries"
    assert "Rent/Mortgage" in LABELS
    assert "Cash Withdrawal" in LABELS


def test_amount_buckets():
    assert amount_bucket(5.0) == "micro"
    assert amount_bucket(9.99) == "micro"
    assert amount_bucket(10.0) == "small"
    assert amount_bucket(49.99) == "small"
    assert amount_bucket(50.0) == "medium"
    assert amount_bucket(199.99) == "medium"
    assert amount_bucket(200.0) == "large"
    assert amount_bucket(999.99) == "large"
    assert amount_bucket(1000.0) == "very_large"
    assert amount_bucket(None) == "micro"
    assert amount_bucket(-42.0) == "small"


def test_direction_normalisation():
    assert normalise_direction("debit") == "debit"
    assert normalise_direction("out") == "debit"
    assert normalise_direction("credit") == "credit"
    assert normalise_direction("in") == "credit"
    assert normalise_direction(None) == "debit"


def test_payment_rail_detection():
    assert detect_payment_rail("CARD", "") == "CARD"
    assert detect_payment_rail("DD", "") == "DD"
    assert detect_payment_rail(None, "BACS ACME LTD") == "BACS"
    assert detect_payment_rail(None, "PAYPAL *NETFLIX") == "PAYPAL"
    assert detect_payment_rail(None, "SQ *COFFEE HOUSE") == "SQUARE"
    assert detect_payment_rail(None, "TFL TRAVEL CHARGE") == "OTHER"
    assert detect_payment_rail(None, "APPLE PAY TESCO") == "APPLE_PAY"
    assert detect_payment_rail(None, "FASTER PAYMENT TO SAVINGS") == "FPS"


def test_mcc_buckets():
    assert mcc_bucket("5411") == 0
    assert mcc_bucket(None) == mcc_bucket("")
    unknown = mcc_bucket("1234")
    assert 16 <= unknown < 24


def test_metadata_features_and_pii_vector():
    meta = build_metadata_features(42.5, "debit", "CARD", "5411", ["PERSON"])
    assert meta["amount_bucket"] == "small"
    assert meta["direction"] == "debit"
    assert meta["payment_rail"] == "CARD"
    assert meta["pii_detected"] is True
    assert meta["mcc_bucket"] == 0
    assert meta["mcc_present"] is True
    assert pii_type_vector(["PERSON"])[0] == 1
    assert pii_type_vector([]) == [0] * 9


def test_mcc_present_flag():
    assert build_metadata_features(10, "debit", None, None, [])["mcc_present"] is False
    assert build_metadata_features(10, "debit", None, "", [])["mcc_present"] is False
    assert build_metadata_features(10, "debit", None, "5812", [])["mcc_present"] is True
