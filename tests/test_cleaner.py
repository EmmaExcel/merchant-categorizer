from __future__ import annotations


def test_card_payment_example(cleaner):
    result = cleaner.clean("CARD PAYMENT TESCO STORES 3402 LONDON 14/09/2026")
    assert result.cleaned_description == "TESCO STORES LONDON"


def test_bacs_payroll_example(cleaner):
    result = cleaner.clean("BACS ACME LTD PAYROLL REF 12345678")
    assert result.cleaned_description == "ACME LTD PAYROLL"


def test_transfer_example(cleaner):
    result = cleaner.clean("TFR TO SAVINGS 220912")
    assert result.cleaned_description == "TRANSFER SAVINGS"


def test_direct_debit_utility(cleaner):
    result = cleaner.clean("DD BRITISH GAS")
    assert result.cleaned_description == "BRITISH GAS"


def test_paypal_marker_preserved(cleaner):
    result = cleaner.clean("PAYPAL *NETFLIX.COM")
    assert "PAYPAL" in result.cleaned_description
    assert "NETFLIX" in result.cleaned_description


def test_square_marker_preserved(cleaner):
    result = cleaner.clean("SQ *COFFEE HOUSE MANCHESTER")
    assert result.cleaned_description == "SQ COFFEE HOUSE MANCHESTER"


def test_tfl_marker_preserved(cleaner):
    result = cleaner.clean("TFL TRAVEL CHARGE")
    assert result.cleaned_description == "TFL TRAVEL CHARGE"


def test_rent_with_pii_cleans_to_rent(cleaner):
    result = cleaner.clean("BACS JOHN SMITH 20-45-67 12345678 RENT SEPTEMBER")
    assert result.cleaned_description == "RENT"
    assert "JOHN SMITH" not in result.redacted_description
    assert "20-45-67" not in result.redacted_description


def test_unicode_and_whitespace_normalisation(cleaner):
    result = cleaner.clean("CARD\u00a0PAYMENT\u2003TESCO\u00a0STORES\u2002LONDON")
    assert result.cleaned_description == "TESCO STORES LONDON"


def test_terminal_ids_and_long_refs_removed(cleaner):
    result = cleaner.clean("POS SAINSBURY'S 4477 23/09/2026")
    assert "4477" not in result.cleaned_description
    assert "23/09/2026" not in result.cleaned_description


def test_tokenizer(cleaner):
    tokens = cleaner.tokenize("TESCO STORES LONDON M&S CO-OP SAINSBURY'S")
    assert tokens == ["TESCO", "STORES", "LONDON", "M&S", "CO-OP", "SAINSBURY'S"]


def test_cleaned_is_uppercase_and_trimmed(cleaner):
    result = cleaner.clean("card payment tesco 3402 london")
    assert result.cleaned_description == "TESCO LONDON"
