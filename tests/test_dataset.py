"""Dataset and label validation tests."""

from __future__ import annotations

import csv
from pathlib import Path

from config import PROJECT_ROOT
from preprocessing.features import LABELS

DATA_CSV = PROJECT_ROOT / "data" / "raw" / "synthetic_uk_transactions.csv"
REQUIRED_COLUMNS = {
    "raw_description", "amount", "currency", "direction", "transaction_type",
    "merchant_name", "mcc", "label", "source",
}


def _load_rows():
    with DATA_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_dataset_has_at_least_2000_records():
    rows = _load_rows()
    assert len(rows) >= 2000


def test_dataset_has_required_columns():
    rows = _load_rows()
    assert REQUIRED_COLUMNS.issubset(set(rows[0].keys()))


def test_dataset_labels_are_valid():
    rows = _load_rows()
    labels = {row["label"] for row in rows}
    assert labels.issubset(set(LABELS))
    assert len(labels) == len(LABELS)  # every category is represented


def test_dataset_sources_are_valid():
    rows = _load_rows()
    assert {row["source"] for row in rows} <= {"synthetic", "sandbox"}
    assert any(row["source"] == "sandbox" for row in rows)


def test_prepared_dataset_excludes_raw_descriptions(tmp_path, monkeypatch):
    """After redaction the processed dataset must not contain raw text."""
    from training.config import TrainingConfig
    from training.data import prepare_dataset

    config = TrainingConfig()
    config.data_csv = DATA_CSV
    monkeypatch.setattr("config.settings.DATA_PROCESSED_DIR", tmp_path)
    prepared = prepare_dataset(config, save_processed=True)

    processed = tmp_path / "prepared_dataset.csv"
    assert processed.exists()
    with processed.open("r", encoding="utf-8") as fh:
        header = next(fh).strip()
    assert "raw_description" not in header.split(",")
    assert "redacted_description" in header.split(",")
    assert len(prepared.df) >= 2000


def test_group_split_prevents_merchant_leakage(tmp_path, monkeypatch):
    """No merchant signature may appear in more than one split."""
    from training.config import TrainingConfig
    from training.data import prepare_dataset

    config = TrainingConfig()
    config.data_csv = DATA_CSV
    monkeypatch.setattr("config.settings.DATA_PROCESSED_DIR", tmp_path)
    prepared = prepare_dataset(config, save_processed=False)

    # The grouped split guarantees that *within each label* a merchant
    # signature never appears in more than one split. The same signature
    # string may legitimately appear for different labels (e.g. "ASDA" as a
    # supermarket and "ASDA" as a petrol station), which is not leakage.
    for label in prepared.df["label"].unique():
        train_sigs = set(prepared.df.iloc[prepared.train_indices].loc[
            lambda f: f["label"] == label, "merchant_signature"
        ])
        val_sigs = set(prepared.df.iloc[prepared.val_indices].loc[
            lambda f: f["label"] == label, "merchant_signature"
        ])
        test_sigs = set(prepared.df.iloc[prepared.test_indices].loc[
            lambda f: f["label"] == label, "merchant_signature"
        ])
        assert train_sigs.isdisjoint(val_sigs)
        assert train_sigs.isdisjoint(test_sigs)
        assert val_sigs.isdisjoint(test_sigs)

    # Sanity: splits are 70/15/15 within a tolerance.
    n = len(prepared.df)
    assert 0.65 <= len(prepared.train_indices) / n <= 0.78
    assert 0.10 <= len(prepared.val_indices) / n <= 0.20
    assert 0.10 <= len(prepared.test_indices) / n <= 0.20
