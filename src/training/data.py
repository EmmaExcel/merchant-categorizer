
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from preprocessing.features import LABELS, build_metadata_features
from preprocessing.transaction_cleaner import TransactionCleaner
from privacy.redactor_adapter import RedactorAdapter, get_redactor


@dataclass
class PreparedDataset:
    df: pd.DataFrame
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]


def _parse_amount(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(str(value).replace(",", "").replace("£", ""))


def load_raw_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, dtype={"mcc": str})
    frame["amount"] = frame["amount"].apply(_parse_amount)
    frame["mcc"] = frame["mcc"].fillna("").astype(str)
    frame["merchant_name"] = frame["merchant_name"].fillna("")
    frame["raw_description"] = frame["raw_description"].fillna("").astype(str)
    frame = frame[frame["label"].isin(LABELS)]
    return frame.reset_index(drop=True)


def prepare_dataset(
    config,
    redactor: Optional[RedactorAdapter] = None,
    save_processed: bool = True,
) -> PreparedDataset:
    frame = load_raw_frame(config.data_csv)
    cleaner = TransactionCleaner(redactor=redactor or get_redactor())

    redacted_descriptions: List[str] = []
    cleaned_descriptions: List[str] = []
    pii_entity_types: List[str] = []
    signatures: List[str] = []

    for _, row in frame.iterrows():
        cleaned = cleaner.clean(row["raw_description"], redact=True)
        redacted_descriptions.append(cleaned.redacted_description)
        cleaned_descriptions.append(cleaned.cleaned_description)
        pii_entity_types.append("|".join(cleaned.pii_entity_types))
        signatures.append(
            TransactionCleaner.merchant_signature(
                cleaned.cleaned_description, row["merchant_name"] or None
            )
        )

    frame["redacted_description"] = redacted_descriptions
    frame["cleaned_description"] = cleaned_descriptions
    frame["pii_entity_types"] = pii_entity_types
    frame["merchant_signature"] = signatures
    frame["cleaned_empty"] = frame["cleaned_description"].str.strip() == ""


    frame = frame[~frame["cleaned_empty"]].reset_index(drop=True)

    train_idx, val_idx, test_idx = group_stratified_split(
        frame, label_col="label", group_col="merchant_signature",
        ratios=(config.train_ratio, config.val_ratio, config.test_ratio),
        seed=config.split_seed,
    )
    frame["split"] = "train"
    frame.loc[val_idx, "split"] = "val"
    frame.loc[test_idx, "split"] = "test"

    if save_processed:
        processed_path = settings.DATA_PROCESSED_DIR / "prepared_dataset.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "redacted_description", "cleaned_description", "pii_entity_types",
            "amount", "currency", "direction", "transaction_type",
            "merchant_name", "mcc", "label", "source", "split",
        ]
        frame[columns].to_csv(processed_path, index=False)

    return PreparedDataset(
        df=frame,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )


def group_stratified_split(
    frame: pd.DataFrame,
    label_col: str,
    group_col: str,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    total = len(frame)
    targets = {
        "train": ratios[0] * total,
        "val": ratios[1] * total,
        "test": ratios[2] * total,
    }
    current = {"train": 0, "val": 0, "test": 0}

    for label in LABELS:
        sub = frame[frame[label_col] == label]
        if sub.empty:
            continue
        groups: Dict[str, List[int]] = {}
        for pos, signature in zip(sub.index, sub[group_col]):
            groups.setdefault(str(signature), []).append(int(pos))

        group_items = list(groups.items())
        rng.shuffle(group_items)

        if len(group_items) == 1:


            indices = list(sub.index)
            rng.shuffle(indices)
            n = len(indices)
            n_test = max(1, int(round(n * ratios[2]))) if n >= 3 else 0
            n_val = max(1, int(round(n * ratios[1]))) if n - n_test >= 2 else 0
            test_indices.extend(int(i) for i in indices[:n_test])
            val_indices.extend(int(i) for i in indices[n_test:n_test + n_val])
            train_indices.extend(int(i) for i in indices[n_test + n_val:])
            continue

        def _place(group_indices: List[int], split: str) -> None:
            if split == "train":
                train_indices.extend(group_indices)
            elif split == "val":
                val_indices.extend(group_indices)
            else:
                test_indices.extend(group_indices)
            current[split] += len(group_indices)



        assigned_test = group_items.pop()
        _place(assigned_test[1], "test")
        if len(group_items) == 1:
            _place(group_items.pop()[1], "train")
            continue

        assigned_val = group_items.pop()
        _place(assigned_val[1], "val")



        for _, group_indices in group_items:
            deficits = {
                split: (targets[split] - current[split]) / max(targets[split], 1)
                for split in ("train", "val", "test")
            }
            best_split = max(deficits, key=deficits.get)
            _place(group_indices, best_split)

    return train_indices, val_indices, test_indices


def meta_features_from_row(row: pd.Series) -> Dict[str, Any]:
    return build_metadata_features(
        amount=row.get("amount"),
        direction=row.get("direction"),
        transaction_type=row.get("transaction_type"),
        mcc=row.get("mcc"),
        pii_entities=(
            str(row.get("pii_entity_types", "")).split("|")
            if str(row.get("pii_entity_types", ""))
            else []
        ),
        raw_description=str(row.get("redacted_description", "")),
    )
