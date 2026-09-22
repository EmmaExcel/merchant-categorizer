
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from config import settings
from models import get_model_class, pick_device, set_seed
from models.base import BaseCategoriser
from preprocessing.features import ID2LABEL, LABELS
from training.config import TrainingConfig
from training.data import PreparedDataset, meta_features_from_row, prepare_dataset
from training.metrics import compute_all_metrics, confusion_matrix

logger = logging.getLogger(__name__)


def evaluate_artifact(
    model: BaseCategoriser,
    prepared: PreparedDataset,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    test_frame = prepared.df.iloc[prepared.test_indices].reset_index(drop=True)
    model.eval()

    all_probs: List[np.ndarray] = []
    all_labels: List[int] = []
    cleaned_texts = test_frame["cleaned_description"].tolist()
    meta_features = [meta_features_from_row(row) for _, row in test_frame.iterrows()]

    with torch.no_grad():
        for start in range(0, len(cleaned_texts), batch_size):
            texts = cleaned_texts[start:start + batch_size]
            meta = meta_features[start:start + batch_size]
            prediction = model.predict(texts, meta, device=device, top_k=3)
            all_probs.append(prediction["probabilities"])
            all_labels.extend(
                LABELS.index(str(label))
                for label in test_frame["label"].iloc[start:start + batch_size]
            )

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = probs.argmax(axis=1)
    metrics = compute_all_metrics(y_true, probs, LABELS)

    return {
        "metrics": metrics,
        "probs": probs,
        "y_true": y_true,
        "y_pred": y_pred,
        "frame": test_frame,
    }


def save_evaluation_outputs(
    result: Dict[str, Any], output_dir: Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = result["metrics"]
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    frame: pd.DataFrame = result["frame"].copy()
    frame["predicted"] = [ID2LABEL[int(i)] for i in result["y_pred"]]
    frame["true_label"] = [ID2LABEL[int(i)] for i in result["y_true"]]
    frame["max_confidence"] = result["probs"].max(axis=1).round(4)


    incorrect = frame[frame["predicted"] != frame["true_label"]]
    columns = [
        "redacted_description", "cleaned_description", "pii_entity_types",
        "amount", "direction", "transaction_type", "mcc", "label", "predicted",
        "max_confidence", "source",
    ]
    incorrect[columns].to_csv(output_dir / "incorrect_predictions.csv", index=False)

    _plot_confusion_matrix(result["y_true"], result["y_pred"], output_dir / "confusion_matrix.png")
    _plot_confidence_distribution(result["probs"], result["y_true"], output_dir / "confidence_distribution.png")


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y_true, y_pred, len(LABELS))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(LABELS)), LABELS, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (synthetic/sandbox test set)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confidence_distribution(probs: np.ndarray, y_true: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(max_conf[correct], bins=25, alpha=0.6, label="Correct", color="tab:green")
    ax.hist(max_conf[~correct], bins=25, alpha=0.6, label="Incorrect", color="tab:red")
    ax.set_xlabel("Max predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution (synthetic/sandbox test set)")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(settings.MODEL_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=settings.SEED)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    set_seed(args.seed)
    device = pick_device(args.device or settings.DEVICE)

    artifact_dir = Path(args.artifact)
    config_data = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    model_class = get_model_class(config_data["model_type"])
    model = model_class.load_artifact(artifact_dir, device=device)

    training_config = TrainingConfig(model_type=config_data["model_type"])
    prepared = prepare_dataset(training_config, save_processed=False)

    result = evaluate_artifact(model, prepared, device, args.batch_size)
    metrics = result["metrics"]
    logger.info(
        "Macro F1: %.4f | Weighted F1: %.4f | Top-1: %.4f | Top-3: %.4f",
        metrics["macro_f1"], metrics["weighted_f1"],
        metrics["top1_accuracy"], metrics["top3_accuracy"],
    )

    output_dir = Path(args.output_dir or (artifact_dir / "evaluation"))
    save_evaluation_outputs(result, output_dir)
    logger.info("Evaluation outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
