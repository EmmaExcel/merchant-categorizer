"""Training entry point.

Trains either the default MiniLM classifier or the educational BiLSTM-attention
baseline on the synthetic/sandbox dataset, with:

* stratified 70/15/15 split grouped by merchant signature
* CrossEntropyLoss with class weights computed only from the training split
* AdamW with linear warmup and cosine decay
* early stopping on validation macro F1
* best-checkpoint persistence and full inference artifact
* a generated model card and local file-based run logging

Usage:
    python -m src.training.train --model-type minilm --epochs 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import settings
from models import build_model, pick_device, set_seed
from models.base import ModelConfig, cosine_warmup_schedule, count_parameters
from preprocessing.features import LABEL2ID, LABELS, encode_metadata_tensor
from training.config import TrainingConfig
from training.data import PreparedDataset, meta_features_from_row, prepare_dataset
from training.experiment_logger import FileExperimentLogger
from training.metrics import compute_all_metrics, macro_f1

logger = logging.getLogger(__name__)


class TransactionDataset(Dataset):
    """Rows of a prepared split with precomputed MiniLM embeddings (optional)."""

    def __init__(
        self,
        frame,
        indices: List[int],
        model_type: str,
        embeddings: Optional[np.ndarray] = None,
        mcc_dropout: float = 0.0,
    ) -> None:
        self.frame = frame
        self.indices = indices
        self.model_type = model_type
        self.embeddings = embeddings
        self.mcc_dropout = mcc_dropout

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        pos = self.indices[i]
        row = self.frame.iloc[pos]
        meta = meta_features_from_row(row)
        # MCC dropout: teach the model to classify from text + other metadata
        # when the MCC is unavailable at inference time.
        if self.mcc_dropout > 0 and random.random() < self.mcc_dropout:
            meta["mcc_present"] = False
        item = {
            "cleaned": str(row["cleaned_description"]) or "[UNK]",
            "label": LABEL2ID[str(row["label"])],
            "meta": meta,
        }
        if self.model_type == "minilm" and self.embeddings is not None:
            item["text_embedding"] = torch.tensor(
                self.embeddings[pos], dtype=torch.float
            )
        return item


def collate_fn(batch: List[Dict[str, Any]], model_type: str, model=None) -> Dict[str, Any]:
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    meta = encode_metadata_tensor([item["meta"] for item in batch])
    text_inputs: Dict[str, torch.Tensor] = {}
    if model_type == "minilm":
        text_inputs["text_embedding"] = torch.stack(
            [item["text_embedding"] for item in batch]
        )
    else:
        text_inputs = model.encode_texts([item["cleaned"] for item in batch])
    return {"text_inputs": text_inputs, "meta": meta, "labels": labels}


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights computed only from the training split."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero for missing classes
    weights = len(labels) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


def get_minilm_embeddings(model, texts: List[str], cache_path: Optional[Path]) -> np.ndarray:
    """Compute (or load cached) MiniLM embeddings for all cleaned texts."""
    digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if str(cached["digest"]) == digest:
            logger.info("Loaded cached MiniLM embeddings from %s", cache_path)
            return cached["embeddings"]
    logger.info("Encoding %d descriptions with MiniLM (this may take a minute)...", len(texts))
    embeddings = model._ensure_encoder().encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings, digest=digest)
    return embeddings


@torch.no_grad()
def evaluate_split(
    model,
    dataset: TransactionDataset,
    device: torch.device,
    model_type: str,
    batch_size: int,
) -> Tuple[float, Dict[str, Any]]:
    """Return macro F1 and the full metric set for a split."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, model_type, model),
    )
    all_probs = []
    all_labels = []
    for batch in loader:
        text_inputs = {k: v.to(device) for k, v in batch["text_inputs"].items()}
        meta = {k: v.to(device) for k, v in batch["meta"].items()}
        logits = model(text_inputs, meta)
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    metrics = compute_all_metrics(labels, probs, LABELS)
    return metrics["macro_f1"], metrics


def train(config: TrainingConfig) -> Dict[str, Any]:
    set_seed(config.seed)
    device = pick_device(config.device)
    logger.info("Device: %s", device)

    prepared = prepare_dataset(config)
    frame = prepared.df
    logger.info(
        "Prepared dataset: %d rows (train=%d, val=%d, test=%d)",
        len(frame),
        len(prepared.train_indices),
        len(prepared.val_indices),
        len(prepared.test_indices),
    )

    model_config = ModelConfig(
        model_type=config.model_type,
        dropout=config.dropout,
        hidden=config.hidden,
        meta_embed_dim=config.meta_embed_dim,
        max_length=config.max_length,
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        lstm_hidden=config.lstm_hidden,
        lstm_layers=config.lstm_layers,
        seed=config.seed,
    )
    model = build_model(config.model_type, model_config).to(device)

    # BiLSTM tokenizer is trained only on the training corpus.
    if config.model_type == "bilstm":
        train_cleaned = frame.iloc[prepared.train_indices]["cleaned_description"].tolist()
        logger.info("Training BiLSTM WordPiece tokenizer on %d train rows...", len(train_cleaned))
        tokenizer = model.train_tokenizer(
            train_cleaned,
            vocab_size=config.vocab_size,
            max_length=config.max_length,
        )
        model.tokenizer = tokenizer
        model_config.vocab_size = tokenizer.get_vocab_size()

    # MiniLM text embeddings are computed once and cached.
    embeddings: Optional[np.ndarray] = None
    if config.model_type == "minilm":
        all_cleaned = frame["cleaned_description"].tolist()
        embeddings = get_minilm_embeddings(model, all_cleaned, config.embedding_cache)

    train_dataset = TransactionDataset(
        frame, prepared.train_indices, config.model_type, embeddings,
        mcc_dropout=config.mcc_dropout,
    )
    val_dataset = TransactionDataset(frame, prepared.val_indices, config.model_type, embeddings)
    test_dataset = TransactionDataset(frame, prepared.test_indices, config.model_type, embeddings)

    # Class weights from the training split only.
    train_labels = frame.iloc[prepared.train_indices]["label"].map(LABEL2ID).to_numpy()
    class_weights = compute_class_weights(train_labels, model_config.n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn(batch, config.model_type, model),
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    experiment_logger = FileExperimentLogger()
    run_id = experiment_logger.start_run(
        config.model_type,
        params={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "seed": config.seed,
            "device": str(device),
            "train_rows": len(train_dataset),
            "val_rows": len(val_dataset),
            "test_rows": len(test_dataset),
            "trainable_parameters": count_parameters(model),
        },
    )

    best_val_f1 = -1.0
    best_state = None
    patience_left = config.early_stopping_patience
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            text_inputs = {k: v.to(device) for k, v in batch["text_inputs"].items()}
            meta = {k: v.to(device) for k, v in batch["meta"].items()}
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(text_inputs, meta)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimizer.step()

            global_step += 1
            cosine_warmup_schedule(optimizer, warmup_steps, total_steps, global_step)
            total_loss += float(loss.item())

        val_f1, val_metrics = evaluate_split(
            model, val_dataset, device, config.model_type, config.batch_size
        )
        avg_loss = total_loss / max(1, len(train_loader))
        logger.info(
            "Epoch %02d/%d loss=%.4f val_macro_f1=%.4f (best %.4f)",
            epoch, config.epochs, avg_loss, val_f1, max(best_val_f1, 0.0),
        )
        experiment_logger.log_metrics(
            run_id,
            {"epoch": epoch, "train_loss": round(avg_loss, 4),
             "val_macro_f1": round(val_f1, 4),
             "val_top3_accuracy": val_metrics["top3_accuracy"]},
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping after %d epochs without improvement.", config.early_stopping_patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on the held-out test split.
    test_f1, test_metrics = evaluate_split(
        model, test_dataset, device, config.model_type, config.batch_size
    )
    logger.info("Test macro F1: %.4f | Top-3 accuracy: %.4f", test_f1, test_metrics["top3_accuracy"])

    # Persist the full inference artifact.
    output_dir = Path(config.output_dir)
    training_info = {
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model_type": config.model_type,
        "device": str(device),
        "epochs_trained": epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "data_source": "synthetic and TrueLayer sandbox fixture (no real consumer data)",
        "seed": config.seed,
    }
    model.save_artifact(
        output_dir,
        metrics=test_metrics,
        preprocessing_config=None,
        training_info=training_info,
    )
    write_model_card(output_dir, config, test_metrics, training_info)

    run_metrics = {
        "val_macro_f1": round(best_val_f1, 4),
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_top1_accuracy": test_metrics["top1_accuracy"],
        "test_top3_accuracy": test_metrics["top3_accuracy"],
    }
    experiment_logger.end_run(run_id, run_metrics)
    logger.info("Artifact saved to %s", output_dir)
    return test_metrics


def write_model_card(
    output_dir: Path,
    config: TrainingConfig,
    metrics: Dict[str, Any],
    training_info: Dict[str, Any],
) -> None:
    """Generate a model card describing data, limitations, intended use, metrics."""
    per_class_rows = "\n".join(
        f"| {label} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |"
        for label, m in metrics["per_class"].items()
    )
    card = f"""# Model Card: {config.model_type}

**This is a proof-of-concept model trained on synthetic and sandbox data.**
It does **not** claim production-grade performance on real consumer banking
data. Real data would require informed user consent and privacy controls.

## Intended use
Local, privacy-first merchant categorisation of UK bank transaction
descriptions for personal finance tooling and research prototypes.

## Training data
- Synthetic UK transaction descriptions generated programmatically (seeded).
- TrueLayer Data API Sandbox example fixture (also synthetic).
- No real personal bank transaction data was used.

## Training details
- Model: `{config.model_type}`
- Splits: 70/15/15 grouped by merchant signature to prevent leakage of
  near-duplicate descriptions.
- Loss: CrossEntropyLoss with class weights from the training split only.
- Optimiser: AdamW with linear warmup and cosine decay.
- Early stopping: validation macro F1.
- Seed: `{config.seed}`
- Timestamp: `{training_info.get('training_timestamp')}`

## Evaluation (synthetic/sandbox held-out test set)
These scores are **synthetic/sandbox evaluation results**, not real-world
production performance.

| Metric | Value |
|---|---|
| Macro F1 | {metrics['macro_f1']} |
| Weighted F1 | {metrics['weighted_f1']} |
| Top-1 accuracy | {metrics['top1_accuracy']} |
| Top-3 accuracy | {metrics['top3_accuracy']} |
| Support | {metrics['support']} |

### Per-class metrics (precision / recall / F1 / support)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
{per_class_rows}

## Limitations
- Synthetic merchants and descriptions do not capture the full long tail of
  real UK transaction data.
- Merchant-grouped splitting means the test set contains merchants unseen in
  training; real-world generalisation is likely lower.
- PII redaction may miss novel personal-data formats; human review is required
  for production use.
- Pseudonymised data can still be personal data and must be protected.

## Privacy
Raw descriptions are redacted before preprocessing, training, evaluation
exports, and persistence. Feedback storage retains only redacted text.
"""
    (output_dir / "model-card.md").write_text(card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", default="minilm", choices=["minilm", "bilstm"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-embedding-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = TrainingConfig(model_type=args.model_type)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.seed is not None:
        config.seed = args.seed
        config.split_seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.output_dir is not None:
        config.output_dir = Path(args.output_dir)
    if args.no_embedding_cache:
        config.embedding_cache = None

    train(config)


if __name__ == "__main__":
    main()
