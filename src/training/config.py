from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import PROJECT_ROOT, settings


@dataclass
class TrainingConfig:
    model_type: str = "minilm"
    data_csv: Path = PROJECT_ROOT / "data" / "raw" / "synthetic_uk_transactions.csv"
    output_dir: Path = PROJECT_ROOT / "artifacts" / "current"


    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 1


    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 6
    grad_clip: float = 1.0


    dropout: float = 0.3
    hidden: int = 256
    meta_embed_dim: int = 16
    max_length: int = 48
    vocab_size: int = 4000
    embedding_dim: int = 128
    lstm_hidden: int = 128
    lstm_layers: int = 2


    seed: int = settings.SEED
    device: str = settings.DEVICE
    num_workers: int = 0



    mcc_dropout: float = 0.7


    embedding_cache: Optional[Path] = PROJECT_ROOT / "data" / "processed" / "minilm_embeddings.npz"
