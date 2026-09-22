"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import PROJECT_ROOT, settings


@dataclass
class TrainingConfig:
    """Configuration for a single training run."""

    model_type: str = "minilm"
    data_csv: Path = PROJECT_ROOT / "data" / "raw" / "synthetic_uk_transactions.csv"
    sandbox_json: Path = PROJECT_ROOT / "data" / "raw" / "truelayer_sandbox_example.json"
    output_dir: Path = PROJECT_ROOT / "artifacts" / "current"

    # Data splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 1  # keeps the documented example merchants in training

    # Training loop
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 6
    grad_clip: float = 1.0

    # Model hyperparameters
    dropout: float = 0.3
    hidden: int = 256
    meta_embed_dim: int = 16
    max_length: int = 48
    vocab_size: int = 4000
    embedding_dim: int = 128
    lstm_hidden: int = 128
    lstm_layers: int = 2

    # Reproducibility / device
    seed: int = settings.SEED
    device: str = settings.DEVICE
    num_workers: int = 0

    # Mask MCC for a fraction of training rows so the model does not over-rely
    # on MCC and still classifies correctly when the API receives mcc=null.
    mcc_dropout: float = 0.7

    # Text embedding cache for MiniLM (speeds repeated runs)
    embedding_cache: Optional[Path] = PROJECT_ROOT / "data" / "processed" / "minilm_embeddings.npz"

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}
