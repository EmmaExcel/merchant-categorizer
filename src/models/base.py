"""Shared model interface and metadata embedding utilities."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from preprocessing.features import (
    AMOUNT_BUCKET2ID,
    DIRECTION2ID,
    MCC_BUCKET_COUNT,
    RAIL2ID,
    ID2LABEL,
    LABELS,
)


@dataclass
class ModelConfig:
    """Serialisable configuration shared by all model variants."""

    model_type: str = "minilm"
    n_classes: int = len(LABELS)
    amount_vocab: int = 5
    direction_vocab: int = 2
    rail_vocab: int = 15
    mcc_vocab: int = MCC_BUCKET_COUNT
    pii_dim: int = 9
    meta_embed_dim: int = 16
    dropout: float = 0.3
    hidden: int = 256
    max_length: int = 48
    seed: int = 42
    # MiniLM-specific
    text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_dim: int = 384
    freeze_text_encoder: bool = True
    # BiLSTM-specific
    vocab_size: int = 4000
    embedding_dim: int = 128
    lstm_hidden: int = 128
    lstm_layers: int = 2
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class MetaEmbedder(nn.Module):
    """Embeds categorical transaction metadata and appends the PII type vector."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.meta_embed_dim
        self.amount_emb = nn.Embedding(config.amount_vocab, dim)
        self.direction_emb = nn.Embedding(config.direction_vocab, dim)
        self.rail_emb = nn.Embedding(config.rail_vocab, dim)
        self.mcc_emb = nn.Embedding(config.mcc_vocab, dim)
        self.output_dim = 4 * dim + config.pii_dim

    def forward(self, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        amount = self.amount_emb(meta["amount_bucket"])
        direction = self.direction_emb(meta["direction"])
        rail = self.rail_emb(meta["payment_rail"])
        mcc = self.mcc_emb(meta["mcc"])
        # Gate the MCC embedding so a missing MCC contributes no signal
        # instead of biasing the head towards the "missing" bucket.
        if "mcc_present" in meta:
            mcc = mcc * meta["mcc_present"].unsqueeze(-1)
        pii = meta["pii_types"]
        return torch.cat([amount, direction, rail, mcc, pii], dim=-1)


class BaseCategoriser(nn.Module, ABC):
    """Common interface every categoriser model implements."""

    name: str = "base"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    @abstractmethod
    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize/encode cleaned descriptions into model input tensors."""

    @abstractmethod
    def forward(
        self, text_inputs: Dict[str, torch.Tensor], meta: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return class logits of shape ``(batch, n_classes)``."""

    @abstractmethod
    def _save_text_encoder(self, directory: Path) -> None:
        """Persist the text encoder/tokenizer part of the artifact."""

    @abstractmethod
    def _load_text_encoder(self, directory: Path) -> None:
        """Restore the text encoder/tokenizer part of the artifact."""

    # ------------------------------------------------------------------
    # Shared prediction / serialisation logic
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        texts: List[str],
        meta_features: List[Dict[str, Any]],
        device: Optional[torch.device] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Return probabilities and Top-K predictions for a batch."""
        self.eval()
        device = device or next(self.parameters()).device
        text_inputs = self.encode_texts(texts)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        from preprocessing.features import encode_metadata_tensor

        meta = encode_metadata_tensor(meta_features)
        meta = {k: v.to(device) for k, v in meta.items()}

        logits = self.forward(text_inputs, meta)
        probs = torch.softmax(logits, dim=-1)
        topk_values, topk_indices = torch.topk(probs, k=min(top_k, self.config.n_classes), dim=-1)

        return {
            "logits": logits.cpu().numpy(),
            "probabilities": probs.cpu().numpy(),
            "topk_indices": topk_indices.cpu().numpy(),
            "topk_probs": topk_values.cpu().numpy(),
        }

    def save_artifact(
        self,
        directory: Path,
        metrics: Optional[Dict[str, Any]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        training_info: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save the full inference artifact."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        (directory / "model.pt").write_bytes(
            _serialise_torch(self._state_dict_for_artifact())
        )
        (directory / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        label_mapping = {
            "labels": LABELS,
            "label2id": {label: i for i, label in enumerate(LABELS)},
            "id2label": {str(i): label for i, label in ID2LABEL.items()},
        }
        (directory / "label_mapping.json").write_text(
            json.dumps(label_mapping, indent=2), encoding="utf-8"
        )
        (directory / "preprocessing_config.json").write_text(
            json.dumps(
                preprocessing_config
                or {
                    "pipeline": [
                        "pii_redaction",
                        "uppercase",
                        "unicode_whitespace_normalisation",
                        "banking_code_tagging",
                        "numeric_noise_removal",
                        "marker_preservation",
                        "tokenization",
                    ],
                    "redaction_placeholders": {
                        "PERSON": "[PERSON]",
                        "SORT_CODE": "[SORT_CODE]",
                        "ACCOUNT_NUMBER": "[ACCOUNT_NUMBER]",
                        "EMAIL": "[EMAIL]",
                        "PHONE": "[PHONE]",
                        "ADDRESS": "[ADDRESS]",
                    },
                    "amount_buckets": ["micro", "small", "medium", "large", "very_large"],
                    "payment_rails": list(RAIL2ID.keys()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (directory / "metrics.json").write_text(
            json.dumps(metrics or {}, indent=2), encoding="utf-8"
        )
        (directory / "training_info.json").write_text(
            json.dumps(training_info or {}, indent=2), encoding="utf-8"
        )
        self._save_text_encoder(directory)
        return directory

    def _state_dict_for_artifact(self) -> Dict[str, torch.Tensor]:
        """State dict to persist in ``model.pt``.

        Text encoders that are saved separately (e.g. the MiniLM transformer)
        are excluded so the artifact stays small and loads independently of
        the text-encoder files.
        """
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("_encoder.")
        }

    @classmethod
    def load_artifact(cls, directory: Path, device: Optional[torch.device] = None):
        """Load a saved artifact into a model instance."""
        from models import get_model_class

        directory = Path(directory)
        config = ModelConfig.from_dict(
            json.loads((directory / "config.json").read_text(encoding="utf-8"))
        )
        model_class = get_model_class(config.model_type)
        model = model_class(config)
        model._load_text_encoder(directory)

        state = _deserialise_torch((directory / "model.pt").read_bytes())
        model_state = model.state_dict()
        missing = [
            key for key in model_state
            if key not in state and not key.startswith("_encoder.")
        ]
        unexpected = [
            key for key in state
            if key not in model_state and not key.startswith("_encoder.")
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"Artifact state dict mismatch for '{config.model_type}'. "
                f"Missing: {missing[:5]} Unexpected: {unexpected[:5]}"
            )
        model.load_state_dict(
            {key: value for key, value in state.items() if key in model_state},
            strict=False,
        )
        if device is not None:
            model = model.to(device)
        model.eval()
        return model


def _serialise_torch(state_dict: Dict[str, torch.Tensor]) -> bytes:
    import io

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def _deserialise_torch(raw: bytes) -> Dict[str, torch.Tensor]:
    import io

    buffer = io.BytesIO(raw)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def pick_device(requested: Optional[str] = None) -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda``/``mps`` to a torch device."""
    if requested is None:
        requested = "auto"
    requested = requested.strip().lower()
    if requested in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Deterministic random seed configuration for PyTorch/numpy/Python."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    current_step: int,
) -> None:
    """Linear warmup followed by cosine decay to zero."""
    if current_step < warmup_steps and warmup_steps > 0:
        lr_scale = current_step / max(1, warmup_steps)
    else:
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["initial_lr"] * max(lr_scale, 1e-8)
