"""Model A (default): sentence-transformers MiniLM + metadata + MLP head."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn

from models.base import BaseCategoriser, MetaEmbedder, ModelConfig

logger = logging.getLogger(__name__)


class MiniLMClassifier(BaseCategoriser):
    """Pooled MiniLM text embedding + categorical metadata + dropout MLP head."""

    name = "minilm"

    def __init__(self, config: Optional[ModelConfig] = None):
        config = config or ModelConfig(model_type="minilm")
        super().__init__(config)
        self.meta_embedder = MetaEmbedder(config)
        text_dim = config.text_dim
        meta_dim = self.meta_embedder.output_dim

        self.head = nn.Sequential(
            nn.Linear(text_dim + meta_dim, config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.hidden // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden // 2, config.n_classes),
        )

        self._encoder = None  # lazy: SentenceTransformer instance

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------
    def _ensure_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.config.text_encoder_name)
        return self._encoder

    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        encoder = self._ensure_encoder()
        embeddings = encoder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return {"text_embedding": torch.tensor(embeddings, dtype=torch.float)}

    def forward(
        self, text_inputs: Dict[str, torch.Tensor], meta: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        text_embedding = text_inputs["text_embedding"]
        meta_embedding = self.meta_embedder(meta)
        combined = torch.cat([text_embedding, meta_embedding], dim=-1)
        return self.head(combined)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_text_encoder(self, directory: Path) -> None:
        encoder = self._ensure_encoder()
        target = directory / "text_encoder"
        target.mkdir(parents=True, exist_ok=True)
        encoder.save(str(target))

    def _load_text_encoder(self, directory: Path) -> None:
        from sentence_transformers import SentenceTransformer

        target = directory / "text_encoder"
        self._encoder = SentenceTransformer(str(target))
