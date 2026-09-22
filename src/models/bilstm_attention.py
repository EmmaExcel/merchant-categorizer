"""Model B (educational baseline): BiLSTM + attention pooling + metadata.

A from-scratch PyTorch baseline:

* subword (WordPiece) tokenizer trained only on the synthetic training corpus
* learned embedding layer
* bidirectional LSTM
* additive attention pooling implemented directly in PyTorch
* concatenated categorical metadata embeddings
* linear classification head
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.base import BaseCategoriser, MetaEmbedder, ModelConfig

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]


class BiLSTMAttentionClassifier(BaseCategoriser):
    name = "bilstm"

    def __init__(self, config: Optional[ModelConfig] = None):
        config = config or ModelConfig(model_type="bilstm")
        super().__init__(config)
        self.tokenizer = None

        self.embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        # Additive (Bahdanau-style) attention over the bidirectional outputs.
        self.attention = nn.Linear(2 * config.lstm_hidden, 1)

        self.meta_embedder = MetaEmbedder(config)
        self.head = nn.Linear(
            2 * config.lstm_hidden + self.meta_embedder.output_dim, config.n_classes
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    @staticmethod
    def train_tokenizer(
        corpus: Iterable[str],
        vocab_size: int = 4000,
        max_length: int = 48,
        save_path: Optional[Path] = None,
    ):
        """Train a WordPiece tokenizer only on the given corpus."""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS
        )
        tokenizer.train_from_iterator(corpus, trainer)
        tokenizer.enable_padding(
            direction="right", pad_id=0, pad_token="[PAD]", length=max_length
        )
        tokenizer.enable_truncation(max_length=max_length)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(save_path))
        return tokenizer

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError(
                "BiLSTM tokenizer is not loaded. Train it or load an artifact first."
            )
        return self.tokenizer

    def _load_tokenizer(self, path: Path) -> None:
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(path))

    # ------------------------------------------------------------------
    # Encoding / forward
    # ------------------------------------------------------------------
    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        tokenizer = self._ensure_tokenizer()
        encodings = tokenizer.encode_batch([t or "[UNK]" for t in texts])
        ids = torch.tensor([e.ids for e in encodings], dtype=torch.long)
        mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    def forward(
        self, text_inputs: Dict[str, torch.Tensor], meta: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        ids = text_inputs["input_ids"]
        mask = text_inputs["attention_mask"]

        embedded = self.embedding(ids)  # (B, T, E)
        lengths = mask.sum(dim=1).clamp(min=1).cpu().to(torch.long)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=ids.size(1)
        )  # (B, T, 2H)

        # Attention pooling implemented directly in PyTorch.
        scores = self.attention(outputs).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        context = (outputs * weights).sum(dim=1)  # (B, 2H)

        meta_embedding = self.meta_embedder(meta)
        combined = torch.cat([context, meta_embedding], dim=-1)
        return self.head(combined)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_text_encoder(self, directory: Path) -> None:
        self._ensure_tokenizer().save(str(directory / "tokenizer.json"))

    def _load_text_encoder(self, directory: Path) -> None:
        self._load_tokenizer(directory / "tokenizer.json")
