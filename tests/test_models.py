from __future__ import annotations

import importlib
import os

import pytest

from preprocessing.features import build_metadata_features

META = [build_metadata_features(42.5, "debit", "CARD", "5411", []) for _ in range(4)]


def _make_bilstm():
    from models.base import ModelConfig
    from models.bilstm_attention import BiLSTMAttentionClassifier

    config = ModelConfig(
        model_type="bilstm",
        vocab_size=64,
        embedding_dim=16,
        lstm_hidden=16,
        lstm_layers=1,
        hidden=32,
        max_length=24,
        meta_embed_dim=8,
    )
    model = BiLSTMAttentionClassifier(config)
    model.tokenizer = model.train_tokenizer(
        ["TESCO STORES LONDON", "TRANSFER SAVINGS", "COSTA COFFEE"],
        vocab_size=64,
        max_length=24,
    )
    return model


def test_bilstm_forward_pass_shape():
    from preprocessing.features import encode_metadata_tensor

    model = _make_bilstm()
    text_inputs = model.encode_texts(["TESCO STORES LONDON", "TRANSFER SAVINGS"])
    meta = encode_metadata_tensor(META[:2])
    logits = model(text_inputs, meta)
    assert logits.shape == (2, 21)


def test_bilstm_artifact_roundtrip(tmp_path):
    from models.bilstm_attention import BiLSTMAttentionClassifier

    model = _make_bilstm()
    directory = tmp_path / "artifact"
    model.save_artifact(
        directory,
        metrics={"macro_f1": 0.8},
        training_info={"training_timestamp": "2026-09-22T00:00:00+00:00"},
    )

    loaded = BiLSTMAttentionClassifier.load_artifact(directory, device=None)
    assert loaded.config.model_type == "bilstm"
    texts = ["TESCO STORES LONDON"]
    prediction = loaded.predict(texts, META[:1])
    assert prediction["probabilities"].shape == (1, 21)
    assert prediction["topk_indices"].shape == (1, 3)


def test_mcc_embedding_gated_when_missing():
    import torch

    from models.base import MetaEmbedder, ModelConfig
    from preprocessing.features import build_metadata_features, encode_metadata_tensor

    config = ModelConfig()
    embedder = MetaEmbedder(config)
    meta_with_mcc = encode_metadata_tensor(
        [build_metadata_features(10, "debit", "CARD", "5411", [])]
    )
    meta_without_mcc = encode_metadata_tensor(
        [build_metadata_features(10, "debit", "CARD", None, [])]
    )
    output_without = embedder(meta_without_mcc)
    output_with = embedder(meta_with_mcc)


    start = 3 * config.meta_embed_dim
    end = 4 * config.meta_embed_dim
    assert torch.all(output_without[:, start:end] == 0)
    assert not torch.all(output_with[:, start:end] == 0)


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None
    or os.environ.get("UKMC_CI") == "true",
    reason="requires downloading the MiniLM model; skipped when unavailable or in CI",
)
def test_minilm_forward_pass_shape():
    from models.base import ModelConfig
    from models.minilm_classifier import MiniLMClassifier
    from preprocessing.features import encode_metadata_tensor

    config = ModelConfig(model_type="minilm", hidden=64, meta_embed_dim=8)
    model = MiniLMClassifier(config)
    text_inputs = model.encode_texts(["TESCO STORES LONDON", "TRANSFER SAVINGS"])
    meta = encode_metadata_tensor(META[:2])
    logits = model(text_inputs, meta)
    assert logits.shape == (2, 21)
