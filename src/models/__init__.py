"""Model implementations and artifact helpers."""

from models.base import (
    BaseCategoriser,
    MetaEmbedder,
    ModelConfig,
    cosine_warmup_schedule,
    count_parameters,
    pick_device,
    set_seed,
)
from models.bilstm_attention import BiLSTMAttentionClassifier
from models.minilm_classifier import MiniLMClassifier

MODEL_REGISTRY = {
    "minilm": MiniLMClassifier,
    "bilstm": BiLSTMAttentionClassifier,
}


def get_model_class(model_type: str):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from {sorted(MODEL_REGISTRY)}."
        )
    return MODEL_REGISTRY[model_type]


def build_model(model_type: str, config: ModelConfig):
    """Build a fresh model instance for training."""
    return get_model_class(model_type)(config)


__all__ = [
    "BaseCategoriser",
    "MetaEmbedder",
    "ModelConfig",
    "MiniLMClassifier",
    "BiLSTMAttentionClassifier",
    "MODEL_REGISTRY",
    "get_model_class",
    "build_model",
    "pick_device",
    "set_seed",
    "cosine_warmup_schedule",
    "count_parameters",
]
