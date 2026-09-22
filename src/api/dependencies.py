from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings
from models import get_model_class
from models.base import pick_device
from models.bootstrap import BootstrapCategoriser, bootstrap_model_info
from preprocessing.features import LABELS
from preprocessing.transaction_cleaner import TransactionCleaner
from privacy.redactor_adapter import RedactorAdapter, get_redactor

logger = logging.getLogger(__name__)

_model_cache: Dict[str, Any] = {}
_cleaner_cache: Optional[TransactionCleaner] = None


def _load_artifact_model() -> Any:
    artifact_dir = Path(settings.MODEL_DIR)
    config_data = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    model_class = get_model_class(config_data["model_type"])
    device = pick_device(settings.DEVICE)
    model = model_class.load_artifact(artifact_dir, device=device)
    logger.info("Loaded '%s' artifact from %s", config_data["model_type"], artifact_dir)
    return model


def get_model() -> Any:
    cache_key = str(settings.MODEL_DIR)
    if cache_key not in _model_cache:
        artifact_dir = Path(settings.MODEL_DIR)
        if (artifact_dir / "config.json").exists() and (artifact_dir / "model.pt").exists():
            _model_cache[cache_key] = _load_artifact_model()
        else:
            logger.warning(
                "No trained artifact at %s; serving bootstrap keyword model. "
                "Run `make train` for the real model.",
                artifact_dir,
            )
            _model_cache[cache_key] = BootstrapCategoriser()
    return _model_cache[cache_key]


def clear_model_cache() -> None:
    _model_cache.clear()


def get_model_info() -> Dict[str, Any]:
    model = get_model()
    if isinstance(model, BootstrapCategoriser):
        return bootstrap_model_info()

    training_info = {}
    training_path = Path(settings.MODEL_DIR) / "training_info.json"
    if training_path.exists():
        training_info = json.loads(training_path.read_text(encoding="utf-8"))
    metrics = {}
    metrics_path = Path(settings.MODEL_DIR) / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    return {
        "model_name": getattr(model, "name", "unknown"),
        "version": settings.APP_VERSION,
        "supported_labels": LABELS,
        "training_timestamp": training_info.get("training_timestamp"),
        "metrics": {
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "top1_accuracy": metrics.get("top1_accuracy"),
            "top3_accuracy": metrics.get("top3_accuracy"),
        },
    }


def get_redactor_dep() -> RedactorAdapter:
    return get_redactor()


def get_cleaner_dep() -> TransactionCleaner:
    global _cleaner_cache
    if _cleaner_cache is None:
        _cleaner_cache = TransactionCleaner(redactor=get_redactor())
    return _cleaner_cache


def model_version() -> str:
    model = get_model()
    if isinstance(model, BootstrapCategoriser):
        return model.version
    return settings.APP_VERSION
