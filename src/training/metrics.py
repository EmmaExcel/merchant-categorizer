"""Training and evaluation metrics computed with numpy (no sklearn needed)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 1) -> float:
    """Fraction of examples whose true label is in the Top-K predictions.

    Top-3 accuracy is defined as: the correct label appears anywhere in the
    three highest-probability predictions.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if len(y_true) == 0:
        return 0.0
    topk = np.argsort(-probs, axis=1)[:, :k]
    hits = np.any(topk == y_true.reshape(-1, 1), axis=1)
    return float(hits.mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1 scores."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        scores.append(f1)
    return float(np.mean(scores))


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Support-weighted mean of per-class F1 scores."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    supports = np.bincount(y_true, minlength=n_classes)
    scores = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        scores.append(f1)
    total = supports.sum()
    if total == 0:
        return 0.0
    return float(np.sum(supports * np.asarray(scores)) / total)


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """Per-class precision, recall, F1 and support."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    report: Dict[str, Dict[str, float]] = {}
    for c, label in enumerate(labels):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        support = int(np.sum(y_true == c))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return report


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(np.asarray(y_true), np.asarray(y_pred)):
        matrix[int(true), int(pred)] += 1
    return matrix


def compute_all_metrics(y_true: np.ndarray, probs: np.ndarray, labels: List[str]) -> Dict:
    """Compute the full evaluation metric set."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    n_classes = len(labels)

    return {
        "macro_f1": round(macro_f1(y_true, y_pred, n_classes), 4),
        "weighted_f1": round(weighted_f1(y_true, y_pred, n_classes), 4),
        "top1_accuracy": round(top_k_accuracy(y_true, probs, k=1), 4),
        "top3_accuracy": round(top_k_accuracy(y_true, probs, k=3), 4),
        "per_class": classification_report(y_true, y_pred, labels),
        "support": int(len(y_true)),
    }
