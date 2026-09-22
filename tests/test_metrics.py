"""Tests for metric calculations."""

from __future__ import annotations

import numpy as np

from preprocessing.features import LABELS
from training.metrics import (
    classification_report,
    compute_all_metrics,
    macro_f1,
    top_k_accuracy,
    weighted_f1,
)


def test_top1_accuracy():
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    y_true = np.array([0, 1, 1])
    assert top_k_accuracy(y_true, probs, k=1) == 1.0
    assert top_k_accuracy(np.array([0]), np.array([[0.4, 0.6]]), k=1) == 0.0


def test_top3_accuracy_definition():
    # Correct label at rank 3 counts as a hit; at rank 4 it does not.
    probs = np.array([[0.4, 0.3, 0.2, 0.1]])
    y_true = np.array([2])  # rank 3
    assert top_k_accuracy(y_true, probs, k=3) == 1.0
    assert top_k_accuracy(y_true, probs, k=2) == 0.0


def test_macro_f1_perfect_and_zero():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    assert macro_f1(y_true, y_pred, n_classes=3) == 1.0
    y_bad = np.array([1, 1, 2, 2, 0, 0])
    assert macro_f1(y_true, y_bad, n_classes=3) == 0.0


def test_weighted_f1():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert 0.5 <= weighted_f1(y_true, y_pred, n_classes=2) <= 1.0


def test_classification_report_support():
    report = classification_report(np.array([0, 0, 1]), np.array([0, 1, 1]), ["a", "b"])
    assert report["a"]["support"] == 2
    assert report["b"]["support"] == 1
    assert report["a"]["precision"] == 1.0
    assert report["a"]["recall"] == 0.5


def test_compute_all_metrics_shape():
    rng = np.random.default_rng(0)
    probs = rng.random((30, len(LABELS)))
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_true = rng.integers(0, len(LABELS), size=30)
    metrics = compute_all_metrics(y_true, probs, LABELS)
    assert "macro_f1" in metrics
    assert "top3_accuracy" in metrics
    assert len(metrics["per_class"]) == len(LABELS)
