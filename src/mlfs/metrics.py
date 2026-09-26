"""Classification metrics from scratch (NumPy only).

Fill each body. Conventions: y_true, y_pred in {0,1}; y_score is P(y=1) in [0,1].
Positive class = 1. Run `pytest tests/test_metrics.py` until green.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) — counts, in that order."""
    raise NotImplementedError


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """tp / (tp + fp). Return 0.0 if no positive predictions."""
    raise NotImplementedError


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """tp / (tp + fn). Return 0.0 if no actual positives."""
    raise NotImplementedError


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Harmonic mean of precision and recall. 0.0 if both are 0."""
    raise NotImplementedError


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep every distinct score as a threshold (descending).

    Return (fpr, tpr, thresholds), each starting at (0, 0) and ending at (1, 1).
    fpr = fp / (fp + tn), tpr = tp / (tp + fn), predicted positive when score >= threshold.
    """
    raise NotImplementedError


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve by the trapezoid rule (np.trapz over fpr, tpr)."""
    raise NotImplementedError


def pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same sweep as roc_curve. Return (precision, recall, thresholds)."""
    raise NotImplementedError


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision: sum over thresholds of (recall_k - recall_{k-1}) * precision_k.

    This is what sklearn.metrics.average_precision_score computes (no interpolation).
    """
    raise NotImplementedError


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray, step: float = 0.05) -> np.ndarray:
    """Rows of (threshold, precision, recall, f1) for thresholds 0, step, 2*step, ..., 1."""
    raise NotImplementedError
