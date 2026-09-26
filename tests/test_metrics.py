import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mlfs.metrics import (
    confusion_matrix, precision, recall, f1,
    roc_curve, roc_auc, pr_curve, pr_auc, threshold_sweep,
)


@pytest.fixture(scope="module")
def imbalanced():
    X, y = make_classification(n_samples=5000, n_features=10, weights=[0.99, 0.01],
                               random_state=0, flip_y=0.01)
    score = LogisticRegression(max_iter=2000).fit(X, y).predict_proba(X)[:, 1]
    return y, score


def test_confusion_matrix_matches_sklearn(imbalanced):
    y, s = imbalanced
    pred = (s >= 0.5).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, pred).ravel()
    assert confusion_matrix(y, pred) == (tp, fp, fn, tn)


def test_prf_match_sklearn(imbalanced):
    y, s = imbalanced
    pred = (s >= 0.3).astype(int)
    assert precision(y, pred) == pytest.approx(skm.precision_score(y, pred))
    assert recall(y, pred) == pytest.approx(skm.recall_score(y, pred))
    assert f1(y, pred) == pytest.approx(skm.f1_score(y, pred))


def test_f1_is_at_most_arithmetic_mean(imbalanced):
    y, s = imbalanced
    pred = (s >= 0.3).astype(int)
    assert f1(y, pred) <= (precision(y, pred) + recall(y, pred)) / 2 + 1e-12


def test_roc_auc_matches_sklearn(imbalanced):
    y, s = imbalanced
    assert roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-6)


def test_roc_auc_extremes():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000)
    assert roc_auc(y, rng.random(20000)) == pytest.approx(0.5, abs=0.02)


def test_pr_auc_matches_sklearn(imbalanced):
    y, s = imbalanced
    assert pr_auc(y, s) == pytest.approx(skm.average_precision_score(y, s), abs=1e-6)


def test_pr_auc_random_model_is_positive_rate():
    rng = np.random.default_rng(1)
    y = (rng.random(50000) < 0.02).astype(int)
    assert pr_auc(y, rng.random(50000)) == pytest.approx(0.02, abs=0.01)


def test_curves_endpoints(imbalanced):
    y, s = imbalanced
    fpr, tpr, _ = roc_curve(y, s)
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1


def test_threshold_sweep_shape(imbalanced):
    y, s = imbalanced
    tab = threshold_sweep(y, s, step=0.05)
    assert tab.shape == (21, 4)
    assert np.all(np.diff(tab[:, 2]) <= 1e-12)   # recall never increases as threshold rises
