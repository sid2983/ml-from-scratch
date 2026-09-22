import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mlfs import bce_loss, predict, sigmoid, train_logistic_regression


@pytest.fixture
def data():
    # same synthetic data as the notebook
    X, y = make_classification(
        n_samples=100, n_features=2, n_informative=1, n_redundant=0,
        random_state=41, n_classes=2, n_clusters_per_class=1,
        hypercube=False, class_sep=20,
    )
    return X, y


def test_sigmoid_basics():
    assert sigmoid(np.array([0.0])) == pytest.approx(0.5)
    assert sigmoid(np.array([1000.0])) == pytest.approx(1.0)
    assert sigmoid(np.array([-1000.0])) == pytest.approx(0.0)


def test_bce_perfect_prediction_is_zero():
    y = np.array([0, 1, 1, 0])
    assert bce_loss(y, y.astype(float)) == pytest.approx(0.0, abs=1e-9)


def test_gradient_check_passes(data):
    X, y = data
    out = train_logistic_regression(X, y, epochs=1, grad_check=True)
    assert out["grad_check"]["passed"]
    assert out["grad_check"]["max_abs_diff"] < 1e-6


def test_loss_decreases(data):
    X, y = data
    out = train_logistic_regression(X, y, epochs=2000)
    lh = out["loss_history"]
    assert lh[-1] < lh[0]
    # monotone non-increasing for full-batch GD with a sane learning rate
    assert np.all(np.diff(lh) <= 1e-12)


def test_reproducible(data):
    X, y = data
    a = train_logistic_regression(X, y, epochs=100, seed=0)
    b = train_logistic_regression(X, y, epochs=100, seed=0)
    np.testing.assert_allclose(a["weights"], b["weights"])


def test_matches_sklearn_on_overlapping_data():
    # Overlapping classes -> finite optimum -> our GD should agree with sklearn (no regularisation).
    X, y = make_classification(
        n_samples=500, n_features=3, n_informative=3, n_redundant=0,
        random_state=7, class_sep=0.8,
    )
    ours = train_logistic_regression(X, y, learning_rate=0.5, epochs=20000)
    sk = LogisticRegression(C=np.inf, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(ours["coef"], sk.coef_.ravel(), rtol=1e-2, atol=1e-2)
    assert ours["intercept"] == pytest.approx(sk.intercept_[0], rel=1e-2, abs=1e-2)
    acc = (predict(X, ours["weights"]) == y).mean()
    assert acc == pytest.approx(sk.score(X, y), abs=0.01)
