import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from mlfs.regularization import (
    soft_threshold, ridge_closed_form, lasso_cd, elastic_net_cd, count_zeros_along_path,
)


@pytest.fixture(scope="module")
def data():
    # 20 features, only 5 actually matter -> Lasso should find the other 15 are useless
    X, y = make_regression(n_samples=300, n_features=20, n_informative=5,
                           noise=10.0, random_state=0)
    return X, y


def test_soft_threshold():
    z = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    np.testing.assert_allclose(soft_threshold(z, 1.0), [-2.0, 0.0, 0.0, 0.0, 2.0])


def test_ridge_zero_lambda_is_ols(data):
    X, y = data
    b, w = ridge_closed_form(X, y, lam=0.0)
    ols = LinearRegression().fit(X, y)
    np.testing.assert_allclose(w, ols.coef_, atol=1e-6)
    assert b == pytest.approx(ols.intercept_, abs=1e-6)


@pytest.mark.parametrize("lam", [0.1, 10.0, 1000.0])
def test_ridge_matches_sklearn(data, lam):
    X, y = data
    b, w = ridge_closed_form(X, y, lam=lam)
    sk = Ridge(alpha=lam).fit(X, y)
    np.testing.assert_allclose(w, sk.coef_, atol=1e-6)
    assert b == pytest.approx(sk.intercept_, abs=1e-6)


def test_ridge_shrinks_but_never_zeros(data):
    X, y = data
    _, w = ridge_closed_form(X, y, lam=1e4)
    assert np.all(np.abs(w) > 0)                      # small, but not exactly zero


@pytest.mark.parametrize("alpha", [0.1, 1.0, 5.0])
def test_lasso_matches_sklearn(data, alpha):
    X, y = data
    b, w = lasso_cd(X, y, alpha=alpha)
    sk = Lasso(alpha=alpha, tol=1e-10, max_iter=100000).fit(X, y)
    np.testing.assert_allclose(w, sk.coef_, atol=1e-4)
    assert b == pytest.approx(sk.intercept_, abs=1e-4)


def test_lasso_zeroes_useless_features(data):
    X, y = data
    _, w = lasso_cd(X, y, alpha=5.0)
    assert np.sum(w == 0.0) >= 10                     # exact zeros, not "small"


@pytest.mark.parametrize("l1_ratio", [0.2, 0.5, 0.8])
def test_elastic_net_matches_sklearn(data, l1_ratio):
    X, y = data
    b, w = elastic_net_cd(X, y, alpha=1.0, l1_ratio=l1_ratio)
    sk = ElasticNet(alpha=1.0, l1_ratio=l1_ratio, tol=1e-10, max_iter=100000).fit(X, y)
    np.testing.assert_allclose(w, sk.coef_, atol=1e-4)


def test_zeros_grow_with_alpha(data):
    X, y = data
    zeros = count_zeros_along_path(X, y, alphas=[0.01, 0.1, 1.0, 5.0, 20.0])
    assert list(zeros) == sorted(zeros)               # more penalty -> more zeros
    assert zeros[0] < zeros[-1]
