"""Ridge, Lasso, Elastic Net from scratch (NumPy only).

Objectives are written exactly the way sklearn defines them, so the tests can
compare coefficient-for-coefficient. n = number of rows.

  Ridge       :  ||y - Xw - b||^2               + lam * ||w||^2
  Lasso       :  (1/2n) ||y - Xw - b||^2        + alpha * ||w||_1
  Elastic Net :  (1/2n) ||y - Xw - b||^2        + alpha * l1_ratio * ||w||_1
                                                + 0.5 * alpha * (1 - l1_ratio) * ||w||^2

Intercept b is never penalised: centre X and y first, solve for w on the
centred data, then recover b = mean(y) - mean(X) @ w.

Run `pytest tests/test_regularization.py` until green.
"""
from __future__ import annotations

import numpy as np


def soft_threshold(z: np.ndarray, t: float) -> np.ndarray:
    """S(z, t) = sign(z) * max(|z| - t, 0).

    This one line is WHY Lasso gives exact zeros: anything with |z| <= t
    is snapped to 0.
    """
    return np.sign(z) * np.maximum(np.abs(z) - t, 0) 


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """Return (intercept, coef).

    On centred data: w = (X^T X + lam * I)^-1 X^T y.
    Use np.linalg.solve, not np.linalg.inv.
    """
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    n, p = Xc.shape
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(p), Xc.T @ yc)
    return float(y_mean - x_mean @ w), w


def elastic_net_cd(
    X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float,
    max_iter: int = 10000, tol: float = 1e-10,
) -> tuple[float, np.ndarray]:
    """Coordinate descent. Return (intercept, coef).

    On centred data, start w = 0, residual r = y - X w. Repeat until the
    biggest weight change in a full sweep is < tol:
      for each feature j:
        rho_j = X[:, j] @ (r + X[:, j] * w[j])          # correlation with the residual, w_j put back
        z_j   = X[:, j] @ X[:, j]
        new   = soft_threshold(rho_j, n * alpha * l1_ratio) / (z_j + n * alpha * (1 - l1_ratio))
        update r for the change in w[j], then set w[j] = new
    """
    x_mean, y_mean = X.mean(axis=0), y.mean()
    X, y = X - x_mean, y - y_mean      # work on centred data; intercept recovered at the end
    n, p = X.shape
    w = np.zeros(p)
    r = y.copy()
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            rho_j = X[:, j] @ (r + X[:, j] * w[j])
            z_j = X[:, j] @ X[:, j]
            new = soft_threshold(rho_j, n * alpha * l1_ratio) / (z_j + n * alpha * (1 - l1_ratio))
            r -= X[:, j] * (new - w[j])
            w[j] = new
        if np.max(np.abs(w - w_old)) < tol:
            break
    return float(y_mean - x_mean @ w), w   


def lasso_cd(X: np.ndarray, y: np.ndarray, alpha: float, **kw) -> tuple[float, np.ndarray]:
    """Lasso is Elastic Net with l1_ratio = 1. One line."""
    return elastic_net_cd(X, y, alpha, l1_ratio=1.0, **kw)


def count_zeros_along_path(X: np.ndarray, y: np.ndarray, alphas) -> np.ndarray:
    """For each alpha, fit lasso_cd and count coefficients that are exactly 0."""
    return np.array([np.sum(lasso_cd(X, y, alpha)[1] == 0) for alpha in alphas])
