"""Logistic regression from scratch: sigmoid -> log-loss -> gradient descent.

The four beats (memorise these, they are the interview answer):
    1. score      z = Xw            linear score (log-odds)
    2. squash     p = sigmoid(z)    log-odds -> probability in (0, 1)
    3. loss       L = BCE(y, p)     convex in w, unlike MSE-on-sigmoid
    4. gradient   dL/dw = X^T (p - y) / m     -> "prediction minus truth", averaged

Everything below is NumPy only.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """1 / (1 + e^-z). Clipped so exp never overflows."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    """Binary cross-entropy (log-loss), averaged over all rows.

    p is clipped away from 0 and 1 so log() never sees exactly 0.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.5,
    epochs: int = 5000,
    seed: int = 0,
    grad_check: bool = True,
    grad_check_tol: float = 1e-6,
) -> dict:
    """Fit logistic regression with full-batch gradient descent.

    One function does everything the notebook built up to:
      * seeds NumPy so runs are reproducible,
      * adds the bias column, initialises weights,
      * runs gradient descent on log-loss,
      * records the loss every epoch (loss_history),
      * optionally verifies the analytic gradient against a
        finite-difference gradient at the initial weights.

    Parameters
    ----------
    X : (m, n) feature matrix, WITHOUT a bias column.
    y : (m,)   labels in {0, 1}.
    learning_rate, epochs : plain GD hyper-parameters.
    seed : NumPy seed (used only for reproducibility of any randomness).
    grad_check : if True, run the finite-difference check before training.
    grad_check_tol : max allowed |analytic - numeric| per weight.

    Returns
    -------
    dict with keys
      intercept     : float
      coef          : (n,) array
      weights       : (n+1,) array  (bias first)
      loss_history  : (epochs,) array
      grad_check    : dict(max_abs_diff, passed) or None
    """
    np.random.seed(seed)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m = X.shape[0]
    Xb = np.insert(X, 0, 1.0, axis=1)          # bias column first
    w = np.ones(Xb.shape[1])                   # same init as the notebook

    def loss_at(w_):
        return bce_loss(y, sigmoid(Xb @ w_))

    def grad_at(w_):
        p = sigmoid(Xb @ w_)
        return Xb.T @ (p - y) / m              # dL/dw = X^T (p - y) / m

    # ---- finite-difference gradient check (before any training) ----
    check = None
    if grad_check:
        eps = 1e-5
        analytic = grad_at(w)
        numeric = np.zeros_like(w)
        for j in range(w.size):
            e = np.zeros_like(w)
            e[j] = eps
            numeric[j] = (loss_at(w + e) - loss_at(w - e)) / (2 * eps)   # central difference
        max_abs_diff = float(np.max(np.abs(analytic - numeric)))
        check = {"max_abs_diff": max_abs_diff, "passed": max_abs_diff < grad_check_tol}
        if not check["passed"]:
            raise ValueError(
                f"Gradient check failed: max |analytic - numeric| = {max_abs_diff:.3e} "
                f"(tol {grad_check_tol:.0e}). The gradient formula is wrong."
            )

    # ---- gradient descent ----
    loss_history = np.empty(epochs)
    for epoch in range(epochs):
        p = sigmoid(Xb @ w)
        loss_history[epoch] = bce_loss(y, p)
        w -= learning_rate * (Xb.T @ (p - y) / m)

    return {
        "intercept": float(w[0]),
        "coef": w[1:].copy(),
        "weights": w.copy(),
        "loss_history": loss_history,
        "grad_check": check,
    }


def predict_proba(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """P(y=1 | x) for each row. `weights` is bias-first, as returned by train_*."""
    Xb = np.insert(np.asarray(X, dtype=float), 0, 1.0, axis=1)
    return sigmoid(Xb @ weights)


def predict(X: np.ndarray, weights: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Hard 0/1 labels at the given probability threshold."""
    return (predict_proba(X, weights) >= threshold).astype(int)
