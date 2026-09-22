# ml-from-scratch

Core ML algorithms implemented from scratch in NumPy — one file per algorithm, each written to
answer the interview question *"derive it, then implement it"*, and each checked against scikit-learn.

Built during a Sept–Oct 2026 fundamentals sprint. Companion notebooks show the build-up step by step;
`src/mlfs/` holds the clean, tested version.

## Layout

```
notebooks/   step-by-step build-ups (run top to bottom)
src/mlfs/    clean implementations, importable as `mlfs`
tests/       pytest: gradient checks, convergence, agreement with sklearn
```

## Algorithms

| # | Algorithm | Notebook | Module | Verified by |
|---|-----------|----------|--------|-------------|
| 01 | Logistic regression (perceptron → sigmoid → log-loss → GD) | `notebooks/01_logistic_regression.ipynb` | `mlfs.logistic_regression` | finite-difference gradient check (max diff ≈ 1e-11), loss monotone ↓, coefficients match sklearn (`C=inf`) to 1e-2 |

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Usage

```python
from mlfs import train_logistic_regression, predict

out = train_logistic_regression(X, y, learning_rate=0.5, epochs=5000)   # runs a gradient check first
out["intercept"], out["coef"], out["loss_history"], out["grad_check"]
predict(X, out["weights"])
```

## Notes to self (interview-facing)

- Logistic regression = linear score `z = Xw` → `sigmoid(z)` gives a probability → train by minimising log-loss with gradient descent.
- Why sigmoid: it's the inverse of the log-odds, so `z` is interpretable as log-odds and `e^(w_j)` is an odds ratio.
- Why log-loss, not MSE: log-loss on a sigmoid is convex in `w` and its gradient is the clean `X^T (p − y) / m`; MSE on a sigmoid is non-convex and its gradient vanishes when the model is confidently wrong.
- Perfectly separable data → unregularised weights grow without bound (loss → 0 but never reaches it). That is why sklearn regularises by default (`C=1.0`).
