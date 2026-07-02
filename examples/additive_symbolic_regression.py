"""Worked examples for the ``jaxsr.additive`` submodule.

Demonstrates, end to end:

1. Stagewise additive regression (gradient boosting with symbolic weak learners)
2. Robust regression under outliers (Huber / absolute-error losses)
3. Quantile regression (pinball loss -> calibrated coverage)
4. Structural uncertainty via bootstrap (basis inclusion probabilities)
5. Backfitting (GAM-style: revise terms instead of freezing them)
6. Recursive basis expansion (reach compositions a flat library misses)

Run with::

    python examples/additive_symbolic_regression.py
"""

import numpy as np

from jaxsr import fit_symbolic
from jaxsr.additive import (
    BackfittingSymbolicRegressor,
    QuantileLoss,
    RecursiveSymbolicRegressor,
    StagewiseSymbolicRegressor,
    bootstrap_additive,
    bootstrap_predict_additive,
)


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1 - np.sum((y - p) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def stagewise_example():
    section("1. Stagewise additive regression:  y = 2*x0 + 0.5*x1^2")
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 2))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=200)

    model = StagewiseSymbolicRegressor(
        n_terms=5, learning_rate=0.2, max_complexity=6, refit_coefficients=True
    ).fit(X, y)
    print(model)
    print("R^2:", round(model.score(X, y), 4))
    print("combined:", model.to_expression())


def robust_example():
    section("2. Robust regression: 8% heavy outliers, evaluate on clean signal")
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(500, 2))
    clean = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2
    y = clean + rng.normal(0, 0.1, 500)
    y[rng.choice(500, 40, replace=False)] += 25.0  # outliers

    for loss in ["squared_error", "huber", "absolute_error"]:
        m = StagewiseSymbolicRegressor(
            n_terms=8, max_complexity=3, learning_rate=0.5, loss=loss, refit_coefficients=False
        ).fit(X, y)
        mae = float(np.mean(np.abs(np.asarray(m.predict(X)) - clean)))
        print(f"  {loss:15s} MAE vs clean signal = {mae:.3f}")


def quantile_example():
    section("3. Quantile regression: empirical coverage should match q")
    rng = np.random.default_rng(2)
    X = rng.uniform(-2, 2, size=(500, 2))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.normal(0, 1.0, 500)

    for q in [0.1, 0.5, 0.9]:
        m = StagewiseSymbolicRegressor(
            n_terms=10,
            max_complexity=3,
            learning_rate=0.5,
            loss=QuantileLoss(q),
            refit_coefficients=False,
        ).fit(X, y)
        coverage = float(np.mean(y <= np.asarray(m.predict(X))))
        print(f"  q={q}: fraction below prediction = {coverage:.3f}")


def uncertainty_example():
    section("4. Structural uncertainty: bootstrap inclusion probabilities")
    # Collinear features -> the structure is not well determined.
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 1, 400)
    x1 = 0.9 * x0 + 0.1 * rng.normal(0, 1, 400)
    x2 = 0.8 * x0 + 0.2 * rng.normal(0, 1, 400)
    X = np.column_stack([x0, x1, x2])
    y = x0 - x1 + 0.5 * x2 + rng.normal(0, 0.1, 400)

    est = StagewiseSymbolicRegressor(n_terms=3, max_complexity=1)
    res = bootstrap_additive(est, X, y, n_bootstrap=80, random_state=0)
    for name, prob in list(res["inclusion_probabilities"].items())[:5]:
        print(f"  {name:8s} selected in {prob:.0%} of resamples")
    pi = bootstrap_predict_additive(res["models"], X[:3], alpha=0.1)
    print("  90% prediction intervals (first 3):")
    for i in range(3):
        print(f"    x[{i}]: [{pi['lower'][i]:+.2f}, {pi['upper'][i]:+.2f}]")


def backfitting_example():
    section("5. Backfitting (GAM-style): revise terms instead of freezing")
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 2, size=(300, 2))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.1, 300)

    bf = BackfittingSymbolicRegressor(n_terms=3, n_sweeps=6, max_complexity=3).fit(X, y)
    print("R^2:", round(bf.score(X, y), 4), "| sweeps run:", len(bf.training_history_) - 1)


def recursive_example():
    section("6. Recursive expansion: reach compositions a flat library misses")
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(400, 2))
    y = np.exp(X[:, 0] * X[:, 1]) + 0.02 * rng.normal(size=400)
    Xtr, ytr, Xte, yte = X[:200], y[:200], X[200:], y[200:]

    rec = RecursiveSymbolicRegressor(n_expansions=3, max_terms=6, beam_width=25).fit(Xtr, ytr)
    flat = fit_symbolic(Xtr, ytr, max_terms=6, include_transcendental=True)
    print("  target: exp(x0*x1)")
    print(f"  recursive test R^2 = {r2(yte, rec.predict(Xte)):.4f}  ->  {rec.expression_[:48]}")
    print(f"  flat-library test R^2 = {r2(yte, flat.predict(Xte)):.4f}")


def main() -> None:
    stagewise_example()
    robust_example()
    quantile_example()
    uncertainty_example()
    backfitting_example()
    recursive_example()


if __name__ == "__main__":
    main()
