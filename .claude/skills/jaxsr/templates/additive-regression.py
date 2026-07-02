"""
Additive Symbolic Regression — boosting-style ensembles of small expressions.

This template shows the ``jaxsr.additive`` workflows:
1. Stagewise additive regression (gradient boosting with symbolic weak learners)
2. Robust regression under outliers (Huber / absolute-error losses)
3. Quantile regression (prediction intervals via the pinball loss)
4. Structural uncertainty (bootstrap basis-inclusion probabilities)
5. Backfitting (GAM-style: revise terms instead of freezing them)
6. Recursive expansion (reach compositions a flat library misses)

Pick the section you need; each block is self-contained after the imports.
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

# Replace with your own data. X shape (n_samples, n_features), y shape (n_samples,).
rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(300, 2))
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=300)

# =============================================================================
# 1. Stagewise additive regression
#    Many small interpretable terms; keep max_complexity small.
#    refit_coefficients=True re-solves all weights by least squares each stage.
# =============================================================================
model = StagewiseSymbolicRegressor(
    n_terms=10,
    learning_rate=0.2,
    max_complexity=4,
    refit_coefficients=True,
    early_stopping=False,  # set True + validation_fraction to guard overfitting
)
model.fit(X, y)
print(model)  # pretty structural summary
print(model.expressions_)  # per-term expression strings
print(model.intercept_, model.coefficients_)
print("combined:", model.to_expression())
model.save("additive_model.json")  # JSON round-trip (models are NOT picklable)

# =============================================================================
# 2. Robust regression (outliers) — use refit_coefficients=False for any
#    non-squared loss (OLS refit only applies to squared error).
# =============================================================================
robust = StagewiseSymbolicRegressor(
    loss="huber",  # or "absolute_error"
    n_terms=8,
    max_complexity=3,
    learning_rate=0.5,
    refit_coefficients=False,
).fit(X, y)

# =============================================================================
# 3. Quantile regression — fit several quantiles to build a prediction band.
# =============================================================================
q_models = {
    q: StagewiseSymbolicRegressor(
        loss=QuantileLoss(q),
        n_terms=10,
        max_complexity=3,
        learning_rate=0.5,
        refit_coefficients=False,
    ).fit(X, y)
    for q in (0.1, 0.5, 0.9)
}
lower, median, upper = (q_models[q].predict(X) for q in (0.1, 0.5, 0.9))

# =============================================================================
# 4. Structural uncertainty — how stable is the discovered structure?
# =============================================================================
res = bootstrap_additive(model, X, y, n_bootstrap=100, random_state=0)
print(res["inclusion_probabilities"])  # {basis: fraction selected}
pi = bootstrap_predict_additive(res["models"], X)  # mean / lower / upper / ...

# =============================================================================
# 5. Backfitting (GAM-style) — fixed set of terms, revised across sweeps.
# =============================================================================
bf = BackfittingSymbolicRegressor(n_terms=4, n_sweeps=6, max_complexity=3).fit(X, y)

# =============================================================================
# 6. Recursive expansion (experimental) — discover compositions like exp(x0*x1)
#    that a flat library cannot reach.
# =============================================================================
Xc = rng.uniform(-2, 2, size=(400, 2))
yc = np.exp(Xc[:, 0] * Xc[:, 1])
rec = RecursiveSymbolicRegressor(n_expansions=3, max_terms=6, beam_width=25).fit(Xc, yc)
print(rec.expression_)  # e.g. "exp((x0)*(x1))"
# Compare with a flat library (which misses it):
flat = fit_symbolic(Xc, yc, max_terms=6, include_transcendental=True)
