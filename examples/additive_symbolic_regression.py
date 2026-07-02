"""Minimal example: stagewise additive symbolic regression.

Fits ``y = 2*x0 + 0.5*x1**2 + noise`` as a sum of small symbolic expressions,
analogous to gradient boosting with symbolic weak learners.

Run with::

    python examples/additive_symbolic_regression.py
"""

import numpy as np

from jaxsr.additive import StagewiseSymbolicRegressor


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 2))
    noise = rng.normal(size=200)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * noise

    model = StagewiseSymbolicRegressor(
        n_terms=5,
        learning_rate=0.2,
        max_complexity=6,
        refit_coefficients=True,
    )
    model.fit(X, y)

    print(model)
    print()
    print("R^2 on training data:", model.score(X, y))
    print("Combined expression:", model.to_expression())


if __name__ == "__main__":
    main()
