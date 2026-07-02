"""
Experimental: residual-guided recursive basis expansion.

This is a deterministic, bounded cousin of genetic-programming symbolic
regression -- and a partial escape from the fixed-library ceiling.  Instead of
enumerating a huge depth-``d`` composition space up front (which explodes
combinatorially), it *grows the basis library lazily along the residual*:

1. Fit a sparse symbolic model over the current library.
2. Take the residual.
3. Build candidate basis functions by composing the currently useful
   "building blocks" (selected terms + raw features) with a small operator set
   (unary functions, products, ratios) -- one new layer of composition.
4. Screen hard: drop non-finite candidates, deduplicate, keep the top few by
   correlation with the residual.
5. Add the survivors to the library and refit.  Repeat.

The effective composition depth is the number of expansion rounds, because a
term discovered in one round becomes an input to the operators in the next.
This is essentially Fast Function Extraction (FFX) / symbolic feature
construction: it can reach compositional targets (e.g. ``x0*sin(x1)``) that a
single flat library misses, at bounded cost -- but it re-enters search-based
territory and will not match a mature GP engine (PySR, Operon).

The result is an ordinary fitted :class:`jaxsr.SymbolicRegressor` over the grown
library, so it inherits predict/expression/scoring (and the non-finite-basis
guard and negligible-term pruning of the base regressor).  Note: the composed
bases are Python closures, so the fitted model is not serialisable via
``save``/``load``, and ``to_sympy`` may not parse deeply nested term names.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from .._compat import _SklearnCompatMixin
from ..basis import BasisLibrary, _safe_exp, _safe_log, _safe_sqrt
from ..regressor import SymbolicRegressor

# Unary operators: (name template, function, complexity cost).
_UNARY_OPS: dict[str, tuple[str, Callable, int]] = {
    "sin": ("sin({})", jnp.sin, 2),
    "cos": ("cos({})", jnp.cos, 2),
    "exp": ("exp({})", _safe_exp, 2),
    "log": ("log({})", _safe_log, 2),
    "sqrt": ("sqrt({})", _safe_sqrt, 2),
    "square": ("({})^2", jnp.square, 2),
}

# A "block" is (function X->(n,), complexity, set of feature indices used).
_Block = tuple[Callable, int, frozenset]


def _compose_unary(func: Callable, unary: Callable) -> Callable:
    """Return X -> unary(func(X))."""
    return lambda X: unary(func(X))


def _compose_product(f1: Callable, f2: Callable) -> Callable:
    """Return X -> f1(X) * f2(X)."""
    return lambda X: f1(X) * f2(X)


def _compose_ratio(f1: Callable, f2: Callable) -> Callable:
    """Return X -> f1(X) / f2(X)."""
    return lambda X: f1(X) / f2(X)


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation, robust to constant inputs."""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return abs(float((a @ b) / (na * nb)))


class RecursiveSymbolicRegressor(_SklearnCompatMixin):
    """
    Residual-guided recursive basis expansion (experimental).

    Grows a symbolic basis library over several rounds, composing the most
    useful discovered terms with unary functions and products/ratios, guided by
    correlation with the current residual.  Produces a fitted
    :class:`jaxsr.SymbolicRegressor` over the grown library.

    Parameters
    ----------
    n_expansions : int
        Number of expansion rounds (roughly the maximum composition depth).
    max_terms : int
        Maximum number of terms in the sparse model fit each round.
    beam_width : int
        Number of new candidate bases (highest residual correlation) kept per
        round.  Controls the combinatorial cost.
    base_degree : int
        Degree of the initial polynomial seed terms (before any composition).
    unary_ops : tuple of str
        Unary operators to compose with.  Subset of ``sin``, ``cos``, ``exp``,
        ``log``, ``sqrt``, ``square``.
    binary_ops : tuple of str
        Binary operators: any of ``mul`` (products) and ``div`` (ratios).
    strategy : str
        Selection strategy for the per-round sparse fit.
    information_criterion : str
        Information criterion for the per-round sparse fit.
    feature_names : list of str, optional
        Names for the input features.
    random_state : int, optional
        Unused placeholder for API symmetry (the search is deterministic).

    Attributes
    ----------
    model_ : SymbolicRegressor
        The fitted sparse model over the final grown library.
    library_size_ : int
        Number of candidate basis functions in the final library.
    history_ : list of dict
        Per-round diagnostics (library size, number of terms, train R^2).

    Examples
    --------
    >>> from jaxsr.additive import RecursiveSymbolicRegressor
    >>> model = RecursiveSymbolicRegressor(n_expansions=3)
    >>> model.fit(X, y)  # doctest: +SKIP
    >>> print(model.expression_)  # doctest: +SKIP
    """

    def __init__(
        self,
        n_expansions: int = 3,
        max_terms: int = 8,
        beam_width: int = 25,
        base_degree: int = 2,
        unary_ops: tuple[str, ...] = ("sin", "cos", "exp", "log", "sqrt", "square"),
        binary_ops: tuple[str, ...] = ("mul", "div"),
        strategy: str = "greedy_forward",
        information_criterion: str = "bic",
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.n_expansions = n_expansions
        self.max_terms = max_terms
        self.beam_width = beam_width
        self.base_degree = base_degree
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.strategy = strategy
        self.information_criterion = information_criterion
        self.feature_names = feature_names
        self.random_state = random_state

        self.model_: SymbolicRegressor | None = None
        self.library_size_: int | None = None
        self.history_: list[dict] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    def _validate_params(self) -> None:
        """Validate constructor parameters."""
        if self.n_expansions < 0:
            raise ValueError(f"n_expansions must be >= 0, got {self.n_expansions}.")
        if self.max_terms < 1:
            raise ValueError(f"max_terms must be >= 1, got {self.max_terms}.")
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}.")
        bad_unary = set(self.unary_ops) - set(_UNARY_OPS)
        if bad_unary:
            raise ValueError(f"Unknown unary_ops {sorted(bad_unary)}; valid: {sorted(_UNARY_OPS)}.")
        bad_binary = set(self.binary_ops) - {"mul", "div"}
        if bad_binary:
            raise ValueError(f"Unknown binary_ops {sorted(bad_binary)}; valid: ['div', 'mul'].")

    def _seed_library(self, feature_names: list[str]) -> dict[str, _Block]:
        """Build the depth-0 library: constant, features, and polynomial seeds."""
        library: dict[str, _Block] = {
            "1": (lambda X: jnp.ones(X.shape[0]), 0, frozenset()),
        }
        for i, name in enumerate(feature_names):
            library[name] = (lambda X, i=i: X[:, i], 1, frozenset({i}))
        for d in range(2, self.base_degree + 1):
            for i, name in enumerate(feature_names):
                library[f"{name}^{d}"] = (lambda X, i=i, d=d: X[:, i] ** d, d, frozenset({i}))
        return library

    def _generate(self, blocks: dict[str, _Block]) -> dict[str, _Block]:
        """Compose one new layer of candidates from the current blocks."""
        new: dict[str, _Block] = {}
        items = list(blocks.items())

        for name, (func, comp, feats) in items:
            for op in self.unary_ops:
                template, unary_fn, cost = _UNARY_OPS[op]
                new[template.format(name)] = (_compose_unary(func, unary_fn), comp + cost, feats)

        if "mul" in self.binary_ops:
            for i, (n1, (f1, c1, ft1)) in enumerate(items):
                for n2, (f2, c2, ft2) in items[i:]:
                    new[f"({n1})*({n2})"] = (_compose_product(f1, f2), c1 + c2 + 1, ft1 | ft2)

        if "div" in self.binary_ops:
            for n1, (f1, c1, ft1) in items:
                for n2, (f2, c2, ft2) in items:
                    if n1 != n2:
                        new[f"({n1})/({n2})"] = (_compose_ratio(f1, f2), c1 + c2 + 1, ft1 | ft2)

        return new

    def _screen(
        self,
        candidates: dict[str, _Block],
        X: jnp.ndarray,
        residual: np.ndarray,
        existing: dict[str, _Block],
    ) -> dict[str, _Block]:
        """Keep the finite, non-duplicate, most residual-correlated candidates."""
        scored: list[tuple[float, str, _Block, np.ndarray]] = []
        for name, block in candidates.items():
            if name in existing:
                continue
            values = np.asarray(block[0](X), dtype=float)
            if values.shape != residual.shape or not np.all(np.isfinite(values)):
                continue
            if values.std() < 1e-12:
                continue
            scored.append((_abs_corr(values, residual), name, block, values))

        scored.sort(key=lambda t: -t[0])

        kept: dict[str, _Block] = {}
        kept_values: list[np.ndarray] = []
        for corr, name, block, values in scored:
            if corr < 1e-6:
                break
            # Deduplicate against already-kept candidates by near-perfect correlation.
            if any(_abs_corr(values, kv) > 0.9999 for kv in kept_values):
                continue
            kept[name] = block
            kept_values.append(values)
            if len(kept) >= self.beam_width:
                break
        return kept

    def _fit_library(
        self, library: dict[str, _Block], X: jnp.ndarray, y: jnp.ndarray, n_features: int
    ) -> SymbolicRegressor:
        """Fit a sparse SymbolicRegressor over the current library."""
        basis = BasisLibrary(n_features, self.feature_names)
        for name, (func, complexity, feats) in library.items():
            basis.add_custom(
                name, func, complexity=complexity, feature_indices=tuple(sorted(feats))
            )
        model = SymbolicRegressor(
            basis_library=basis,
            max_terms=self.max_terms,
            strategy=self.strategy,
            information_criterion=self.information_criterion,
        )
        return model.fit(X, y)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> RecursiveSymbolicRegressor:
        """
        Fit by residual-guided recursive basis expansion.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RecursiveSymbolicRegressor
            The fitted estimator.

        Raises
        ------
        ValueError
            If parameters are invalid or ``X``/``y`` are mismatched.
        """
        self._validate_params()
        X = jnp.atleast_2d(jnp.asarray(X))
        y = jnp.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}."
            )

        n_features = X.shape[1]
        feature_names = self.feature_names or [f"x{i}" for i in range(n_features)]

        library = self._seed_library(feature_names)
        model = self._fit_library(library, X, y, n_features)
        self.history_ = [
            {
                "round": 0,
                "library_size": len(library),
                "n_terms": len(model.selected_features_),
                "train_r2": float(model.score(X, y)),
            }
        ]

        for rnd in range(1, self.n_expansions + 1):
            residual = np.asarray(y - model.predict(X), dtype=float)
            if np.linalg.norm(residual) < 1e-9 * (np.linalg.norm(np.asarray(y)) + 1e-12):
                break

            # Building blocks for composition: raw features/seeds plus the
            # terms the model actually selected (bounded and useful).
            blocks = self._seed_library(feature_names)
            for name in model.selected_features_:
                if name in library:
                    blocks[name] = library[name]

            candidates = self._generate(blocks)
            survivors = self._screen(candidates, X, residual, library)
            if not survivors:
                break

            library.update(survivors)
            model = self._fit_library(library, X, y, n_features)
            self.history_.append(
                {
                    "round": rnd,
                    "library_size": len(library),
                    "n_terms": len(model.selected_features_),
                    "train_r2": float(model.score(X, y)),
                }
            )

        self.model_ = model
        self.library_size_ = len(library)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        """Raise if the model has not been fitted yet."""
        if not self._is_fitted or self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Predict with the fitted model."""
        self._check_is_fitted()
        return self.model_.predict(X)

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Return the R^2 score of the fitted model on ``(X, y)``."""
        self._check_is_fitted()
        return self.model_.score(X, y)

    @property
    def expression_(self) -> str:
        """Human-readable expression of the fitted model."""
        self._check_is_fitted()
        return self.model_.expression_

    @property
    def selected_features_(self) -> list[str]:
        """Names of the selected (possibly composed) basis functions."""
        self._check_is_fitted()
        return self.model_.selected_features_
