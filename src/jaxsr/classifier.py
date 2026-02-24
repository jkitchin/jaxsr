"""
Symbolic Classification for JAXSR.

Provides a ``SymbolicClassifier`` that discovers interpretable logistic
models from data.  Binary classification uses the sigmoid link; multiclass
uses one-vs-rest (OVR) with per-class expressions.

The classifier reuses ``BasisLibrary`` for design-matrix construction and
IRLS for coefficient fitting, keeping the same sparse-selection workflow
as ``SymbolicRegressor``.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from ._compat import _SklearnCompatMixin
from .basis import BasisLibrary
from .constraints import Constraints
from .metrics import (
    compute_accuracy,
    compute_all_classification_metrics,
    compute_log_loss,
)
from .selection import (
    ClassificationPath,
    ClassificationResult,
    _sigmoid,
    compute_pareto_front_classification,
    select_features_classification,
)
from .uncertainty import (
    classification_coefficient_intervals,
    conformal_classification_split,
)


class SymbolicClassifier(_SklearnCompatMixin):
    """
    JAX-accelerated symbolic classification using sparse selection.

    Discovers interpretable logistic models by selecting sparse subsets
    of basis functions and fitting via IRLS.  Binary problems use a
    single sigmoid link; multiclass problems are handled via one-vs-rest
    (OVR), giving each class its own interpretable expression.

    Parameters
    ----------
    basis_library : BasisLibrary, optional
        Library of candidate basis functions.
    max_terms : int
        Maximum number of terms in each expression.
    strategy : str
        Selection strategy: ``"greedy_forward"``, ``"greedy_backward"``,
        ``"exhaustive"``, or ``"lasso_path"``.
    information_criterion : str
        IC for model selection: ``"aic"``, ``"aicc"``, ``"bic"``.
    regularization : float, optional
        L2 ridge penalty for IRLS.
    constraints : Constraints, optional
        Physical constraints (applied to linear predictor).
    random_state : int, optional
        Random seed for reproducibility.
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        IRLS convergence tolerance.

    Attributes
    ----------
    expression_ : str
        Human-readable linear-predictor expression (binary) or dict
        of expressions (multiclass).
    coefficients_ : jnp.ndarray
        Fitted coefficients.
    selected_features_ : list of str
        Selected basis function names.
    classes_ : jnp.ndarray
        Unique class labels found during ``fit()``.
    metrics_ : dict
        Classification metrics computed on training data.

    Examples
    --------
    >>> from jaxsr import BasisLibrary
    >>> from jaxsr.classifier import SymbolicClassifier
    >>> lib = BasisLibrary(2).add_constant().add_linear().add_polynomials()
    >>> clf = SymbolicClassifier(basis_library=lib, max_terms=4)
    >>> clf.fit(X_train, y_train)
    >>> print(clf.expression_)
    >>> proba = clf.predict_proba(X_test)
    """

    def __init__(
        self,
        basis_library: BasisLibrary | None = None,
        max_terms: int = 5,
        strategy: str = "greedy_forward",
        information_criterion: str = "bic",
        regularization: float | None = None,
        constraints: Constraints | None = None,
        random_state: int | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.basis_library = basis_library
        self.max_terms = max_terms
        self.strategy = strategy
        self.information_criterion = information_criterion
        self.regularization = regularization
        self.constraints = constraints
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

        # Fitted state
        self._result: ClassificationResult | None = None
        self._selection_path: ClassificationPath | None = None
        self._ovr_results: list[ClassificationResult] | None = None
        self._ovr_paths: list[ClassificationPath] | None = None
        self._classes: jnp.ndarray | None = None
        self._is_binary: bool = True
        self._X_train: jnp.ndarray | None = None
        self._y_train: jnp.ndarray | None = None
        self._is_fitted = False

    # =================================================================
    # Fitted properties
    # =================================================================

    @property
    def expression_(self) -> str | dict[int, str]:
        """
        Human-readable expression for the linear predictor.

        For binary classification, returns a string.  For multiclass,
        returns a dict mapping class label to expression string.
        """
        self._check_is_fitted()
        if self._is_binary:
            return self._result.expression()
        return {
            int(c): r.expression() for c, r in zip(self._classes, self._ovr_results, strict=False)
        }

    @property
    def coefficients_(self) -> jnp.ndarray:
        """Fitted coefficients (binary: 1-D; multiclass: list)."""
        self._check_is_fitted()
        if self._is_binary:
            return self._result.coefficients
        return jnp.array([r.coefficients for r in self._ovr_results])

    @property
    def selected_features_(self) -> list[str]:
        """Names of selected basis functions (binary model)."""
        self._check_is_fitted()
        if self._is_binary:
            return self._result.selected_names
        # Union of all OVR features
        seen = []
        for r in self._ovr_results:
            for n in r.selected_names:
                if n not in seen:
                    seen.append(n)
        return seen

    @property
    def classes_(self) -> jnp.ndarray:
        """Unique class labels found during fit."""
        self._check_is_fitted()
        return self._classes

    @property
    def metrics_(self) -> dict[str, float]:
        """Training classification metrics."""
        self._check_is_fitted()
        y_pred = self.predict(self._X_train)
        proba = self.predict_proba(self._X_train)
        if proba.ndim == 2 and proba.shape[1] == 2:
            proba_for_metrics = proba[:, 1]
        else:
            proba_for_metrics = proba
        n_params = (
            len(self._result.coefficients)
            if self._is_binary
            else sum(len(r.coefficients) for r in self._ovr_results)
        )
        return compute_all_classification_metrics(
            self._y_train, y_pred, proba_for_metrics, n_params
        )

    @property
    def pareto_front_(self) -> list[ClassificationResult]:
        """Pareto-optimal models from the selection path."""
        self._check_is_fitted()
        return compute_pareto_front_classification(self._selection_path.results)

    def _check_is_fitted(self):
        """Raise if model has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    # =================================================================
    # Fitting
    # =================================================================

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> SymbolicClassifier:
        """
        Fit the symbolic classifier.

        Automatically detects binary vs multiclass from ``unique(y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels (integers or floats castable to int).

        Returns
        -------
        self : SymbolicClassifier
            Fitted model.

        Raises
        ------
        ValueError
            If *basis_library* is ``None``, shapes are inconsistent,
            or labels contain fewer than 2 classes.
        """
        X = jnp.atleast_2d(jnp.asarray(X, dtype=float))
        y = jnp.asarray(y).ravel()

        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y must have same number of samples. " f"Got X: {X.shape[0]}, y: {len(y)}"
            )
        if self.basis_library is None:
            raise ValueError("basis_library must be specified")
        if X.shape[1] != self.basis_library.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features but basis_library expects "
                f"{self.basis_library.n_features}"
            )

        classes = jnp.unique(y)
        if len(classes) < 2:
            raise ValueError("y must contain at least 2 distinct classes.")

        self._classes = classes
        self._X_train = X
        self._y_train = y

        # Evaluate basis functions
        Phi = self.basis_library.evaluate(X)
        invalid_mask = ~jnp.all(jnp.isfinite(Phi), axis=0)
        if jnp.any(invalid_mask):
            n_invalid = int(jnp.sum(invalid_mask))
            warnings.warn(
                f"Removing {n_invalid} basis functions with non-finite values",
                stacklevel=2,
            )
            Phi = jnp.where(jnp.isfinite(Phi), Phi, 0)

        if len(classes) == 2:
            self._is_binary = True
            self._fit_binary(Phi, y, classes)
        else:
            self._is_binary = False
            self._fit_multiclass(Phi, y, classes)

        self._is_fitted = True
        return self

    def _fit_binary(
        self,
        Phi: jnp.ndarray,
        y: jnp.ndarray,
        classes: jnp.ndarray,
    ):
        """Fit binary logistic model."""
        # Recode y to {0, 1} if needed
        y_bin = jnp.where(y == classes[1], 1.0, 0.0)

        self._selection_path = select_features_classification(
            Phi=Phi,
            y=y_bin,
            basis_names=self.basis_library.names,
            complexities=self.basis_library.complexities,
            strategy=self.strategy,
            max_terms=self.max_terms,
            information_criterion=self.information_criterion,
            regularization=self.regularization,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self._result = self._selection_path.best

    def _fit_multiclass(
        self,
        Phi: jnp.ndarray,
        y: jnp.ndarray,
        classes: jnp.ndarray,
    ):
        """Fit one-vs-rest logistic models."""
        self._ovr_results = []
        self._ovr_paths = []

        for c in classes:
            y_bin = jnp.where(y == c, 1.0, 0.0)
            path = select_features_classification(
                Phi=Phi,
                y=y_bin,
                basis_names=self.basis_library.names,
                complexities=self.basis_library.complexities,
                strategy=self.strategy,
                max_terms=self.max_terms,
                information_criterion=self.information_criterion,
                regularization=self.regularization,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            self._ovr_results.append(path.best)
            self._ovr_paths.append(path)

        # Use first class's path and result as the "primary" for
        # properties that expect a single result
        self._selection_path = self._ovr_paths[0]
        self._result = self._ovr_results[0]

    # =================================================================
    # Prediction
    # =================================================================

    def decision_function(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute raw logits (linear predictor values).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        logits : jnp.ndarray
            Shape ``(n,)`` for binary, ``(n, K)`` for multiclass.
        """
        self._check_is_fitted()
        X = jnp.atleast_2d(jnp.asarray(X, dtype=float))

        if self._is_binary:
            Phi = self.basis_library.evaluate_subset(X, self._result.selected_indices)
            return Phi @ self._result.coefficients
        else:
            logits = []
            for r in self._ovr_results:
                Phi = self.basis_library.evaluate_subset(X, r.selected_indices)
                logits.append(Phi @ r.coefficients)
            return jnp.column_stack(logits)

    def predict_proba(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : jnp.ndarray
            Shape ``(n, K)`` probability matrix.  Columns correspond
            to ``self.classes_``.
        """
        self._check_is_fitted()
        logits = self.decision_function(X)

        if self._is_binary:
            p1 = _sigmoid(logits)
            return jnp.column_stack([1 - p1, p1])
        else:
            # OVR: sigmoid each column, then normalise
            proba = _sigmoid(logits)
            row_sums = jnp.sum(proba, axis=1, keepdims=True)
            row_sums = jnp.maximum(row_sums, 1e-10)
            return proba / row_sums

    def predict_log_proba(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict log class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        log_proba : jnp.ndarray
            Shape ``(n, K)`` log-probability matrix.
        """
        proba = self.predict_proba(X)
        return jnp.log(jnp.clip(proba, 1e-15, 1.0))

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : jnp.ndarray
            Predicted class labels of shape ``(n,)``.
        """
        proba = self.predict_proba(X)
        idx = jnp.argmax(proba, axis=1)
        return self._classes[idx]

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Compute classification accuracy (sklearn convention).

        Parameters
        ----------
        X : array-like
            Test data.
        y : array-like
            True labels.

        Returns
        -------
        accuracy : float
            Fraction of correct predictions.
        """
        self._check_is_fitted()
        y = jnp.asarray(y).ravel()
        y_pred = self.predict(X)
        return compute_accuracy(y, y_pred)

    # =================================================================
    # Expression output
    # =================================================================

    def to_sympy(self):
        """
        Convert expression to SymPy.

        For binary: returns ``1 / (1 + exp(-linear_predictor))``.
        For multiclass: returns a dict mapping class → SymPy expression.

        Returns
        -------
        expr : sympy.Expr or dict
            SymPy representation of the probability model.
        """
        self._check_is_fitted()
        import sympy

        symbols = {name: sympy.Symbol(name) for name in self.basis_library.feature_names}

        if self._is_binary:
            linear = self._build_sympy_linear(self._result, symbols)
            return 1 / (1 + sympy.exp(-linear))
        else:
            result = {}
            for c, r in zip(self._classes, self._ovr_results, strict=False):
                linear = self._build_sympy_linear(r, symbols)
                result[int(c)] = 1 / (1 + sympy.exp(-linear))
            return result

    def _build_sympy_linear(self, result, symbols):
        """Build sympy expression for a linear predictor."""
        import sympy

        # Reuse SymbolicRegressor's parser via a temporary instance
        from .regressor import SymbolicRegressor

        parser = SymbolicRegressor.__new__(SymbolicRegressor)
        parser.basis_library = self.basis_library

        expr = sympy.Integer(0)
        for coef, name in zip(result.coefficients, result.selected_names, strict=False):
            coef_val = float(coef)
            if abs(coef_val) < 1e-10:
                continue
            term = parser._parse_basis_to_sympy(name, symbols)
            expr = expr + coef_val * term
        return sympy.simplify(expr)

    def to_latex(self) -> str:
        """
        Convert expression to LaTeX string.

        Returns
        -------
        latex : str
            LaTeX representation.
        """
        self._check_is_fitted()
        import sympy

        expr = self.to_sympy()
        if isinstance(expr, dict):
            parts = []
            for cls, ex in expr.items():
                parts.append(f"P(y={cls}) = {sympy.latex(ex)}")
            return " \\\\ ".join(parts)
        return sympy.latex(expr)

    def to_callable(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Convert to a pure NumPy callable that returns probabilities.

        Returns
        -------
        func : callable
            ``func(X) -> proba`` where ``proba`` has shape ``(n, K)``.
        """
        self._check_is_fitted()

        from .regressor import SymbolicRegressor

        if self._is_binary:
            # Build a temporary SymbolicRegressor to get its to_callable
            reg = SymbolicRegressor.__new__(SymbolicRegressor)
            reg.basis_library = self.basis_library
            reg._result = type(self._result)(
                coefficients=self._result.coefficients,
                selected_indices=self._result.selected_indices,
                selected_names=self._result.selected_names,
                neg_log_likelihood=self._result.neg_log_likelihood,
                complexity=self._result.complexity,
                aic=self._result.aic,
                bic=self._result.bic,
                aicc=self._result.aicc,
                n_samples=self._result.n_samples,
            )
            reg._is_fitted = True

            # Use basis library to evaluate subset + apply sigmoid
            coefficients = np.array(self._result.coefficients)
            indices = self._result.selected_indices
            lib = self.basis_library

            def predict_binary(X: np.ndarray) -> np.ndarray:
                X = np.atleast_2d(X)
                Phi = np.array(lib.evaluate_subset(jnp.asarray(X), indices))
                eta = Phi @ coefficients
                p1 = 1.0 / (1.0 + np.exp(-eta))
                return np.column_stack([1 - p1, p1])

            return predict_binary
        else:
            all_coeffs = [np.array(r.coefficients) for r in self._ovr_results]
            all_indices = [r.selected_indices for r in self._ovr_results]
            lib = self.basis_library

            def predict_multiclass(X: np.ndarray) -> np.ndarray:
                X = np.atleast_2d(X)
                logits = []
                for coefficients, indices in zip(all_coeffs, all_indices, strict=False):
                    Phi = np.array(lib.evaluate_subset(jnp.asarray(X), indices))
                    logits.append(Phi @ coefficients)
                logits_arr = np.column_stack(logits)
                proba = 1.0 / (1.0 + np.exp(-logits_arr))
                row_sums = proba.sum(axis=1, keepdims=True)
                row_sums = np.maximum(row_sums, 1e-10)
                return proba / row_sums

            return predict_multiclass

    # =================================================================
    # Uncertainty Quantification
    # =================================================================

    def coefficient_intervals(
        self, alpha: float = 0.05
    ) -> dict[str, tuple[float, float, float, float]]:
        """
        Wald confidence intervals for coefficients (binary only).

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        intervals : dict
            ``{name: (estimate, lower, upper, se)}``.
        """
        self._check_is_fitted()
        if not self._is_binary:
            raise ValueError(
                "coefficient_intervals is only supported for binary "
                "classification. Use the OVR results directly for multiclass."
            )
        Phi = self.basis_library.evaluate_subset(self._X_train, self._result.selected_indices)
        y_bin = jnp.where(self._y_train == self._classes[1], 1.0, 0.0)
        return classification_coefficient_intervals(
            Phi,
            y_bin,
            self._result.coefficients,
            self._result.selected_names,
            alpha,
        )

    def predict_conformal(
        self,
        X: jnp.ndarray,
        alpha: float = 0.05,
        X_cal: jnp.ndarray | None = None,
        y_cal: jnp.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Conformal prediction sets for classification.

        Parameters
        ----------
        X : jnp.ndarray
            New input data.
        alpha : float
            Significance level.
        X_cal : jnp.ndarray, optional
            Calibration features.  If ``None``, uses training data.
        y_cal : jnp.ndarray, optional
            Calibration labels.  If ``None``, uses training data.

        Returns
        -------
        result : dict
            Keys: ``"prediction_sets"``, ``"quantile"``, ``"y_pred"``.
        """
        self._check_is_fitted()
        if X_cal is None or y_cal is None:
            X_cal = self._X_train
            y_cal = self._y_train
        return conformal_classification_split(self, X_cal, y_cal, X, alpha)

    # =================================================================
    # Persistence
    # =================================================================

    def save(self, filepath: str) -> None:
        """
        Save model to JSON file.

        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        self._check_is_fitted()

        data: dict[str, Any] = {
            "model_type": "SymbolicClassifier",
            "config": {
                "max_terms": self.max_terms,
                "strategy": self.strategy,
                "information_criterion": self.information_criterion,
                "regularization": self.regularization,
                "random_state": self.random_state,
                "max_iter": self.max_iter,
                "tol": self.tol,
            },
            "basis_library": self.basis_library.to_dict(),
            "classes": np.array(self._classes).tolist(),
            "is_binary": self._is_binary,
            "constraints": (self.constraints.to_dict() if self.constraints else None),
        }

        if self._is_binary:
            data["result"] = self._result.to_dict()
        else:
            data["ovr_results"] = [r.to_dict() for r in self._ovr_results]

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> SymbolicClassifier:
        """
        Load model from JSON file.

        Parameters
        ----------
        filepath : str
            Path to the model file.

        Returns
        -------
        model : SymbolicClassifier
            Loaded model.
        """
        with open(filepath) as f:
            data = json.load(f)

        basis_library = BasisLibrary.from_dict(data["basis_library"])

        constraints = None
        if data.get("constraints"):
            constraints = Constraints.from_dict(data["constraints"])

        model = cls(
            basis_library=basis_library,
            constraints=constraints,
            **data["config"],
        )

        model._classes = jnp.array(data["classes"])
        model._is_binary = data["is_binary"]

        def _dict_to_result(d):
            return ClassificationResult(
                coefficients=jnp.array(d["coefficients"]),
                selected_indices=jnp.array(d["selected_indices"]),
                selected_names=d["selected_names"],
                neg_log_likelihood=d["neg_log_likelihood"],
                complexity=d["complexity"],
                aic=d["aic"],
                bic=d["bic"],
                aicc=d["aicc"],
                n_samples=d["n_samples"],
                n_iter=d.get("n_iter", 0),
                converged=d.get("converged", True),
            )

        if model._is_binary:
            model._result = _dict_to_result(data["result"])
            model._selection_path = ClassificationPath(
                results=[model._result], strategy="loaded", best_index=0
            )
        else:
            model._ovr_results = [_dict_to_result(d) for d in data["ovr_results"]]
            model._result = model._ovr_results[0]
            model._selection_path = ClassificationPath(
                results=[model._result], strategy="loaded", best_index=0
            )
            model._ovr_paths = [
                ClassificationPath(results=[r], strategy="loaded", best_index=0)
                for r in model._ovr_results
            ]

        model._is_fitted = True
        return model

    # =================================================================
    # Display
    # =================================================================

    def summary(self) -> str:
        """
        Return a human-readable summary of the fitted model.

        Returns
        -------
        summary : str
            Model summary string.
        """
        self._check_is_fitted()

        lines = [
            "=" * 60,
            "JAXSR Symbolic Classification Model",
            "=" * 60,
            "",
        ]

        if self._is_binary:
            lines.append(f"Type: Binary (classes: {np.array(self._classes).tolist()})")
            lines.append(f"Linear predictor: {self._result.expression()}")
            lines.append("")
            lines.append(f"Selected terms ({len(self._result.selected_names)}):")
            for name, coef in zip(
                self._result.selected_names,
                self._result.coefficients,
                strict=False,
            ):
                lines.append(f"  {name}: {float(coef):.6g}")
            lines.append("")
            lines.append("Metrics:")
            lines.append(f"  NLL: {self._result.neg_log_likelihood:.4f}")
            lines.append(f"  BIC: {self._result.bic:.2f}")
            lines.append(f"  AIC: {self._result.aic:.2f}")

            y_pred = self.predict(self._X_train)
            acc = compute_accuracy(self._y_train, y_pred)
            lines.append(f"  Train accuracy: {acc:.4f}")

            proba = self.predict_proba(self._X_train)[:, 1]
            y_bin = jnp.where(self._y_train == self._classes[1], 1.0, 0.0)
            ll = compute_log_loss(y_bin, proba)
            lines.append(f"  Log-loss: {ll:.4f}")
        else:
            lines.append(
                f"Type: Multiclass OVR ({len(self._classes)} classes: "
                f"{np.array(self._classes).tolist()})"
            )
            for c, r in zip(self._classes, self._ovr_results, strict=False):
                lines.append(f"\n  Class {int(c)}: {r.expression()}")
                lines.append(f"    Terms: {len(r.selected_names)}, NLL: {r.neg_log_likelihood:.4f}")

            y_pred = self.predict(self._X_train)
            acc = compute_accuracy(self._y_train, y_pred)
            lines.append(f"\n  Overall train accuracy: {acc:.4f}")

        lines.extend(
            [
                "",
                f"Training samples: {self._result.n_samples}",
                f"Strategy: {self.strategy}",
                "=" * 60,
            ]
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"SymbolicClassifier(max_terms={self.max_terms}, "
            f"strategy='{self.strategy}', {status})"
        )


# =============================================================================
# Convenience Function
# =============================================================================


def fit_symbolic_classification(
    X: jnp.ndarray,
    y: jnp.ndarray,
    feature_names: list[str] | None = None,
    max_terms: int = 5,
    max_poly_degree: int = 3,
    include_transcendental: bool = False,
    strategy: str = "greedy_forward",
    information_criterion: str = "bic",
) -> SymbolicClassifier:
    """
    Convenience function for quick symbolic classification.

    Builds a default ``BasisLibrary``, constructs a
    ``SymbolicClassifier``, and fits it.

    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like
        Class labels.
    feature_names : list of str, optional
        Names for features.
    max_terms : int
        Maximum terms in expression.
    max_poly_degree : int
        Maximum polynomial degree.
    include_transcendental : bool
        Include log, exp, sqrt, inv.
    strategy : str
        Selection strategy.
    information_criterion : str
        Information criterion.

    Returns
    -------
    model : SymbolicClassifier
        Fitted classifier.
    """
    X = jnp.atleast_2d(jnp.asarray(X, dtype=float))
    n_features = X.shape[1]

    library = BasisLibrary(n_features, feature_names)
    library.add_constant()
    library.add_linear()
    library.add_polynomials(max_poly_degree)
    library.add_interactions()

    if include_transcendental:
        library.add_transcendental()

    model = SymbolicClassifier(
        basis_library=library,
        max_terms=max_terms,
        strategy=strategy,
        information_criterion=information_criterion,
    )

    return model.fit(X, y)
