"""
Main SymbolicRegressor Class for JAXSR.

Provides a scikit-learn compatible interface for symbolic regression.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from .basis import BasisLibrary
from .constraints import Constraints, fit_constrained_ols
from .metrics import compute_information_criterion
from .selection import (
    SelectionPath,
    SelectionResult,
    compute_pareto_front,
    select_features,
)
from .uncertainty import (
    BayesianModelAverage,
    compute_coeff_covariance,
    compute_unbiased_variance,
    conformal_predict_jackknife_plus,
    conformal_predict_split,
    ensemble_predict,
)
from .uncertainty import (
    coefficient_intervals as _coefficient_intervals,
)
from .uncertainty import (
    prediction_interval as _prediction_interval,
)

# =============================================================================
# Main Regressor Class
# =============================================================================


class SymbolicRegressor:
    """
    JAX-accelerated symbolic regression using sparse selection.

    Discovers interpretable algebraic expressions from data by selecting
    sparse subsets of basis functions using information criteria.

    Parameters
    ----------
    basis_library : BasisLibrary, optional
        Library of candidate basis functions. If None, must be specified
        when calling fit() via the `library_config` parameter.
    max_terms : int
        Maximum number of terms in the expression.
    strategy : str
        Selection strategy: "greedy_forward", "greedy_backward",
        "exhaustive", or "lasso_path".
    information_criterion : str
        Criterion for model selection: "aic", "aicc", "bic".
    cv_folds : int
        Number of cross-validation folds (if using CV for selection).
    regularization : float, optional
        L2 regularization parameter (ridge penalty).
    constraints : Constraints, optional
        Physical constraints to enforce.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    expression_ : str
        Human-readable expression after fitting.
    coefficients_ : jnp.ndarray
        Fitted coefficients.
    selected_features_ : list of str
        Names of selected basis functions.
    complexity_ : int
        Total complexity score of the expression.
    metrics_ : dict
        Dictionary of evaluation metrics.
    pareto_front_ : list of SelectionResult
        Pareto-optimal models.

    Examples
    --------
    >>> from jaxsr import BasisLibrary, SymbolicRegressor
    >>> library = (BasisLibrary(n_features=2)
    ...     .add_constant()
    ...     .add_linear()
    ...     .add_polynomials(max_degree=3)
    ... )
    >>> model = SymbolicRegressor(basis_library=library, max_terms=5)
    >>> model.fit(X, y)
    >>> print(model.expression_)
    >>> y_pred = model.predict(X_new)
    """

    def __init__(
        self,
        basis_library: BasisLibrary | None = None,
        max_terms: int = 5,
        strategy: str = "greedy_forward",
        information_criterion: str = "bic",
        cv_folds: int = 5,
        regularization: float | None = None,
        constraints: Constraints | None = None,
        random_state: int | None = None,
        param_optimizer: str = "scipy",
        param_optimization_budget: int = 50,
    ):
        self.basis_library = basis_library
        self.max_terms = max_terms
        self.strategy = strategy
        self.information_criterion = information_criterion
        self.cv_folds = cv_folds
        self.regularization = regularization
        self.constraints = constraints
        self.random_state = random_state
        self.param_optimizer = param_optimizer
        self.param_optimization_budget = param_optimization_budget

        # Fitted attributes
        self._result: SelectionResult | None = None
        self._selection_path: SelectionPath | None = None
        self._X_train: jnp.ndarray | None = None
        self._y_train: jnp.ndarray | None = None
        self._is_fitted = False

    @property
    def expression_(self) -> str:
        """Human-readable expression."""
        self._check_is_fitted()
        return self._result.expression()

    @property
    def coefficients_(self) -> jnp.ndarray:
        """Fitted coefficients."""
        self._check_is_fitted()
        return self._result.coefficients

    @property
    def selected_features_(self) -> list[str]:
        """Names of selected basis functions."""
        self._check_is_fitted()
        return self._result.selected_names

    @property
    def selected_indices_(self) -> jnp.ndarray:
        """Indices of selected basis functions."""
        self._check_is_fitted()
        return self._result.selected_indices

    @property
    def complexity_(self) -> int:
        """Total complexity score."""
        self._check_is_fitted()
        return self._result.complexity

    @property
    def metrics_(self) -> dict[str, float]:
        """Evaluation metrics."""
        self._check_is_fitted()
        return {
            "mse": self._result.mse,
            "aic": self._result.aic,
            "bic": self._result.bic,
            "aicc": self._result.aicc,
            "r2": self._compute_r2(),
        }

    @property
    def pareto_front_(self) -> list[SelectionResult]:
        """Pareto-optimal models."""
        self._check_is_fitted()
        return compute_pareto_front(self._selection_path.results)

    def _check_is_fitted(self):
        """Check if model is fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def _compute_r2(self) -> float:
        """Compute R² score on training data."""
        y_pred = self.predict(self._X_train)
        ss_res = jnp.sum((self._y_train - y_pred) ** 2)
        ss_tot = jnp.sum((self._y_train - jnp.mean(self._y_train)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-10))

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        sample_weight: jnp.ndarray | None = None,
    ) -> SymbolicRegressor:
        """
        Fit symbolic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights (not yet implemented).

        Returns
        -------
        self : SymbolicRegressor
            Fitted model.
        """
        # Convert to JAX arrays
        X = jnp.atleast_2d(jnp.asarray(X))
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

        self._X_train = X
        self._y_train = y

        # Evaluate basis functions
        Phi = self.basis_library.evaluate(X)

        # Handle invalid values in design matrix
        invalid_mask = ~jnp.all(jnp.isfinite(Phi), axis=0)
        if jnp.any(invalid_mask):
            n_invalid = int(jnp.sum(invalid_mask))
            warnings.warn(
                f"Removing {n_invalid} basis functions with non-finite values", stacklevel=2
            )
            # Replace invalid columns with zeros (they won't be selected)
            Phi = jnp.where(jnp.isfinite(Phi), Phi, 0)

        # Run selection
        extra_kw: dict[str, Any] = {}
        if self.basis_library.has_parametric:
            extra_kw.update(
                X=X,
                basis_library=self.basis_library,
                param_optimizer=self.param_optimizer,
                param_optimization_budget=self.param_optimization_budget,
            )

        self._selection_path = select_features(
            Phi=Phi,
            y=y,
            basis_names=self.basis_library.names,
            complexities=self.basis_library.complexities,
            strategy=self.strategy,
            max_terms=self.max_terms,
            information_criterion=self.information_criterion,
            regularization=self.regularization,
            **extra_kw,
        )

        self._result = self._selection_path.best

        # Apply constraints if specified
        if self.constraints is not None:
            self._apply_constraints(Phi, y, X)

        # Resolve parametric parameters so predict() uses optimised values
        if self.basis_library.has_parametric and self._result.parametric_params:
            self._resolve_parametric_params()

        self._is_fitted = True
        return self

    def _apply_constraints(
        self,
        Phi: jnp.ndarray,
        y: jnp.ndarray,
        X: jnp.ndarray,
    ):
        """Refit with constraints."""
        # Get selected subset
        indices = self._result.selected_indices
        Phi_subset = Phi[:, indices]
        basis_names_subset = [self.basis_library.names[int(i)] for i in indices]

        # Fit with constraints
        coeffs, mse = fit_constrained_ols(
            Phi=Phi_subset,
            y=y,
            constraints=self.constraints,
            basis_names=basis_names_subset,
            feature_names=self.basis_library.feature_names,
            X=X,
            basis_library=self.basis_library,
            selected_indices=indices,
        )

        # Update result with recalculated information criteria
        n = len(y)
        k = len(coeffs)
        self._result = SelectionResult(
            coefficients=coeffs,
            selected_indices=indices,
            selected_names=basis_names_subset,
            mse=mse,
            complexity=self._result.complexity,
            aic=compute_information_criterion(n, k, mse, "aic"),
            bic=compute_information_criterion(n, k, mse, "bic"),
            aicc=compute_information_criterion(n, k, mse, "aicc"),
            n_samples=n,
            parametric_params=self._result.parametric_params,
        )

    def _resolve_parametric_params(self):
        """Update library basis functions with optimised parametric values."""
        import re

        for p_info in self.basis_library._parametric_info:
            if (
                self._result.parametric_params
                and p_info.basis_index in self._result.parametric_params
            ):
                params = self._result.parametric_params[p_info.basis_index]
                p_info.resolved_params = params

                bf = self.basis_library.basis_functions[p_info.basis_index]

                # Pin the evaluation closure to the optimised values
                def _make_func(f, p):
                    return lambda X: f(X, **p)

                bf.func = _make_func(p_info.func, params)

                # Update the human-readable name
                resolved_name = p_info.name
                for pname, val in params.items():
                    resolved_name = re.sub(
                        r"\b" + re.escape(pname) + r"\b",
                        f"{val:.4g}",
                        resolved_name,
                    )
                bf.name = resolved_name

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict using fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : jnp.ndarray of shape (n_samples,)
            Predicted values.
        """
        self._check_is_fitted()

        X = jnp.atleast_2d(jnp.asarray(X))
        Phi = self.basis_library.evaluate_subset(X, self._result.selected_indices)
        return Phi @ self._result.coefficients

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Compute R² score.

        Parameters
        ----------
        X : array-like
            Test samples.
        y : array-like
            True values.

        Returns
        -------
        score : float
            R² score.
        """
        self._check_is_fitted()
        y = jnp.asarray(y).ravel()
        y_pred = self.predict(X)
        ss_res = jnp.sum((y - y_pred) ** 2)
        ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-10))

    # =================================================================
    # Uncertainty Quantification
    # =================================================================

    def _get_Phi_train(self) -> jnp.ndarray:
        """Get the training design matrix for the selected features."""
        self._check_is_fitted()
        return self.basis_library.evaluate_subset(self._X_train, self._result.selected_indices)

    def _warn_if_constrained_or_regularized(self):
        """Emit a warning if constraints or regularization are active."""
        if self.constraints is not None:
            warnings.warn(
                "Constraints are active. Classical OLS intervals may not be "
                "valid. Consider using bootstrap methods instead.",
                stacklevel=3,
            )
        if self.regularization is not None and self.regularization > 0:
            warnings.warn(
                "Regularization is active. Classical OLS intervals may not be "
                "valid. Consider using bootstrap methods instead.",
                stacklevel=3,
            )

    @property
    def sigma_(self) -> float:
        """
        Estimated noise standard deviation: sqrt(SSR / (n - p)).

        Returns
        -------
        sigma : float
            Noise standard deviation estimate.
        """
        self._check_is_fitted()
        self._warn_if_constrained_or_regularized()
        Phi = self._get_Phi_train()
        sigma_sq = compute_unbiased_variance(Phi, self._y_train, self._result.coefficients)
        return float(jnp.sqrt(sigma_sq))

    @property
    def covariance_matrix_(self) -> jnp.ndarray:
        """
        Coefficient covariance matrix: s^2 * (Phi^T Phi)^{-1}.

        Returns
        -------
        cov : jnp.ndarray
            Covariance matrix of shape (p, p).
        """
        self._check_is_fitted()
        self._warn_if_constrained_or_regularized()
        Phi = self._get_Phi_train()
        sigma_sq = compute_unbiased_variance(Phi, self._y_train, self._result.coefficients)
        return compute_coeff_covariance(Phi, sigma_sq)

    def coefficient_intervals(self, alpha: float = 0.05) -> dict[str, tuple]:
        """
        Confidence intervals for all coefficients.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 for 95% CIs).

        Returns
        -------
        intervals : dict
            {name: (estimate, lower, upper, se)} for each coefficient.
        """
        self._check_is_fitted()
        self._warn_if_constrained_or_regularized()
        Phi = self._get_Phi_train()
        return _coefficient_intervals(
            Phi,
            self._y_train,
            self._result.coefficients,
            self._result.selected_names,
            alpha,
        )

    def predict_interval(
        self, X: jnp.ndarray, alpha: float = 0.05
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Prediction intervals for new observations.

        Parameters
        ----------
        X : jnp.ndarray
            New input data of shape (n_samples, n_features).
        alpha : float
            Significance level (default 0.05 for 95% intervals).

        Returns
        -------
        y_pred : jnp.ndarray
            Predicted values.
        lower : jnp.ndarray
            Lower prediction interval bound.
        upper : jnp.ndarray
            Upper prediction interval bound.
        """
        self._check_is_fitted()
        self._warn_if_constrained_or_regularized()
        X = jnp.atleast_2d(jnp.asarray(X))
        Phi_train = self._get_Phi_train()
        Phi_new = self.basis_library.evaluate_subset(X, self._result.selected_indices)
        result = _prediction_interval(
            Phi_train,
            self._y_train,
            self._result.coefficients,
            Phi_new,
            alpha,
        )
        return result["y_pred"], result["pred_lower"], result["pred_upper"]

    def confidence_band(
        self, X: jnp.ndarray, alpha: float = 0.05
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Confidence band on mean response E[y|x].

        Parameters
        ----------
        X : jnp.ndarray
            New input data of shape (n_samples, n_features).
        alpha : float
            Significance level (default 0.05 for 95% band).

        Returns
        -------
        y_pred : jnp.ndarray
            Predicted mean values.
        lower : jnp.ndarray
            Lower confidence band bound.
        upper : jnp.ndarray
            Upper confidence band bound.
        """
        self._check_is_fitted()
        self._warn_if_constrained_or_regularized()
        X = jnp.atleast_2d(jnp.asarray(X))
        Phi_train = self._get_Phi_train()
        Phi_new = self.basis_library.evaluate_subset(X, self._result.selected_indices)
        result = _prediction_interval(
            Phi_train,
            self._y_train,
            self._result.coefficients,
            Phi_new,
            alpha,
        )
        return result["y_pred"], result["conf_lower"], result["conf_upper"]

    def predict_ensemble(self, X: jnp.ndarray) -> dict[str, Any]:
        """
        Predictions from Pareto-front models.

        Parameters
        ----------
        X : jnp.ndarray
            New input data.

        Returns
        -------
        result : dict
            Keys: y_mean, y_std, y_min, y_max, y_all, models.
        """
        return ensemble_predict(self, X)

    def predict_bma(
        self,
        X: jnp.ndarray,
        criterion: str = "bic",
        alpha: float = 0.05,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Bayesian Model Averaging prediction with intervals.

        Parameters
        ----------
        X : jnp.ndarray
            New input data.
        criterion : str
            IC for weighting ("bic" or "aic").
        alpha : float
            Significance level.

        Returns
        -------
        y_pred : jnp.ndarray
            BMA mean prediction.
        lower : jnp.ndarray
            Lower bound.
        upper : jnp.ndarray
            Upper bound.
        """
        bma = BayesianModelAverage(self, criterion=criterion)
        return bma.predict_interval(X, alpha)

    def predict_conformal(
        self,
        X: jnp.ndarray,
        alpha: float = 0.05,
        method: str = "jackknife+",
        X_cal: jnp.ndarray | None = None,
        y_cal: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Conformal prediction intervals.

        Parameters
        ----------
        X : jnp.ndarray
            New input data.
        alpha : float
            Significance level.
        method : str
            "jackknife+" or "split".
        X_cal : jnp.ndarray, optional
            Calibration features (required for "split" method).
        y_cal : jnp.ndarray, optional
            Calibration targets (required for "split" method).

        Returns
        -------
        y_pred : jnp.ndarray
            Point predictions.
        lower : jnp.ndarray
            Lower interval bound.
        upper : jnp.ndarray
            Upper interval bound.
        """
        self._check_is_fitted()
        X = jnp.atleast_2d(jnp.asarray(X))

        if method == "jackknife+":
            result = conformal_predict_jackknife_plus(self, X, alpha)
        elif method == "split":
            if X_cal is None or y_cal is None:
                raise ValueError("X_cal and y_cal are required for split conformal prediction.")
            result = conformal_predict_split(self, X_cal, y_cal, X, alpha)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'jackknife+' or 'split'.")

        return result["y_pred"], result["lower"], result["upper"]

    def update(
        self,
        X_new: jnp.ndarray,
        y_new: jnp.ndarray,
        refit: bool = True,
    ) -> SymbolicRegressor:
        """
        Update model with new data points.

        Parameters
        ----------
        X_new : array-like
            New input samples.
        y_new : array-like
            New target values.
        refit : bool
            If True, rerun full selection. If False, only refit coefficients.

        Returns
        -------
        self : SymbolicRegressor
            Updated model.
        """
        self._check_is_fitted()

        X_new = jnp.atleast_2d(jnp.asarray(X_new))
        y_new = jnp.asarray(y_new).ravel()

        # Combine with existing data
        X_combined = jnp.vstack([self._X_train, X_new])
        y_combined = jnp.concatenate([self._y_train, y_new])

        if refit:
            # Full refit
            return self.fit(X_combined, y_combined)
        else:
            # Only refit coefficients with existing selection
            Phi = self.basis_library.evaluate_subset(X_combined, self._result.selected_indices)
            coeffs, mse = (
                fit_constrained_ols(
                    Phi=Phi,
                    y=y_combined,
                    constraints=self.constraints or Constraints(),
                    basis_names=self._result.selected_names,
                    feature_names=self.basis_library.feature_names,
                    X=X_combined,
                    basis_library=self.basis_library,
                    selected_indices=self._result.selected_indices,
                )
                if self.constraints
                else (
                    jnp.linalg.lstsq(Phi, y_combined, rcond=None)[0],
                    float(
                        jnp.mean(
                            (y_combined - Phi @ jnp.linalg.lstsq(Phi, y_combined, rcond=None)[0])
                            ** 2
                        )
                    ),
                )
            )

            self._result = SelectionResult(
                coefficients=coeffs,
                selected_indices=self._result.selected_indices,
                selected_names=self._result.selected_names,
                mse=mse,
                complexity=self._result.complexity,
                aic=self._result.aic,
                bic=self._result.bic,
                aicc=self._result.aicc,
                n_samples=len(y_combined),
            )

            self._X_train = X_combined
            self._y_train = y_combined

            return self

    def to_sympy(self):
        """
        Convert expression to SymPy for symbolic manipulation.

        Returns
        -------
        expr : sympy.Expr
            SymPy expression.
        """
        self._check_is_fitted()

        import sympy

        # Create symbols for features
        symbols = {name: sympy.Symbol(name) for name in self.basis_library.feature_names}

        # Build expression
        expr = sympy.Integer(0)
        for coef, name in zip(self._result.coefficients, self._result.selected_names, strict=False):
            coef = float(coef)
            if abs(coef) < 1e-10:
                continue

            # Parse basis function name to sympy
            term = self._parse_basis_to_sympy(name, symbols)
            expr = expr + coef * term

        return sympy.simplify(expr)

    def _parse_basis_to_sympy(self, name: str, symbols: dict):
        """Parse a basis function name to SymPy expression."""
        import sympy

        # Handle constant
        if name == "1":
            return sympy.Integer(1)

        # Handle linear terms
        if name in symbols:
            return symbols[name]

        # Handle categorical interaction: I(color=2)*temperature
        if name.startswith("I(") and ")*" in name:
            indicator_part, cont_part = name.split(")*", 1)
            inner = indicator_part[2:]  # "color=2"
            feat_name, cat_val_str = inner.split("=", 1)
            x = symbols.get(feat_name, sympy.Symbol(feat_name))
            cat_val = float(cat_val_str)
            indicator = sympy.Piecewise((1, sympy.Eq(x, cat_val)), (0, True))
            cont_sym = symbols.get(cont_part, sympy.Symbol(cont_part))
            return indicator * cont_sym

        # Handle indicator: I(color=2)
        if name.startswith("I(") and name.endswith(")") and "=" in name:
            inner = name[2:-1]  # "color=2"
            feat_name, cat_val_str = inner.split("=", 1)
            x = symbols.get(feat_name, sympy.Symbol(feat_name))
            cat_val = float(cat_val_str)
            return sympy.Piecewise((1, sympy.Eq(x, cat_val)), (0, True))

        # Handle powers: x^2, x^0.8, x^(1/3), etc.
        if "^" in name and "/" not in name:
            # Split only on first '^'
            idx = name.index("^")
            base = name[:idx]
            power_str = name[idx + 1 :].strip("()")
            if base in symbols:
                try:
                    return symbols[base] ** float(power_str)
                except ValueError:
                    pass

        # Handle transcendental functions (before interactions so that
        # names like "exp(-0.3*x)" are not mis-parsed as products)
        for func_name, sympy_func in [
            ("log", sympy.log),
            ("exp", sympy.exp),
            ("sqrt", sympy.sqrt),
            ("sin", sympy.sin),
            ("cos", sympy.cos),
        ]:
            if name.startswith(f"{func_name}(") and name.endswith(")"):
                inner = name[len(func_name) + 1 : -1]
                if inner in symbols:
                    return sympy_func(symbols[inner])
                # Try to parse a complex inner expression via sympify
                try:
                    inner_expr = sympy.sympify(inner, locals=dict(symbols))
                    return sympy_func(inner_expr)
                except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
                    pass

        # Handle interactions: x*y, x*y*z (only when ALL parts are symbols)
        if "*" in name and "/" not in name and "^" not in name:
            parts = name.split("*")
            if all(p.strip() in symbols for p in parts):
                result = sympy.Integer(1)
                for part in parts:
                    result = result * symbols[part.strip()]
                return result

        # Handle inverse: 1/x
        if name.startswith("1/") and name[2:] in symbols:
            return 1 / symbols[name[2:]]

        # Handle ratios: x/y
        if "/" in name and not name.startswith("1/"):
            parts = name.split("/")
            if len(parts) == 2 and parts[0] in symbols and parts[1] in symbols:
                return symbols[parts[0]] / symbols[parts[1]]

        # Last resort: try sympify on the whole name
        try:
            expr = sympy.sympify(name, locals=dict(symbols))
            if expr != sympy.Symbol(name):
                return expr
        except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
            pass

        # Fallback: return as symbol
        return sympy.Symbol(name)

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
        return sympy.latex(expr)

    def to_callable(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Convert to pure Python/NumPy callable (no JAX dependency).

        Returns
        -------
        func : callable
            Function that takes NumPy array and returns predictions.
        """
        self._check_is_fitted()

        # Extract coefficients and indices
        coefficients = np.array(self._result.coefficients)
        names = self._result.selected_names
        feature_names = self.basis_library.feature_names

        def predict_numpy(X: np.ndarray) -> np.ndarray:
            """Pure NumPy prediction function."""
            X = np.atleast_2d(X)
            n_samples = X.shape[0]
            result = np.zeros(n_samples)

            for coef, name in zip(coefficients, names, strict=False):
                if name == "1":
                    term = np.ones(n_samples)
                elif name in feature_names:
                    idx = feature_names.index(name)
                    term = X[:, idx]
                elif "^" in name and "*" not in name:
                    base, power = name.split("^")
                    idx = feature_names.index(base)
                    term = X[:, idx] ** float(power)
                elif "*" in name and "/" not in name:
                    parts = name.split("*")
                    term = np.ones(n_samples)
                    for part in parts:
                        part = part.strip()
                        if "^" in part:
                            base, power = part.split("^")
                            idx = feature_names.index(base)
                            term = term * X[:, idx] ** float(power)
                        else:
                            idx = feature_names.index(part)
                            term = term * X[:, idx]
                elif name.startswith("log("):
                    inner = name[4:-1]
                    idx = feature_names.index(inner)
                    term = np.log(X[:, idx])
                elif name.startswith("exp("):
                    inner = name[4:-1]
                    idx = feature_names.index(inner)
                    term = np.exp(X[:, idx])
                elif name.startswith("sqrt("):
                    inner = name[5:-1]
                    idx = feature_names.index(inner)
                    term = np.sqrt(X[:, idx])
                elif name.startswith("I(") and ")*" in name:
                    # Categorical interaction: I(color=2)*temperature
                    indicator_part, cont_part = name.split(")*", 1)
                    inner = indicator_part[2:]  # "color=2"
                    feat_name, cat_val_str = inner.split("=", 1)
                    cat_idx = feature_names.index(feat_name)
                    cat_val = float(cat_val_str)
                    cont_idx = feature_names.index(cont_part)
                    term = (X[:, cat_idx] == cat_val).astype(float) * X[:, cont_idx]
                elif name.startswith("I(") and name.endswith(")") and "=" in name:
                    # Indicator: I(color=2)
                    inner = name[2:-1]  # "color=2"
                    feat_name, cat_val_str = inner.split("=", 1)
                    idx = feature_names.index(feat_name)
                    cat_val = float(cat_val_str)
                    term = (X[:, idx] == cat_val).astype(float)
                elif name.startswith("1/"):
                    inner = name[2:]
                    idx = feature_names.index(inner)
                    term = 1.0 / X[:, idx]
                elif "/" in name:
                    num, den = name.split("/")
                    idx_num = feature_names.index(num)
                    idx_den = feature_names.index(den)
                    term = X[:, idx_num] / X[:, idx_den]
                else:
                    # Fallback for parametric / complex basis functions:
                    # evaluate via the stored callable and convert to numpy
                    import jax.numpy as _jnp

                    _bf = None
                    for _i, _n in enumerate(self.basis_library.names):  # type: ignore[union-attr]
                        if _n == name:
                            _bf = self.basis_library.basis_functions[_i]
                            break
                    if _bf is not None:
                        term = np.asarray(_bf.func(_jnp.array(X)))
                    else:
                        raise ValueError(f"Cannot convert basis function: {name}")

                result = result + coef * term

            return result

        return predict_numpy

    def save(self, filepath: str) -> None:
        """
        Save model to JSON file.

        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        self._check_is_fitted()

        data = {
            "config": {
                "max_terms": self.max_terms,
                "strategy": self.strategy,
                "information_criterion": self.information_criterion,
                "cv_folds": self.cv_folds,
                "regularization": self.regularization,
                "random_state": self.random_state,
                "param_optimizer": self.param_optimizer,
                "param_optimization_budget": self.param_optimization_budget,
            },
            "basis_library": self.basis_library.to_dict(),
            "result": self._result.to_dict(),
            "constraints": self.constraints.to_dict() if self.constraints else None,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> SymbolicRegressor:
        """
        Load model from JSON file.

        Parameters
        ----------
        filepath : str
            Path to the model file.

        Returns
        -------
        model : SymbolicRegressor
            Loaded model.
        """
        with open(filepath) as f:
            data = json.load(f)

        # Reconstruct basis library
        basis_library = BasisLibrary.from_dict(data["basis_library"])

        # Reconstruct constraints
        constraints = None
        if data.get("constraints"):
            constraints = Constraints.from_dict(data["constraints"])

        # Create model
        model = cls(
            basis_library=basis_library,
            constraints=constraints,
            **data["config"],
        )

        # Reconstruct result
        result_data = data["result"]
        parametric_params = result_data.get("parametric_params")
        if parametric_params is not None:
            parametric_params = {int(k): v for k, v in parametric_params.items()}
        model._result = SelectionResult(
            coefficients=jnp.array(result_data["coefficients"]),
            selected_indices=jnp.array(result_data["selected_indices"]),
            selected_names=result_data["selected_names"],
            mse=result_data["mse"],
            complexity=result_data["complexity"],
            aic=result_data["aic"],
            bic=result_data["bic"],
            aicc=result_data["aicc"],
            n_samples=result_data["n_samples"],
            parametric_params=parametric_params,
        )

        model._is_fitted = True
        return model

    def summary(self) -> str:
        """
        Return a summary of the fitted model.

        Returns
        -------
        summary : str
            Model summary.
        """
        self._check_is_fitted()

        lines = [
            "=" * 60,
            "JAXSR Symbolic Regression Model",
            "=" * 60,
            "",
            f"Expression: {self.expression_}",
            "",
            f"Selected terms ({len(self._result.selected_names)}):",
        ]

        for name, coef in zip(self._result.selected_names, self._result.coefficients, strict=False):
            lines.append(f"  {name}: {float(coef):.6g}")

        lines.extend(
            [
                "",
                "Metrics:",
                f"  MSE: {self._result.mse:.6g}",
                f"  R²: {self._compute_r2():.6f}",
                f"  BIC: {self._result.bic:.2f}",
                f"  AIC: {self._result.aic:.2f}",
                f"  Complexity: {self._result.complexity}",
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
            f"SymbolicRegressor(max_terms={self.max_terms}, "
            f"strategy='{self.strategy}', {status})"
        )


# =============================================================================
# Convenience Function
# =============================================================================


def fit_symbolic(
    X: jnp.ndarray,
    y: jnp.ndarray,
    feature_names: list[str] | None = None,
    max_terms: int = 5,
    max_poly_degree: int = 3,
    include_transcendental: bool = True,
    include_ratios: bool = False,
    strategy: str = "greedy_forward",
    information_criterion: str = "bic",
) -> SymbolicRegressor:
    """
    Convenience function for quick symbolic regression.

    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like
        Target values.
    feature_names : list of str, optional
        Names for features.
    max_terms : int
        Maximum terms in expression.
    max_poly_degree : int
        Maximum polynomial degree.
    include_transcendental : bool
        Include log, exp, sqrt, inv.
    include_ratios : bool
        Include x_i / x_j terms.
    strategy : str
        Selection strategy.
    information_criterion : str
        Information criterion.

    Returns
    -------
    model : SymbolicRegressor
        Fitted model.

    Examples
    --------
    >>> model = fit_symbolic(X, y, feature_names=["T", "P"])
    >>> print(model.expression_)
    """
    X = jnp.atleast_2d(jnp.asarray(X))
    n_features = X.shape[1]

    library = BasisLibrary(n_features, feature_names)
    library.add_constant()
    library.add_linear()
    library.add_polynomials(max_poly_degree)
    library.add_interactions()

    if include_transcendental:
        library.add_transcendental()
    if include_ratios:
        library.add_ratios()

    model = SymbolicRegressor(
        basis_library=library,
        max_terms=max_terms,
        strategy=strategy,
        information_criterion=information_criterion,
    )

    return model.fit(X, y)
