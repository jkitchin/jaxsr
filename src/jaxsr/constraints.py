"""
Physical Constraints for JAXSR.

Provides functionality to incorporate domain knowledge through:
- Output bounds
- Monotonicity constraints
- Convexity constraints
- Sign constraints on coefficients
- Linear constraints on coefficients
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap


# =============================================================================
# Constraint Types
# =============================================================================


class ConstraintType(Enum):
    """Types of constraints supported."""
    BOUND = "bound"
    MONOTONIC = "monotonic"
    CONVEX = "convex"
    CONCAVE = "concave"
    SIGN = "sign"
    LINEAR = "linear"
    FIXED = "fixed"
    CUSTOM = "custom"  # For arbitrary nonlinear constraints


@dataclass
class Constraint:
    """
    Base constraint specification.

    Parameters
    ----------
    constraint_type : ConstraintType
        Type of constraint.
    target : str
        Target of constraint ("y" for output, feature name for input).
    params : dict
        Constraint-specific parameters.
    weight : float
        Weight for soft constraint penalty.
    hard : bool
        If True, enforce as hard constraint.
    """
    constraint_type: ConstraintType
    target: str
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    hard: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "constraint_type": self.constraint_type.value,
            "target": self.target,
            "params": self.params,
            "weight": self.weight,
            "hard": self.hard,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Constraint:
        """Deserialize from dictionary."""
        return cls(
            constraint_type=ConstraintType(data["constraint_type"]),
            target=data["target"],
            params=data["params"],
            weight=data["weight"],
            hard=data["hard"],
        )


# =============================================================================
# Constraints Builder
# =============================================================================


class Constraints:
    """
    Builder for constraint specifications.

    Supports method chaining for convenient constraint construction.

    Examples
    --------
    >>> constraints = (Constraints()
    ...     .add_bounds("y", lower=0)
    ...     .add_monotonic("T", direction="increasing")
    ...     .add_sign_constraint("T", sign="positive")
    ... )
    """

    def __init__(self):
        self.constraints: List[Constraint] = []
        self._feature_names: Optional[List[str]] = None

    def add_bounds(
        self,
        target: str = "y",
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        weight: float = 1.0,
        hard: bool = False,
    ) -> Constraints:
        """
        Add bounds constraint.

        Parameters
        ----------
        target : str
            "y" for output bounds, or feature name.
        lower : float, optional
            Lower bound.
        upper : float, optional
            Upper bound.
        weight : float
            Penalty weight for soft constraint.
        hard : bool
            If True, project to satisfy constraint.

        Returns
        -------
        self : Constraints
            For method chaining.
        """
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.BOUND,
            target=target,
            params={"lower": lower, "upper": upper},
            weight=weight,
            hard=hard,
        ))
        return self

    def add_monotonic(
        self,
        feature: str,
        direction: str = "increasing",
        weight: float = 1.0,
        hard: bool = False,
    ) -> Constraints:
        """
        Add monotonicity constraint.

        Parameters
        ----------
        feature : str
            Feature name for which output should be monotonic.
        direction : str
            "increasing" or "decreasing".
        weight : float
            Penalty weight.
        hard : bool
            If True, enforce strictly.

        Returns
        -------
        self : Constraints
        """
        if direction not in ["increasing", "decreasing"]:
            raise ValueError(f"direction must be 'increasing' or 'decreasing', got {direction}")

        self.constraints.append(Constraint(
            constraint_type=ConstraintType.MONOTONIC,
            target=feature,
            params={"direction": direction},
            weight=weight,
            hard=hard,
        ))
        return self

    def add_convex(
        self,
        feature: str,
        weight: float = 1.0,
        hard: bool = False,
    ) -> Constraints:
        """
        Add convexity constraint (positive second derivative).

        Parameters
        ----------
        feature : str
            Feature name.
        weight : float
            Penalty weight.
        hard : bool
            If True, enforce strictly.

        Returns
        -------
        self : Constraints
        """
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.CONVEX,
            target=feature,
            params={},
            weight=weight,
            hard=hard,
        ))
        return self

    def add_concave(
        self,
        feature: str,
        weight: float = 1.0,
        hard: bool = False,
    ) -> Constraints:
        """
        Add concavity constraint (negative second derivative).

        Parameters
        ----------
        feature : str
            Feature name.
        weight : float
            Penalty weight.
        hard : bool
            If True, enforce strictly.

        Returns
        -------
        self : Constraints
        """
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.CONCAVE,
            target=feature,
            params={},
            weight=weight,
            hard=hard,
        ))
        return self

    def add_sign_constraint(
        self,
        basis_name: str,
        sign: str = "positive",
        weight: float = 1.0,
        hard: bool = True,
    ) -> Constraints:
        """
        Add sign constraint on coefficient.

        Parameters
        ----------
        basis_name : str
            Name of basis function whose coefficient is constrained.
        sign : str
            "positive" or "negative".
        weight : float
            Penalty weight.
        hard : bool
            If True, project coefficient to satisfy constraint.

        Returns
        -------
        self : Constraints
        """
        if sign not in ["positive", "negative"]:
            raise ValueError(f"sign must be 'positive' or 'negative', got {sign}")

        self.constraints.append(Constraint(
            constraint_type=ConstraintType.SIGN,
            target=basis_name,
            params={"sign": sign},
            weight=weight,
            hard=hard,
        ))
        return self

    def add_linear_constraint(
        self,
        A: jnp.ndarray,
        b: jnp.ndarray,
        weight: float = 1.0,
        hard: bool = False,
    ) -> Constraints:
        """
        Add linear constraint: A @ coefficients <= b.

        Parameters
        ----------
        A : jnp.ndarray
            Constraint matrix of shape (n_constraints, n_basis).
        b : jnp.ndarray
            Constraint bounds of shape (n_constraints,).
        weight : float
            Penalty weight.
        hard : bool
            If True, project to satisfy constraint.

        Returns
        -------
        self : Constraints
        """
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.LINEAR,
            target="coefficients",
            params={"A": np.array(A).tolist(), "b": np.array(b).tolist()},
            weight=weight,
            hard=hard,
        ))
        return self

    def add_known_coefficient(
        self,
        basis_name: str,
        value: float,
        fixed: bool = True,
    ) -> Constraints:
        """
        Fix a coefficient to a known value.

        Parameters
        ----------
        basis_name : str
            Name of basis function.
        value : float
            Value to fix coefficient to.
        fixed : bool
            If True, coefficient is fixed during fitting.

        Returns
        -------
        self : Constraints
        """
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.FIXED,
            target=basis_name,
            params={"value": value},
            weight=0.0,
            hard=fixed,
        ))
        return self

    def add_custom(
        self,
        name: str,
        constraint_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float],
        weight: float = 1.0,
    ) -> Constraints:
        """
        Add a custom nonlinear constraint.

        The constraint function should return a penalty value (0 if satisfied,
        positive otherwise). It receives (coefficients, X, y_pred) as arguments.

        Parameters
        ----------
        name : str
            Name for this constraint (for reporting).
        constraint_fn : callable
            Function (coefficients, X, y_pred) -> float penalty.
            Should return 0 when satisfied, positive when violated.
        weight : float
            Penalty weight.

        Returns
        -------
        self : Constraints

        Examples
        --------
        >>> # Constraint: sum of coefficients must equal 1
        >>> def sum_to_one(coeffs, X, y_pred):
        ...     return (jnp.sum(coeffs) - 1.0) ** 2
        >>> constraints = Constraints().add_custom("sum_to_one", sum_to_one)

        >>> # Constraint: prediction at x=0 should be near 0
        >>> def zero_at_origin(coeffs, X, y_pred):
        ...     origin = jnp.zeros((1, X.shape[1]))
        ...     # Evaluate at origin requires basis evaluation
        ...     return 0.0  # Placeholder
        >>> constraints = Constraints().add_custom("zero_origin", zero_at_origin)

        >>> # Constraint: ratio of coefficients
        >>> def coeff_ratio(coeffs, X, y_pred):
        ...     # coeffs[0] / coeffs[1] should be approximately 2
        ...     if len(coeffs) < 2 or abs(coeffs[1]) < 1e-8:
        ...         return 0.0
        ...     ratio = coeffs[0] / coeffs[1]
        ...     return (ratio - 2.0) ** 2
        """
        # Store the function in params (note: can't serialize to JSON)
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.CUSTOM,
            target=name,
            params={"fn": constraint_fn},
            weight=weight,
            hard=False,  # Custom constraints are always soft
        ))
        return self

    def add_physics_constraint(
        self,
        name: str,
        constraint_type: str,
        params: Dict[str, Any],
        weight: float = 1.0,
    ) -> Constraints:
        """
        Add a physics-based constraint using predefined templates.

        Parameters
        ----------
        name : str
            Name for the constraint.
        constraint_type : str
            Type of physics constraint:
            - "asymptotic": y -> value as x -> inf
            - "periodic": y(x) = y(x + period)
            - "symmetric": y(x) = y(-x) or y(x) = -y(-x)
            - "scaling": y(ax) = a^n * y(x) for some n
            - "passthrough": y(x0) = y0 (passes through a specific point)
        params : dict
            Parameters for the constraint type.
        weight : float
            Penalty weight.

        Returns
        -------
        self : Constraints

        Examples
        --------
        >>> # y -> 0 as x -> infinity
        >>> constraints = Constraints().add_physics_constraint(
        ...     "asymptotic_zero", "asymptotic",
        ...     {"feature": "x", "value": 0.0, "at": "infinity"}
        ... )

        >>> # y passes through (0, 1)
        >>> constraints = Constraints().add_physics_constraint(
        ...     "initial_condition", "passthrough",
        ...     {"point": [0.0], "value": 1.0}
        ... )
        """
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.CUSTOM,
            target=name,
            params={"physics_type": constraint_type, **params},
            weight=weight,
            hard=False,
        ))
        return self

    def __len__(self) -> int:
        return len(self.constraints)

    def __iter__(self):
        return iter(self.constraints)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "constraints": [c.to_dict() for c in self.constraints],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Constraints:
        """Deserialize from dictionary."""
        obj = cls()
        obj.constraints = [Constraint.from_dict(c) for c in data["constraints"]]
        return obj


# =============================================================================
# Constraint Evaluation
# =============================================================================


class ConstraintEvaluator:
    """
    Evaluates constraints for a given model.

    Parameters
    ----------
    constraints : Constraints
        Constraint specifications.
    basis_names : list of str
        Names of basis functions.
    feature_names : list of str
        Names of input features.
    """

    def __init__(
        self,
        constraints: Constraints,
        basis_names: List[str],
        feature_names: List[str],
    ):
        self.constraints = constraints
        self.basis_names = basis_names
        self.feature_names = feature_names
        self._basis_name_to_idx = {name: i for i, name in enumerate(basis_names)}
        self._feature_name_to_idx = {name: i for i, name in enumerate(feature_names)}

    def compute_penalty(
        self,
        coefficients: jnp.ndarray,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        X: jnp.ndarray,
        y: Optional[jnp.ndarray] = None,
    ) -> float:
        """
        Compute total penalty for constraint violations.

        Parameters
        ----------
        coefficients : jnp.ndarray
            Current coefficients.
        predict_fn : callable
            Function that predicts y given X.
        X : jnp.ndarray
            Input points to check constraints.
        y : jnp.ndarray, optional
            Target values (for output bounds).

        Returns
        -------
        penalty : float
            Total weighted penalty.
        """
        total_penalty = 0.0

        for constraint in self.constraints:
            if not constraint.hard:
                penalty = self._evaluate_constraint(
                    constraint, coefficients, predict_fn, X
                )
                total_penalty += constraint.weight * penalty

        return total_penalty

    def _evaluate_constraint(
        self,
        constraint: Constraint,
        coefficients: jnp.ndarray,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        X: jnp.ndarray,
    ) -> float:
        """Evaluate a single constraint."""
        if constraint.constraint_type == ConstraintType.BOUND:
            return self._eval_bound(constraint, predict_fn, X)
        elif constraint.constraint_type == ConstraintType.MONOTONIC:
            return self._eval_monotonic(constraint, predict_fn, X)
        elif constraint.constraint_type == ConstraintType.CONVEX:
            return self._eval_convex(constraint, predict_fn, X)
        elif constraint.constraint_type == ConstraintType.CONCAVE:
            return self._eval_concave(constraint, predict_fn, X)
        elif constraint.constraint_type == ConstraintType.SIGN:
            return self._eval_sign(constraint, coefficients)
        elif constraint.constraint_type == ConstraintType.LINEAR:
            return self._eval_linear(constraint, coefficients)
        elif constraint.constraint_type == ConstraintType.CUSTOM:
            return self._eval_custom(constraint, coefficients, predict_fn, X)
        else:
            return 0.0

    def _eval_bound(
        self,
        constraint: Constraint,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        X: jnp.ndarray,
    ) -> float:
        """Evaluate bound constraint."""
        y_pred = predict_fn(X)
        lower = constraint.params.get("lower")
        upper = constraint.params.get("upper")

        penalty = 0.0
        if lower is not None:
            violations = jnp.maximum(lower - y_pred, 0)
            penalty += jnp.sum(violations ** 2)
        if upper is not None:
            violations = jnp.maximum(y_pred - upper, 0)
            penalty += jnp.sum(violations ** 2)

        return float(penalty)

    def _eval_monotonic(
        self,
        constraint: Constraint,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        X: jnp.ndarray,
    ) -> float:
        """Evaluate monotonicity constraint via finite differences."""
        feature = constraint.target
        direction = constraint.params["direction"]

        if feature not in self._feature_name_to_idx:
            return 0.0

        feature_idx = self._feature_name_to_idx[feature]

        # Compute gradient with respect to feature at each point
        eps = 1e-5
        X_plus = X.at[:, feature_idx].add(eps)
        X_minus = X.at[:, feature_idx].add(-eps)

        y_plus = predict_fn(X_plus)
        y_minus = predict_fn(X_minus)

        gradient = (y_plus - y_minus) / (2 * eps)

        if direction == "increasing":
            # Penalize negative gradients
            violations = jnp.maximum(-gradient, 0)
        else:
            # Penalize positive gradients
            violations = jnp.maximum(gradient, 0)

        return float(jnp.sum(violations ** 2))

    def _eval_convex(
        self,
        constraint: Constraint,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        X: jnp.ndarray,
    ) -> float:
        """Evaluate convexity constraint via second derivatives."""
        feature = constraint.target

        if feature not in self._feature_name_to_idx:
            return 0.0

        feature_idx = self._feature_name_to_idx[feature]

        # Compute second derivative via finite differences
        eps = 1e-4
        X_plus = X.at[:, feature_idx].add(eps)
        X_minus = X.at[:, feature_idx].add(-eps)

        y_center = predict_fn(X)
        y_plus = predict_fn(X_plus)
        y_minus = predict_fn(X_minus)

        second_deriv = (y_plus - 2 * y_center + y_minus) / (eps ** 2)

        # Convex means second derivative >= 0
        violations = jnp.maximum(-second_deriv, 0)

        return float(jnp.sum(violations ** 2))

    def _eval_concave(
        self,
        constraint: Constraint,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        X: jnp.ndarray,
    ) -> float:
        """Evaluate concavity constraint."""
        feature = constraint.target

        if feature not in self._feature_name_to_idx:
            return 0.0

        feature_idx = self._feature_name_to_idx[feature]

        eps = 1e-4
        X_plus = X.at[:, feature_idx].add(eps)
        X_minus = X.at[:, feature_idx].add(-eps)

        y_center = predict_fn(X)
        y_plus = predict_fn(X_plus)
        y_minus = predict_fn(X_minus)

        second_deriv = (y_plus - 2 * y_center + y_minus) / (eps ** 2)

        # Concave means second derivative <= 0
        violations = jnp.maximum(second_deriv, 0)

        return float(jnp.sum(violations ** 2))

    def _eval_sign(
        self,
        constraint: Constraint,
        coefficients: jnp.ndarray,
    ) -> float:
        """Evaluate sign constraint on coefficient."""
        basis_name = constraint.target
        sign = constraint.params["sign"]

        if basis_name not in self._basis_name_to_idx:
            return 0.0

        idx = self._basis_name_to_idx[basis_name]
        coef = coefficients[idx]

        if sign == "positive":
            return float(jnp.maximum(-coef, 0) ** 2)
        else:
            return float(jnp.maximum(coef, 0) ** 2)

    def _eval_linear(
        self,
        constraint: Constraint,
        coefficients: jnp.ndarray,
    ) -> float:
        """Evaluate linear constraint A @ coefficients <= b."""
        A = jnp.array(constraint.params["A"])
        b = jnp.array(constraint.params["b"])

        # Penalty for violations
        violations = jnp.maximum(A @ coefficients - b, 0)
        return float(jnp.sum(violations ** 2))

    def _eval_custom(
        self,
        constraint: Constraint,
        coefficients: jnp.ndarray,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        X: jnp.ndarray,
    ) -> float:
        """Evaluate custom nonlinear constraint."""
        # Check if it's a function-based constraint
        if "fn" in constraint.params:
            fn = constraint.params["fn"]
            y_pred = predict_fn(X)
            return float(fn(coefficients, X, y_pred))

        # Check if it's a physics-based constraint template
        physics_type = constraint.params.get("physics_type")
        if physics_type == "passthrough":
            return self._eval_passthrough(constraint, predict_fn)
        elif physics_type == "asymptotic":
            return self._eval_asymptotic(constraint, predict_fn, X)
        elif physics_type == "symmetric":
            return self._eval_symmetric(constraint, predict_fn, X)

        return 0.0

    def _eval_passthrough(
        self,
        constraint: Constraint,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> float:
        """Evaluate passthrough constraint: y(x0) = y0."""
        point = jnp.array(constraint.params["point"]).reshape(1, -1)
        expected_value = constraint.params["value"]
        y_pred = predict_fn(point)
        return float((y_pred[0] - expected_value) ** 2)

    def _eval_asymptotic(
        self,
        constraint: Constraint,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        X: jnp.ndarray,
    ) -> float:
        """Evaluate asymptotic constraint: y -> value as x -> limit."""
        feature = constraint.params.get("feature")
        limit_value = constraint.params.get("value", 0.0)
        at = constraint.params.get("at", "infinity")

        if feature not in self._feature_name_to_idx:
            return 0.0

        feature_idx = self._feature_name_to_idx[feature]

        # Create test point at large value
        if at == "infinity":
            large_val = 1e6
        elif at == "neg_infinity":
            large_val = -1e6
        else:
            large_val = float(at)

        test_point = jnp.mean(X, axis=0, keepdims=True)
        test_point = test_point.at[0, feature_idx].set(large_val)

        y_pred = predict_fn(test_point)
        return float((y_pred[0] - limit_value) ** 2)

    def _eval_symmetric(
        self,
        constraint: Constraint,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        X: jnp.ndarray,
    ) -> float:
        """Evaluate symmetry constraint: y(x) = y(-x) or y(x) = -y(-x)."""
        feature = constraint.params.get("feature")
        sym_type = constraint.params.get("type", "even")  # "even" or "odd"

        if feature not in self._feature_name_to_idx:
            return 0.0

        feature_idx = self._feature_name_to_idx[feature]

        # Create reflected X
        X_reflected = X.at[:, feature_idx].multiply(-1)

        y_pos = predict_fn(X)
        y_neg = predict_fn(X_reflected)

        if sym_type == "even":
            # y(x) = y(-x)
            violations = y_pos - y_neg
        else:  # odd
            # y(x) = -y(-x)
            violations = y_pos + y_neg

        return float(jnp.mean(violations ** 2))

    def apply_hard_constraints(
        self,
        coefficients: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Project coefficients to satisfy hard constraints.

        Parameters
        ----------
        coefficients : jnp.ndarray
            Current coefficients.

        Returns
        -------
        coefficients : jnp.ndarray
            Projected coefficients.
        """
        coefficients = coefficients.copy()

        for constraint in self.constraints:
            if not constraint.hard:
                continue

            if constraint.constraint_type == ConstraintType.SIGN:
                coefficients = self._project_sign(constraint, coefficients)
            elif constraint.constraint_type == ConstraintType.FIXED:
                coefficients = self._project_fixed(constraint, coefficients)

        return coefficients

    def _project_sign(
        self,
        constraint: Constraint,
        coefficients: jnp.ndarray,
    ) -> jnp.ndarray:
        """Project coefficient to satisfy sign constraint."""
        basis_name = constraint.target
        sign = constraint.params["sign"]

        if basis_name not in self._basis_name_to_idx:
            return coefficients

        idx = self._basis_name_to_idx[basis_name]

        if sign == "positive":
            coefficients = coefficients.at[idx].set(jnp.maximum(coefficients[idx], 0))
        else:
            coefficients = coefficients.at[idx].set(jnp.minimum(coefficients[idx], 0))

        return coefficients

    def _project_fixed(
        self,
        constraint: Constraint,
        coefficients: jnp.ndarray,
    ) -> jnp.ndarray:
        """Fix coefficient to known value."""
        basis_name = constraint.target
        value = constraint.params["value"]

        if basis_name not in self._basis_name_to_idx:
            return coefficients

        idx = self._basis_name_to_idx[basis_name]
        coefficients = coefficients.at[idx].set(value)

        return coefficients

    def get_fixed_indices(self) -> List[Tuple[int, float]]:
        """
        Get indices and values of fixed coefficients.

        Returns
        -------
        fixed : list of (int, float)
            Indices and fixed values.
        """
        fixed = []
        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.FIXED and constraint.hard:
                basis_name = constraint.target
                if basis_name in self._basis_name_to_idx:
                    idx = self._basis_name_to_idx[basis_name]
                    value = constraint.params["value"]
                    fixed.append((idx, value))
        return fixed

    def check_satisfaction(
        self,
        coefficients: jnp.ndarray,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        X: jnp.ndarray,
        tolerance: float = 1e-6,
    ) -> Dict[str, bool]:
        """
        Check which constraints are satisfied.

        Parameters
        ----------
        coefficients : jnp.ndarray
            Current coefficients.
        predict_fn : callable
            Prediction function.
        X : jnp.ndarray
            Test points.
        tolerance : float
            Tolerance for constraint satisfaction.

        Returns
        -------
        satisfied : dict
            Constraint names mapped to satisfaction status.
        """
        results = {}

        for i, constraint in enumerate(self.constraints):
            penalty = self._evaluate_constraint(constraint, coefficients, predict_fn, X)
            name = f"{constraint.constraint_type.value}_{constraint.target}"
            results[name] = penalty < tolerance

        return results


# =============================================================================
# Constrained Fitting
# =============================================================================


def fit_constrained_ols(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    constraints: Constraints,
    basis_names: List[str],
    feature_names: List[str],
    X: jnp.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[jnp.ndarray, float]:
    """
    Fit least squares with constraints.

    Uses iterative projection for hard constraints and penalty for soft.

    Parameters
    ----------
    Phi : jnp.ndarray
        Design matrix.
    y : jnp.ndarray
        Target vector.
    constraints : Constraints
        Constraint specifications.
    basis_names : list of str
        Names of basis functions.
    feature_names : list of str
        Names of input features.
    X : jnp.ndarray
        Input data (for evaluating constraints).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    coefficients : jnp.ndarray
        Fitted coefficients.
    mse : float
        Mean squared error.
    """
    evaluator = ConstraintEvaluator(constraints, basis_names, feature_names)

    # Get fixed coefficients
    fixed = evaluator.get_fixed_indices()

    # Initial OLS solution
    coeffs, residuals, rank, s = jnp.linalg.lstsq(Phi, y, rcond=None)

    # Apply hard constraints
    coeffs = evaluator.apply_hard_constraints(coeffs)

    # For fixed coefficients, adjust the problem
    if fixed:
        # Solve reduced problem excluding fixed coefficients
        free_indices = [i for i in range(len(coeffs)) if i not in [f[0] for f in fixed]]

        if free_indices:
            # Adjust target for fixed coefficients
            y_adjusted = y.copy()
            for idx, value in fixed:
                y_adjusted = y_adjusted - Phi[:, idx] * value

            Phi_free = Phi[:, free_indices]
            coeffs_free, _, _, _ = jnp.linalg.lstsq(Phi_free, y_adjusted, rcond=None)

            # Reconstruct full coefficient vector
            coeffs = jnp.zeros(len(basis_names))
            for idx, value in fixed:
                coeffs = coeffs.at[idx].set(value)
            for i, idx in enumerate(free_indices):
                coeffs = coeffs.at[idx].set(coeffs_free[i])

    # Apply remaining hard constraints
    coeffs = evaluator.apply_hard_constraints(coeffs)

    # Compute MSE
    y_pred = Phi @ coeffs
    mse = float(jnp.mean((y - y_pred) ** 2))

    return coeffs, mse
