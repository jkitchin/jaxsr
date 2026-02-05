"""Tests for constraints module."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr.constraints import (
    Constraint,
    ConstraintType,
    Constraints,
    ConstraintEvaluator,
)


class TestConstraint:
    """Tests for Constraint dataclass."""

    def test_creation(self):
        """Test Constraint creation."""
        constraint = Constraint(
            constraint_type=ConstraintType.BOUND,
            target="y",
            params={"lower": 0.0},
            weight=1.0,
            hard=False,
        )
        assert constraint.constraint_type == ConstraintType.BOUND
        assert constraint.target == "y"
        assert constraint.params["lower"] == 0.0

    def test_serialization(self):
        """Test Constraint serialization."""
        constraint = Constraint(
            constraint_type=ConstraintType.MONOTONIC,
            target="x",
            params={"direction": "increasing"},
            weight=2.0,
            hard=True,
        )

        data = constraint.to_dict()
        loaded = Constraint.from_dict(data)

        assert loaded.constraint_type == constraint.constraint_type
        assert loaded.target == constraint.target
        assert loaded.params == constraint.params
        assert loaded.weight == constraint.weight
        assert loaded.hard == constraint.hard


class TestConstraints:
    """Tests for Constraints builder."""

    def test_empty(self):
        """Test empty constraints."""
        constraints = Constraints()
        assert len(constraints) == 0

    def test_add_bounds(self):
        """Test adding bound constraints."""
        constraints = Constraints().add_bounds("y", lower=0.0, upper=1.0)
        assert len(constraints) == 1

        c = constraints.constraints[0]
        assert c.constraint_type == ConstraintType.BOUND
        assert c.params["lower"] == 0.0
        assert c.params["upper"] == 1.0

    def test_add_monotonic(self):
        """Test adding monotonicity constraints."""
        constraints = Constraints().add_monotonic("x", direction="increasing")
        assert len(constraints) == 1

        c = constraints.constraints[0]
        assert c.constraint_type == ConstraintType.MONOTONIC
        assert c.params["direction"] == "increasing"

    def test_add_monotonic_invalid_direction(self):
        """Test error for invalid monotonic direction."""
        with pytest.raises(ValueError, match="direction"):
            Constraints().add_monotonic("x", direction="upward")

    def test_add_convex(self):
        """Test adding convexity constraint."""
        constraints = Constraints().add_convex("x")
        assert len(constraints) == 1
        assert constraints.constraints[0].constraint_type == ConstraintType.CONVEX

    def test_add_concave(self):
        """Test adding concavity constraint."""
        constraints = Constraints().add_concave("x")
        assert len(constraints) == 1
        assert constraints.constraints[0].constraint_type == ConstraintType.CONCAVE

    def test_add_sign_constraint(self):
        """Test adding sign constraint."""
        constraints = Constraints().add_sign_constraint("x", sign="positive")
        assert len(constraints) == 1

        c = constraints.constraints[0]
        assert c.constraint_type == ConstraintType.SIGN
        assert c.params["sign"] == "positive"

    def test_add_sign_constraint_invalid(self):
        """Test error for invalid sign."""
        with pytest.raises(ValueError, match="sign"):
            Constraints().add_sign_constraint("x", sign="nonnegative")

    def test_add_linear_constraint(self):
        """Test adding linear constraint."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([1.0, 2.0])

        constraints = Constraints().add_linear_constraint(A, b)
        assert len(constraints) == 1
        assert constraints.constraints[0].constraint_type == ConstraintType.LINEAR

    def test_add_known_coefficient(self):
        """Test adding fixed coefficient."""
        constraints = Constraints().add_known_coefficient("1", value=0.0)
        assert len(constraints) == 1

        c = constraints.constraints[0]
        assert c.constraint_type == ConstraintType.FIXED
        assert c.params["value"] == 0.0

    def test_method_chaining(self):
        """Test method chaining."""
        constraints = (
            Constraints()
            .add_bounds("y", lower=0)
            .add_monotonic("x", direction="increasing")
            .add_sign_constraint("x", sign="positive")
        )
        assert len(constraints) == 3

    def test_serialization(self):
        """Test Constraints serialization."""
        constraints = (
            Constraints()
            .add_bounds("y", lower=0)
            .add_monotonic("x", direction="increasing")
        )

        data = constraints.to_dict()
        loaded = Constraints.from_dict(data)

        assert len(loaded) == len(constraints)


class TestConstraintEvaluator:
    """Tests for ConstraintEvaluator."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple test setup."""
        basis_names = ["1", "x", "y"]
        feature_names = ["x", "y"]

        def predict_fn(X):
            # Simple linear model: y = 1 + 2*x + 3*y_feature
            return 1.0 + 2.0 * X[:, 0] + 3.0 * X[:, 1]

        return basis_names, feature_names, predict_fn

    def test_bound_constraint_penalty(self, simple_setup):
        """Test bound constraint penalty computation."""
        basis_names, feature_names, predict_fn = simple_setup

        constraints = Constraints().add_bounds("y", lower=10.0)
        evaluator = ConstraintEvaluator(constraints, basis_names, feature_names)

        # Points where prediction is below bound
        X = jnp.array([[0.0, 0.0]])  # predict = 1.0 < 10.0
        coefficients = jnp.array([1.0, 2.0, 3.0])

        penalty = evaluator.compute_penalty(coefficients, predict_fn, X)
        assert penalty > 0  # Should have positive penalty

        # Points where prediction satisfies bound
        X_ok = jnp.array([[10.0, 10.0]])  # predict = 1 + 20 + 30 = 51 > 10
        penalty_ok = evaluator.compute_penalty(coefficients, predict_fn, X_ok)
        assert penalty_ok < penalty

    def test_sign_constraint_penalty(self, simple_setup):
        """Test sign constraint penalty."""
        basis_names, feature_names, predict_fn = simple_setup

        constraints = Constraints().add_sign_constraint("x", sign="positive", hard=False)
        evaluator = ConstraintEvaluator(constraints, basis_names, feature_names)

        X = jnp.array([[0.0, 0.0]])

        # Positive coefficient - no penalty
        coefficients_pos = jnp.array([1.0, 2.0, 3.0])
        penalty_pos = evaluator.compute_penalty(coefficients_pos, predict_fn, X)

        # Negative coefficient - penalty
        coefficients_neg = jnp.array([1.0, -2.0, 3.0])
        penalty_neg = evaluator.compute_penalty(coefficients_neg, predict_fn, X)

        assert penalty_neg > penalty_pos

    def test_apply_hard_sign_constraint(self, simple_setup):
        """Test applying hard sign constraint."""
        basis_names, feature_names, predict_fn = simple_setup

        constraints = Constraints().add_sign_constraint("x", sign="positive", hard=True)
        evaluator = ConstraintEvaluator(constraints, basis_names, feature_names)

        # Negative coefficient should be projected to 0
        coefficients = jnp.array([1.0, -2.0, 3.0])
        projected = evaluator.apply_hard_constraints(coefficients)

        assert projected[1] >= 0  # x coefficient should be non-negative

    def test_apply_fixed_coefficient(self, simple_setup):
        """Test applying fixed coefficient constraint."""
        basis_names, feature_names, predict_fn = simple_setup

        constraints = Constraints().add_known_coefficient("1", value=5.0, fixed=True)
        evaluator = ConstraintEvaluator(constraints, basis_names, feature_names)

        coefficients = jnp.array([1.0, 2.0, 3.0])
        projected = evaluator.apply_hard_constraints(coefficients)

        assert projected[0] == 5.0  # Constant should be fixed to 5.0

    def test_get_fixed_indices(self, simple_setup):
        """Test getting fixed coefficient indices."""
        basis_names, feature_names, predict_fn = simple_setup

        constraints = (
            Constraints()
            .add_known_coefficient("1", value=0.0, fixed=True)
            .add_known_coefficient("y", value=1.5, fixed=True)
        )
        evaluator = ConstraintEvaluator(constraints, basis_names, feature_names)

        fixed = evaluator.get_fixed_indices()
        assert len(fixed) == 2
        assert (0, 0.0) in fixed  # index 0, value 0.0
        assert (2, 1.5) in fixed  # index 2, value 1.5

    def test_check_satisfaction(self, simple_setup):
        """Test constraint satisfaction checking."""
        basis_names, feature_names, predict_fn = simple_setup

        constraints = (
            Constraints()
            .add_sign_constraint("x", sign="positive", hard=False)
            .add_bounds("y", lower=0.0)
        )
        evaluator = ConstraintEvaluator(constraints, basis_names, feature_names)

        X = jnp.array([[1.0, 1.0]])  # predict = 6.0 > 0

        # All constraints satisfied
        coefficients_ok = jnp.array([1.0, 2.0, 3.0])
        satisfied = evaluator.check_satisfaction(coefficients_ok, predict_fn, X)
        assert all(satisfied.values())

        # Sign constraint violated
        coefficients_bad = jnp.array([1.0, -2.0, 3.0])
        satisfied_bad = evaluator.check_satisfaction(coefficients_bad, predict_fn, X)
        assert not satisfied_bad["sign_x"]
