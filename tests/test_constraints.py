"""Tests for constraints module."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr.basis import BasisLibrary
from jaxsr.constraints import (
    Constraint,
    ConstraintEvaluator,
    Constraints,
    ConstraintType,
    fit_constrained_ols,
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
            Constraints().add_bounds("y", lower=0).add_monotonic("x", direction="increasing")
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

    def test_compute_hard_penalty(self, simple_setup):
        """Test compute_hard_penalty for hard shape constraints."""
        basis_names, feature_names, predict_fn = simple_setup

        # Hard monotonic constraint
        constraints = Constraints().add_monotonic("x", direction="decreasing", hard=True)
        evaluator = ConstraintEvaluator(constraints, basis_names, feature_names)

        X = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        coefficients = jnp.array([1.0, 2.0, 3.0])

        # predict_fn = 1 + 2*x + 3*y, which is increasing in x
        # So hard penalty for "decreasing" should be > 0
        hard_penalty = evaluator.compute_hard_penalty(coefficients, predict_fn, X)
        assert hard_penalty > 0

        # Soft monotonic constraint should NOT appear in hard penalty
        constraints_soft = Constraints().add_monotonic("x", direction="decreasing", hard=False)
        evaluator_soft = ConstraintEvaluator(constraints_soft, basis_names, feature_names)
        hard_penalty_soft = evaluator_soft.compute_hard_penalty(coefficients, predict_fn, X)
        assert hard_penalty_soft == 0.0


class TestConstraintEnforcement:
    """Tests for constraint enforcement through optimization."""

    def test_monotonic_constraint_enforcement(self):
        """Test that monotonic constraint produces a monotonic model."""
        np.random.seed(42)
        # Data: y = -x^2 + 3x (increasing for x in [0,1.5], decreasing after)
        X = np.linspace(0, 3, 50).reshape(-1, 1)
        y = -X[:, 0] ** 2 + 3 * X[:, 0]

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=3)
        )

        Phi = library.evaluate(jnp.array(X))
        basis_names = library.names
        # Select all basis functions
        n_basis = len(basis_names)
        selected_indices = jnp.arange(n_basis)

        # Fit with increasing monotonicity constraint (hard)
        constraints = Constraints().add_monotonic("x", direction="increasing", hard=True)

        coeffs, mse = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=basis_names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=selected_indices,
        )

        # Verify monotonicity: predictions should be non-decreasing
        y_pred = np.array(Phi @ coeffs)
        diffs = np.diff(y_pred)
        # Allow small numerical tolerance
        assert np.all(diffs >= -1e-3), f"Model is not monotonic: min diff = {diffs.min()}"

    def test_convex_constraint_enforcement(self):
        """Test that convexity constraint produces a convex model."""
        np.random.seed(42)
        # Data: y = -x^2 (concave), fit with convexity constraint
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        y = -X[:, 0] ** 2

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=3)
        )

        Phi = library.evaluate(jnp.array(X))
        basis_names = library.names
        selected_indices = jnp.arange(len(basis_names))

        constraints = Constraints().add_convex("x", hard=True)

        coeffs, mse = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=basis_names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=selected_indices,
        )

        # Verify convexity: second differences should be >= 0
        y_pred = np.array(Phi @ coeffs)
        second_diffs = np.diff(y_pred, n=2)
        assert np.all(
            second_diffs >= -1e-2
        ), f"Model is not convex: min second diff = {second_diffs.min()}"

    def test_soft_bounds_affect_fitting(self):
        """Test that soft bound penalties actually change the fit."""
        np.random.seed(42)
        X = np.linspace(0, 5, 50).reshape(-1, 1)
        y = 2 * X[:, 0] + 1  # y ranges from 1 to 11

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()

        Phi = library.evaluate(jnp.array(X))
        basis_names = library.names
        selected_indices = jnp.arange(len(basis_names))

        # Fit without constraints
        constraints_none = Constraints()
        coeffs_plain, _ = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints_none,
            basis_names=basis_names,
            feature_names=["x"],
            X=jnp.array(X),
        )

        # Fit with soft upper bound at y=5 (tight, will force change)
        constraints_bounded = Constraints().add_bounds("y", upper=5.0, weight=10.0, hard=False)
        coeffs_bounded, _ = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints_bounded,
            basis_names=basis_names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=selected_indices,
            penalty_weight=10.0,
        )

        # Bounded fit should produce different coefficients
        assert not np.allclose(
            np.array(coeffs_plain), np.array(coeffs_bounded), atol=0.1
        ), "Soft bounds did not affect fitting"

    def test_combined_hard_soft_constraints(self):
        """Test fixed coefficient + monotonic constraint together."""
        np.random.seed(42)
        X = np.linspace(0, 3, 50).reshape(-1, 1)
        y = 2 * X[:, 0] + 1

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
        )

        Phi = library.evaluate(jnp.array(X))
        basis_names = library.names
        selected_indices = jnp.arange(len(basis_names))

        # Fix intercept to 0 and require increasing monotonicity
        constraints = (
            Constraints()
            .add_known_coefficient("1", value=0.0)
            .add_monotonic("x", direction="increasing", hard=True)
        )

        coeffs, mse = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=basis_names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=selected_indices,
        )

        # Check intercept is fixed to 0
        const_idx = basis_names.index("1")
        assert abs(float(coeffs[const_idx])) < 1e-10, "Intercept should be fixed to 0"

        # Check monotonicity
        y_pred = np.array(Phi @ coeffs)
        diffs = np.diff(y_pred)
        assert np.all(diffs >= -1e-3), "Model should be monotonically increasing"

    def test_no_basis_library_fallback(self):
        """Test that coefficient-only constraints work without basis_library."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 2 * X[:, 0] + 1

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()

        Phi = library.evaluate(jnp.array(X))
        basis_names = library.names

        # Only SIGN/FIXED constraints - should work without basis_library
        constraints = Constraints().add_sign_constraint("x", sign="positive")

        coeffs, mse = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=basis_names,
            feature_names=["x"],
            X=jnp.array(X),
            # No basis_library or selected_indices
        )

        # x coefficient should be positive
        x_idx = basis_names.index("x")
        assert float(coeffs[x_idx]) >= 0


# =============================================================================
# Multi-Level Constraint Enforcement Tests
# =============================================================================


def _make_monotonic_test_problem():
    """Helper: create a test problem where unconstrained fit violates monotonicity."""
    np.random.seed(42)
    # Data: y = -x^2 + 3x (increasing for x in [0,1.5], decreasing after)
    X = np.linspace(0, 3, 50).reshape(-1, 1)
    y = -X[:, 0] ** 2 + 3 * X[:, 0]

    library = (
        BasisLibrary(n_features=1, feature_names=["x"])
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=3)
    )

    Phi = library.evaluate(jnp.array(X))
    basis_names = library.names
    selected_indices = jnp.arange(len(basis_names))
    constraints = Constraints().add_monotonic("x", direction="increasing", hard=True)

    return {
        "Phi": Phi,
        "y": jnp.array(y),
        "X": jnp.array(X),
        "basis_names": basis_names,
        "library": library,
        "selected_indices": selected_indices,
        "constraints": constraints,
    }


class TestConstraintEnforcementLevels:
    """Tests for the multi-level constraint enforcement system."""

    # ---- Backward compatibility ----

    def test_penalty_default_matches_old_behavior(self):
        """enforcement='penalty' (default) produces same result as before."""
        p = _make_monotonic_test_problem()
        coeffs_default, mse_default = fit_constrained_ols(
            Phi=p["Phi"],
            y=p["y"],
            constraints=p["constraints"],
            basis_names=p["basis_names"],
            feature_names=["x"],
            X=p["X"],
            basis_library=p["library"],
            selected_indices=p["selected_indices"],
        )
        coeffs_explicit, mse_explicit = fit_constrained_ols(
            Phi=p["Phi"],
            y=p["y"],
            constraints=p["constraints"],
            basis_names=p["basis_names"],
            feature_names=["x"],
            X=p["X"],
            basis_library=p["library"],
            selected_indices=p["selected_indices"],
            enforcement="penalty",
        )
        np.testing.assert_allclose(np.array(coeffs_default), np.array(coeffs_explicit), atol=1e-8)
        assert abs(mse_default - mse_explicit) < 1e-10

    # ---- Constrained (trust-constr) path ----

    def test_constrained_monotonic(self):
        """enforcement='constrained' produces a monotonic model."""
        p = _make_monotonic_test_problem()
        coeffs, mse = fit_constrained_ols(
            Phi=p["Phi"],
            y=p["y"],
            constraints=p["constraints"],
            basis_names=p["basis_names"],
            feature_names=["x"],
            X=p["X"],
            basis_library=p["library"],
            selected_indices=p["selected_indices"],
            enforcement="constrained",
        )
        y_pred = np.array(p["Phi"] @ coeffs)
        diffs = np.diff(y_pred)
        assert np.all(diffs >= -1e-4), f"trust-constr not monotonic: min diff = {diffs.min()}"

    def test_constrained_convex(self):
        """enforcement='constrained' enforces convexity."""
        np.random.seed(42)
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        y = -X[:, 0] ** 2  # concave data

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=3)
        )
        Phi = library.evaluate(jnp.array(X))
        constraints = Constraints().add_convex("x", hard=True)

        coeffs, _ = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=library.names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=jnp.arange(len(library.names)),
            enforcement="constrained",
        )
        y_pred = np.array(Phi @ coeffs)
        second_diffs = np.diff(y_pred, n=2)
        assert np.all(
            second_diffs >= -1e-2
        ), f"trust-constr not convex: min 2nd diff = {second_diffs.min()}"

    def test_constrained_bounds(self):
        """enforcement='constrained' enforces hard output bounds."""
        np.random.seed(42)
        X = np.linspace(0, 5, 50).reshape(-1, 1)
        y = 2 * X[:, 0] + 1  # y ranges 1..11

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()
        Phi = library.evaluate(jnp.array(X))
        constraints = Constraints().add_bounds("y", upper=8.0, hard=True)

        coeffs, _ = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=library.names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=jnp.arange(len(library.names)),
            enforcement="constrained",
        )
        y_pred = np.array(Phi @ coeffs)
        # Allow solver tolerance
        assert np.all(y_pred <= 8.0 + 1e-4), f"trust-constr bound violated: max = {y_pred.max()}"

    # ---- Exact (QP) path ----

    def test_exact_monotonic(self):
        """enforcement='exact' produces a monotonic model."""
        cvxpy = pytest.importorskip("cvxpy")  # noqa: F841
        p = _make_monotonic_test_problem()
        coeffs, mse = fit_constrained_ols(
            Phi=p["Phi"],
            y=p["y"],
            constraints=p["constraints"],
            basis_names=p["basis_names"],
            feature_names=["x"],
            X=p["X"],
            basis_library=p["library"],
            selected_indices=p["selected_indices"],
            enforcement="exact",
        )
        y_pred = np.array(p["Phi"] @ coeffs)
        diffs = np.diff(y_pred)
        assert np.all(diffs >= -1e-5), f"QP not monotonic: min diff = {diffs.min()}"

    def test_exact_convex(self):
        """enforcement='exact' enforces convexity."""
        cvxpy = pytest.importorskip("cvxpy")  # noqa: F841
        np.random.seed(42)
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        y = -X[:, 0] ** 2

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=3)
        )
        Phi = library.evaluate(jnp.array(X))
        constraints = Constraints().add_convex("x", hard=True)

        coeffs, _ = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=library.names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=jnp.arange(len(library.names)),
            enforcement="exact",
        )
        y_pred = np.array(Phi @ coeffs)
        second_diffs = np.diff(y_pred, n=2)
        assert np.all(second_diffs >= -1e-3), f"QP not convex: min 2nd diff = {second_diffs.min()}"

    def test_exact_bounds(self):
        """enforcement='exact' enforces hard output bounds."""
        cvxpy = pytest.importorskip("cvxpy")  # noqa: F841
        np.random.seed(42)
        X = np.linspace(0, 5, 50).reshape(-1, 1)
        y = 2 * X[:, 0] + 1

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()
        Phi = library.evaluate(jnp.array(X))
        constraints = Constraints().add_bounds("y", upper=8.0, hard=True)

        coeffs, _ = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=library.names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=jnp.arange(len(library.names)),
            enforcement="exact",
        )
        y_pred = np.array(Phi @ coeffs)
        assert np.all(y_pred <= 8.0 + 1e-5), f"QP bound violated: max = {y_pred.max()}"

    # ---- Error cases ----

    def test_invalid_enforcement_value(self):
        """Invalid enforcement value raises ValueError."""
        with pytest.raises(ValueError, match="enforcement"):
            fit_constrained_ols(
                Phi=jnp.eye(2),
                y=jnp.array([1.0, 2.0]),
                constraints=Constraints(),
                basis_names=["a", "b"],
                feature_names=["x"],
                X=jnp.array([[1.0], [2.0]]),
                enforcement="invalid",
            )

    def test_exact_hard_custom_raises(self):
        """enforcement='exact' with hard=True CUSTOM constraint raises ValueError."""
        cvxpy = pytest.importorskip("cvxpy")  # noqa: F841
        p = _make_monotonic_test_problem()

        # Add a custom constraint and force hard=True
        constraints = Constraints().add_monotonic("x", direction="increasing", hard=True)
        # Manually add a hard custom constraint
        constraints.constraints.append(
            Constraint(
                constraint_type=ConstraintType.CUSTOM,
                target="test",
                params={"fn": lambda c, X, y: 0.0},
                weight=1.0,
                hard=True,
            )
        )

        with pytest.raises(ValueError, match="CUSTOM"):
            fit_constrained_ols(
                Phi=p["Phi"],
                y=p["y"],
                constraints=constraints,
                basis_names=p["basis_names"],
                feature_names=["x"],
                X=p["X"],
                basis_library=p["library"],
                selected_indices=p["selected_indices"],
                enforcement="exact",
            )

    # ---- Soft constraints unaffected by enforcement level ----

    def test_soft_constraints_same_across_levels(self):
        """Soft (hard=False) constraints behave similarly regardless of enforcement level."""
        np.random.seed(42)
        X = np.linspace(0, 5, 50).reshape(-1, 1)
        y = 2 * X[:, 0] + 1

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()
        Phi = library.evaluate(jnp.array(X))

        # Soft bound only — no hard constraints to enforce differently
        constraints = Constraints().add_bounds("y", upper=5.0, weight=10.0, hard=False)

        coeffs_penalty, _ = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=library.names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=jnp.arange(len(library.names)),
            penalty_weight=10.0,
            enforcement="penalty",
        )
        coeffs_constr, _ = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=library.names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=jnp.arange(len(library.names)),
            penalty_weight=10.0,
            enforcement="constrained",
        )
        # Both should produce similar coefficients (soft penalty is the same)
        np.testing.assert_allclose(np.array(coeffs_penalty), np.array(coeffs_constr), atol=0.5)

    # ---- SymbolicRegressor integration ----

    def test_regressor_enforcement_param(self):
        """SymbolicRegressor accepts constraint_enforcement parameter."""
        from jaxsr import BasisLibrary, SymbolicRegressor

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()
        model = SymbolicRegressor(
            basis_library=library,
            max_terms=2,
            constraint_enforcement="constrained",
        )
        assert model.constraint_enforcement == "constrained"

    def test_regressor_invalid_enforcement(self):
        """SymbolicRegressor rejects invalid constraint_enforcement."""
        from jaxsr import BasisLibrary, SymbolicRegressor

        with pytest.raises(ValueError, match="constraint_enforcement"):
            SymbolicRegressor(
                basis_library=BasisLibrary(n_features=1),
                constraint_enforcement="bad",
            )

    def test_regressor_save_load_roundtrip(self, tmp_path):
        """constraint_enforcement persists through save/load."""
        from jaxsr import BasisLibrary, SymbolicRegressor

        np.random.seed(42)
        X = np.random.randn(30, 1)
        y = 2 * X[:, 0] + 1

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()
        model = SymbolicRegressor(
            basis_library=library,
            max_terms=2,
            constraint_enforcement="constrained",
        )
        model.fit(X, y)

        filepath = str(tmp_path / "model.json")
        model.save(filepath)

        loaded = SymbolicRegressor.load(filepath)
        assert loaded.constraint_enforcement == "constrained"

    def test_load_old_format_defaults_to_penalty(self, tmp_path):
        """Loading a save without constraint_enforcement defaults to 'penalty'."""
        import json

        from jaxsr import BasisLibrary, SymbolicRegressor

        np.random.seed(42)
        X = np.random.randn(30, 1)
        y = 2 * X[:, 0] + 1

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()
        model = SymbolicRegressor(basis_library=library, max_terms=2)
        model.fit(X, y)

        filepath = str(tmp_path / "old_model.json")
        model.save(filepath)

        # Remove the key to simulate an old save file
        with open(filepath) as f:
            data = json.load(f)
        del data["config"]["constraint_enforcement"]
        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = SymbolicRegressor.load(filepath)
        assert loaded.constraint_enforcement == "penalty"

    # ---- Constrained path with fixed coefficients ----

    def test_constrained_with_fixed_coeff(self):
        """enforcement='constrained' works with fixed coefficient + monotonicity."""
        np.random.seed(42)
        X = np.linspace(0, 3, 50).reshape(-1, 1)
        y = 2 * X[:, 0] + 1

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
        )
        Phi = library.evaluate(jnp.array(X))

        constraints = (
            Constraints()
            .add_known_coefficient("1", value=0.0)
            .add_monotonic("x", direction="increasing", hard=True)
        )

        coeffs, _ = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=library.names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=jnp.arange(len(library.names)),
            enforcement="constrained",
        )

        const_idx = library.names.index("1")
        assert abs(float(coeffs[const_idx])) < 1e-10, "Intercept should be fixed to 0"

        y_pred = np.array(Phi @ coeffs)
        diffs = np.diff(y_pred)
        assert np.all(diffs >= -1e-3), "Model should be monotonically increasing"

    # ---- Linear constraint enforcement ----

    def test_constrained_linear_constraint(self):
        """enforcement='constrained' enforces hard linear constraints."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 3 * X[:, 0] + 5

        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()
        Phi = library.evaluate(jnp.array(X))

        # Constraint: coeff_0 + coeff_1 <= 2 (sum of coefficients <= 2)
        A = np.array([[1.0, 1.0]])
        b = np.array([2.0])
        constraints = Constraints().add_linear_constraint(A, b, hard=True)

        coeffs, _ = fit_constrained_ols(
            Phi=Phi,
            y=jnp.array(y),
            constraints=constraints,
            basis_names=library.names,
            feature_names=["x"],
            X=jnp.array(X),
            basis_library=library,
            selected_indices=jnp.arange(len(library.names)),
            enforcement="constrained",
        )

        coeff_sum = float(np.sum(np.array(coeffs)))
        assert coeff_sum <= 2.0 + 1e-4, f"Linear constraint violated: sum = {coeff_sum}"
