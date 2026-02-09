"""
Tests for the SymbolicClassifier and supporting classification infrastructure.
"""

from __future__ import annotations

import json
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr.basis import BasisLibrary
from jaxsr.classifier import SymbolicClassifier, fit_symbolic_classification
from jaxsr.metrics import (
    compute_accuracy,
    compute_all_classification_metrics,
    compute_auc_roc,
    compute_classification_ic,
    compute_confusion_matrix,
    compute_f1_score,
    compute_log_loss,
    compute_matthews_corrcoef,
    compute_precision,
    compute_recall,
)
from jaxsr.selection import (
    ClassificationPath,
    ClassificationResult,
    _sigmoid,
    compute_pareto_front_classification,
    fit_classification_subset,
    fit_irls,
    fit_logistic_lasso,
    greedy_backward_classification,
    greedy_forward_classification,
)
from jaxsr.uncertainty import (
    bootstrap_classification_coefficients,
    calibration_curve,
    classification_coefficient_intervals,
    conformal_classification_split,
)

# =============================================================================
# Fixtures
# =============================================================================


def _make_binary_data(n=200, seed=42):
    """Generate a simple linearly separable binary dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    # True decision boundary: 1 + 2*x0 - 1*x1
    logits = 1 + 2 * X[:, 0] - 1 * X[:, 1]
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(n) < prob).astype(float)
    return jnp.array(X), jnp.array(y)


def _make_multiclass_data(n=300, seed=42):
    """Generate a simple 3-class dataset."""
    rng = np.random.RandomState(seed)
    n_per = n // 3
    X0 = rng.randn(n_per, 2) + np.array([0, 2])
    X1 = rng.randn(n_per, 2) + np.array([2, -1])
    X2 = rng.randn(n_per, 2) + np.array([-2, -1])
    X = np.vstack([X0, X1, X2])
    y = np.concatenate([np.zeros(n_per), np.ones(n_per), 2 * np.ones(n_per)])
    return jnp.array(X), jnp.array(y)


def _make_library(n_features=2):
    """Build a small basis library for testing."""
    lib = BasisLibrary(n_features=n_features)
    lib.add_constant()
    lib.add_linear()
    lib.add_polynomials(max_degree=2)
    return lib


# =============================================================================
# Test Classification Metrics
# =============================================================================


class TestClassificationMetrics:
    """Tests for pure classification metric functions."""

    def test_accuracy_perfect(self):
        y = jnp.array([0, 1, 0, 1])
        assert compute_accuracy(y, y) == 1.0

    def test_accuracy_half(self):
        y_true = jnp.array([0, 0, 1, 1])
        y_pred = jnp.array([0, 1, 0, 1])
        assert compute_accuracy(y_true, y_pred) == 0.5

    def test_log_loss_binary(self):
        y_true = jnp.array([0, 0, 1, 1])
        y_pred_proba = jnp.array([0.1, 0.2, 0.9, 0.8])
        loss = compute_log_loss(y_true, y_pred_proba)
        assert loss > 0
        assert loss < 1.0  # Should be low for good predictions

    def test_log_loss_multiclass(self):
        y_true = jnp.array([0, 1, 2])
        y_pred_proba = jnp.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )
        loss = compute_log_loss(y_true, y_pred_proba)
        assert loss > 0
        assert loss < 1.0

    def test_precision_recall_f1(self):
        y_true = jnp.array([1, 1, 0, 0, 1, 0])
        y_pred = jnp.array([1, 0, 0, 0, 1, 1])

        p = compute_precision(y_true, y_pred)
        r = compute_recall(y_true, y_pred)
        f1 = compute_f1_score(y_true, y_pred)

        assert 0 <= p <= 1
        assert 0 <= r <= 1
        assert 0 <= f1 <= 1
        # F1 should be harmonic mean
        if p + r > 0:
            assert abs(f1 - 2 * p * r / (p + r)) < 1e-10

    def test_precision_no_positive_predictions(self):
        y_true = jnp.array([1, 1, 0])
        y_pred = jnp.array([0, 0, 0])
        assert compute_precision(y_true, y_pred) == 0.0

    def test_recall_no_positive_samples(self):
        y_true = jnp.array([0, 0, 0])
        y_pred = jnp.array([1, 0, 1])
        assert compute_recall(y_true, y_pred) == 0.0

    def test_auc_roc_perfect(self):
        y_true = jnp.array([0, 0, 1, 1])
        y_score = jnp.array([0.1, 0.2, 0.9, 0.8])
        auc = compute_auc_roc(y_true, y_score)
        assert auc == 1.0

    def test_auc_roc_random(self):
        rng = np.random.RandomState(123)
        y_true = jnp.array(rng.randint(0, 2, 100))
        y_score = jnp.array(rng.rand(100))
        auc = compute_auc_roc(y_true, y_score)
        assert 0 <= auc <= 1

    def test_auc_roc_single_class_raises(self):
        y_true = jnp.array([0, 0, 0])
        y_score = jnp.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="at least two"):
            compute_auc_roc(y_true, y_score)

    def test_confusion_matrix(self):
        y_true = jnp.array([0, 0, 1, 1, 2, 2])
        y_pred = jnp.array([0, 1, 1, 1, 2, 0])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert cm.sum() == 6
        assert cm[0, 0] == 1  # TN for class 0
        assert cm[1, 1] == 2  # TP for class 1

    def test_matthews_corrcoef_perfect(self):
        y_true = jnp.array([0, 0, 1, 1])
        y_pred = jnp.array([0, 0, 1, 1])
        mcc = compute_matthews_corrcoef(y_true, y_pred)
        assert abs(mcc - 1.0) < 1e-10

    def test_matthews_corrcoef_opposite(self):
        y_true = jnp.array([0, 0, 1, 1])
        y_pred = jnp.array([1, 1, 0, 0])
        mcc = compute_matthews_corrcoef(y_true, y_pred)
        assert abs(mcc + 1.0) < 1e-10

    def test_classification_ic(self):
        aic = compute_classification_ic(100, 3, 50.0, "aic")
        bic = compute_classification_ic(100, 3, 50.0, "bic")
        aicc = compute_classification_ic(100, 3, 50.0, "aicc")

        assert aic == 2 * 50.0 + 2 * 3  # 106
        assert bic > aic  # BIC penalises more for n > e^2 ~ 7
        assert aicc >= aic

    def test_classification_ic_invalid_criterion(self):
        with pytest.raises(ValueError, match="Unknown criterion"):
            compute_classification_ic(100, 3, 50.0, "cv")

    def test_compute_all_classification_metrics(self):
        y_true = jnp.array([0, 0, 1, 1, 1])
        y_pred = jnp.array([0, 1, 1, 1, 0])
        proba = jnp.array([0.2, 0.6, 0.9, 0.8, 0.3])

        metrics = compute_all_classification_metrics(y_true, y_pred, proba, n_params=2)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "mcc" in metrics
        assert "log_loss" in metrics
        assert "aic" in metrics
        assert "bic" in metrics
        assert "auc_roc" in metrics


# =============================================================================
# Test IRLS
# =============================================================================


class TestIRLS:
    """Tests for IRLS logistic regression solver."""

    def test_convergence(self):
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        Phi = lib.evaluate(X)
        coeffs, nll, n_iter, converged = fit_irls(Phi, y)
        assert converged
        assert n_iter < 100
        assert nll > 0

    def test_known_solution_intercept_only(self):
        """Intercept-only model: w = log(p/(1-p)) where p = mean(y)."""
        rng = np.random.RandomState(42)
        n = 500
        y = jnp.array((rng.rand(n) < 0.7).astype(float))
        Phi = jnp.ones((n, 1))  # intercept only

        coeffs, nll, n_iter, converged = fit_irls(Phi, y)
        assert converged

        p = float(jnp.mean(y))
        expected = np.log(p / (1 - p))
        assert abs(float(coeffs[0]) - expected) < 0.05

    def test_ridge_regularization(self):
        X, y = _make_binary_data(n=50)
        lib = _make_library()
        Phi = lib.evaluate(X)

        coeffs_unreg, _, _, _ = fit_irls(Phi, y)
        coeffs_reg, _, _, _ = fit_irls(Phi, y, regularization=10.0)

        # Regularised coefficients should be smaller
        assert float(jnp.sum(coeffs_reg**2)) < float(jnp.sum(coeffs_unreg**2))

    def test_predictions_reasonable(self):
        X, y = _make_binary_data(n=200)
        lib = _make_library()
        Phi = lib.evaluate(X)

        coeffs, _, _, _ = fit_irls(Phi, y)
        proba = _sigmoid(Phi @ coeffs)
        preds = (proba > 0.5).astype(float)
        acc = float(jnp.mean(preds == y))
        assert acc > 0.7  # Should do well on linearly separable data


# =============================================================================
# Test Logistic LASSO
# =============================================================================


class TestLogisticLasso:
    """Tests for FISTA-based logistic LASSO solver."""

    def test_sparsity(self):
        """High alpha should produce sparse solutions."""
        X, y = _make_binary_data(n=200)
        lib = _make_library()
        Phi = lib.evaluate(X)

        coeffs = fit_logistic_lasso(Phi, y, alpha=1.0)
        n_nonzero = int(jnp.sum(jnp.abs(coeffs) > 1e-8))
        assert n_nonzero < Phi.shape[1]

    def test_warm_start(self):
        """Warm start should converge faster / give same result."""
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        Phi = lib.evaluate(X)

        coeffs1 = fit_logistic_lasso(Phi, y, alpha=0.1)
        coeffs2 = fit_logistic_lasso(Phi, y, alpha=0.1, warm_start=coeffs1)
        # Should give essentially the same result
        assert float(jnp.max(jnp.abs(coeffs1 - coeffs2))) < 0.1

    def test_elastic_net(self):
        """l1_ratio < 1 should give elastic-net behaviour."""
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        Phi = lib.evaluate(X)

        coeffs = fit_logistic_lasso(Phi, y, alpha=0.1, l1_ratio=0.5)
        assert coeffs.shape[0] == Phi.shape[1]


# =============================================================================
# Test Classification Selection
# =============================================================================


class TestClassificationSelection:
    """Tests for classification selection strategies."""

    def test_fit_classification_subset(self):
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        Phi = lib.evaluate(X)

        result = fit_classification_subset(Phi, y, [0, 1, 2], lib.names, lib.complexities)
        assert isinstance(result, ClassificationResult)
        assert len(result.coefficients) == 3
        assert result.neg_log_likelihood > 0
        assert result.aic > 0
        assert result.bic > 0

    def test_greedy_forward(self):
        X, y = _make_binary_data(n=150)
        lib = _make_library()
        Phi = lib.evaluate(X)

        path = greedy_forward_classification(Phi, y, lib.names, lib.complexities, max_terms=3)
        assert isinstance(path, ClassificationPath)
        assert len(path.results) > 0
        assert path.best_index >= 0
        best = path.best
        assert isinstance(best, ClassificationResult)
        assert len(best.coefficients) <= 3

    def test_greedy_backward(self):
        X, y = _make_binary_data(n=100)
        lib = BasisLibrary(2).add_constant().add_linear()
        Phi = lib.evaluate(X)

        path = greedy_backward_classification(Phi, y, lib.names, lib.complexities)
        assert isinstance(path, ClassificationPath)
        assert len(path.results) > 0

    def test_exhaustive(self):
        X, y = _make_binary_data(n=100)
        lib = BasisLibrary(2).add_constant().add_linear()
        Phi = lib.evaluate(X)

        path = greedy_forward_classification(Phi, y, lib.names, lib.complexities, max_terms=3)
        assert len(path.results) > 0

    def test_lasso_path(self):
        from jaxsr.selection import lasso_path_classification

        X, y = _make_binary_data(n=100)
        lib = _make_library()
        Phi = lib.evaluate(X)

        path = lasso_path_classification(Phi, y, lib.names, lib.complexities, max_terms=3)
        assert isinstance(path, ClassificationPath)
        assert len(path.results) > 0

    def test_pareto_front(self):
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        Phi = lib.evaluate(X)

        path = greedy_forward_classification(
            Phi, y, lib.names, lib.complexities, max_terms=4, early_stop=False
        )
        pareto = compute_pareto_front_classification(path.results)
        assert len(pareto) > 0
        # Pareto front should be sorted by complexity
        for i in range(1, len(pareto)):
            assert pareto[i].complexity >= pareto[i - 1].complexity

    def test_classification_result_to_dict(self):
        X, y = _make_binary_data(n=50)
        lib = _make_library()
        Phi = lib.evaluate(X)

        result = fit_classification_subset(Phi, y, [0, 1], lib.names, lib.complexities)
        d = result.to_dict()
        assert "coefficients" in d
        assert "neg_log_likelihood" in d
        assert "converged" in d


# =============================================================================
# Test SymbolicClassifier
# =============================================================================


class TestSymbolicClassifier:
    """Tests for the main SymbolicClassifier class."""

    def test_fit_predict(self):
        X, y = _make_binary_data(n=200)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=4)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert preds.shape == (200,)
        acc = clf.score(X, y)
        assert acc > 0.7

    def test_predict_proba(self):
        X, y = _make_binary_data(n=200)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (200, 2)
        # Probabilities should sum to 1
        row_sums = jnp.sum(proba, axis=1)
        np.testing.assert_allclose(np.array(row_sums), 1.0, atol=1e-6)
        # All probabilities in [0, 1]
        assert float(jnp.min(proba)) >= 0
        assert float(jnp.max(proba)) <= 1

    def test_predict_log_proba(self):
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        log_proba = clf.predict_log_proba(X)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(np.array(log_proba), np.log(np.array(proba)), atol=1e-5)

    def test_decision_function(self):
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        logits = clf.decision_function(X)
        assert logits.shape == (100,)

    def test_unfitted_raises(self):
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib)

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(jnp.ones((5, 2)))
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(jnp.ones((5, 2)))
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = clf.expression_
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = clf.coefficients_
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.score(jnp.ones((5, 2)), jnp.ones(5))

    def test_no_library_raises(self):
        clf = SymbolicClassifier()
        with pytest.raises(ValueError, match="basis_library"):
            clf.fit(jnp.ones((10, 2)), jnp.ones(10))

    def test_shape_mismatch_raises(self):
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib)
        with pytest.raises(ValueError, match="same number of samples"):
            clf.fit(jnp.ones((10, 2)), jnp.ones(5))

    def test_single_class_raises(self):
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib)
        with pytest.raises(ValueError, match="at least 2"):
            clf.fit(jnp.ones((10, 2)), jnp.zeros(10))

    def test_feature_mismatch_raises(self):
        lib = _make_library(n_features=3)
        clf = SymbolicClassifier(basis_library=lib)
        with pytest.raises(ValueError, match="features"):
            clf.fit(jnp.ones((10, 2)), jnp.array([0, 1] * 5))

    def test_fitted_properties(self):
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        assert isinstance(clf.expression_, str)
        assert isinstance(clf.coefficients_, jnp.ndarray)
        assert isinstance(clf.selected_features_, list)
        assert len(clf.classes_) == 2
        assert isinstance(clf.metrics_, dict)
        assert "accuracy" in clf.metrics_

    def test_strategies(self):
        """All selection strategies should work."""
        X, y = _make_binary_data(n=100)
        lib = BasisLibrary(2).add_constant().add_linear()

        for strategy in ["greedy_forward", "greedy_backward", "lasso_path"]:
            clf = SymbolicClassifier(basis_library=lib, max_terms=3, strategy=strategy)
            clf.fit(X, y)
            acc = clf.score(X, y)
            assert acc > 0.6, f"Strategy {strategy} got accuracy {acc}"

    def test_regularization(self):
        X, y = _make_binary_data(n=100)
        lib = _make_library()

        clf_unreg = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf_unreg.fit(X, y)

        clf_reg = SymbolicClassifier(basis_library=lib, max_terms=3, regularization=1.0)
        clf_reg.fit(X, y)

        # Both should work
        assert clf_unreg.score(X, y) > 0.5
        assert clf_reg.score(X, y) > 0.5

    def test_repr(self):
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        assert "not fitted" in repr(clf)

        X, y = _make_binary_data(n=50)
        clf.fit(X, y)
        assert "fitted" in repr(clf)

    def test_summary(self):
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        s = clf.summary()
        assert "Binary" in s
        assert "accuracy" in s.lower() or "Accuracy" in s or "Train accuracy" in s


# =============================================================================
# Test Multiclass
# =============================================================================


class TestMulticlass:
    """Tests for multiclass (OVR) classification."""

    def test_multiclass_fit_predict(self):
        X, y = _make_multiclass_data(n=300)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert preds.shape == (300,)
        assert set(np.unique(np.array(preds))).issubset({0, 1, 2})
        acc = clf.score(X, y)
        assert acc > 0.5

    def test_multiclass_predict_proba_sums_to_one(self):
        X, y = _make_multiclass_data(n=300)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (300, 3)
        row_sums = jnp.sum(proba, axis=1)
        np.testing.assert_allclose(np.array(row_sums), 1.0, atol=1e-6)

    def test_multiclass_expression(self):
        X, y = _make_multiclass_data(n=150)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=2)
        clf.fit(X, y)

        expr = clf.expression_
        assert isinstance(expr, dict)
        assert len(expr) == 3

    def test_multiclass_summary(self):
        X, y = _make_multiclass_data(n=150)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=2)
        clf.fit(X, y)

        s = clf.summary()
        assert "Multiclass" in s or "OVR" in s


# =============================================================================
# Test Expression Output
# =============================================================================


class TestExpressionOutput:
    """Tests for to_sympy, to_latex, to_callable."""

    def test_to_sympy(self):
        pytest.importorskip("sympy")
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        expr = clf.to_sympy()
        assert expr is not None

    def test_to_latex(self):
        pytest.importorskip("sympy")
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        latex = clf.to_latex()
        assert isinstance(latex, str)
        assert len(latex) > 0

    def test_to_callable(self):
        X, y = _make_binary_data(n=200)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        func = clf.to_callable()
        X_np = np.array(X)
        proba_func = func(X_np)
        proba_clf = np.array(clf.predict_proba(X))

        # callable should match predict_proba
        np.testing.assert_allclose(proba_func, proba_clf, atol=1e-5)

    def test_to_callable_multiclass(self):
        X, y = _make_multiclass_data(n=150)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=2)
        clf.fit(X, y)

        func = clf.to_callable()
        X_np = np.array(X)
        proba_func = func(X_np)
        proba_clf = np.array(clf.predict_proba(X))

        np.testing.assert_allclose(proba_func, proba_clf, atol=1e-5)

    def test_to_sympy_multiclass(self):
        pytest.importorskip("sympy")
        X, y = _make_multiclass_data(n=150)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=2)
        clf.fit(X, y)

        expr = clf.to_sympy()
        assert isinstance(expr, dict)
        assert len(expr) == 3


# =============================================================================
# Test Save/Load
# =============================================================================


class TestSaveLoad:
    """Tests for model persistence."""

    def test_save_load_binary(self):
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        clf.save(filepath)

        clf2 = SymbolicClassifier.load(filepath)

        proba1 = clf.predict_proba(X)
        proba2 = clf2.predict_proba(X)
        np.testing.assert_allclose(np.array(proba1), np.array(proba2), atol=1e-6)

    def test_save_load_multiclass(self):
        X, y = _make_multiclass_data(n=150)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=2)
        clf.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        clf.save(filepath)
        clf2 = SymbolicClassifier.load(filepath)

        preds1 = clf.predict(X)
        preds2 = clf2.predict(X)
        np.testing.assert_array_equal(np.array(preds1), np.array(preds2))

    def test_save_creates_valid_json(self):
        X, y = _make_binary_data(n=50)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=2)
        clf.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        clf.save(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert data["model_type"] == "SymbolicClassifier"
        assert data["is_binary"] is True


# =============================================================================
# Test Classification UQ
# =============================================================================


class TestClassificationUQ:
    """Tests for classification uncertainty quantification."""

    def test_wald_intervals(self):
        X, y = _make_binary_data(n=200)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        intervals = clf.coefficient_intervals(alpha=0.05)
        assert isinstance(intervals, dict)
        for _name, (est, lo, hi, se) in intervals.items():
            assert lo <= est <= hi
            assert se >= 0

    def test_wald_intervals_standalone(self):
        X, y = _make_binary_data(n=200)
        lib = _make_library()
        Phi = lib.evaluate(X)
        coeffs, _, _, _ = fit_irls(Phi, y)

        intervals = classification_coefficient_intervals(Phi, y, coeffs, lib.names, alpha=0.05)
        assert len(intervals) == len(lib.names)

    def test_bootstrap_coefficients(self):
        X, y = _make_binary_data(n=100)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        result = bootstrap_classification_coefficients(clf, n_bootstrap=20, seed=42)
        assert "coefficients" in result
        assert "lower" in result
        assert "upper" in result
        assert "names" in result

    def test_conformal_prediction_sets(self):
        X, y = _make_binary_data(n=200)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        # Use training data as calibration (not ideal, but tests the API)
        result = clf.predict_conformal(X[:10], alpha=0.1)
        assert "prediction_sets" in result
        assert "quantile" in result
        assert len(result["prediction_sets"]) == 10
        for pset in result["prediction_sets"]:
            assert isinstance(pset, set)
            assert len(pset) >= 1

    def test_conformal_standalone(self):
        X, y = _make_binary_data(n=200)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=3)
        clf.fit(X, y)

        result = conformal_classification_split(clf, X[:100], y[:100], X[100:], alpha=0.1)
        assert "prediction_sets" in result
        assert len(result["prediction_sets"]) == 100

    def test_calibration_curve(self):
        rng = np.random.RandomState(42)
        y_true = jnp.array(rng.randint(0, 2, 500))
        y_prob = jnp.array(rng.rand(500))

        result = calibration_curve(y_true, y_prob, n_bins=5)
        assert "fraction_of_positives" in result
        assert "mean_predicted_value" in result
        assert "bin_counts" in result
        assert len(result["fraction_of_positives"]) <= 5
        assert all(0 <= f <= 1 for f in result["fraction_of_positives"])

    def test_multiclass_coefficient_intervals_raises(self):
        X, y = _make_multiclass_data(n=150)
        lib = _make_library()
        clf = SymbolicClassifier(basis_library=lib, max_terms=2)
        clf.fit(X, y)

        with pytest.raises(ValueError, match="binary"):
            clf.coefficient_intervals()


# =============================================================================
# Test Convenience Function
# =============================================================================


class TestConvenienceFunction:
    """Tests for fit_symbolic_classification."""

    def test_basic(self):
        X, y = _make_binary_data(n=100)
        clf = fit_symbolic_classification(X, y, feature_names=["x0", "x1"], max_terms=3)
        assert isinstance(clf, SymbolicClassifier)
        assert clf.score(X, y) > 0.6

    def test_multiclass(self):
        X, y = _make_multiclass_data(n=150)
        clf = fit_symbolic_classification(X, y, max_terms=2)
        assert isinstance(clf, SymbolicClassifier)
        assert len(clf.classes_) == 3


# =============================================================================
# Test Sigmoid
# =============================================================================


class TestSigmoid:
    """Tests for the sigmoid function."""

    def test_basic_values(self):
        assert abs(float(_sigmoid(jnp.array(0.0))) - 0.5) < 1e-7
        assert float(_sigmoid(jnp.array(10.0))) > 0.999
        assert float(_sigmoid(jnp.array(-10.0))) < 0.001

    def test_numerical_stability_large(self):
        """Should not overflow for large positive/negative inputs."""
        val_pos = _sigmoid(jnp.array(500.0))
        val_neg = _sigmoid(jnp.array(-500.0))
        assert jnp.isfinite(val_pos)
        assert jnp.isfinite(val_neg)

    def test_vectorized(self):
        x = jnp.array([-5.0, 0.0, 5.0])
        result = _sigmoid(x)
        assert result.shape == (3,)
        assert float(result[0]) < 0.01
        assert abs(float(result[1]) - 0.5) < 1e-7
        assert float(result[2]) > 0.99
