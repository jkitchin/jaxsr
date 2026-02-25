"""Tests for scikit-learn estimator protocol compatibility."""

from __future__ import annotations

import numpy as np
import pytest

from jaxsr import BasisLibrary, MultiOutputSymbolicRegressor, SymbolicRegressor
from jaxsr.classifier import SymbolicClassifier
from jaxsr.regressor import _clone_estimator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def library():
    return BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(max_degree=2)


@pytest.fixture()
def model(library):
    return SymbolicRegressor(basis_library=library, max_terms=3)


@pytest.fixture()
def classifier(library):
    return SymbolicClassifier(basis_library=library, max_terms=3)


@pytest.fixture()
def multi_model(model):
    return MultiOutputSymbolicRegressor(estimator=model)


@pytest.fixture()
def X():
    rng = np.random.RandomState(42)
    return rng.randn(30, 2)


@pytest.fixture()
def y(X):
    return 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.1 * np.random.RandomState(42).randn(30)


@pytest.fixture()
def y_class(X):
    return (X[:, 0] + X[:, 1] > 0).astype(int)


@pytest.fixture()
def Y(X):
    rng = np.random.RandomState(42)
    return np.column_stack(
        [
            2.0 * X[:, 0] + 0.1 * rng.randn(30),
            3.0 * X[:, 1] + 0.1 * rng.randn(30),
        ]
    )


# ===========================================================================
# Protocol tests (no sklearn needed)
# ===========================================================================


class TestGetParams:
    """Test get_params returns all constructor parameters."""

    def test_symbolic_regressor_params(self, model):
        params = model.get_params(deep=False)
        expected_keys = {
            "basis_library",
            "max_terms",
            "strategy",
            "information_criterion",
            "cv_folds",
            "regularization",
            "constraints",
            "random_state",
            "param_optimizer",
            "param_optimization_budget",
            "constraint_enforcement",
            "constraint_selection_weight",
        }
        assert set(params.keys()) == expected_keys

    def test_symbolic_regressor_values(self, model):
        params = model.get_params(deep=False)
        assert params["max_terms"] == 3
        assert params["strategy"] == "greedy_forward"
        assert params["information_criterion"] == "bic"

    def test_classifier_params(self, classifier):
        params = classifier.get_params(deep=False)
        expected_keys = {
            "basis_library",
            "max_terms",
            "strategy",
            "information_criterion",
            "regularization",
            "constraints",
            "random_state",
            "max_iter",
            "tol",
        }
        assert set(params.keys()) == expected_keys

    def test_multi_output_params_shallow(self, multi_model):
        params = multi_model.get_params(deep=False)
        assert set(params.keys()) == {"estimator", "target_names"}
        assert isinstance(params["estimator"], SymbolicRegressor)

    def test_multi_output_params_deep(self, multi_model):
        params = multi_model.get_params(deep=True)
        assert "estimator__max_terms" in params
        assert params["estimator__max_terms"] == 3
        assert "estimator__strategy" in params


class TestSetParams:
    """Test set_params modifies parameters correctly."""

    def test_set_direct_param(self, model):
        model.set_params(max_terms=8)
        assert model.max_terms == 8
        assert model.get_params()["max_terms"] == 8

    def test_set_multiple_params(self, model):
        model.set_params(max_terms=10, strategy="exhaustive")
        assert model.max_terms == 10
        assert model.strategy == "exhaustive"

    def test_set_nested_param(self, multi_model):
        multi_model.set_params(estimator__max_terms=8)
        assert multi_model.estimator.max_terms == 8

    def test_set_invalid_param_raises(self, model):
        with pytest.raises(ValueError, match="Invalid parameter"):
            model.set_params(bad_param=1)

    def test_set_invalid_nested_prefix_raises(self, multi_model):
        with pytest.raises(ValueError, match="Invalid parameter"):
            multi_model.set_params(nonexistent__foo=1)

    def test_returns_self(self, model):
        result = model.set_params(max_terms=7)
        assert result is model


class TestCloneEstimator:
    """Test _clone_estimator creates correct unfitted copies."""

    def test_clone_symbolic_regressor(self, model):
        clone = _clone_estimator(model)
        assert type(clone) is SymbolicRegressor
        assert clone.max_terms == model.max_terms
        assert clone.strategy == model.strategy
        assert clone.basis_library is model.basis_library
        assert not clone._is_fitted

    def test_clone_multi_output(self, multi_model):
        clone = _clone_estimator(multi_model)
        assert type(clone) is MultiOutputSymbolicRegressor
        assert clone.estimator is multi_model.estimator
        assert not clone._is_fitted

    def test_clone_classifier(self, classifier):
        clone = _clone_estimator(classifier)
        assert type(clone) is SymbolicClassifier
        assert clone.max_terms == classifier.max_terms
        assert not clone._is_fitted


class TestRepr:
    """Test __repr__ output."""

    def test_repr_regressor(self, model):
        r = repr(model)
        assert r.startswith("SymbolicRegressor(")
        assert "max_terms=3" in r

    def test_repr_classifier(self, classifier):
        r = repr(classifier)
        assert r.startswith("SymbolicClassifier(")

    def test_repr_multi_output(self, multi_model):
        r = repr(multi_model)
        assert r.startswith("MultiOutputSymbolicRegressor(")


# ===========================================================================
# Sklearn integration tests (skip if sklearn not installed)
# ===========================================================================

sklearn = pytest.importorskip("sklearn")


class TestSklearnClone:
    """Test sklearn.base.clone works with JAXSR estimators."""

    def test_clone_regressor(self, model):
        from sklearn.base import clone

        cloned = clone(model)
        assert type(cloned) is SymbolicRegressor
        assert cloned.max_terms == model.max_terms
        assert not cloned._is_fitted

    def test_clone_classifier(self, classifier):
        from sklearn.base import clone

        cloned = clone(classifier)
        assert type(cloned) is SymbolicClassifier
        assert cloned.max_terms == classifier.max_terms

    def test_clone_multi_output(self, multi_model):
        from sklearn.base import clone

        cloned = clone(multi_model)
        assert type(cloned) is MultiOutputSymbolicRegressor
        assert cloned.estimator.max_terms == multi_model.estimator.max_terms


class TestCrossValScore:
    """Test sklearn cross_val_score works."""

    def test_cross_val_score_regressor(self, model, X, y):
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(model, X, y, cv=3, scoring="r2")
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_cross_val_score_classifier(self, classifier, X, y_class):
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(classifier, X, y_class, cv=3, scoring="accuracy")
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestGridSearchCV:
    """Test sklearn GridSearchCV works."""

    def test_grid_search_max_terms(self, library, X, y):
        from sklearn.model_selection import GridSearchCV

        model = SymbolicRegressor(basis_library=library)
        grid = GridSearchCV(
            model,
            param_grid={"max_terms": [2, 3]},
            cv=2,
            scoring="r2",
        )
        grid.fit(X, y)
        assert grid.best_params_["max_terms"] in [2, 3]
        assert hasattr(grid, "best_score_")


class TestPipeline:
    """Test sklearn Pipeline works."""

    def test_pipeline_fit_predict(self, library, X, y):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("sr", SymbolicRegressor(basis_library=library, max_terms=3)),
            ]
        )
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        assert y_pred.shape == y.shape
