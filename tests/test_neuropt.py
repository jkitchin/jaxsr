"""Tests for neuropt (LLM-guided hyperparameter search) integration."""

from __future__ import annotations

import numpy as np
import pytest

neuropt = pytest.importorskip("neuropt")

from jaxsr import BasisLibrary, SymbolicRegressor


@pytest.fixture()
def library():
    return BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(max_degree=2)


@pytest.fixture()
def X():
    rng = np.random.RandomState(42)
    return rng.randn(30, 2)


@pytest.fixture()
def y(X):
    return 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.1 * np.random.RandomState(42).randn(30)


class TestNeuroptIntegration:
    """Test that neuropt.ArchSearch.from_model works with JAXSR estimators."""

    def test_from_model_detects_sklearn_compat(self, library):
        from neuropt.introspect import is_sklearn_compatible

        model = SymbolicRegressor(basis_library=library)
        assert is_sklearn_compatible(model)

    def test_from_model_introspects_params(self, library):
        from neuropt.introspect import introspect_sklearn

        model = SymbolicRegressor(basis_library=library, max_terms=5)
        info = introspect_sklearn(model)
        assert info["model_type"] == "SymbolicRegressor"
        assert "max_terms" in info["tunable_params"]
        assert info["tunable_params"]["max_terms"] == 5

    def test_search_runs(self, library, X, y, tmp_path):
        model = SymbolicRegressor(basis_library=library, max_terms=3)

        def train_fn(config):
            m = config["model"]
            m.fit(X, y)
            y_pred = np.asarray(m.predict(X))
            mse = float(np.mean((y - y_pred) ** 2))
            return {"score": mse}

        search = neuropt.ArchSearch.from_model(
            model,
            train_fn,
            backend="none",
            log_path=str(tmp_path / "search.jsonl"),
        )
        search.run(max_evals=3)
        assert search.total_experiments == 3
        assert search.best_score < float("inf")
