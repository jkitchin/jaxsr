"""
Tests for categorical / discrete variable support in JAXSR.
"""

from __future__ import annotations

import json
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr import BasisLibrary, SymbolicRegressor
from jaxsr.sampling import (
    grid_sample,
    halton_sample,
    latin_hypercube_sample,
    sobol_sample,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_library():
    """BasisLibrary with 1 continuous + 1 categorical (3 levels) feature."""
    return BasisLibrary(
        n_features=2,
        feature_names=["T", "catalyst"],
        feature_types=["continuous", "categorical"],
        categories={1: [0, 1, 2]},
    )


@pytest.fixture
def two_cat_library():
    """BasisLibrary with 2 categorical features."""
    return BasisLibrary(
        n_features=2,
        feature_names=["color", "size"],
        feature_types=["categorical", "categorical"],
        categories={0: [0, 1, 2], 1: [0, 1]},
    )


@pytest.fixture
def mixed_data():
    """Synthetic dataset: y = 2*T + 5*I(catalyst=1) + 10*I(catalyst=2)."""
    rng = np.random.RandomState(42)
    n = 60
    T = rng.uniform(300, 500, n)
    catalyst = rng.choice([0, 1, 2], n)
    y = 2.0 * T + 5.0 * (catalyst == 1) + 10.0 * (catalyst == 2) + rng.normal(0, 0.5, n)
    X = np.column_stack([T, catalyst])
    return X, y


# ---------------------------------------------------------------------------
# BasisLibrary feature types
# ---------------------------------------------------------------------------


class TestBasisLibraryFeatureTypes:
    def test_default_all_continuous(self):
        lib = BasisLibrary(n_features=3)
        assert lib.feature_types == ["continuous", "continuous", "continuous"]
        assert lib.continuous_indices == [0, 1, 2]
        assert lib.categorical_indices == []

    def test_mixed_types(self, mixed_library):
        assert mixed_library.feature_types == ["continuous", "categorical"]
        assert mixed_library.continuous_indices == [0]
        assert mixed_library.categorical_indices == [1]

    def test_invalid_feature_type_raises(self):
        with pytest.raises(ValueError, match="Invalid feature type"):
            BasisLibrary(
                n_features=1,
                feature_types=["ordinal"],
            )

    def test_missing_categories_raises(self):
        with pytest.raises(ValueError, match="no categories provided"):
            BasisLibrary(
                n_features=1,
                feature_types=["categorical"],
            )

    def test_feature_types_length_mismatch(self):
        with pytest.raises(ValueError, match="must match n_features"):
            BasisLibrary(
                n_features=2,
                feature_types=["continuous"],
            )


# ---------------------------------------------------------------------------
# Categorical indicators
# ---------------------------------------------------------------------------


class TestCategoricalIndicators:
    def test_add_indicators_count(self, mixed_library):
        """3 categories -> 2 indicator functions (reference encoding)."""
        mixed_library.add_categorical_indicators()
        indicator_names = [bf.name for bf in mixed_library.basis_functions]
        assert "I(catalyst=1)" in indicator_names
        assert "I(catalyst=2)" in indicator_names
        assert len(indicator_names) == 2  # K-1

    def test_indicator_evaluation(self, mixed_library):
        mixed_library.add_categorical_indicators()
        X = jnp.array([[300.0, 0.0], [400.0, 1.0], [500.0, 2.0]])
        Phi = mixed_library.evaluate(X)

        # I(catalyst=1): should be [0, 1, 0]
        np.testing.assert_array_equal(Phi[:, 0], [0.0, 1.0, 0.0])
        # I(catalyst=2): should be [0, 0, 1]
        np.testing.assert_array_equal(Phi[:, 1], [0.0, 0.0, 1.0])

    def test_indicator_on_non_categorical_raises(self, mixed_library):
        with pytest.raises(ValueError, match="not categorical"):
            mixed_library.add_categorical_indicators(features=[0])

    def test_multiple_categorical_features(self, two_cat_library):
        two_cat_library.add_categorical_indicators()
        names = [bf.name for bf in two_cat_library.basis_functions]
        # color: 3 cats -> 2 indicators; size: 2 cats -> 1 indicator
        assert len(names) == 3
        assert "I(color=1)" in names
        assert "I(color=2)" in names
        assert "I(size=1)" in names


# ---------------------------------------------------------------------------
# Categorical interactions
# ---------------------------------------------------------------------------


class TestCategoricalInteractions:
    def test_interaction_count(self, mixed_library):
        mixed_library.add_categorical_interactions()
        names = [bf.name for bf in mixed_library.basis_functions]
        # 2 indicator levels * 1 continuous feature = 2 interactions
        assert len(names) == 2
        assert "I(catalyst=1)*T" in names
        assert "I(catalyst=2)*T" in names

    def test_interaction_evaluation(self, mixed_library):
        mixed_library.add_categorical_interactions()
        X = jnp.array([[300.0, 0.0], [400.0, 1.0], [500.0, 2.0]])
        Phi = mixed_library.evaluate(X)

        # I(catalyst=1)*T: [0*300, 1*400, 0*500] = [0, 400, 0]
        np.testing.assert_array_almost_equal(Phi[:, 0], [0.0, 400.0, 0.0])
        # I(catalyst=2)*T: [0*300, 0*400, 1*500] = [0, 0, 500]
        np.testing.assert_array_almost_equal(Phi[:, 1], [0.0, 0.0, 500.0])


# ---------------------------------------------------------------------------
# Continuous methods skip categorical
# ---------------------------------------------------------------------------


class TestContinuousMethodsSkipCategorical:
    def test_add_linear_skips_categorical(self, mixed_library):
        mixed_library.add_linear()
        names = [bf.name for bf in mixed_library.basis_functions]
        assert "T" in names
        assert "catalyst" not in names

    def test_add_polynomials_skips_categorical(self, mixed_library):
        mixed_library.add_polynomials(max_degree=3)
        names = [bf.name for bf in mixed_library.basis_functions]
        assert "T^2" in names
        assert "T^3" in names
        # No polynomial of the categorical feature
        assert not any("catalyst^" in n for n in names)

    def test_add_transcendental_skips_categorical(self, mixed_library):
        mixed_library.add_transcendental(["log", "exp"])
        names = [bf.name for bf in mixed_library.basis_functions]
        assert "log(T)" in names
        assert "exp(T)" in names
        assert not any("catalyst" in n for n in names)

    def test_add_interactions_skips_categorical(self, mixed_library):
        """With only 1 continuous feature, no continuous interactions possible."""
        mixed_library.add_interactions(max_order=2)
        assert len(mixed_library.basis_functions) == 0

    def test_add_ratios_skips_categorical(self, mixed_library):
        mixed_library.add_ratios()
        # 1 continuous feature -> no ratios possible
        assert len(mixed_library.basis_functions) == 0

    def test_build_default_includes_indicators(self, mixed_library):
        mixed_library.build_default()
        names = [bf.name for bf in mixed_library.basis_functions]
        # Should include indicators and cat*cont interactions
        assert "I(catalyst=1)" in names
        assert "I(catalyst=2)" in names
        assert "I(catalyst=1)*T" in names
        assert "I(catalyst=2)*T" in names
        # Should NOT include categorical polynomials
        assert not any("catalyst^" in n for n in names)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestCategoricalSerialization:
    def test_to_dict_includes_types(self, mixed_library):
        mixed_library.add_categorical_indicators()
        d = mixed_library.to_dict()
        assert d["feature_types"] == ["continuous", "categorical"]
        assert "1" in d["categories"]  # JSON keys are strings
        assert d["categories"]["1"] == [0, 1, 2]

    def test_roundtrip(self, mixed_library):
        mixed_library.add_constant().add_linear().add_categorical_indicators()
        d = mixed_library.to_dict()
        json_str = json.dumps(d)
        restored = BasisLibrary.from_dict(json.loads(json_str))

        assert restored.feature_types == mixed_library.feature_types
        assert restored.categories == mixed_library.categories
        assert len(restored) == len(mixed_library)
        assert restored.names == mixed_library.names

    def test_indicator_from_dict_evaluates(self, mixed_library):
        mixed_library.add_categorical_indicators()
        d = mixed_library.to_dict()
        restored = BasisLibrary.from_dict(d)

        X = jnp.array([[300.0, 1.0], [400.0, 2.0]])
        Phi_orig = mixed_library.evaluate(X)
        Phi_restored = restored.evaluate(X)
        np.testing.assert_array_almost_equal(Phi_orig, Phi_restored)

    def test_save_load_roundtrip(self, mixed_library):
        mixed_library.add_constant().add_linear().add_categorical_indicators()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mixed_library.save(f.name)
            loaded = BasisLibrary.load(f.name)
        assert loaded.feature_types == mixed_library.feature_types
        assert loaded.names == mixed_library.names


# ---------------------------------------------------------------------------
# Sampling with discrete dims
# ---------------------------------------------------------------------------


class TestCategoricalSampling:
    def test_latin_hypercube_discrete(self):
        bounds = [(300, 500), (0, 2)]
        samples = latin_hypercube_sample(20, bounds, random_state=42, discrete_dims={1: [0, 1, 2]})
        # All values in dim 1 should be exactly 0, 1, or 2
        assert set(np.array(samples[:, 1]).tolist()).issubset({0.0, 1.0, 2.0})
        # Dim 0 should be continuous in [300, 500]
        assert np.all(samples[:, 0] >= 300)
        assert np.all(samples[:, 0] <= 500)

    def test_sobol_discrete(self):
        bounds = [(0, 10), (0, 1)]
        samples = sobol_sample(16, bounds, random_state=42, discrete_dims={1: [0, 1]})
        assert set(np.array(samples[:, 1]).tolist()).issubset({0.0, 1.0})

    def test_halton_discrete(self):
        bounds = [(0, 10), (0, 2)]
        samples = halton_sample(20, bounds, random_state=42, discrete_dims={1: [0, 1, 2]})
        assert set(np.array(samples[:, 1]).tolist()).issubset({0.0, 1.0, 2.0})

    def test_grid_sample_discrete(self):
        bounds = [(0, 10), (0, 2)]
        samples = grid_sample(5, bounds, discrete_dims={1: [0, 1, 2]})
        # 5 continuous * 3 discrete = 15 points
        assert samples.shape == (15, 2)
        assert set(np.array(samples[:, 1]).tolist()) == {0.0, 1.0, 2.0}


# ---------------------------------------------------------------------------
# Integration: end-to-end regression with categorical features
# ---------------------------------------------------------------------------


class TestCategoricalRegression:
    def test_fit_with_categorical(self, mixed_data):
        X, y = mixed_data
        library = (
            BasisLibrary(
                n_features=2,
                feature_names=["T", "catalyst"],
                feature_types=["continuous", "categorical"],
                categories={1: [0, 1, 2]},
            )
            .add_constant()
            .add_linear()
            .add_categorical_indicators()
        )
        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(X, y)

        # Should achieve high R²
        r2 = model.score(X, y)
        assert r2 > 0.95

    def test_to_callable_with_indicators(self, mixed_data):
        X, y = mixed_data
        library = (
            BasisLibrary(
                n_features=2,
                feature_names=["T", "catalyst"],
                feature_types=["continuous", "categorical"],
                categories={1: [0, 1, 2]},
            )
            .add_constant()
            .add_linear()
            .add_categorical_indicators()
        )
        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(X, y)

        predict_fn = model.to_callable()
        y_pred_jax = np.array(model.predict(X))
        y_pred_np = predict_fn(np.array(X))
        np.testing.assert_array_almost_equal(y_pred_jax, y_pred_np, decimal=4)

    def test_fit_mixed_with_interactions(self, mixed_data):
        X, y = mixed_data
        library = (
            BasisLibrary(
                n_features=2,
                feature_names=["T", "catalyst"],
                feature_types=["continuous", "categorical"],
                categories={1: [0, 1, 2]},
            )
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
            .add_categorical_indicators()
            .add_categorical_interactions()
        )
        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(X, y)
        assert model.score(X, y) > 0.95

    def test_build_default_with_categorical(self, mixed_data):
        """build_default should auto-add indicators and cat interactions."""
        X, y = mixed_data
        library = BasisLibrary(
            n_features=2,
            feature_names=["T", "catalyst"],
            feature_types=["continuous", "categorical"],
            categories={1: [0, 1, 2]},
        ).build_default()

        model = SymbolicRegressor(basis_library=library, max_terms=5)
        model.fit(X, y)
        assert model.score(X, y) > 0.90
