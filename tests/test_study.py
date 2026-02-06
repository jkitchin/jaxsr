"""Tests for DOEStudy persistence and workflow."""

from __future__ import annotations

import json
import os
import zipfile

import numpy as np
import pytest

from jaxsr.study import _SCHEMA_VERSION, DOEStudy, Iteration

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_study():
    """A simple 2-factor continuous study."""
    return DOEStudy(
        name="test_study",
        factor_names=["x1", "x2"],
        bounds=[(0, 1), (0, 1)],
    )


@pytest.fixture
def categorical_study():
    """A study with mixed continuous/categorical factors."""
    return DOEStudy(
        name="cat_study",
        factor_names=["temperature", "pressure", "catalyst"],
        bounds=[(300, 500), (1, 10), (0, 2)],
        feature_types=["continuous", "continuous", "categorical"],
        categories={2: ["A", "B", "C"]},
        description="Testing categorical support",
    )


@pytest.fixture
def tmp_path_file(tmp_path):
    """Return a temporary file path for saving studies."""
    return str(tmp_path / "test_study.jaxsr")


# =============================================================================
# Construction and validation
# =============================================================================


class TestDOEStudyConstruction:
    """Test DOEStudy initialization and validation."""

    def test_basic_creation(self, simple_study):
        assert simple_study.name == "test_study"
        assert simple_study.n_factors == 2
        assert simple_study.factor_names == ["x1", "x2"]
        assert simple_study.n_observations == 0
        assert not simple_study.is_fitted
        assert simple_study.X is None
        assert simple_study.y is None
        assert simple_study.design_points is None

    def test_categorical_creation(self, categorical_study):
        assert categorical_study.n_factors == 3
        assert categorical_study.feature_types == ["continuous", "continuous", "categorical"]
        assert categorical_study.categories == {2: ["A", "B", "C"]}
        assert categorical_study.description == "Testing categorical support"

    def test_mismatched_bounds_raises(self):
        with pytest.raises(ValueError, match="bounds"):
            DOEStudy(
                name="bad",
                factor_names=["x1", "x2"],
                bounds=[(0, 1)],  # only 1 bound for 2 factors
            )

    def test_mismatched_feature_types_raises(self):
        with pytest.raises(ValueError, match="feature_types"):
            DOEStudy(
                name="bad",
                factor_names=["x1", "x2"],
                bounds=[(0, 1), (0, 1)],
                feature_types=["continuous"],  # only 1 type for 2 factors
            )

    def test_invalid_feature_type_raises(self):
        with pytest.raises(ValueError, match="must be 'continuous' or 'categorical'"):
            DOEStudy(
                name="bad",
                factor_names=["x1"],
                bounds=[(0, 1)],
                feature_types=["numeric"],
            )

    def test_empty_categories_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            DOEStudy(
                name="bad",
                factor_names=["x1"],
                bounds=[(0, 1)],
                feature_types=["categorical"],
                categories={0: []},
            )

    def test_metadata_populated(self, simple_study):
        meta = simple_study.meta
        assert meta["schema_version"] == _SCHEMA_VERSION
        assert "created" in meta
        assert "modified" in meta
        assert "jaxsr_version" in meta

    def test_repr(self, simple_study):
        r = repr(simple_study)
        assert "test_study" in r
        assert "factors=2" in r
        assert "observations=0" in r
        assert "unfitted" in r


# =============================================================================
# Design creation
# =============================================================================


class TestDesignCreation:
    """Test experimental design generation."""

    def test_latin_hypercube(self, simple_study):
        X = simple_study.create_design(method="latin_hypercube", n_points=10, random_state=42)
        assert X.shape == (10, 2)
        assert np.all(X >= 0) and np.all(X <= 1)
        assert simple_study.design_points is not None
        assert len(simple_study._observation_status) == 10
        assert all(s == "pending" for s in simple_study._observation_status)

    def test_sobol(self, simple_study):
        X = simple_study.create_design(method="sobol", n_points=8, random_state=42)
        assert X.shape == (8, 2)

    def test_halton(self, simple_study):
        X = simple_study.create_design(method="halton", n_points=10, random_state=42)
        assert X.shape == (10, 2)

    def test_grid(self, simple_study):
        X = simple_study.create_design(method="grid", n_points=5, n_per_dim=5)
        assert X.shape[1] == 2
        assert X.shape[0] == 25  # 5x5 grid

    def test_unknown_method_raises(self, simple_study):
        with pytest.raises(ValueError, match="Unknown design method"):
            simple_study.create_design(method="magic")

    def test_pending_points(self, simple_study):
        simple_study.create_design(method="latin_hypercube", n_points=5, random_state=42)
        pending = simple_study.pending_points
        assert pending is not None
        assert len(pending) == 5

    def test_pending_points_no_design(self, simple_study):
        assert simple_study.pending_points is None

    def test_completed_points_initially_empty(self, simple_study):
        simple_study.create_design(method="latin_hypercube", n_points=5, random_state=42)
        completed = simple_study.completed_points
        assert completed is not None
        assert len(completed) == 0

    def test_design_config_stored(self, simple_study):
        simple_study.create_design(method="latin_hypercube", n_points=10, random_state=42)
        assert simple_study._design_config["method"] == "latin_hypercube"
        assert simple_study._design_config["n_points"] == 10
        assert simple_study._design_config["random_state"] == 42


# =============================================================================
# Observation management
# =============================================================================


class TestObservations:
    """Test adding observations and tracking status."""

    def test_add_observations(self, simple_study):
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([1.0, 2.0])
        simple_study.add_observations(X, y)

        assert simple_study.n_observations == 2
        np.testing.assert_array_equal(simple_study.X, X)
        np.testing.assert_array_equal(simple_study.y, y)

    def test_add_observations_incremental(self, simple_study):
        X1 = np.array([[0.1, 0.2]])
        y1 = np.array([1.0])
        simple_study.add_observations(X1, y1)
        assert simple_study.n_observations == 1

        X2 = np.array([[0.3, 0.4]])
        y2 = np.array([2.0])
        simple_study.add_observations(X2, y2)
        assert simple_study.n_observations == 2

    def test_add_observations_marks_design_completed(self, simple_study):
        X_design = simple_study.create_design(method="latin_hypercube", n_points=5, random_state=42)
        # Add first 2 design points as observations
        simple_study.add_observations(X_design[:2], np.array([1.0, 2.0]))

        assert sum(1 for s in simple_study._observation_status if s == "completed") == 2
        assert sum(1 for s in simple_study._observation_status if s == "pending") == 3
        assert len(simple_study.pending_points) == 3
        assert len(simple_study.completed_points) == 2

    def test_add_observations_shape_mismatch_raises(self, simple_study):
        with pytest.raises(ValueError, match="columns"):
            simple_study.add_observations(np.array([[0.1, 0.2, 0.3]]), np.array([1.0]))

        with pytest.raises(ValueError, match="rows"):
            simple_study.add_observations(np.array([[0.1, 0.2]]), np.array([1.0, 2.0]))

    def test_add_observations_records_iteration(self, simple_study):
        simple_study.add_observations(np.array([[0.1, 0.2]]), np.array([1.0]))
        assert len(simple_study.iterations) == 1
        it = simple_study.iterations[0]
        assert it.round_number == 1
        assert it.n_points_added == 1

    def test_add_observations_no_iteration(self, simple_study):
        simple_study.add_observations(
            np.array([[0.1, 0.2]]), np.array([1.0]), record_iteration=False
        )
        assert len(simple_study.iterations) == 0

    def test_add_1d_x(self, simple_study):
        """Single observation with 1D X array should be reshaped."""
        simple_study.add_observations(np.array([0.1, 0.2]), np.array([1.0]))
        assert simple_study.n_observations == 1
        assert simple_study.X.shape == (1, 2)


# =============================================================================
# Model fitting
# =============================================================================


class TestModelFitting:
    """Test fitting models within a study."""

    def test_fit_basic(self, simple_study):
        np.random.seed(42)
        X = np.random.rand(30, 2)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.01 * np.random.randn(30)
        simple_study.add_observations(X, y, record_iteration=False)

        model = simple_study.fit(max_terms=3)
        assert simple_study.is_fitted
        assert model is not None
        assert model.expression_ is not None

    def test_fit_no_data_raises(self, simple_study):
        with pytest.raises(RuntimeError, match="No observations"):
            simple_study.fit()

    def test_fit_updates_iteration(self, simple_study):
        np.random.seed(42)
        X = np.random.rand(30, 2)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.01 * np.random.randn(30)
        simple_study.add_observations(X, y)
        simple_study.fit(max_terms=3)

        last = simple_study.iterations[-1]
        assert last.model_expression is not None
        assert last.model_metrics is not None
        assert "mse" in last.model_metrics


# =============================================================================
# Save / Load round-trip
# =============================================================================


class TestSaveLoad:
    """Test serialization to and from .jaxsr archives."""

    def test_save_empty_study(self, simple_study, tmp_path_file):
        simple_study.save(tmp_path_file)
        assert os.path.exists(tmp_path_file)

        # Should be a valid ZIP
        assert zipfile.is_zipfile(tmp_path_file)

    def test_save_archive_contents(self, simple_study, tmp_path_file):
        simple_study.create_design(method="latin_hypercube", n_points=5, random_state=42)
        simple_study.save(tmp_path_file)

        with zipfile.ZipFile(tmp_path_file, "r") as zf:
            names = zf.namelist()
            assert "meta.json" in names
            assert "study.json" in names
            assert "X_design.npy" in names

    def test_roundtrip_empty(self, simple_study, tmp_path_file):
        simple_study.save(tmp_path_file)
        loaded = DOEStudy.load(tmp_path_file)

        assert loaded.name == "test_study"
        assert loaded.factor_names == ["x1", "x2"]
        assert loaded.bounds == [(0, 1), (0, 1)]
        assert loaded.n_observations == 0
        assert not loaded.is_fitted

    def test_roundtrip_with_design(self, simple_study, tmp_path_file):
        X_design = simple_study.create_design(
            method="latin_hypercube", n_points=10, random_state=42
        )
        simple_study.save(tmp_path_file)
        loaded = DOEStudy.load(tmp_path_file)

        np.testing.assert_array_almost_equal(loaded.design_points, X_design)
        assert len(loaded._observation_status) == 10

    def test_roundtrip_with_observations(self, simple_study, tmp_path_file):
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y = np.array([1.0, 2.0, 3.0])
        simple_study.add_observations(X, y)
        simple_study.save(tmp_path_file)

        loaded = DOEStudy.load(tmp_path_file)
        assert loaded.n_observations == 3
        np.testing.assert_array_almost_equal(loaded.X, X)
        np.testing.assert_array_almost_equal(loaded.y, y)

    def test_roundtrip_with_fitted_model(self, simple_study, tmp_path_file):
        np.random.seed(42)
        X = np.random.rand(30, 2)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.01 * np.random.randn(30)
        simple_study.add_observations(X, y, record_iteration=False)
        simple_study.fit(max_terms=3)

        expr_before = simple_study.model.expression_
        simple_study.save(tmp_path_file)

        loaded = DOEStudy.load(tmp_path_file)
        assert loaded.is_fitted
        assert loaded.model.expression_ == expr_before

    def test_roundtrip_categorical(self, categorical_study, tmp_path_file):
        categorical_study.save(tmp_path_file)
        loaded = DOEStudy.load(tmp_path_file)

        assert loaded.feature_types == ["continuous", "continuous", "categorical"]
        assert loaded.categories == {2: ["A", "B", "C"]}
        assert loaded.description == "Testing categorical support"

    def test_roundtrip_iterations(self, simple_study, tmp_path_file):
        X1 = np.array([[0.1, 0.2], [0.3, 0.4]])
        y1 = np.array([1.0, 2.0])
        simple_study.add_observations(X1, y1, notes="First batch")

        X2 = np.array([[0.5, 0.6]])
        y2 = np.array([3.0])
        simple_study.add_observations(X2, y2, notes="Second batch")

        simple_study.save(tmp_path_file)
        loaded = DOEStudy.load(tmp_path_file)

        assert len(loaded.iterations) == 2
        assert loaded.iterations[0].notes == "First batch"
        assert loaded.iterations[1].notes == "Second batch"
        assert loaded.iterations[0].round_number == 1
        assert loaded.iterations[1].round_number == 2

    def test_roundtrip_preserves_metadata(self, simple_study, tmp_path_file):
        simple_study.save(tmp_path_file)
        loaded = DOEStudy.load(tmp_path_file)

        assert loaded.meta["schema_version"] == _SCHEMA_VERSION
        assert "created" in loaded.meta
        assert "modified" in loaded.meta

    def test_load_future_major_version_raises(self, simple_study, tmp_path_file):
        simple_study.save(tmp_path_file)

        # Tamper with the schema version
        with zipfile.ZipFile(tmp_path_file, "r") as zf:
            meta = json.loads(zf.read("meta.json"))
            study = zf.read("study.json")
            other_files = {}
            for name in zf.namelist():
                if name not in ("meta.json", "study.json"):
                    other_files[name] = zf.read(name)

        meta["schema_version"] = "99.0.0"
        with zipfile.ZipFile(tmp_path_file, "w") as zf:
            zf.writestr("meta.json", json.dumps(meta))
            zf.writestr("study.json", study)
            for name, data in other_files.items():
                zf.writestr(name, data)

        with pytest.raises(ValueError, match="newer than supported"):
            DOEStudy.load(tmp_path_file)

    def test_meta_json_readable(self, simple_study, tmp_path_file):
        """Verify that meta.json is human-readable JSON inside the ZIP."""
        simple_study.save(tmp_path_file)
        with zipfile.ZipFile(tmp_path_file, "r") as zf:
            meta = json.loads(zf.read("meta.json"))
        assert isinstance(meta, dict)
        assert "schema_version" in meta


# =============================================================================
# Iteration tracking
# =============================================================================


class TestIteration:
    """Test the Iteration dataclass."""

    def test_iteration_roundtrip(self):
        it = Iteration(
            round_number=1,
            timestamp="2026-02-06T12:00:00+00:00",
            n_points_added=5,
            model_expression="2.0*x1 + 3.0*x2",
            model_metrics={"mse": 0.01, "aic": 42.0},
            notes="First round",
        )
        d = it.to_dict()
        restored = Iteration.from_dict(d)

        assert restored.round_number == 1
        assert restored.n_points_added == 5
        assert restored.model_expression == "2.0*x1 + 3.0*x2"
        assert restored.model_metrics["mse"] == 0.01
        assert restored.notes == "First round"

    def test_iteration_defaults(self):
        it = Iteration(round_number=1, timestamp="now", n_points_added=3)
        assert it.model_expression is None
        assert it.model_metrics is None
        assert it.notes == ""


# =============================================================================
# Summary display
# =============================================================================


class TestSummary:
    """Test the summary display."""

    def test_summary_unfitted(self, simple_study):
        text = simple_study.summary()
        assert "test_study" in text
        assert "x1, x2" in text
        assert "not fitted" in text

    def test_summary_with_design(self, simple_study):
        simple_study.create_design(method="latin_hypercube", n_points=5, random_state=42)
        text = simple_study.summary()
        assert "5 points" in text
        assert "pending" in text

    def test_summary_with_observations(self, simple_study):
        simple_study.add_observations(
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            np.array([1.0, 2.0]),
        )
        text = simple_study.summary()
        assert "Observations: 2" in text

    def test_summary_fitted(self, simple_study):
        np.random.seed(42)
        X = np.random.rand(30, 2)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1]
        simple_study.add_observations(X, y, record_iteration=False)
        simple_study.fit(max_terms=3)
        text = simple_study.summary()
        assert "Model:" in text
        assert "MSE:" in text


# =============================================================================
# Suggest next points
# =============================================================================


class TestSuggestNext:
    """Test adaptive suggestion of next experimental points."""

    def test_suggest_next_raises_without_fit(self, simple_study):
        with pytest.raises(RuntimeError, match="No fitted model"):
            simple_study.suggest_next()

    def test_suggest_next_basic(self, simple_study):
        np.random.seed(42)
        X = np.random.rand(30, 2)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.01 * np.random.randn(30)
        simple_study.add_observations(X, y, record_iteration=False)
        simple_study.fit(max_terms=3)

        next_pts = simple_study.suggest_next(n_points=3, strategy="space_filling")
        assert next_pts.shape == (3, 2)
        # Points should be within bounds
        assert np.all(next_pts >= 0) and np.all(next_pts <= 1)
