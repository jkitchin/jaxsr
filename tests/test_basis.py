"""Tests for basis function library."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsr.basis import BasisFunction, BasisLibrary


class TestBasisFunction:
    """Tests for BasisFunction dataclass."""

    def test_creation(self):
        """Test basic BasisFunction creation."""
        bf = BasisFunction(
            name="x^2",
            func=lambda X: X[:, 0] ** 2,
            complexity=2,
        )
        assert bf.name == "x^2"
        assert bf.complexity == 2

    def test_evaluate(self):
        """Test BasisFunction evaluation."""
        bf = BasisFunction(
            name="x",
            func=lambda X: X[:, 0],
            complexity=1,
        )
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = bf.evaluate(X)
        np.testing.assert_array_almost_equal(result, jnp.array([1.0, 3.0]))


class TestBasisLibrary:
    """Tests for BasisLibrary."""

    def test_creation(self):
        """Test library creation."""
        library = BasisLibrary(n_features=2, feature_names=["x", "y"])
        assert library.n_features == 2
        assert library.feature_names == ["x", "y"]
        assert len(library) == 0

    def test_default_feature_names(self):
        """Test default feature names."""
        library = BasisLibrary(n_features=3)
        assert library.feature_names == ["x0", "x1", "x2"]

    def test_add_constant(self):
        """Test adding constant term."""
        library = BasisLibrary(n_features=2).add_constant()
        assert len(library) == 1
        assert library.names[0] == "1"

        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Phi = library.evaluate(X)
        np.testing.assert_array_almost_equal(Phi, jnp.ones((3, 1)))

    def test_add_linear(self):
        """Test adding linear terms."""
        library = BasisLibrary(n_features=2, feature_names=["x", "y"]).add_linear()
        assert len(library) == 2
        assert library.names == ["x", "y"]

        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        Phi = library.evaluate(X)
        np.testing.assert_array_almost_equal(Phi, X)

    def test_add_polynomials(self):
        """Test adding polynomial terms."""
        library = BasisLibrary(n_features=2, feature_names=["x", "y"]).add_polynomials(max_degree=3)
        # Should have x^2, x^3, y^2, y^3 = 4 terms
        assert len(library) == 4

        X = jnp.array([[2.0, 3.0]])
        Phi = library.evaluate(X)
        expected = jnp.array([[4.0, 8.0, 9.0, 27.0]])  # x^2, x^3, y^2, y^3
        np.testing.assert_array_almost_equal(Phi, expected)

    def test_add_interactions(self):
        """Test adding interaction terms."""
        library = BasisLibrary(n_features=3, feature_names=["x", "y", "z"]).add_interactions(
            max_order=2
        )
        # Should have x*y, x*z, y*z = 3 terms
        assert len(library) == 3
        assert "x*y" in library.names
        assert "x*z" in library.names
        assert "y*z" in library.names

        X = jnp.array([[2.0, 3.0, 4.0]])
        Phi = library.evaluate(X)
        expected = jnp.array([[6.0, 8.0, 12.0]])  # x*y, x*z, y*z
        np.testing.assert_array_almost_equal(Phi, expected)

    def test_add_transcendental(self):
        """Test adding transcendental terms."""
        library = BasisLibrary(n_features=1, feature_names=["x"]).add_transcendental(
            ["exp", "sqrt"]
        )
        assert len(library) == 2
        assert "exp(x)" in library.names
        assert "sqrt(x)" in library.names

        X = jnp.array([[1.0], [4.0]])
        Phi = library.evaluate(X)
        expected = jnp.array(
            [
                [jnp.exp(1.0), 1.0],
                [jnp.exp(4.0), 2.0],
            ]
        )
        np.testing.assert_array_almost_equal(Phi, expected)

    def test_add_ratios(self):
        """Test adding ratio terms."""
        library = BasisLibrary(n_features=2, feature_names=["x", "y"]).add_ratios()
        # Should have x/y, y/x = 2 terms
        assert len(library) == 2

        X = jnp.array([[4.0, 2.0]])
        Phi = library.evaluate(X)
        expected = jnp.array([[2.0, 0.5]])  # x/y, y/x
        np.testing.assert_array_almost_equal(Phi, expected)

    def test_add_custom(self):
        """Test adding custom basis function."""
        library = BasisLibrary(n_features=2, feature_names=["x", "y"]).add_custom(
            name="x^2*y",
            func=lambda X: X[:, 0] ** 2 * X[:, 1],
            complexity=3,
        )
        assert len(library) == 1
        assert library.names[0] == "x^2*y"

        X = jnp.array([[2.0, 3.0]])
        Phi = library.evaluate(X)
        expected = jnp.array([[12.0]])  # 2^2 * 3 = 12
        np.testing.assert_array_almost_equal(Phi, expected)

    def test_build_default(self):
        """Test building default library."""
        library = BasisLibrary(n_features=2).build_default(max_poly_degree=2)
        # Should have: 1, x0, x1, x0^2, x1^2, x0*x1, + transcendental
        assert len(library) > 5

    def test_method_chaining(self):
        """Test method chaining."""
        library = (
            BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(max_degree=2)
        )
        # 1 + 2 + 2 = 5 terms
        assert len(library) == 5

    def test_complexities(self):
        """Test complexity scores."""
        library = (
            BasisLibrary(n_features=2)
            .add_constant()  # complexity 0
            .add_linear()  # complexity 1 each
            .add_polynomials(max_degree=2)  # complexity 2 each
        )
        complexities = library.complexities
        assert complexities[0] == 0  # constant
        assert complexities[1] == 1  # x0
        assert complexities[2] == 1  # x1
        assert complexities[3] == 2  # x0^2
        assert complexities[4] == 2  # x1^2

    def test_serialization(self, tmp_path):
        """Test save/load functionality."""
        library = (
            BasisLibrary(n_features=2, feature_names=["a", "b"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
        )

        filepath = tmp_path / "library.json"
        library.save(str(filepath))

        loaded = BasisLibrary.load(str(filepath))
        assert len(loaded) == len(library)
        assert loaded.feature_names == library.feature_names

        # Test evaluation produces same results
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        Phi_original = library.evaluate(X)
        Phi_loaded = loaded.evaluate(X)
        np.testing.assert_array_almost_equal(Phi_original, Phi_loaded)

    def test_evaluate_subset(self):
        """Test evaluating subset of basis functions."""
        library = (
            BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(max_degree=2)
        )
        X = jnp.array([[2.0, 3.0]])

        # Evaluate only linear terms (indices 1, 2)
        Phi_subset = library.evaluate_subset(X, [1, 2])
        np.testing.assert_array_almost_equal(Phi_subset, jnp.array([[2.0, 3.0]]))

    def test_filter_by_complexity(self):
        """Test filtering by complexity."""
        library = (
            BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(max_degree=3)
        )

        # Get indices of low complexity terms
        indices = library.filter_by_complexity(max_complexity=1)
        names = [library.names[i] for i in indices]
        assert "1" in names
        assert "x0" in names
        assert "x1" in names
        assert "x0^2" not in names

    def test_feature_mismatch_error(self):
        """Test error when feature count doesn't match."""
        library = BasisLibrary(n_features=2).add_linear()
        X = jnp.array([[1.0, 2.0, 3.0]])  # 3 features

        with pytest.raises(ValueError):
            library.evaluate(X)

    def test_empty_library_error(self):
        """Test error when evaluating empty library."""
        library = BasisLibrary(n_features=2)

        with pytest.raises(ValueError):
            library.evaluate(jnp.array([[1.0, 2.0]]))

    def test_safe_log(self):
        """Test safe log handles non-positive values."""
        library = BasisLibrary(n_features=1).add_transcendental(["log"])
        X = jnp.array([[1.0], [-1.0], [0.0]])
        Phi = library.evaluate(X)

        assert jnp.isfinite(Phi[0, 0])  # log(1) = 0
        assert jnp.isnan(Phi[1, 0])  # log(-1) = NaN
        assert jnp.isnan(Phi[2, 0])  # log(0) = NaN

    def test_safe_sqrt(self):
        """Test safe sqrt handles negative values."""
        library = BasisLibrary(n_features=1).add_transcendental(["sqrt"])
        X = jnp.array([[4.0], [-1.0]])
        Phi = library.evaluate(X)

        np.testing.assert_almost_equal(float(Phi[0, 0]), 2.0)
        assert jnp.isnan(Phi[1, 0])

    def test_repr(self):
        """Test string representation shows basis functions."""
        library = (
            BasisLibrary(n_features=2, feature_names=["x", "y"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
        )
        repr_str = repr(library)

        # Check header information
        assert "BasisLibrary" in repr_str
        assert "n_features=2" in repr_str
        assert "n_basis=5" in repr_str

        # Check basis functions are listed
        assert "Basis functions:" in repr_str
        assert "1" in repr_str  # constant
        assert "x" in repr_str  # linear
        assert "y" in repr_str  # linear

    def test_repr_truncation(self):
        """Test that repr truncates long lists of basis functions."""
        library = BasisLibrary(n_features=3).build_default(max_poly_degree=3)
        # This should have many basis functions

        # Test with small max_display
        repr_str = library.__repr__(max_display=5)
        assert "... and" in repr_str
        assert "more" in repr_str

        # Count displayed functions (should be 5 + header + truncation message)
        lines = repr_str.split("\n")
        function_lines = [line for line in lines if line.strip().startswith("[")]
        assert len(function_lines) == 5

    def test_repr_html(self):
        """Test HTML representation for Jupyter notebooks."""
        library = BasisLibrary(n_features=2, feature_names=["x", "y"]).add_constant().add_linear()
        html = library._repr_html_()

        # Check HTML structure
        assert "<table" in html
        assert "<thead>" in html
        assert "<tbody>" in html
        assert "BasisLibrary" in html

        # Check content
        assert "2 features" in html
        assert "3 basis functions" in html
        assert "x, y" in html  # feature names

    def test_repr_html_truncation(self):
        """Test HTML repr truncates correctly."""
        library = BasisLibrary(n_features=3).build_default(max_poly_degree=3)

        html = library._repr_html_(max_display=5)
        assert "... and" in html
        assert "more basis functions" in html

    def test_repr_markdown(self):
        """Test markdown representation."""
        library = BasisLibrary(n_features=2, feature_names=["x", "y"]).add_constant().add_linear()
        md = library._repr_markdown_()

        # Check markdown structure
        assert "**BasisLibrary**" in md
        assert "2 features" in md
        assert "3 basis functions" in md
        assert "| Index | Basis Function |" in md  # table header

        # Check content
        assert "`1`" in md  # constant
        assert "`x`" in md  # linear
        assert "`y`" in md  # linear

    def test_repr_markdown_truncation(self):
        """Test markdown repr truncates correctly."""
        library = BasisLibrary(n_features=3).build_default(max_poly_degree=3)

        md = library._repr_markdown_(max_display=5)
        assert "and" in md.lower()
        assert "more" in md.lower()


class TestSafeExponential:
    """``_safe_exp`` must signal out-of-domain results the same way at any dtype.

    Regression test: ``_safe_exp`` used to clip its argument to a hardcoded
    +/-500.  At float32 that overflows to ``inf`` (which the regressor's
    non-finite column filter removes), but at float64 it yields a finite
    ~1e217 that survives selection and overflows ``Phi.T @ Phi`` instead.
    """

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_out_of_range_argument_is_nan(self, dtype):
        """An argument too large to square yields NaN, not a huge finite value."""
        from jaxsr.basis import _safe_exp

        out = np.asarray(_safe_exp(np.array([1e6], dtype=dtype)))
        assert np.isnan(out[0])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_in_range_argument_is_exact(self, dtype):
        """Ordinary arguments are unaffected."""
        from jaxsr.basis import _safe_exp

        out = np.asarray(_safe_exp(np.array([0.0, 1.0, 2.0], dtype=dtype)))
        assert np.allclose(out, np.exp([0.0, 1.0, 2.0]), rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_result_survives_being_squared(self, dtype):
        """Whatever survives must be usable in a Gram matrix."""
        from jaxsr.basis import _safe_exp

        x = np.linspace(-1000, 1000, 4001, dtype=dtype)
        out = np.asarray(_safe_exp(x), dtype=dtype)
        finite = out[np.isfinite(out)]
        gram = (finite.astype(dtype) ** 2).sum()
        assert np.isfinite(gram)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_large_negative_argument_underflows_to_zero(self, dtype):
        """Very negative arguments are legitimate (exp -> 0), not out-of-domain."""
        from jaxsr.basis import _safe_exp

        out = np.asarray(_safe_exp(np.array([-1e6], dtype=dtype)))
        assert out[0] == 0.0

    def test_pole_bearing_composition_is_excluded(self):
        """exp(a/b) evaluated across a zero in b is dropped, not kept as 1e217."""
        from jaxsr import SymbolicRegressor

        rng = np.random.default_rng(0)
        X = jnp.array(rng.uniform(-2, 2, size=(200, 2)))
        y = jnp.exp(X[:, 0] * X[:, 1])
        library = (
            BasisLibrary(n_features=2)
            .add_constant()
            .add_linear()
            .add_interactions()
            .add_compositions(["exp"], ["product", "ratio"])
        )
        with pytest.warns(UserWarning, match="non-finite"):
            model = SymbolicRegressor(basis_library=library, max_terms=4).fit(X, y)

        assert "exp(x0*x1)" in model.selected_features_
        assert np.all(np.isfinite(np.asarray(model.predict(X))))


class TestCanonicalNames:
    """Tests for basis identity that survives a parametric fit."""

    def _parametric_library(self):
        return (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_parametric(
                name="exp(-a*x)",
                func=lambda X, a: jnp.exp(-a * X[:, 0]),
                param_bounds={"a": (0.01, 5.0)},
                feature_indices=(0,),
            )
        )

    def test_canonical_name_matches_name_for_ordinary_basis(self):
        """Non-parametric basis functions keep their ordinary name."""
        library = BasisLibrary(n_features=1, feature_names=["x"]).add_constant().add_linear()
        assert library.canonical_names == library.names
        assert library.canonical_name(1) == "x"

    def test_canonical_name_keeps_parameter_symbol(self):
        """A parametric basis reports its registered template, not the rendering."""
        library = self._parametric_library()
        assert library.names[2] == "exp(-2.505*x)"
        assert library.canonical_name(2) == "exp(-a*x)"
        assert library.canonical_names[:2] == library.names[:2]

    def test_canonical_name_stable_across_fits(self):
        """Fitting rewrites names but leaves the canonical name unchanged."""
        from jaxsr import SymbolicRegressor

        rng = np.random.RandomState(0)
        X = jnp.array(rng.uniform(0, 5, (60, 1)))
        y = jnp.array(3.0 * np.exp(-0.5 * np.asarray(X)[:, 0]) + 1.0)

        library = self._parametric_library()
        SymbolicRegressor(basis_library=library, max_terms=3).fit(X, y)

        assert library.names[2] != "exp(-a*x)"  # rendered with the fitted value
        assert library.canonical_name(2) == "exp(-a*x)"

    def test_canonical_name_out_of_range(self):
        """An index outside the library raises IndexError."""
        library = BasisLibrary(n_features=1).add_constant()
        with pytest.raises(IndexError, match="out of range"):
            library.canonical_name(5)


class TestBasisLibraryCopy:
    """Tests for BasisLibrary.copy()."""

    def test_copy_is_independent(self):
        """Mutating the copy leaves the original alone."""
        library = BasisLibrary(n_features=2, feature_names=["a", "b"]).add_constant().add_linear()
        other = library.copy()

        assert other.names == library.names
        assert other.feature_names == library.feature_names

        other.basis_functions[0].name = "renamed"
        other.add_polynomials(max_degree=2)
        assert library.names[0] == "1"
        assert len(library) == 3

    def test_copy_preserves_block_labels(self):
        """Block membership survives a copy, so blocks/filter_by_block still work."""
        theta = BasisLibrary(n_features=1, feature_names=["q"]).add_constant().add_linear()
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by="y_x", block_name="horizontal"
        )
        assert library.copy().blocks == library.blocks

    def test_copy_evaluates_identically(self):
        """The copy shares the basis callables, so it evaluates the same."""
        rng = np.random.RandomState(1)
        X = jnp.array(rng.uniform(-1, 1, (20, 2)))
        library = (
            BasisLibrary(n_features=2).add_constant().add_linear().add_polynomials(max_degree=2)
        )
        assert np.allclose(np.asarray(library.copy().evaluate(X)), np.asarray(library.evaluate(X)))

    def test_copy_isolates_parametric_state(self):
        """Fitting on a copy must not repin the original's parametric basis."""
        from jaxsr import SymbolicRegressor

        rng = np.random.RandomState(0)
        X = jnp.array(rng.uniform(0, 5, (60, 1)))
        y = jnp.array(3.0 * np.exp(-0.5 * np.asarray(X)[:, 0]) + 1.0)

        library = (
            BasisLibrary(n_features=1, feature_names=["x"])
            .add_constant()
            .add_linear()
            .add_parametric(
                name="exp(-a*x)",
                func=lambda X, a: jnp.exp(-a * X[:, 0]),
                param_bounds={"a": (0.01, 5.0)},
                feature_indices=(0,),
            )
        )
        before_name = library.names[2]
        before_column = np.asarray(library.evaluate(X)[:, 2])

        SymbolicRegressor(basis_library=library.copy(), max_terms=3).fit(X, y)

        assert library.names[2] == before_name
        assert np.allclose(np.asarray(library.evaluate(X)[:, 2]), before_column)
        assert library._parametric_info[0].resolved_params is None


class TestAddBlock:
    """Tests for structured basis blocks (Theta(a) times a data column)."""

    @pytest.fixture
    def theta(self):
        """A small basis over a single variable q."""
        return (
            BasisLibrary(n_features=1, feature_names=["q"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=2)
        )

    @pytest.fixture
    def X(self):
        """Data over (q, y_x)."""
        rng = np.random.default_rng(0)
        return jnp.array(rng.uniform(0.5, 2.0, size=(20, 2)))

    def test_names_are_generated(self, theta):
        """Block names are the source names times the multiplying column."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by="y_x"
        )
        assert library.names == ["y_x", "q*y_x", "q^2*y_x"]

    def test_unmultiplied_block_keeps_source_names(self, theta):
        """Without multiply_by the block is a plain copy of the source."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(theta)
        assert library.names == theta.names

    def test_evaluates_as_product(self, theta, X):
        """Each column is the source column times the data column."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by="y_x"
        )
        Phi = np.asarray(library.evaluate(X))
        q, y_x = np.asarray(X[:, 0]), np.asarray(X[:, 1])
        np.testing.assert_allclose(Phi[:, 0], y_x, rtol=1e-6)
        np.testing.assert_allclose(Phi[:, 1], q * y_x, rtol=1e-6)
        np.testing.assert_allclose(Phi[:, 2], q**2 * y_x, rtol=1e-6)

    def test_multiply_by_index(self, theta, X):
        """multiply_by accepts a column index as well as a name."""
        by_name = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by="y_x"
        )
        by_index = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by=1
        )
        assert by_index.names == by_name.names
        np.testing.assert_allclose(
            np.asarray(by_index.evaluate(X)), np.asarray(by_name.evaluate(X)), rtol=1e-6
        )

    def test_features_are_remapped(self, X):
        """A source written against its own columns is re-expressed on ours."""
        theta = BasisLibrary(n_features=1, feature_names=["y_x"]).add_linear()
        # 'y_x' is column 0 of the source but column 1 here
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(theta)
        Phi = np.asarray(library.evaluate(X))
        np.testing.assert_allclose(Phi[:, 0], np.asarray(X[:, 1]), rtol=1e-6)

    def test_complexity_inherited_plus_one_for_the_product(self, theta):
        """Multiplying by a column costs one, on top of the inherited score."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by="y_x"
        )
        assert list(np.asarray(library.complexities)) == [1, 2, 3]

    def test_complexity_offset(self, theta):
        """complexity_offset shifts the whole block."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, complexity_offset=2
        )
        assert list(np.asarray(library.complexities)) == [2, 3, 4]

    def test_feature_indices_include_the_multiplier(self, theta):
        """Feature indices are mapped to this library and include the column."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by="y_x"
        )
        assert library.basis_functions[0].feature_indices == (1,)
        assert library.basis_functions[1].feature_indices == (0, 1)

    def test_source_library_is_reusable(self, theta, X):
        """Adding a block copies functions; the source is untouched."""
        library = (
            BasisLibrary(n_features=2, feature_names=["q", "y_x"])
            .add_block(theta, multiply_by="y_x", block_name="horizontal")
            .add_block(theta, block_name="vertical")
        )
        assert len(theta) == 3
        assert theta.names == ["1", "q", "q^2"]
        assert len(library) == 6
        # The unmultiplied half still evaluates on q alone
        Phi = np.asarray(library.evaluate(X))
        np.testing.assert_allclose(Phi[:, 4], np.asarray(X[:, 0]), rtol=1e-6)

    def test_feature_map(self, X):
        """feature_map matches a source feature whose name differs."""
        theta = BasisLibrary(n_features=1, feature_names=["x"]).add_linear()
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by="y_x", feature_map={"x": "q"}
        )
        assert library.names == ["x*y_x"]
        Phi = np.asarray(library.evaluate(X))
        np.testing.assert_allclose(Phi[:, 0], np.asarray(X[:, 0] * X[:, 1]), rtol=1e-6)

    def test_sum_names_are_parenthesized(self, X):
        """A bare sum is parenthesized so the generated name stays correct."""
        theta = BasisLibrary(n_features=1, feature_names=["q"]).add_custom(
            "1+q", lambda Xs: 1.0 + Xs[:, 0], complexity=2, feature_indices=(0,)
        )
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by="y_x"
        )
        assert library.names == ["(1+q)*y_x"]
        Phi = np.asarray(library.evaluate(X))
        np.testing.assert_allclose(Phi[:, 0], np.asarray((1.0 + X[:, 0]) * X[:, 1]), rtol=1e-6)

    def test_custom_source_functions_are_carried_over(self, X):
        """add_custom terms in the source work in the block."""
        theta = BasisLibrary(n_features=1, feature_names=["q"]).add_custom(
            "1/(1+q)^2", lambda Xs: 1.0 / (1.0 + Xs[:, 0]) ** 2, complexity=3, feature_indices=(0,)
        )
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by="y_x"
        )
        assert library.names == ["1/(1+q)^2*y_x"]
        assert library.basis_functions[0].complexity == 4
        Phi = np.asarray(library.evaluate(X))
        np.testing.assert_allclose(Phi[:, 0], np.asarray(X[:, 1] / (1.0 + X[:, 0]) ** 2), rtol=1e-6)

    def test_parametric_passes_through(self, X):
        """Parametric source terms stay parametric inside the block."""
        theta = BasisLibrary(n_features=1, feature_names=["q"]).add_parametric(
            name="1/(c2+q)^2",
            func=lambda Xs, c2: 1.0 / (c2 + Xs[:, 0]) ** 2,
            param_bounds={"c2": (0.2, 0.8)},
            complexity=3,
            feature_indices=(0,),
        )
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by="y_x", block_name="horizontal"
        )
        assert library.has_parametric
        info = library._parametric_info[0]
        assert info.name == "1/(c2+q)^2*y_x"
        assert info.param_bounds == {"c2": (0.2, 0.8)}
        assert info.basis_index == 0
        assert library.basis_functions[0].block == "horizontal"
        assert library.basis_functions[0].complexity == 4
        # The registered function multiplies by the data column
        got = np.asarray(info.func(X, c2=0.5))
        np.testing.assert_allclose(got, np.asarray(X[:, 1] / (0.5 + X[:, 0]) ** 2), rtol=1e-6)

    def test_blocks_property(self, theta):
        """Blocks report the indices they own."""
        library = (
            BasisLibrary(n_features=2, feature_names=["q", "y_x"])
            .add_block(theta, multiply_by="y_x", block_name="horizontal")
            .add_block(theta, block_name="vertical")
        )
        assert library.blocks == {"horizontal": [0, 1, 2], "vertical": [3, 4, 5]}

    def test_unlabelled_functions_have_no_block(self, theta):
        """Functions added outside a block are not part of one."""
        library = (
            BasisLibrary(n_features=2, feature_names=["q", "y_x"])
            .add_constant()
            .add_block(theta, multiply_by="y_x", block_name="horizontal")
        )
        assert library.blocks == {"horizontal": [1, 2, 3]}
        assert library.basis_functions[0].block is None

    def test_filter_by_block(self, theta):
        """filter_by_block selects and drops whole blocks."""
        library = (
            BasisLibrary(n_features=2, feature_names=["q", "y_x"])
            .add_constant()
            .add_block(theta, multiply_by="y_x", block_name="horizontal")
            .add_block(theta, block_name="vertical")
        )
        assert library.filter_by_block(include="horizontal") == [1, 2, 3]
        assert library.filter_by_block(exclude=["vertical"]) == [0, 1, 2, 3]
        assert library.filter_by_block() == list(range(7))

    def test_filter_by_block_unknown_name(self, theta):
        """A typo in a block name is an error, not an empty result."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, block_name="vertical"
        )
        with pytest.raises(ValueError, match="Unknown block"):
            library.filter_by_block(include="verticle")

    def test_without_blocks(self, theta, X):
        """Dropping a block leaves a working library and the original intact."""
        library = (
            BasisLibrary(n_features=2, feature_names=["q", "y_x"])
            .add_block(theta, multiply_by="y_x", block_name="horizontal")
            .add_block(theta, block_name="vertical")
        )
        reduced = library.without_blocks("vertical")

        assert len(library) == 6
        assert reduced.names == ["y_x", "q*y_x", "q^2*y_x"]
        assert reduced.blocks == {"horizontal": [0, 1, 2]}
        np.testing.assert_allclose(
            np.asarray(reduced.evaluate(X)), np.asarray(library.evaluate(X))[:, :3], rtol=1e-6
        )

    def test_without_blocks_reindexes_parametric(self, X):
        """Parametric bookkeeping follows the surviving functions."""
        theta = BasisLibrary(n_features=1, feature_names=["q"]).add_parametric(
            name="1/(c2+q)^2",
            func=lambda Xs, c2: 1.0 / (c2 + Xs[:, 0]) ** 2,
            param_bounds={"c2": (0.2, 0.8)},
            feature_indices=(0,),
        )
        library = (
            BasisLibrary(n_features=2, feature_names=["q", "y_x"])
            .add_block(theta, block_name="vertical")
            .add_block(theta, multiply_by="y_x", block_name="horizontal")
        )
        assert [p.basis_index for p in library._parametric_info] == [0, 1]

        reduced = library.without_blocks("vertical")
        assert len(reduced) == 1
        assert [p.basis_index for p in reduced._parametric_info] == [0]
        assert reduced._parametric_info[0].name == "1/(c2+q)^2*y_x"

    def test_empty_source_raises(self):
        """An empty source library is a mistake worth reporting."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"])
        with pytest.raises(ValueError, match="no basis functions"):
            library.add_block(BasisLibrary(n_features=1, feature_names=["q"]))

    def test_unmatched_source_feature_raises(self, theta):
        """A source feature with no counterpart names the fix."""
        library = BasisLibrary(n_features=2, feature_names=["c", "y_x"])
        with pytest.raises(ValueError, match="feature_map"):
            library.add_block(theta, multiply_by="y_x")

    def test_unknown_multiply_by_raises(self, theta):
        """multiply_by must name a feature of this library."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"])
        with pytest.raises(ValueError, match="multiply_by"):
            library.add_block(theta, multiply_by="y_c")

    def test_out_of_range_multiply_by_raises(self, theta):
        """An index past the end of the feature space is an error."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"])
        with pytest.raises(ValueError, match="out of range"):
            library.add_block(theta, multiply_by=5)

    def test_bad_multiply_by_type_raises(self, theta):
        """multiply_by is a name or an index, nothing else."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"])
        with pytest.raises(TypeError, match="multiply_by"):
            library.add_block(theta, multiply_by=1.5)

    def test_bad_source_type_raises(self):
        """The source must be a BasisLibrary."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"])
        with pytest.raises(TypeError, match="BasisLibrary"):
            library.add_block(["1", "q"])

    def test_block_survives_to_dict(self, theta):
        """The block label is serialized; the function itself is not."""
        library = BasisLibrary(n_features=2, feature_names=["q", "y_x"]).add_block(
            theta, multiply_by="y_x", block_name="horizontal"
        )
        d = library.to_dict()
        assert d["basis_functions"][0]["block"] == "horizontal"
        with pytest.raises(ValueError, match="add_block"):
            BasisLibrary.from_dict(d)

    def test_recovers_a_coefficient_function(self):
        """The motivating case: y_c = s'(c)*y_x with s'(c) = 1 + 2c."""
        from jaxsr import SymbolicRegressor

        rng = np.random.default_rng(0)
        c = rng.uniform(0.1, 1.0, size=200)
        y_x = rng.uniform(0.5, 2.0, size=200)
        y_c = (1.0 + 2.0 * c) * y_x

        theta = (
            BasisLibrary(n_features=1, feature_names=["c"])
            .add_constant()
            .add_linear()
            .add_polynomials(max_degree=3)
        )
        library = BasisLibrary(n_features=2, feature_names=["c", "y_x"]).add_block(
            theta, multiply_by="y_x", block_name="horizontal"
        )

        X = jnp.array(np.column_stack([c, y_x]))
        model = SymbolicRegressor(basis_library=library, max_terms=4).fit(X, jnp.array(y_c))

        assert set(model.selected_features_) == {"y_x", "c*y_x"}
        coefs = dict(zip(model.selected_features_, np.asarray(model.coefficients_), strict=False))
        np.testing.assert_allclose(coefs["y_x"], 1.0, atol=1e-4)
        np.testing.assert_allclose(coefs["c*y_x"], 2.0, atol=1e-4)
