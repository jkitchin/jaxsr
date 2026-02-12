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
