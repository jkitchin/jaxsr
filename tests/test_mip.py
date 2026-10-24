"""Tests for embedding fitted models in Pyomo and discopt (jaxsr.mip)."""

import numpy as np
import pytest

from jaxsr import BasisLibrary, SymbolicRegressor
from jaxsr.mip import milp_incompatible_terms, to_discopt, to_pyomo

pyo = pytest.importorskip("pyomo.environ")


def _highs():
    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        pytest.skip("HiGHS is not available to Pyomo")
    return solver


@pytest.fixture(scope="module")
def pwl_model():
    """Piecewise-linear model using constant, linear, hinge and abs terms."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 2))
    y = np.abs(X[:, 0] - 0.5) + 0.5 * np.maximum(0, X[:, 1]) - 0.3 * X[:, 1] + 1.0
    lib = (
        BasisLibrary(2, ["x", "y"])
        .add_constant()
        .add_linear()
        .add_hinges(knots={"x": [-1.0, 0.5, 1.0], "y": [-0.5, 0.0, 0.5]})
        .add_transcendental(["abs"])
    )
    return SymbolicRegressor(basis_library=lib, max_terms=6).fit(X, y)


@pytest.fixture(scope="module")
def categorical_model():
    """Model with categorical indicators and indicator-by-continuous terms."""
    rng = np.random.default_rng(1)
    c = rng.integers(0, 3, size=150)
    x = rng.uniform(0, 3, size=150)
    y = 1.0 + x + 2.0 * (c == 1) - 1.5 * x * (c == 2)
    X = np.column_stack([c, x]).astype(float)
    lib = (
        BasisLibrary(
            2,
            ["c", "x"],
            feature_types=["categorical", "continuous"],
            categories={0: ["a", "b", "z"]},
        )
        .add_constant()
        .add_linear()
        .add_categorical_indicators()
        .add_categorical_interactions()
    )
    return SymbolicRegressor(basis_library=lib, max_terms=5).fit(X, y)


@pytest.fixture(scope="module")
def quadratic_model():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(60, 1))
    y = 1.0 + X[:, 0] ** 2
    lib = BasisLibrary(1, ["x"]).add_constant().add_linear().add_polynomials(2)
    return SymbolicRegressor(basis_library=lib, max_terms=2).fit(X, y)


def _pyomo_predict(model, point, **kwargs):
    """Solve the embedding with inputs fixed at ``point`` and return the output."""
    m = pyo.ConcreteModel()
    m.sr = pyo.Block()
    to_pyomo(model, m.sr, **kwargs)
    for name, value in zip(model.basis_library.feature_names, point, strict=True):
        m.sr.jaxsr_inputs[name].fix(value)
    m.obj = pyo.Objective(expr=0)
    _highs().solve(m)
    return pyo.value(m.sr.jaxsr_output)


class TestMilpIncompatibleTerms:
    def test_pwl_model_is_milp(self, pwl_model):
        assert milp_incompatible_terms(pwl_model) == []

    def test_quadratic_model_is_not(self, quadratic_model):
        assert milp_incompatible_terms(quadratic_model) == ["x^2"]

    def test_unfitted_raises(self):
        model = SymbolicRegressor(basis_library=BasisLibrary(1).add_linear())
        with pytest.raises(ValueError, match="not fitted"):
            milp_incompatible_terms(model)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            milp_incompatible_terms("model")


class TestToPyomo:
    @pytest.mark.parametrize(
        "point", [(-1.5, 1.2), (0.3, -0.7), (1.9, 0.0), (0.5, 0.5), (-1.9, -1.9)]
    )
    def test_matches_predict(self, pwl_model, point):
        expected = float(pwl_model.predict(np.array([point]))[0])
        assert _pyomo_predict(pwl_model, point) == pytest.approx(expected, abs=1e-4)

    @pytest.mark.parametrize("point", [(0, 0.5), (1, 2.0), (2, 1.5), (2, 0.1)])
    def test_categorical_matches_predict(self, categorical_model, point):
        expected = float(categorical_model.predict(np.array([point], dtype=float))[0])
        assert _pyomo_predict(categorical_model, point) == pytest.approx(expected, abs=1e-4)

    def test_minimize_finds_global_min(self, pwl_model):
        m = pyo.ConcreteModel()
        m.sr = pyo.Block()
        pwl_model.to_pyomo(m.sr)
        m.obj = pyo.Objective(expr=m.sr.jaxsr_output)
        _highs().solve(m)
        x_opt = np.array([[pyo.value(m.sr.jaxsr_inputs[n]) for n in ("x", "y")]])
        # The reported optimum is the model's value there ...
        assert pyo.value(m.obj) == pytest.approx(float(pwl_model.predict(x_opt)[0]), abs=1e-4)
        # ... and no grid point beats it.
        g = np.linspace(-2, 2, 201)
        grid = np.array(np.meshgrid(g, g)).reshape(2, -1).T
        assert pyo.value(m.obj) <= float(np.min(pwl_model.predict(grid))) + 1e-4

    def test_output_sense_drops_binaries(self, pwl_model):
        exact = to_pyomo(pwl_model)
        relaxed = to_pyomo(pwl_model, output_sense="min")
        assert len(relaxed.binaries) < len(exact.binaries)
        for blk in (exact, relaxed):
            blk.obj = pyo.Objective(expr=blk.jaxsr_output)
            _highs().solve(blk)
        assert pyo.value(relaxed.obj) == pytest.approx(pyo.value(exact.obj), abs=1e-6)

    def test_links_user_variables(self, pwl_model):
        m = pyo.ConcreteModel()
        m.a = pyo.Var(bounds=(-2, 2))
        m.b = pyo.Var(bounds=(-2, 2))
        m.f = pyo.Var()
        m.sr = pyo.Block()
        to_pyomo(pwl_model, m.sr, inputs={"x": m.a, "y": m.b}, output=m.f)
        m.a.fix(0.25)
        m.b.fix(-1.0)
        m.obj = pyo.Objective(expr=0)
        _highs().solve(m)
        expected = float(pwl_model.predict(np.array([[0.25, -1.0]]))[0])
        assert pyo.value(m.f) == pytest.approx(expected, abs=1e-4)

    def test_explicit_bounds_are_enforced(self, pwl_model):
        blk = to_pyomo(pwl_model, bounds={"x": (1.5, 2.0), "y": (-0.1, 0.1)})
        assert blk.jaxsr_inputs["x"].lb == 1.5
        assert blk.jaxsr_inputs["y"].ub == 0.1
        assert _pyomo_predict(pwl_model, (1.75, 0.05), bounds=[(1.5, 2.0), (-0.1, 0.1)]) == (
            pytest.approx(float(pwl_model.predict(np.array([[1.75, 0.05]]))[0]), abs=1e-4)
        )

    def test_nonlinear_requires_opt_in(self, quadratic_model):
        with pytest.raises(ValueError, match="not MILP-representable"):
            to_pyomo(quadratic_model)
        blk = to_pyomo(quadratic_model, allow_nonlinear=True)
        blk.jaxsr_inputs["x"].fix(0.5)
        blk.jaxsr_output.set_value(float(quadratic_model.predict(np.array([[0.5]]))[0]))
        for con in blk.cons.values():
            assert abs(pyo.value(con.body) - pyo.value(con.upper)) < 1e-4

    def test_missing_bounds_raise(self, pwl_model):
        import copy

        model = copy.deepcopy(pwl_model)
        model._X_train = None
        with pytest.raises(ValueError, match="No bounds"):
            to_pyomo(model)

    def test_bad_arguments_raise(self, pwl_model):
        with pytest.raises(ValueError, match="output_sense"):
            to_pyomo(pwl_model, output_sense="up")
        with pytest.raises(ValueError, match="Unknown feature"):
            to_pyomo(pwl_model, inputs={"z": 1.0})
        with pytest.raises(ValueError, match="Invalid bounds"):
            to_pyomo(pwl_model, bounds={"x": (1.0, -1.0)})
        with pytest.raises(ValueError, match="entries"):
            to_pyomo(pwl_model, bounds=[(0, 1)])


class TestToDiscopt:
    @pytest.fixture(autouse=True)
    def _discopt(self):
        self.discopt = pytest.importorskip("discopt")

    @pytest.mark.parametrize("point", [(-1.5, 1.2), (0.3, -0.7)])
    def test_matches_predict(self, pwl_model, point):
        m = self.discopt.Model("fixed")
        emb = pwl_model.to_discopt(m)
        for name, value in zip(("x", "y"), point, strict=True):
            m.subject_to(emb.inputs[name] == value)
        m.minimize(emb.output)
        result = m.solve()
        expected = float(pwl_model.predict(np.array([point]))[0])
        assert result.objective == pytest.approx(expected, abs=1e-4)

    def test_minimize_matches_pyomo(self, pwl_model):
        m = self.discopt.Model("opt")
        emb = to_discopt(pwl_model, m)
        m.minimize(emb.output)
        result = m.solve()

        p = to_pyomo(pwl_model)
        p.obj = pyo.Objective(expr=p.jaxsr_output)
        _highs().solve(p)
        assert result.objective == pytest.approx(pyo.value(p.obj), abs=1e-5)

    def test_categorical_and_prefix(self, categorical_model):
        m = self.discopt.Model("cat")
        emb = to_discopt(categorical_model, m, prefix="s1")
        assert set(emb.onehot) == {("c", "a"), ("c", "b"), ("c", "z")}
        m.subject_to(emb.inputs["c"] == 2)
        m.subject_to(emb.inputs["x"] == 1.5)
        m.minimize(emb.output)
        result = m.solve()
        expected = float(categorical_model.predict(np.array([[2.0, 1.5]]))[0])
        assert result.objective == pytest.approx(expected, abs=1e-4)

    def test_nonlinear_embedding(self, quadratic_model):
        m = self.discopt.Model("quad")
        emb = to_discopt(quadratic_model, m, allow_nonlinear=True)
        m.minimize(emb.output)
        result = m.solve()
        g = np.linspace(-1, 1, 401).reshape(-1, 1)
        assert result.objective == pytest.approx(
            float(np.min(quadratic_model.predict(g))), abs=1e-4
        )
