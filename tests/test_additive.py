"""Tests for the additive symbolic regression submodule."""

import jax.numpy as jnp
import numpy as np
import pytest


def _additive_data(n=200, noise=0.0, seed=0):
    """y = 2.0 * x0 + 0.5 * x1**2 (+ optional noise)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2
    if noise:
        y = y + rng.normal(0, noise, size=n)
    return jnp.array(X), jnp.array(y)


def test_module_imports():
    """The additive module and its public API import correctly."""
    from jaxsr.additive import (
        AdditiveSymbolicModel,
        BackfittingSymbolicRegressor,
        Loss,
        SquaredError,
        StagewiseSymbolicRegressor,
        get_loss,
        refit_ols,
    )

    assert StagewiseSymbolicRegressor is not None
    assert BackfittingSymbolicRegressor is not None
    assert AdditiveSymbolicModel is not None
    assert issubclass(SquaredError, Loss)
    assert callable(get_loss)
    assert callable(refit_ols)


def test_fit_runs_and_predict_shape():
    """fit runs on a simple dataset and predict returns the right shape."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data()
    model = StagewiseSymbolicRegressor(n_terms=3, max_complexity=4)
    model.fit(X, y)

    assert model._is_fitted
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    assert model.n_terms_ >= 1


def test_training_loss_decreases():
    """Training loss should not increase as terms are added (refit path)."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data(noise=0.05)
    model = StagewiseSymbolicRegressor(n_terms=4, max_complexity=3, refit_coefficients=True)
    model.fit(X, y)

    losses = [h["train_loss"] for h in model.training_history_]
    assert len(losses) >= 2
    # OLS refit over a growing feature set is monotone non-increasing.
    for prev, curr in zip(losses[:-1], losses[1:], strict=False):
        assert curr <= prev + 1e-8
    # And it genuinely improved over the intercept-only model.
    assert losses[-1] < losses[0]


def test_recovers_simple_additive_function():
    """The model should recover y = 2*x0 + 0.5*x1**2 accurately."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data(noise=0.0)
    model = StagewiseSymbolicRegressor(n_terms=5, max_complexity=4, refit_coefficients=True)
    model.fit(X, y)

    r2 = model.score(X, y)
    assert r2 > 0.99


def test_refit_coefficients_false_predicts():
    """refit_coefficients=False produces a working model."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data(noise=0.05)
    model = StagewiseSymbolicRegressor(
        n_terms=6, learning_rate=0.5, max_complexity=3, refit_coefficients=False
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Stagewise weights are the learning rate.
    assert all(abs(c - 0.5) < 1e-12 for c in model.coefficients_)
    # Should still reduce error relative to the mean baseline.
    assert model.score(X, y) > 0.5


def test_repr_includes_terms():
    """The string representation lists the learned terms."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data()
    model = StagewiseSymbolicRegressor(n_terms=3, max_complexity=4)
    model.fit(X, y)

    text = str(model)
    assert "StagewiseSymbolicRegressor" in text
    assert "intercept" in text
    assert "terms" in text
    # At least one coefficient/term line rendered.
    assert "*" in text


def test_repr_before_fit_is_sklearn_style():
    """Before fitting, repr falls back to the sklearn-style parameter form."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    model = StagewiseSymbolicRegressor(n_terms=7)
    text = repr(model)
    assert text.startswith("StagewiseSymbolicRegressor(")
    assert "n_terms=7" in text


def test_early_stopping():
    """Early stopping stops before n_terms on a small validation split."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data(n=120, noise=0.02)
    model = StagewiseSymbolicRegressor(
        n_terms=20,
        max_complexity=3,
        refit_coefficients=True,
        early_stopping=True,
        validation_fraction=0.25,
        patience=2,
        random_state=0,
    )
    model.fit(X, y)

    # Every stage recorded a validation loss.
    assert all(h["val_loss"] is not None for h in model.training_history_)
    # It should stop early rather than using all 20 terms.
    assert model.n_terms_ < 20
    assert model.score(X, y) > 0.9


def test_to_expression():
    """to_expression returns a combined SymPy expression."""
    pytest.importorskip("sympy")
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data()
    model = StagewiseSymbolicRegressor(n_terms=3, max_complexity=4)
    model.fit(X, y)

    expr = model.to_expression()
    assert expr is not None
    # Should reference at least one feature symbol.
    assert any(sym.name in {"x0", "x1"} for sym in expr.free_symbols)


def test_fitted_attributes_consistent():
    """coefficients_, expressions_, and terms_ have matching lengths."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data()
    model = StagewiseSymbolicRegressor(n_terms=4, max_complexity=3)
    model.fit(X, y)

    n = model.n_terms_
    assert len(model.coefficients_) == n
    assert len(model.expressions_) == n
    assert len(model.terms_) == n
    assert len(model.learning_rates_) == n
    assert isinstance(model.intercept_, float)


def test_predict_before_fit_raises():
    """Calling predict before fit raises a clear error."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    model = StagewiseSymbolicRegressor()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((3, 2)))


def test_invalid_params_raise():
    """Invalid constructor parameters are rejected at fit time."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data(n=40)
    with pytest.raises(ValueError):
        StagewiseSymbolicRegressor(n_terms=0).fit(X, y)
    with pytest.raises(ValueError):
        StagewiseSymbolicRegressor(learning_rate=0.0).fit(X, y)


def test_non_finite_inputs_raise():
    """Non-finite X or y are rejected rather than silently producing NaN."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data(n=40)

    y_nan = y.at[0].set(jnp.nan)
    with pytest.raises(ValueError, match="non-finite"):
        StagewiseSymbolicRegressor(n_terms=2, max_complexity=2).fit(X, y_nan)

    X_inf = X.at[0, 0].set(jnp.inf)
    with pytest.raises(ValueError, match="non-finite"):
        StagewiseSymbolicRegressor(n_terms=2, max_complexity=2).fit(X_inf, y)


def test_predict_feature_count_mismatch_raises():
    """predict rejects inputs with the wrong number of features."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data(n=50, seed=1)
    model = StagewiseSymbolicRegressor(n_terms=2, max_complexity=2).fit(X, y)

    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((5, 5)))
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((5, 1)))


def test_single_feature_and_tiny_sample():
    """The model handles a single feature and very small sample sizes."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    rng = np.random.default_rng(3)
    X = jnp.array(rng.uniform(-2, 2, size=(60, 1)))
    y = 2.0 * X[:, 0]
    model = StagewiseSymbolicRegressor(n_terms=3, max_complexity=2).fit(X, y)
    assert model.score(X, y) > 0.99

    X_small, y_small = _additive_data(n=3)
    model2 = StagewiseSymbolicRegressor(n_terms=2, max_complexity=2).fit(X_small, y_small)
    assert model2.predict(X_small).shape == (3,)


def test_transcendental_terms_stay_finite():
    """Transcendental bases invalid on the domain must not yield NaN models.

    A ``log``/``sqrt``/``1/x`` basis can be selected yet produce NaN on data
    with the wrong sign; the stage must fall back to a finite basis so the
    ensemble never predicts NaN.
    """
    from jaxsr.additive import StagewiseSymbolicRegressor

    rng = np.random.default_rng(6)
    X = jnp.array(rng.uniform(-1.5, 1.5, size=(300, 2)))
    y = jnp.exp(0.5 * X[:, 0]) + X[:, 1]
    model = StagewiseSymbolicRegressor(
        n_terms=5, max_complexity=3, include_transcendental=True, include_ratios=True
    ).fit(X, y)

    preds = np.array(model.predict(X))
    assert np.all(np.isfinite(preds)), "model produced non-finite predictions"
    assert model.score(X, y) > 0.9


def test_determinism_with_random_state():
    """Two fits with the same random_state produce identical predictions."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data(n=80, noise=0.05, seed=2)
    kw = {
        "n_terms": 5,
        "max_complexity": 3,
        "early_stopping": True,
        "validation_fraction": 0.25,
        "random_state": 42,
    }
    a = StagewiseSymbolicRegressor(**kw).fit(X, y)
    b = StagewiseSymbolicRegressor(**kw).fit(X, y)
    assert np.allclose(np.array(a.predict(X)), np.array(b.predict(X)))


def test_save_load_roundtrip(tmp_path):
    """save/load reconstructs a model with identical predictions and state."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data(n=120, noise=0.05, seed=4)
    model = StagewiseSymbolicRegressor(n_terms=4, max_complexity=3, feature_names=["a", "b"]).fit(
        X, y
    )

    path = tmp_path / "additive_model.json"
    model.save(str(path))
    loaded = StagewiseSymbolicRegressor.load(str(path))

    assert np.allclose(np.array(model.predict(X)), np.array(loaded.predict(X)), atol=1e-6)
    assert loaded.intercept_ == pytest.approx(model.intercept_)
    assert np.allclose(loaded.coefficients_, model.coefficients_)
    assert loaded.expressions_ == model.expressions_
    assert loaded.n_terms_ == model.n_terms_


def test_backfitting_scaffold_raises():
    """The backfitting scaffold raises NotImplementedError on fit."""
    from jaxsr.additive import BackfittingSymbolicRegressor

    X, y = _additive_data(n=40)
    model = BackfittingSymbolicRegressor()
    with pytest.raises(NotImplementedError):
        model.fit(X, y)


def test_get_loss_and_squared_error():
    """The loss registry resolves names and squared error behaves correctly."""
    from jaxsr.additive import SquaredError, get_loss

    loss = get_loss("squared_error")
    assert isinstance(loss, SquaredError)

    y = jnp.array([1.0, 2.0, 3.0])
    assert loss.initial_prediction(y) == pytest.approx(2.0)
    resid = loss.negative_gradient(y, jnp.array([0.0, 0.0, 0.0]))
    assert np.allclose(np.array(resid), np.array(y))
    assert loss.loss(y, y) == pytest.approx(0.0)

    with pytest.raises(ValueError):
        get_loss("not_a_loss")


def test_loss_registry_and_parameters():
    """All registered losses resolve; parameterized losses validate inputs."""
    from jaxsr.additive import (
        AbsoluteError,
        HuberLoss,
        QuantileLoss,
        SquaredError,
        get_loss,
    )

    assert isinstance(get_loss("squared_error"), SquaredError)
    assert isinstance(get_loss("absolute_error"), AbsoluteError)
    assert isinstance(get_loss("huber"), HuberLoss)
    assert isinstance(get_loss("quantile"), QuantileLoss)
    # instances pass through unchanged
    q = QuantileLoss(0.9)
    assert get_loss(q) is q

    with pytest.raises(ValueError):
        HuberLoss(delta=0.0)
    with pytest.raises(ValueError):
        QuantileLoss(quantile=0.0)
    with pytest.raises(ValueError):
        QuantileLoss(quantile=1.0)


def test_loss_gradients_and_initial_predictions():
    """Loss initial predictions and pseudo-residuals are correct."""
    from jaxsr.additive import AbsoluteError, HuberLoss, QuantileLoss

    y = jnp.array([1.0, 2.0, 3.0, 100.0])  # last is an outlier

    mae = AbsoluteError()
    assert mae.initial_prediction(y) == pytest.approx(2.5)  # median
    grad = np.array(mae.negative_gradient(y, jnp.full(4, 2.5)))
    assert np.allclose(grad, [-1.0, -1.0, 1.0, 1.0])  # sign(y - pred)

    huber = HuberLoss(delta=1.0)
    r = np.array(huber.negative_gradient(jnp.array([0.0, 0.0]), jnp.array([-0.5, -5.0])))
    assert r[0] == pytest.approx(0.5)  # within delta -> raw residual
    assert r[1] == pytest.approx(1.0)  # beyond delta -> clipped to delta

    q = QuantileLoss(0.9)
    assert q.initial_prediction(jnp.arange(0.0, 101.0)) == pytest.approx(90.0, abs=1.0)
    g = np.array(q.negative_gradient(jnp.array([1.0, -1.0]), jnp.array([0.0, 0.0])))
    assert g[0] == pytest.approx(0.9)  # y > pred
    assert g[1] == pytest.approx(-0.1)  # y < pred  (q - 1)


def test_huber_and_absolute_error_robust_to_outliers():
    """Robust losses recover the clean signal far better under contamination."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    rng = np.random.default_rng(1)
    X = jnp.array(rng.uniform(-2, 2, size=(500, 2)))
    clean = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2
    y = np.array(clean) + rng.normal(0, 0.1, 500)
    idx = rng.choice(500, 40, replace=False)
    y[idx] += 30.0  # heavy one-sided outliers
    y = jnp.array(y)

    kw = {
        "n_terms": 8,
        "max_complexity": 4,
        "learning_rate": 0.5,
        "refit_coefficients": False,
    }

    def mae_vs_clean(loss):
        m = StagewiseSymbolicRegressor(loss=loss, **kw).fit(X, y)
        return float(np.mean(np.abs(np.array(m.predict(X)) - np.array(clean))))

    squared = mae_vs_clean("squared_error")
    huber = mae_vs_clean("huber")
    absolute = mae_vs_clean("absolute_error")

    assert huber < squared
    assert absolute < squared


def test_quantile_coverage():
    """Quantile regression yields approximately calibrated coverage."""
    from jaxsr.additive import QuantileLoss, StagewiseSymbolicRegressor

    rng = np.random.default_rng(2)
    X = jnp.array(rng.uniform(-2, 2, size=(500, 2)))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 + jnp.array(rng.normal(0, 1.0, 500))

    for target in (0.1, 0.9):
        m = StagewiseSymbolicRegressor(
            n_terms=10,
            max_complexity=3,
            learning_rate=0.5,
            loss=QuantileLoss(target),
            refit_coefficients=False,
        ).fit(X, y)
        coverage = float(np.mean(np.array(y) <= np.array(m.predict(X))))
        assert abs(coverage - target) < 0.06, f"q={target}: coverage {coverage:.3f}"


def test_refit_with_nonsquared_loss_warns_and_fits():
    """refit_coefficients=True with a non-squared loss warns and still fits."""
    from jaxsr.additive import StagewiseSymbolicRegressor

    X, y = _additive_data(n=200, noise=0.1, seed=3)
    with pytest.warns(UserWarning, match="least squares"):
        model = StagewiseSymbolicRegressor(
            n_terms=4, max_complexity=3, loss="huber", refit_coefficients=True
        ).fit(X, y)
    assert np.all(np.isfinite(np.array(model.predict(X))))


def test_save_load_preserves_parameterized_loss(tmp_path):
    """save/load round-trips a quantile loss with its parameter."""
    from jaxsr.additive import QuantileLoss, StagewiseSymbolicRegressor

    X, y = _additive_data(n=150, noise=0.2, seed=5)
    model = StagewiseSymbolicRegressor(
        n_terms=4, max_complexity=3, loss=QuantileLoss(0.75), refit_coefficients=False
    ).fit(X, y)

    path = tmp_path / "quantile_model.json"
    model.save(str(path))
    loaded = StagewiseSymbolicRegressor.load(str(path))

    assert isinstance(loaded.loss, QuantileLoss)
    assert loaded.loss.quantile == pytest.approx(0.75)
    assert np.allclose(np.array(model.predict(X)), np.array(loaded.predict(X)), atol=1e-6)


def test_scale_equivariance_and_tiny_targets():
    """Predictions scale with y, and tiny-magnitude targets still fit.

    Guards against float32 ill-conditioning in the coefficient refit: fitting
    an intercept by augmenting with a ones-column fails when the term columns
    are tiny (y ~ 1e-6); centering keeps the fit stable at any scale.
    """
    from jaxsr.additive import StagewiseSymbolicRegressor

    rng = np.random.default_rng(17)
    X = jnp.array(rng.uniform(-2, 2, size=(300, 2)))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 + jnp.array(rng.normal(0, 0.1, 300))

    base = StagewiseSymbolicRegressor(n_terms=5, max_complexity=4).fit(X, y)
    pbase = np.array(base.predict(X))

    scaled = StagewiseSymbolicRegressor(n_terms=5, max_complexity=4).fit(X, 3.7 * y)
    rel = np.max(np.abs(np.array(scaled.predict(X)) - 3.7 * pbase)) / (np.max(np.abs(pbase)) + 1e-9)
    assert rel < 1e-3, f"scale-equivariance broken: rel={rel:.2e}"

    for factor in (1e-6, 1e6):
        m = StagewiseSymbolicRegressor(n_terms=5, max_complexity=4).fit(X, factor * y)
        assert m.score(X, factor * y) > 0.98, f"failed at scale {factor:g}"


def test_refit_ols_zero_terms():
    """refit_ols with no columns returns the mean and empty coefficients."""
    from jaxsr.additive import refit_ols

    y = jnp.array([1.0, 3.0, 5.0])
    intercept, coefs = refit_ols(jnp.zeros((3, 0)), y)
    assert intercept == pytest.approx(3.0)
    assert coefs.shape == (0,)
