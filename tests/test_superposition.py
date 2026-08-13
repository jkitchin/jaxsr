"""Tests for symbolic superposition (time-temperature superposition and friends)."""

import math
import warnings

import numpy as np
import pytest

from jaxsr.superposition import (
    GAS_CONSTANT,
    MasterCurve,
    ShiftTerm,
    SuperpositionRegressor,
    ValidityReport,
    collapse_rmse,
)

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

T_REF = 300.0
E_TRUE = 55.9e3  # J/mol


def true_shift(temperature, energy=E_TRUE, reference=T_REF):
    """Arrhenius shift factor log10(a_T), anchored at the reference temperature."""
    temperature = np.asarray(temperature, dtype=float)
    return energy / (math.log(10) * GAS_CONSTANT) * (1.0 / temperature - 1.0 / reference)


def master(z):
    """A smooth, curved master curve. Curvature is what makes the shift identifiable."""
    return np.tanh(z) + 0.25 * z


def make_table(
    n_conditions=8,
    n_abscissa=16,
    n_replicates=2,
    noise=0.005,
    seed=0,
    domain="frequency",
    energy=E_TRUE,
    vertical=0.0,
    channels=None,
    span=30.0,
):
    """
    Build a tidy superposition table whose collapse is exactly known.

    Parameters
    ----------
    n_conditions, n_abscissa, n_replicates : int
        Grid size.
    noise : float
        Measurement noise standard deviation.
    seed : int
        Seed for the noise.
    domain : str
        ``"frequency"`` builds ``z = x + s``, ``"time"`` builds ``z = x - s``.
    energy : float
        True activation energy, in J/mol.
    vertical : float
        Amplitude of a linear-in-q vertical shift. 0 disables it.
    channels : list, optional
        Channel labels. Each gets its own master curve but the same shift.
    span : float
        Half-width of the temperature window, in kelvin.

    Returns
    -------
    dict
        Column name mapped to a 1-D array.
    """
    rng = np.random.RandomState(seed)
    sign = 1.0 if domain == "frequency" else -1.0
    temperatures = np.linspace(T_REF - span, T_REF + span, n_conditions)
    x_grid = np.linspace(-2.0, 2.0, n_abscissa)
    labels = channels if channels is not None else [None]

    rows = {"T": [], "x": [], "y": [], "channel": []}
    for temperature in temperatures:
        shift = float(true_shift(temperature, energy))
        q = (temperature - T_REF) / T_REF
        for k, label in enumerate(labels):
            for _ in range(n_replicates):
                z = x_grid + sign * shift
                values = master(z) + 0.4 * k * np.tanh(0.5 * z)
                values = values + vertical * q + rng.normal(0.0, noise, x_grid.size)
                rows["T"].extend([temperature] * x_grid.size)
                rows["x"].extend(x_grid)
                rows["y"].extend(values)
                rows["channel"].extend([label] * x_grid.size)

    table = {k: np.asarray(v) for k, v in rows.items()}
    if channels is None:
        table.pop("channel")
    return table


def make_complex_table(energy_low=10e3, energy_high=180e3, noise=0.03, seed=0, span=50.0):
    """
    Build a thermorheologically *complex* table: two relaxation groups whose
    activation energies differ, so no scalar shift factor can collapse the family.

    Returns
    -------
    dict
        Column name mapped to a 1-D array.
    """
    rng = np.random.RandomState(seed)
    temperatures = np.linspace(T_REF - span, T_REF + span, 10)
    x_grid = np.linspace(-2.0, 2.0, 20)
    taus_fast, taus_slow = np.logspace(-1.5, -0.5, 3), np.logspace(0.5, 1.5, 3)

    rows = {"T": [], "x": [], "y": []}
    for temperature in temperatures:
        a_fast = float(true_shift(temperature, energy_low))
        a_slow = float(true_shift(temperature, energy_high))
        for _ in range(2):
            w_fast, w_slow = 10.0 ** (x_grid + a_fast), 10.0 ** (x_grid + a_slow)
            modulus = sum((w_fast * t) ** 2 / (1 + (w_fast * t) ** 2) for t in taus_fast)
            modulus += sum((w_slow * t) ** 2 / (1 + (w_slow * t) ** 2) for t in taus_slow)
            rows["T"].extend([temperature] * x_grid.size)
            rows["x"].extend(x_grid)
            rows["y"].extend(np.log10(modulus) + rng.normal(0.0, noise, x_grid.size))
    return {k: np.asarray(v) for k, v in rows.items()}


def fit_model(table=None, **kwargs):
    """Fit a SuperpositionRegressor with test-friendly defaults."""
    options = {
        "condition": "T",
        "abscissa": "x",
        "response": "y",
        "condition_scale": "kelvin",
        "reference": T_REF,
        "max_terms": 2,
        "validation": "none",
    }
    options.update(kwargs)
    model = SuperpositionRegressor(**options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(make_table() if table is None else table)
    return model


@pytest.fixture(scope="module")
def fitted():
    """A model fitted to clean Arrhenius data, shared across the read-only tests."""
    return fit_model()


# ===========================================================================
# TestShiftTerms
# ===========================================================================


class TestShiftTerms:
    """The antiderivatives must actually integrate the derivatives."""

    @pytest.mark.parametrize(
        "families,poly_degree",
        [(("polynomial",), 2), (("arrhenius",), 0), (("wlf",), 0), (("arrhenius", "wlf"), 1)],
    )
    def test_antiderivative_matches_numerical_integral(self, families, poly_degree):
        """Each term's antiderivative equals the quadrature of its derivative."""
        from jaxsr.superposition import _build_terms

        terms = _build_terms(families, poly_degree, (0.05, 5.0), reciprocal_ok=True)
        grid = np.linspace(-0.15, 0.15, 601)

        for term in terms:
            params = dict.fromkeys(term.param_bounds or {}, 0.4)
            derivative = np.asarray(term.deriv(grid, **params), dtype=float)
            analytic = np.asarray(term.antideriv(grid, **params), dtype=float)
            # Cumulative trapezoid anchored at q = 0, which is where the grid's
            # midpoint sits, matching the antiderivative's anchor.
            cumulative = np.concatenate(
                [[0.0], np.cumsum(np.diff(grid) * (derivative[1:] + derivative[:-1]) / 2)]
            )
            numerical = cumulative - cumulative[grid.size // 2]
            assert np.allclose(analytic, numerical, atol=1e-6), term.name

    def test_antiderivative_is_anchored_at_zero(self):
        """Every term integrates to zero at the reference, so s(c_ref) = 0."""
        from jaxsr.superposition import _build_terms

        terms = _build_terms(("polynomial", "arrhenius", "wlf"), 2, (0.05, 5.0), True)
        for term in terms:
            params = dict.fromkeys(term.param_bounds or {}, 0.4)
            assert float(np.asarray(term.antideriv(np.zeros(1), **params))[0]) == 0.0

    def test_reciprocal_families_rejected_without_kelvin(self):
        """1/T physics is refused for a condition that is not an absolute temperature."""
        from jaxsr.superposition import _build_terms

        with pytest.raises(ValueError, match="reciprocal-temperature physics"):
            _build_terms(("arrhenius",), 2, (0.05, 5.0), reciprocal_ok=False)

    def test_unknown_family_rejected(self):
        """A typo in candidate_families is caught, not silently ignored."""
        from jaxsr.superposition import _build_terms

        with pytest.raises(ValueError, match="Unknown candidate_families"):
            _build_terms(("arhenius",), 2, (0.05, 5.0), reciprocal_ok=True)

    def test_polynomial_only_still_offers_a_constant(self):
        """Dropping the polynomial family still leaves a constant in s'."""
        from jaxsr.superposition import _build_terms

        terms = _build_terms(("arrhenius",), 2, (0.05, 5.0), reciprocal_ok=True)
        assert "1" in [t.name for t in terms]

    def test_shift_term_is_parametric_flag(self):
        """A term with bounds reports itself parametric; a plain one does not."""
        plain = ShiftTerm("1", lambda q: np.ones_like(q), lambda q: q, "q")
        assert not plain.is_parametric


# ===========================================================================
# TestConstructorValidation
# ===========================================================================


class TestConstructorValidation:
    """Every configuration mistake should fail loudly at construction."""

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"domain": "wavelength"}, "domain must be one of"),
            ({"condition_scale": "celsius"}, "condition_scale must be"),
            ({"vertical_shift": "maybe"}, "vertical_shift must be one of"),
            ({"vertical_shift": "per_channel"}, "requires a channel column"),
            ({"validation": "cv"}, "validation must be one of"),
            ({"stability_resampling": "rows"}, "stability_resampling must be one of"),
            ({"weighting": "inverse"}, "weighting must be one of"),
            ({"poly_degree": -1}, "poly_degree must be non-negative"),
            ({"max_terms": 0}, "max_terms must be at least 1"),
            ({"n_stability": -3}, "n_stability must be non-negative"),
            ({"log_base": 1.0}, "log_base must be greater than 1"),
            ({"wlf_c2_bounds": (5.0, 1.0)}, "wlf_c2_bounds must satisfy"),
            ({"collapse_thresholds": (4.0, 2.0)}, "collapse_thresholds must satisfy"),
            ({"max_holdout_conditions": 1}, "max_holdout_conditions must be at least 2"),
        ],
    )
    def test_invalid_arguments_rejected(self, kwargs, match):
        """Bad constructor arguments raise ValueError with a specific message."""
        with pytest.raises(ValueError, match=match):
            SuperpositionRegressor(**kwargs)

    def test_methods_require_fitting(self):
        """Reading results before fit() raises rather than returning junk."""
        model = SuperpositionRegressor()
        for call in (
            model.shift_factors,
            model.vertical_shifts,
            model.transform,
            model.summary,
            model.effective_activation_energy,
        ):
            with pytest.raises(RuntimeError, match="must be fitted"):
                call()
        with pytest.raises(RuntimeError, match="must be fitted"):
            _ = model.shift_expression_


# ===========================================================================
# TestInputHandling
# ===========================================================================


class TestInputHandling:
    """Conventions that are classic sources of silent error are checked, not assumed."""

    def test_missing_column_names_the_columns_it_wanted(self):
        """A missing column raises KeyError listing what was looked for."""
        table = make_table(n_conditions=5, n_abscissa=10, n_replicates=1)
        table.pop("y")
        with pytest.raises(KeyError, match="'y'"):
            fit_model(table)

    def test_unequal_column_lengths_rejected(self):
        """Columns of different lengths are caught before any fitting happens."""
        table = make_table(n_conditions=5, n_abscissa=10, n_replicates=1)
        table["x"] = table["x"][:-3]
        with pytest.raises(ValueError, match="unequal lengths"):
            fit_model(table)

    def test_non_finite_values_rejected(self):
        """A NaN in the response is an error, not something to quietly drop."""
        table = make_table(n_conditions=5, n_abscissa=10, n_replicates=1)
        table["y"][7] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            fit_model(table)

    def test_too_few_conditions_rejected(self):
        """Three conditions cannot separate a shift law from the master curve."""
        table = make_table(n_conditions=3, n_abscissa=16, n_replicates=2)
        with pytest.raises(ValueError, match="at least 4 distinct conditions"):
            fit_model(table)

    def test_non_positive_kelvin_rejected(self):
        """A negative temperature under condition_scale='kelvin' is an error."""
        table = make_table(n_conditions=5, n_abscissa=10, n_replicates=1)
        table["T"] = table["T"] - 400.0
        with pytest.raises(ValueError, match="non-positive values"):
            fit_model(table)

    def test_celsius_looking_column_warns(self):
        """A column that tops out near room temperature is flagged as maybe-Celsius."""
        table = make_table(n_conditions=5, n_abscissa=12, n_replicates=1, span=15.0)
        table["T"] = table["T"] - 250.0  # 35-65 "degrees": positive, but far too cold
        model = SuperpositionRegressor(
            condition="T", abscissa="x", response="y", condition_scale="kelvin", validation="none"
        )
        with pytest.warns(RuntimeWarning, match="looks like Celsius"):
            model.fit(table)

    def test_empty_table_rejected(self):
        """An empty table is an error rather than an empty model."""
        with pytest.raises(ValueError, match="empty"):
            fit_model({"T": np.array([]), "x": np.array([]), "y": np.array([])})

    def test_reference_defaults_to_the_median_condition(self):
        """Without an explicit reference, the median condition anchors the transform."""
        table = make_table(n_conditions=5, n_abscissa=12, n_replicates=1)
        model = fit_model(table, reference=None)
        assert model.reference_ == pytest.approx(np.median(np.unique(table["T"])))


# ===========================================================================
# TestRecovery
# ===========================================================================


class TestRecovery:
    """The transform -- not the expression -- is what has to come back right."""

    def test_arrhenius_shift_recovered(self, fitted):
        """Shift factors match the truth well inside the 0.15-decade criterion."""
        conditions = np.asarray(fitted.conditions_)
        error = np.abs(fitted.shift_factors(conditions) - true_shift(conditions))
        assert np.median(error) < 0.02
        assert np.max(error) < 0.05

    def test_shift_is_zero_at_the_reference(self, fitted):
        """The anchor s(c_ref) = 0 holds exactly, by construction of the integrals."""
        assert fitted.shift_factors([T_REF])[0] == pytest.approx(0.0, abs=1e-12)

    def test_effective_activation_energy_recovered(self, fitted):
        """E_eff lands within a few percent of the truth."""
        energy = fitted.effective_activation_energy()
        assert energy == pytest.approx(E_TRUE, rel=0.10)

    def test_activation_energy_needs_a_temperature(self):
        """A generic condition has no activation energy, and says so."""
        table = make_table(n_conditions=6, n_abscissa=12, n_replicates=1)
        model = fit_model(
            table, condition_scale=None, candidate_families=("polynomial",), reference=T_REF
        )
        with pytest.raises(ValueError, match="absolute temperature"):
            model.effective_activation_energy()

    def test_gas_constant_sets_the_energy_units(self, fitted):
        """Passing R in kJ/(mol K) returns kJ/mol."""
        joules = fitted.effective_activation_energy()
        kilojoules = fitted.effective_activation_energy(gas_constant=GAS_CONSTANT / 1000.0)
        assert kilojoules == pytest.approx(joules / 1000.0)

    def test_time_domain_sign_convention(self):
        """Data built as z = x - s is recovered by domain='time' with the same sign."""
        table = make_table(domain="time", n_conditions=8, n_abscissa=16)
        model = fit_model(table, domain="time")
        conditions = np.asarray(model.conditions_)
        error = np.abs(model.shift_factors(conditions) - true_shift(conditions))
        assert np.median(error) < 0.02

    def test_domain_names_the_transform_rather_than_changing_it(self):
        """
        Reading a frequency table as a time table flips log(a_T) and E_eff.

        The collapse itself is identical either way -- the data fixes the reduced
        coordinate, and the domain only says whether to call the offset +log(a_T) or
        -log(a_T). That is precisely why the convention has to be declared rather than
        inferred from the data, and why getting it wrong is silent in every plot.
        """
        table = make_table(n_conditions=8, n_abscissa=16)
        right = fit_model(table, domain="frequency")
        wrong = fit_model(table, domain="time")
        conditions = np.asarray(right.conditions_)

        assert np.allclose(right.transform()["z"], wrong.transform()["z"])
        assert np.allclose(right.shift_factors(conditions), -wrong.shift_factors(conditions))
        assert right.effective_activation_energy() == pytest.approx(
            -wrong.effective_activation_energy()
        )

    def test_polynomial_only_library_still_recovers_the_transform(self):
        """Even without the true structure available, the transform comes back."""
        model = fit_model(candidate_families=("polynomial",), poly_degree=2, max_terms=2)
        conditions = np.asarray(model.conditions_)
        error = np.abs(model.shift_factors(conditions) - true_shift(conditions))
        assert np.median(error) < 0.05


# ===========================================================================
# TestTransformAndPredict
# ===========================================================================


class TestTransformAndPredict:
    """The reduced coordinates and the reconstructed master curve."""

    def test_transform_adds_reduced_coordinates(self, fitted):
        """transform() returns the input columns plus z and w."""
        out = fitted.transform()
        assert set(out) >= {"T", "x", "y", "z", "w"}
        assert out["z"].shape == out["x"].shape
        assert np.all(np.isfinite(out["z"]))

    def test_transform_collapses_the_family(self, fitted):
        """The reduced coordinates scatter about one curve at roughly the noise level."""
        out = fitted.transform()
        assert collapse_rmse(out["z"], out["w"]) < 0.02

    def test_transform_of_new_rows_reuses_the_fitted_anchor(self, fitted):
        """A subset transformed on its own gets the same reduced coordinates."""
        full = fitted.transform()
        subset = {k: v[:40] for k, v in fitted.transform().items() if k in ("T", "x", "y")}
        assert np.allclose(fitted.transform(subset)["z"], full["z"][:40])

    def test_master_curve_is_reconstructed_with_a_band(self, fitted):
        """Each channel gets a MasterCurve carrying an uncertainty band."""
        curves = fitted.master_curve_
        assert len(curves) == 1
        curve = next(iter(curves.values()))
        assert isinstance(curve, MasterCurve)
        assert curve.z.shape == curve.y.shape == curve.std.shape
        assert np.all(curve.std >= 0)
        assert curve.z_min < curve.z_max

    def test_master_curve_predict_and_covers(self, fitted):
        """The curve evaluates anywhere and reports where it is interpolating."""
        curve = next(iter(fitted.master_curve_.values()))
        inside = np.linspace(curve.z_min, curve.z_max, 20)
        assert np.all(curve.covers(inside))
        assert not curve.covers(np.array([curve.z_max + 5.0]))[0]
        assert np.allclose(curve.predict(inside), master(inside), atol=0.1)

    def test_predict_reproduces_the_training_response(self, fitted):
        """Shifting the master curve back reproduces the measured curves."""
        table = make_table()
        predicted = fitted.predict(table["T"], table["x"])
        assert np.sqrt(np.mean((predicted - table["y"]) ** 2)) < 0.05

    def test_predict_extrapolates_to_an_unmeasured_condition(self, fitted):
        """A condition never measured is shifted by prediction alone."""
        x = np.linspace(-1.0, 1.0, 12)
        unmeasured = 305.0
        predicted = fitted.predict(np.full(x.size, unmeasured), x)
        expected = master(x + true_shift(unmeasured))
        assert np.sqrt(np.mean((predicted - expected) ** 2)) < 0.05

    def test_predict_checks_shapes(self, fitted):
        """Mismatched condition and abscissa lengths are rejected."""
        with pytest.raises(ValueError, match="condition has"):
            fitted.predict(np.zeros(5) + T_REF, np.zeros(4))


# ===========================================================================
# TestChannels
# ===========================================================================


class TestChannels:
    """Several channels share one horizontal shift but get their own master curve."""

    def test_two_channels_share_one_shift(self):
        """A two-channel table yields one transform and two master curves."""
        table = make_table(channels=["Gp", "Gpp"], n_conditions=6, n_abscissa=14, n_replicates=1)
        model = fit_model(table, channel="channel")
        assert set(model.channels_) == {"Gp", "Gpp"}
        assert set(model.master_curve_) == {"Gp", "Gpp"}
        conditions = np.asarray(model.conditions_)
        error = np.abs(model.shift_factors(conditions) - true_shift(conditions))
        assert np.median(error) < 0.03

    def test_channel_required_when_ambiguous(self):
        """predict() refuses to guess which channel is meant."""
        table = make_table(channels=["Gp", "Gpp"], n_conditions=6, n_abscissa=14, n_replicates=1)
        model = fit_model(table, channel="channel")
        with pytest.raises(ValueError, match="pass channel="):
            model.predict([T_REF], [0.0])
        with pytest.raises(ValueError, match="Unknown channel"):
            model.predict([T_REF], [0.0], channel="Gppp")

    def test_single_channel_is_labelled_by_the_response_column(self, fitted):
        """Without a channel column the one channel takes the response column's name."""
        assert fitted.channels_ == ["y"]


# ===========================================================================
# TestVerticalShift
# ===========================================================================


class TestVerticalShift:
    """Vertical shifts are optional, and off by default."""

    def test_no_vertical_shift_by_default(self, fitted):
        """vertical_shift='none' leaves v identically zero."""
        assert fitted.vertical_expressions_ == {}
        assert np.allclose(fitted.vertical_shifts(), 0.0)

    def test_shared_vertical_shift_recovered(self):
        """A linear-in-q vertical offset is recovered along with the horizontal one."""
        table = make_table(vertical=0.6, n_conditions=8, n_abscissa=16)
        model = fit_model(table, vertical_shift="shared", max_terms=3)
        conditions = np.asarray(model.conditions_)
        q = (conditions - T_REF) / T_REF
        assert np.allclose(model.vertical_shifts(conditions), 0.6 * q, atol=0.05)
        error = np.abs(model.shift_factors(conditions) - true_shift(conditions))
        assert np.median(error) < 0.05

    def test_per_channel_vertical_shift_fits_each_channel(self):
        """Per-channel mode builds one vertical block per channel, sharing one shift."""
        table = make_table(channels=["Gp", "Gpp"], n_conditions=6, n_abscissa=14, n_replicates=1)
        model = fit_model(
            table,
            channel="channel",
            vertical_shift="per_channel",
            candidate_families=("polynomial",),
            poly_degree=1,
            max_terms=3,
        )
        assert set(model.vertical_expressions_) == {"Gp", "Gpp"}
        conditions = np.asarray(model.conditions_)
        error = np.abs(model.shift_factors(conditions) - true_shift(conditions))
        assert np.median(error) < 0.05


# ===========================================================================
# TestWeighting
# ===========================================================================


class TestWeighting:
    """The regression target is an estimate, so its precision varies across rows."""

    def test_weights_are_normalized_and_vary(self, fitted):
        """Derivative standard errors become weights averaging one."""
        _, partials = fitted._fit_surfaces(fitted._dataset, fitted.noise_floor_)
        weights = fitted._row_weights(partials)
        assert weights is not None
        assert weights.mean() == pytest.approx(1.0)
        assert np.all(weights > 0)
        assert np.ptp(weights) > 0  # edges really are less certain than the middle

    def test_weighting_can_be_switched_off(self):
        """weighting='none' fits unweighted and still recovers the transform."""
        model = fit_model(weighting="none")
        assert model._row_weights({"y_q_se": np.ones(5)}) is None
        conditions = np.asarray(model.conditions_)
        error = np.abs(model.shift_factors(conditions) - true_shift(conditions))
        assert np.median(error) < 0.05

    def test_degenerate_errors_fall_back_to_unweighted(self, fitted):
        """Constant or unusable standard errors make weighting a no-op, not a crash."""
        assert fitted._row_weights({"y_q_se": np.ones(20)}) is None
        assert fitted._row_weights({"y_q_se": np.zeros(20)}) is None
        assert fitted._row_weights({"y_q_se": np.full(20, np.nan)}) is None

    def test_zero_error_row_cannot_dominate(self, fitted):
        """A zero standard error is floored rather than given infinite precision."""
        errors = np.full(20, 0.1)
        errors[0] = 0.0
        weights = fitted._row_weights({"y_q_se": errors})
        assert np.all(np.isfinite(weights))
        assert weights[0] / weights[1] < 500


# ===========================================================================
# TestValidity
# ===========================================================================


class TestValidity:
    """Held-out collapse is the verdict; nothing else is."""

    def test_genuine_superposition_is_supported(self):
        """Real data collapses on withheld conditions at the noise floor."""
        table = make_table(n_conditions=8, n_abscissa=16, n_replicates=2, noise=0.02)
        model = fit_model(table, validation="loco", max_holdout_conditions=3)
        report = model.validity_report_
        assert report.verdict == "supported"
        assert report.holdout_ratio_median < 2.0
        assert report.noise_floor_source == "replicates"

    def test_thermorheologically_complex_data_is_rejected(self):
        """
        A family with no scalar shift factor fails on withheld conditions.

        This is the negative control the module exists for: the in-sample collapse looks
        respectable and the search still returns a confident shift law, so nothing short
        of a withheld condition catches it.
        """
        model = fit_model(make_complex_table(), validation="loco", max_holdout_conditions=3)
        report = model.validity_report_
        assert report.verdict == "not_supported"
        assert report.holdout_ratio_median > 4.0

    def test_complex_control_looks_fine_in_sample(self):
        """
        The negative control's *in-sample* collapse is nowhere near as damning.

        Guards the module's central claim: judging by the collapse you can see would
        pass a material that has no shift law at all.
        """
        model = fit_model(make_complex_table(), validation="loco", max_holdout_conditions=3)
        report = model.validity_report_
        assert report.in_sample_collapse < report.holdout_collapse_median

    def test_holdout_entries_carry_the_numbers(self):
        """Each withheld condition reports its own collapse and shift error."""
        table = make_table(n_conditions=8, n_abscissa=16, n_replicates=2, noise=0.02)
        model = fit_model(table, validation="loco", max_holdout_conditions=3)
        for entry in model.validity_report_.holdout:
            assert set(entry) >= {
                "condition",
                "collapse_rmse",
                "ratio",
                "shift_predicted",
                "shift_aligned",
                "shift_error",
                "coverage",
            }
            assert entry["ratio"] == pytest.approx(
                entry["collapse_rmse"] / model.validity_report_.noise_floor
            )
            assert 0.0 < entry["coverage"] <= 1.0

    def test_held_out_shift_is_predicted_not_aligned(self):
        """The predicted shift for a withheld condition is close to the best-aligning one."""
        table = make_table(n_conditions=8, n_abscissa=16, n_replicates=2, noise=0.01)
        model = fit_model(table, validation="loco", max_holdout_conditions=3)
        assert model.validity_report_.shift_error_median < 0.15

    def test_validation_can_be_switched_off(self, fitted):
        """validation='none' reports no verdict and flags why."""
        report = fitted.validity_report_
        assert report.verdict == "not_evaluated"
        assert "validation_disabled" in report.flags

    def test_noise_floor_falls_back_without_replicates(self):
        """Without replicates the floor comes from single-condition curves, and says so."""
        table = make_table(n_conditions=6, n_abscissa=20, n_replicates=1, noise=0.01)
        model = fit_model(table)
        assert model.noise_floor_source_ == "curve_smoother"
        assert model.noise_floor_ == pytest.approx(0.01, abs=0.01)

    def test_report_serializes_and_summarizes(self):
        """The report round-trips to a dict and renders as text."""
        table = make_table(n_conditions=8, n_abscissa=16, n_replicates=2, noise=0.02)
        model = fit_model(table, validation="loco", max_holdout_conditions=3)
        report = model.validity_report_
        assert isinstance(report, ValidityReport)
        as_dict = report.to_dict()
        assert as_dict["verdict"] == report.verdict
        assert "held-out collapse" in report.summary()


# ===========================================================================
# TestStability
# ===========================================================================


class TestStability:
    """Stability is reported, and explicitly is not the verdict."""

    def test_ensemble_reports_selection_frequencies(self):
        """The ensemble summarizes which terms get picked, and how E_eff moves."""
        model = fit_model(n_stability=6, random_state=0)
        stability = model.stability_
        assert stability["n_replicates"] + stability["n_failed"] == 6
        assert stability["feature_frequencies"]
        assert set(stability["shift_factor_quantiles"]) == set(model.conditions_)
        energy = stability["effective_activation_energy"]
        assert energy["n"] == stability["n_replicates"]
        assert energy["q05"] <= energy["mean"] <= energy["q95"]

    def test_condition_resampling_runs(self):
        """Whole-curve resampling is available as an alternative to residual draws."""
        model = fit_model(n_stability=4, stability_resampling="conditions", random_state=0)
        assert model.stability_["n_replicates"] >= 1
        assert model.stability_["resampling"] == "pipeline:conditions"

    def test_stability_is_none_when_disabled(self, fitted):
        """No ensemble is run unless one is asked for."""
        assert fitted.stability_ is None


# ===========================================================================
# TestReporting
# ===========================================================================


class TestReporting:
    """The human-facing surface leads with the transform, not the expression."""

    def test_summary_leads_with_the_transform(self, fitted):
        """summary() reports the transform, the law and the verdict, in that order."""
        text = fitted.summary()
        assert text.index("Transform") < text.index("Selected law")
        assert text.index("Selected law") < text.index("validity")
        assert "effective activation energy" in text

    def test_shift_expression_is_a_readable_law(self, fitted):
        """The selected law renders as an expression in q."""
        assert "q" in fitted.shift_expression_

    def test_surfaces_are_kept_for_inspection(self, fitted):
        """The derivative smoothers stay reachable so the smoothing level is visible."""
        surface = next(iter(fitted.surfaces_.values()))
        assert surface.smoothing_source_ in ("gcv", "marginal_likelihood", "sigma", "fixed")
        assert "SurfaceDerivatives" in surface.summary()

    def test_selection_model_is_exposed(self, fitted):
        """The underlying sparse regression is available for the usual diagnostics."""
        assert fitted.selection_model_.selected_features_
        assert fitted.selection_model_.metrics_["mse"] >= 0


# ===========================================================================
# TestCollapseRmse
# ===========================================================================


class TestCollapseRmse:
    """The standalone collapse scorer."""

    def test_exact_curve_scores_near_zero(self):
        """Points on a smooth curve scatter about it by essentially nothing."""
        z = np.linspace(-2, 2, 60)
        assert collapse_rmse(z, master(z)) < 1e-3

    def test_scatter_is_recovered(self):
        """Added noise comes back as roughly the noise level."""
        rng = np.random.RandomState(0)
        z = np.linspace(-2, 2, 200)
        y = master(z) + rng.normal(0, 0.05, z.size)
        assert collapse_rmse(z, y) == pytest.approx(0.05, abs=0.02)

    def test_channels_get_their_own_curve(self):
        """Two channels with different shapes still score as one pooled residual."""
        z = np.linspace(-2, 2, 80)
        stacked_z = np.concatenate([z, z])
        stacked_y = np.concatenate([master(z), -3.0 * master(z)])
        labels = np.array(["a"] * z.size + ["b"] * z.size)
        assert collapse_rmse(stacked_z, stacked_y, channel=labels) < 1e-2

    def test_length_mismatch_rejected(self):
        """Mismatched inputs raise rather than broadcasting."""
        with pytest.raises(ValueError, match="z has"):
            collapse_rmse(np.zeros(10), np.zeros(9))

    def test_non_finite_rejected(self):
        """A NaN cannot be scored."""
        with pytest.raises(ValueError, match="must be finite"):
            collapse_rmse(np.array([0.0, 1.0, np.nan]), np.zeros(3))
