# Superposition: Learning a Master Curve's Shift Law

A large family of experiments produces **a set of curves indexed by a condition**, where
the scientific claim is that some transform of the axes collapses them onto one master
curve:

- time–temperature, time–concentration, time–pressure, time–moisture superposition
- finite-size scaling in critical phenomena
- Larson–Miller and other creep master curves
- dose–response and isoconversional kinetics

In every case the practitioner does two things by hand: shift each curve until things
look aligned, then *assert* a functional form for how the shift depends on the condition
(Arrhenius, WLF, a polynomial). The shift factors are numbers; the law relating them to
the condition is a guess.

`jaxsr.superposition` learns that law symbolically, reconstructs the master curve, and —
most importantly — **tests whether the collapse actually holds**.

## Quick start

The snippets in this section assume `T`, `x` and `y` are your own measured arrays; the
[worked example](#worked-example) at the end of the guide is self-contained and runnable
as-is.

```python
import numpy as np
from jaxsr import SuperpositionRegressor

# A tidy table: one row per measurement. Any mapping works, so a dict of arrays
# and a pandas DataFrame are both fine.
data = {
    "temperature": T,      # kelvin
    "log_omega": x,        # log10 of frequency -- the abscissa is already in log units
    "log_Gp": y,           # log10 of the storage modulus
}

model = SuperpositionRegressor(
    condition="temperature",
    abscissa="log_omega",
    response="log_Gp",
    domain="frequency",              # sets the sign convention: z = x + s(c)
    condition_scale="kelvin",        # enables the reciprocal-condition families
    candidate_families=("arrhenius", "wlf", "polynomial"),
    max_terms=2,
    validation="loco",               # leave-one-condition-out: this is the verdict
)
model.fit(data)

print(model.summary())
```

```
SuperpositionRegressor
==============================================
domain            : frequency (z = x + s(c))
reference         : 300
channels          : ['log_Gp']
conditions        : 10

Transform (the identified quantity)
----------------------------------------------
           270  ->  +1.0406
       276.667  ->  +0.7912
       283.333  ->  +0.5528
           290  ->  +0.3245
       296.667  ->  +0.1059
       303.333  ->  -0.1037
           310  ->  -0.3049
       316.667  ->  -0.4982
       323.333  ->  -0.6840
           330  ->  -0.8627
  effective activation energy : 54.2 kJ/mol

Selected law (not the headline -- see shift_expression_ notes)
----------------------------------------------
  s(q) = -10.798*(q/(1.06991*(1.06991 + q)))

Superposition validity
==============================================
verdict              : supported
noise floor          : 0.03016 (from replicates)
in-sample collapse   : 0.03037
held-out collapse    : 0.03178 (1.05 x noise floor, 4 conditions)
held-out shift error : 0.01625
thresholds           : supported <= 2x, weakly <= 4x noise floor
```

That run is worth reading closely, because it is the module's central lesson in
miniature. The data was generated from a pure **Arrhenius** law with `E = 55.9 kJ/mol`,
and the Arrhenius basis (`1/(1+q)^2`) was in the library. The search did not pick it — it
picked the **WLF** term with a fitted denominator of 1.070. Since Arrhenius is exactly
the WLF term at `c2 = 1`, those two are the same function to within the noise, and no
information criterion can tell them apart.

So the reported *equation* names the wrong family, while the recovered **transform** is
right to about 0.02 decades and `E_eff` lands at 54.2 against a true 55.9.

Report the transform and `E_eff`. Do not report the equation.

## How it works

With the abscissa in log units and a dimensionless condition coordinate
`q = (c - c_ref) / c_ref`, the model is

```
y(x, q) = f(x + sigma * s(q)) + v(q)
```

where `f` is the unknown master curve, `s` the horizontal shift (`s = log10 a_T`), `v` an
optional vertical shift, and `sigma = +1` for a frequency-like abscissa, `-1` for a
time-like one.

Differentiating with respect to `q` **eliminates the unknown master curve entirely**:

```
y_q = sigma * s'(q) * y_x + v'(q)
```

So a sparse regression of `y_q` against the structured blocks
`Theta(q) ⊙ y_x | Theta(q)` recovers `s'` and `v'` as coefficient *functions*, which are
integrated back analytically with the anchor `s(0) = 0`. Both partials come from one
smoothed surface per channel ([`SurfaceDerivatives`](surface-derivatives.md)), never from
finite differences of noisy data.

The candidate library is small and physically motivated. Each entry is a term of `s'`
paired with its exact antiderivative:

| Family | Term of `s'(q)` | Recovered `s(q)` |
|---|---|---|
| `polynomial` | `1`, `q`, `q^2`, … | `q`, `q^2/2`, `q^3/3`, … |
| `arrhenius` | `1/(1+q)^2` | `q/(1+q)` |
| `wlf` | `1/(c2+q)^2`, `c2` fitted | `q/(c2*(c2+q))` |

The WLF denominator constant is a genuine free nonlinear parameter, optimised by profile
likelihood inside the selection loop (`BasisLibrary.add_parametric`).

## Two findings that shape the API

### 1. The transform is identifiable; the equation is not

In a 90-fit synthetic study, only **62%** of fits selected the true Arrhenius basis. The
other 38% picked one of nine distinct structural forms — combinations of a constant, `q`,
and a rational term with a different denominator — that are numerically indistinguishable
over the measured range and produce **the same transform to ~0.01 decades**. Over a
realistic 60 K window, `1/(1+q)^2`, `1/(c2+q)^2` and low-order polynomials span nearly
the same function space.

So do **not** report `shift_expression_` as your result:

```python
model.shift_expression_        # unstable across refits -- structurally
model.shift_factors(T)         # stable -- this is the identified quantity
model.effective_activation_energy() / 1000   # kJ/mol, comparable across structures
```

`E_eff = -ln(b) * R * T_ref * s'(0)` came back at 54.2 ± 1.3 kJ/mol against a true
55.9 across *every* structural variant.

### 2. Expression stability is anti-correlated with validity

The design intuition is that an unstable selected expression signals trouble. The
opposite happens:

| | true superposition | no valid transform exists |
|---|---|---|
| expression stable? | often not (38% structural disagreement) | **yes — 12/12 identical** |
| held-out collapse | at the noise floor | **8× the noise floor** |

A thermorheologically *complex* material — two relaxation groups with different
activation energies, so no scalar shift factor exists — produces a confident, stable,
reproducible shift law and a beautiful in-sample collapse. **Only collapse on withheld
conditions separates the cases.**

That is why `validity_report_` is the module's verdict, and why it is graded rather than
binary:

```python
report = model.validity_report_
report.verdict                  # 'supported' | 'weakly_supported' | 'not_supported'
report.noise_floor              # what the collapse is measured against
report.holdout_ratio_median     # held-out collapse as a multiple of the noise floor
report.holdout                  # per-condition numbers, including the shift error
report.flags                    # machine-readable warnings
print(report.summary())
```

Each withheld condition takes **no part** in the smoothing or in the discovery: the
pipeline is refitted without it and the withheld curve is shifted by prediction alone.
Each entry also records the shift that *would* have aligned that curve best, so a bad
collapse can be attributed either to the law extrapolating wrongly (`shift_error` large)
or to the curve not collapsing under any shift at all (`shift_error` small,
`collapse_rmse` large).

## Weighting

The regression target `y_q` is an *estimate*, and the smoother's uncertainty in it is far
from uniform: derivative standard errors blow up near the edges of the condition range —
exactly where a shift law is most tempted to bend. By default the rows are weighted by
`1/sigma^2` of that estimate (`weighting="derivative_se"`, using the `sample_weight`
support so the weights steer *which terms are selected*, not only their coefficients).

On the synthetic Arrhenius benchmark at 3% noise, over six noise realizations:

| `weighting` | median shift error | worst | `E_eff` (true 55.9) |
|---|---|---|---|
| `"none"` | 0.023 decades | 0.038 | 52.9 ± 1.8 kJ/mol |
| `"derivative_se"` | **0.013 decades** | **0.023** | **54.1 ± 1.2 kJ/mol** |

Pass `weighting="none"` to fit unweighted.

## The noise floor

A collapse cannot be better than the measurement noise, so the verdict is a ratio against
it. It is measured from the raw data alone, before any surface is fitted, by whichever of
these the data supports:

| `noise_floor_source_` | Estimator |
|---|---|
| `replicates` | Pooled within-group scatter of repeated `(channel, condition, x)` points |
| `curve_smoother` | Residual scatter of a smooth curve through each *single* condition |
| `surface` | Residual scatter of the derivative surface — flagged, since it smooths across conditions and can hide a collapse failure |

Measuring it before the surface fit is what lets it double as the smoother's known
`sigma` (`smoothing="sigma"`) without any circularity.

## Conventions worth being strict about

These are the classic source of silent error in superposition work, so the module checks
them rather than assuming:

**The domain names the transform; it does not change it.** The collapse is identical
either way — the data alone fixes the reduced coordinate. What `domain` decides is
whether the offset is reported as `+log a_T` or `-log a_T`. Getting it wrong flips the
sign of every shift factor and of the activation energy, **while every plot still looks
perfect**. There is no guessing from the data.

**Temperatures must be in kelvin.** `condition_scale="kelvin"` refuses non-positive
values and warns on a column that tops out below 150, which is almost certainly Celsius.
Reciprocal-condition families (`arrhenius`, `wlf`) are refused outright for a condition
that is not an absolute temperature — they encode 1/T physics.

**The regression runs in dimensionless `q`.** Design-matrix columns stay order-one
instead of mixing `T^-2` against polynomials.

**The master-curve smoother is separate from the derivative surface.** Reusing the
surface that produced the shift law to also certify it would defeat the validation.

## Multiple channels

`G'` and `G''` share one shift factor but have different shapes. Pass a channel column
and each gets its own surface and its own master curve, while the horizontal block stays
shared across all rows:

```python
model = SuperpositionRegressor(
    condition="temperature", abscissa="log_omega", response="value",
    channel="component",                    # e.g. "Gp" / "Gpp"
    vertical_shift="per_channel",           # none | shared | per_channel
)
model.fit(data)

model.master_curve_["Gp"]                   # a MasterCurve, with an uncertainty band
model.vertical_expressions_                 # {'Gp': ..., 'Gpp': ...}
```

## Stability ensemble

Resampling the *rows* fed to the sparse regression would perturb nothing about the
smoother that produced them, so each replicate regenerates the data and re-runs the whole
pipeline (see `bootstrap_model_selection(..., resample_fn=...)` for the general pattern):

```python
model = SuperpositionRegressor(..., n_stability=50, random_state=0)
model.fit(data)

model.stability_["feature_frequencies"]         # how often each term is selected
model.stability_["n_distinct_structures"]       # how many distinct laws appeared
model.stability_["shift_factor_quantiles"]      # {condition: (q05, q50, q95)}
model.stability_["effective_activation_energy"] # {'mean', 'sd', 'q05', 'q95', 'n'}
```

`stability_resampling="residual"` (default) re-draws measurement noise;
`"conditions"` resamples whole curves with replacement.

Read this as a *spread*, not as evidence. A tight `feature_frequencies` is not a reason
to believe the law — see finding 2 above. The useful outputs here are
`shift_factor_quantiles` and the `effective_activation_energy` spread, both of which are
structure-independent.

## Using the result

```python
model.shift_factors(np.array([290.0, 305.0]))   # log10 a_T at any condition, measured or not
model.transform(data)                            # adds 'z' (reduced abscissa) and 'w'
model.predict(temperature, log_omega)            # master curve shifted back to a condition
model.master_curve_[channel].predict(z)          # the master curve itself
```

`predict` at a condition that was never measured is the payoff of learning the law
symbolically rather than tabulating shift factors.

## Scoring a collapse you produced elsewhere

`collapse_rmse` is public so that a collapse done by hand, or by another tool, can be
scored the same way:

```python
from jaxsr import collapse_rmse

collapse_rmse(z, y)                    # pooled RMS scatter about a smooth curve
collapse_rmse(z, y, channel=labels)    # one master curve per channel
```

## Worked example

Self-contained, and runnable as-is:

```python
import numpy as np
from jaxsr import SuperpositionRegressor

R = 8.314462618
T_ref, E = 300.0, 55.9e3          # true activation energy, J/mol
rng = np.random.default_rng(0)

def master(z):
    """The (unknown, in real life) master curve. Curvature makes the shift identifiable."""
    return np.tanh(z) + 0.25 * z

rows = {"T": [], "x": [], "y": []}
x_grid = np.linspace(-2, 2, 16)
for T in np.linspace(270, 330, 8):
    s = E / (np.log(10) * R) * (1 / T - 1 / T_ref)     # true log10 a_T
    for _ in range(2):                                  # replicates -> a real noise floor
        rows["T"] += [T] * x_grid.size
        rows["x"] += list(x_grid)
        rows["y"] += list(master(x_grid + s) + rng.normal(0, 0.01, x_grid.size))
data = {k: np.asarray(v) for k, v in rows.items()}

model = SuperpositionRegressor(
    condition="T", abscissa="x", response="y",
    domain="frequency", condition_scale="kelvin", reference=T_ref,
    max_terms=2, validation="loco", max_holdout_conditions=3,
)
model.fit(data)

print(model.validity_report_.verdict)                        # 'supported'
print(model.effective_activation_energy() / 1000, "kJ/mol")  # 55.9, against a true 55.9
print(model.shift_factors([305.0]))                          # a condition never measured
```

## Limitations

- **Strategy 1 only.** This is the differential strategy: fit a surface, eliminate the
  master curve, sparse-regress the coefficient functions. It is cheap and it works, but
  it inherits the derivative stage's bias — over-smoothing in the condition direction
  flattens `s'` and biases `E_eff` low by a few percent. The direct-collapse strategy
  (propose `s(c)` symbolically, score by the 1-D collapse residual) removes derivative
  estimation at the cost of an outer symbolic loop, and is not implemented here.
- **At least four distinct conditions** are needed to separate a shift law from the
  master curve, and at least five before leave-one-condition-out validation will run.
- **The abscissa must already be in log units.** The module does not take logs for you,
  because whether your data is `omega` or `log omega` is not something to guess.
- A **vertical shift is off by default**. Turning it on adds a block that can absorb real
  structure, so switch it on because the physics calls for it, not to improve a fit.

## See also

- [Multivariate derivative estimation](surface-derivatives.md) — the derivative stage
- `BasisLibrary.add_block` — the structured `Theta(a) ⊙ column` blocks the regression uses
- `jaxsr.uncertainty.summarize_selection_replicates` — how the stability ensemble is summarised
