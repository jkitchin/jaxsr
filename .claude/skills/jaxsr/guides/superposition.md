# Superposition / Master Curves (`SuperpositionRegressor`)

Discover a **symbolic transform that collapses a family of curves** indexed by a
condition, and test whether the collapse actually holds.

## When to use this

| Situation | Use |
|-----------|-----|
| A set of curves indexed by a condition, believed to collapse under a shift | `SuperpositionRegressor` |
| Time–temperature / concentration / pressure / moisture superposition | `SuperpositionRegressor` |
| Finite-size scaling, Larson–Miller, isoconversional kinetics | `SuperpositionRegressor` |
| You already have shift factors and just want them scored | `collapse_rmse(z, y)` |
| One curve, no condition axis | ordinary `SymbolicRegressor` |

The thing this adds over classic automated superposition is the **law**: existing tools
return a list of per-curve shift numbers, which cannot extrapolate to a condition that
was never measured. This returns `s(c)` symbolically, so `predict` works at any condition.

## API

```python
from jaxsr import SuperpositionRegressor

model = SuperpositionRegressor(
    condition="temperature",     # column name (kelvin)
    abscissa="log_omega",        # column name -- ALREADY in log units
    response="log_Gp",           # column name
    channel=None,                # column name, or None for a single channel
    domain="frequency",          # "frequency" (z = x + s) | "time" (z = x - s)
    condition_scale="kelvin",    # enables arrhenius/wlf and E_eff; None = generic
    vertical_shift="none",       # "none" | "shared" | "per_channel"
    candidate_families=("arrhenius", "wlf", "polynomial"),
    max_terms=2,
    validation="loco",           # leave-one-condition-out -- this is the verdict
    n_stability=0,               # ensemble size; 0 disables
)
model.fit(data)                  # dict of arrays or a pandas DataFrame

model.shift_factors(T)                       # log10 a_T -- THE identified quantity
model.effective_activation_energy() / 1000   # kJ/mol, structure-independent
model.validity_report_.verdict               # supported | weakly_supported | not_supported
model.transform(data)                        # adds "z" (reduced abscissa) and "w"
model.predict(T_new, x_new)                  # works at unmeasured conditions
model.master_curve_[channel]                 # MasterCurve, with uncertainty band
print(model.summary())
```

## Report the transform, NOT the expression

This is the single most important thing to get right when helping a user here.

Only **62%** of fits in a 90-fit study selected the true Arrhenius basis. The other 38%
picked one of **nine** distinct structural forms that are numerically indistinguishable
over the measured range and produce **the same transform to ~0.01 decades**. Over a
realistic 60 K window, `1/(1+q)^2`, `1/(c2+q)^2` and low-order polynomials span nearly
the same function space.

| Report this | Not this |
|-------------|----------|
| `model.shift_factors(T)` | `model.shift_expression_` |
| `model.effective_activation_energy()` | "the data follows a WLF law" |
| `model.stability_["shift_factor_quantiles"]` | the modal selected structure |

`shift_expression_` exists and is accurate for the data — it is just not reproducible,
so a claim built on it will not survive a refit.

## Stability is NOT a validity test — it is anti-correlated with one

The intuition that an unstable expression signals trouble is backwards here:

| | true superposition | no valid transform exists |
|---|---|---|
| expression stable? | often not (38% disagreement) | **yes — 12/12 identical** |
| held-out collapse | at the noise floor | **8× the noise floor** |

A thermorheologically complex material (two relaxation groups, different activation
energies, so no scalar shift factor exists) gives a confident, stable, reproducible shift
law **and** a beautiful in-sample collapse. Never conclude from a visual collapse or from
`stability_`.

**Always read `validity_report_`:**

```python
r = model.validity_report_
r.verdict                  # the graded answer
r.noise_floor              # what the collapse is measured against
r.noise_floor_source       # "replicates" | "curve_smoother" | "surface"
r.holdout_ratio_median     # held-out collapse / noise floor -> the verdict
r.holdout                  # per-condition dicts: collapse_rmse, shift_error, coverage
r.flags                    # machine-readable warnings
print(r.summary())
```

Verdict thresholds (multiples of the noise floor, configurable via
`collapse_thresholds`): `<= 2x` supported, `<= 4x` weakly supported, else not supported.

Each `holdout` entry also carries `shift_aligned` — the shift that *would* have aligned
that curve best. Use it to attribute a bad collapse:

| `shift_error` | `collapse_rmse` | Diagnosis |
|---------------|-----------------|-----------|
| large | large | the law extrapolated wrongly — try more conditions or a different family |
| small | large | the curve does not collapse under *any* shift — superposition fails |

## Conventions that fail silently

| Convention | What goes wrong | Guard |
|------------|-----------------|-------|
| `domain` | Flips the sign of every shift factor and of `E_eff`, while every plot still looks perfect | No default guessing — declare it |
| Kelvin | Reciprocal features become nonsense | Non-positive rejected; a column topping out below 150 warns |
| Log abscissa | The module does **not** take logs for you | Pass `log10(omega)`, not `omega` |
| Vertical shift | Absorbs real structure | Off by default; turn on for physics, not for fit |

`domain` **names** the transform rather than changing it: the collapse is identical
either way because the data alone fixes the reduced coordinate. That is exactly why it
cannot be inferred.

## Choosing `candidate_families`

| Physics | Families |
|---------|----------|
| Thermally activated, Arrhenius expected | `("arrhenius", "polynomial")` |
| Polymer near `T_g`, WLF expected | `("wlf", "arrhenius", "polynomial")` |
| Condition is not a temperature | `("polynomial",)` + `condition_scale=None` |

`arrhenius` and `wlf` encode 1/T physics and are **refused** unless
`condition_scale="kelvin"`. `wlf` fits its denominator constant by profile likelihood, so
it is the slow one; drop it if selection is taking too long.

Note that `arrhenius` is the special case of `wlf` at `c2 = 1` — including both is fine
and is precisely why the *expression* is unstable while the transform is not.

## Weighting (on by default)

`y_q` is an *estimate*, and its standard error blows up near the edges of the condition
range — exactly where a shift law is most tempted to bend. Rows are weighted by
`1/sigma^2` of that estimate by default (`weighting="derivative_se"`), which uses
`sample_weight` so the weights steer *which terms are selected*, not only the
coefficients. On the synthetic Arrhenius benchmark at 3% noise this roughly halves the
shift error (0.023 → 0.013 decades) and moves `E_eff` from 52.9 ± 1.8 to 54.1 ± 1.2
kJ/mol against a true 55.9. Only pass `weighting="none"` if you have a reason.

## Data requirements

- **≥ 4 distinct conditions** to fit at all; **≥ 5** before leave-one-condition-out runs.
- Enough abscissa points per curve to support the surface: at least `(degree+1)^2` points
  per channel, so ~16 for the default cubic.
- **Replicates are valuable** — they give a direct noise floor
  (`noise_floor_source_ == "replicates"`), which is the yardstick the verdict uses.
  Without them the floor comes from single-condition curve smoothing, which is flagged.
- Curves must **overlap** after shifting. Non-overlapping curves give a low-coverage
  flag and are dropped from the held-out score.

## Worked example

```python
import numpy as np
from jaxsr import SuperpositionRegressor

R = 8.314462618
T_ref, E = 300.0, 55.9e3
rng = np.random.default_rng(0)

def master(z):
    return np.tanh(z) + 0.25 * z          # curvature is what makes the shift identifiable

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

## How it works (for explaining to users)

With `q = (c - c_ref)/c_ref` and `y(x, q) = f(x + σ·s(q)) + v(q)`, differentiating in `q`
eliminates the unknown master curve:

```
y_q = σ·s'(q)·y_x + v'(q)
```

so a sparse regression of `y_q` against `Θ(q) ⊙ y_x | Θ(q)` (built with
`BasisLibrary.add_block`) recovers `s'` and `v'` as coefficient *functions*. Each
candidate term carries an exact antiderivative, so integrating back with `s(c_ref) = 0`
stays symbolic. Both partials come from one `SurfaceDerivatives` fit per channel — never
finite differences of noisy data — and the master-curve smoother is a **separate** fit,
because reusing the discovery surface to certify the discovery defeats the validation.

Known limitation: this is the differential strategy, so it inherits the derivative
stage's bias. Over-smoothing in the condition direction flattens `s'` and biases `E_eff`
low by a few percent.

## See also

- `guides/surface-derivatives.md` — the derivative stage
- `guides/basis-library.md` — `add_block` structured blocks
- `guides/uncertainty.md` — `summarize_selection_replicates`, pipeline-level resampling
