# Multivariate Derivative Estimation

Some discovery problems need derivatives *of a surface* rather than of a trajectory.
Two examples:

- **PDE-style discovery.** The target is `u_t` and the candidate library contains
  `u`, `u_x`, `u_xx`, ... — every one of them a partial derivative of the same
  measured field.
- **Transform / shift laws.** Time–temperature superposition rests on the identity

  ```
  y(x, T) = f(x + s(T))   =>   y_T = s'(T) * y_x
  ```

  with `x = log(omega)` and `y = log(G)`. Both partials have to come from **one**
  smoothed surface over `(x, T)` before the symbolic stage can see `s'(T)` at all.

`jaxsr.dynamics.estimate_derivatives` differentiates along a single axis: `X` is
`(n_times, n_states)` and `t` is a 1-D vector. That covers ODE discovery, but not
either case above. `SurfaceDerivatives` fills the gap.

## Quick start

```python
import numpy as np
from jaxsr import SurfaceDerivatives

# A surface sampled on a rectangular grid
x = np.linspace(0.0, 2.0, 25)
T = np.linspace(-1.0, 1.0, 20)
xx, TT = np.meshgrid(x, T, indexing="ij")
Y = np.sin(2 * xx) * np.exp(0.5 * TT)

sigma = 0.01
Y_obs = Y + np.random.default_rng(0).normal(0, sigma, Y.shape)

est = SurfaceDerivatives(method="tensor_spline").fit([x, T], Y_obs, sigma=sigma)

y, dy = est.derivatives(est.coords_, order=[(1, 0), (0, 1)])
y_x, y_T = dy[:, 0], dy[:, 1]

print(est.summary())
```

```
SurfaceDerivatives
========================================
method            : tensor_spline
data              : 500 points, 2 dimensions
penalty lambda    : 0.158489 (chosen by gcv)
effective dof     : 97.89
residual std      : 0.00988053
noise std used    : 0.01
basis per dim     : [12, 12] (degree 3)
```

Gridded data is passed as a list of axis arrays plus an N-D value array; scattered data
as an `(n_points, n_dims)` coordinate array plus a flat value array. Either way,
`est.coords_` holds the flattened sample locations, which is usually what you want to
evaluate at.

The one-call form:

```python
from jaxsr import estimate_partial_derivatives

y, dy = estimate_partial_derivatives(
    [x, T], Y_obs, order=[(1, 0), (0, 1)], method="tensor_spline", sigma=sigma
)
```

## Reading the API

`derivatives()` returns a tuple. The first element is the smoothed surface; the second
is a column per requested order, in the order you asked for them (continuing from the
quick start above):

```python
y, dy = est.derivatives(est.coords_, order=[(1, 0), (0, 1)])   # dy.shape == (n, 2)
y, dy = est.derivatives(est.coords_, order=(2, 0))             # dy.shape == (n, 1)
y, dy, dy_se = est.derivatives(est.coords_, order=[(1, 1)], return_std=True)
```

Each order is a tuple with one entry per coordinate: `(1, 0)` is the first partial with
respect to dimension 0, `(0, 1)` the first partial with respect to dimension 1,
`(2, 0)` the second partial in dimension 0, and `(1, 1)` the mixed partial. `sigma` is a
**standard deviation** (scalar or per point), not a variance.

`predict()` is the order-zero case, with optional standard errors:

```python
mean, std = est.predict(est.coords_, return_std=True)
```

## Choosing a smoother

```python
SurfaceDerivatives(method="tensor_spline")   # default
SurfaceDerivatives(method="local_poly")
SurfaceDerivatives(method="gp")
```

| Method | Best for | Cost | Uncertainty |
|--------|----------|------|-------------|
| `"tensor_spline"` | the default; gridded or scattered data in any dimension | fast — one penalized least-squares solve | posterior of the penalized fit |
| `"local_poly"` | irregular sampling, local structure | moderate — one weighted fit per query point | sandwich variance of the local fit |
| `"gp"` | irregular sampling, honest uncertainty, `n` up to a few hundred | cubic in `n`, capped by `max_points` | exact posterior; grows away from the data |

All three return **analytic** partials of the fitted smoother. None of them finite-differences
the raw data, which is what makes second derivatives usable at all under noise.

`degree` bounds what you can ask for: for `"tensor_spline"` no single dimension may be
differentiated more than `degree` times; for `"local_poly"` the *total* order may not
exceed `degree`. The default `degree=3` covers `u_xx` and mixed second partials.

## Choosing the smoothing level

This is the part that decides whether the downstream symbolic result is trustworthy.

**Never select the smoothing hyperparameter by the downstream symbolic score.** A
smoother tuned against the regression that consumes it can manufacture whichever law the
regression prefers — the fit looks excellent and the failure is silent. JAXSR therefore
only offers criteria that are blind to the symbolic stage:

| `smoothing=` | How the level is chosen |
|--------------|-------------------------|
| `"auto"` (default) | generalized cross-validation for `"tensor_spline"` and `"local_poly"`; log marginal likelihood for `"gp"` |
| `"sigma"` | from the noise you supply to `fit(..., sigma=...)`: the level whose residual sum of squares matches `n_points * sigma**2` |
| a float | used verbatim — the penalty `λ` for `"tensor_spline"`, the bandwidth for `"local_poly"`, the noise variance for `"gp"` |

Replicates are the cleanest source of `sigma`: pool the within-replicate variance and
pass its square root.

## Reporting and diagnosing smoothing bias

Smoothing flattens derivatives, and a flatter derivative reports a smaller coefficient.
The bias tracks the noise level and the smoothing level, and is roughly flat in the
amount of data — so more data does not remove it. It is predictable and calibratable,
but only if the smoothing level is visible (continuing from the quick start above):

```python
est.smoothing_          # λ, bandwidth, or noise variance actually used
est.smoothing_source_   # "gcv", "marginal_likelihood", "sigma", or "fixed"
est.effective_dof_      # effective degrees of freedom of the smoother
est.residual_std_       # residual scatter at the sample points
est.noise_std_          # noise level backing the reported uncertainties
print(est.summary())    # all of the above, formatted
```

`smoothing_scale` multiplies the selected level, which turns "how much does my answer
depend on the derivative stage?" into a three-line experiment:

```python
for scale in (1.0, 3.0, 10.0):
    est = SurfaceDerivatives(smoothing_scale=scale).fit([x, T], Y_obs, sigma=sigma)
    _, dy = est.derivatives(est.coords_, order=[(1, 0), (0, 1)])
    ...  # rerun the symbolic stage, record the coefficient
```

Report the spread across scales alongside the point estimate. A coefficient that moves
by 10% between `×1` and `×10` is telling you where its error bar really comes from.

## Worked example: PDE-style discovery

Recovering the heat equation `u_t = 0.1 * u_xx` from a noisy field:

```python
import numpy as np
from jaxsr import BasisLibrary, SurfaceDerivatives, SymbolicRegressor

x = np.linspace(0, 2 * np.pi, 40)
t = np.linspace(0, 1.0, 30)
xx, tt = np.meshgrid(x, t, indexing="ij")
u = np.exp(-0.1 * tt) * np.sin(xx) + 0.5 * np.exp(-0.9 * tt) * np.sin(3 * xx)

noise = 0.002
u_obs = u + np.random.default_rng(0).normal(0, noise, u.shape)

# One smoothed surface -> every partial the library needs
est = SurfaceDerivatives().fit([x, t], u_obs, sigma=noise)
values, d = est.derivatives(est.coords_, order=[(1, 0), (2, 0), (0, 1)])
u_x, u_xx, u_t = d[:, 0], d[:, 1], d[:, 2]

library = (
    BasisLibrary(n_features=3, feature_names=["u", "u_x", "u_xx"])
    .add_linear()
    .add_interactions(max_order=2)
)
model = SymbolicRegressor(basis_library=library, max_terms=1).fit(
    np.column_stack([values, u_x, u_xx]), u_t
)
print(model.expression_)
```

```
y = 0.09583*u_xx
```

The right term, with a coefficient about 4% low. Raising `smoothing_scale` to `10.0`
moves it to `0.09043` — the derivative stage, not the symbolic stage, is what sets that
digit.

## Worked example: a shift law

For `y(x, T) = f(x + s(T))`, the slope ratio *is* the derivative of the shift law. Here
the shift is Arrhenius, so `s'(T) = E / (R ln(10) T**2)` and the activation energy `E`
falls out of the ratio:

```python
import numpy as np
from jaxsr import SurfaceDerivatives

# Synthetic time-temperature superposition data
gas_r, energy, t_ref = 8.314e-3, 55.85, 350.0        # kJ/mol/K, kJ/mol, K
x_axis = np.linspace(-2.0, 4.0, 20)                  # log frequency
T_axis = np.linspace(320.0, 400.0, 12)               # temperature, K
xx, TT = np.meshgrid(x_axis, T_axis, indexing="ij")
shift = -(energy / (gas_r * np.log(10.0))) * (1.0 / TT - 1.0 / t_ref)

sigma = 0.01
Y_grid = 2.0 + 1.5 * np.tanh(0.8 * (xx + shift - 1.0))
Y_grid = Y_grid + np.random.default_rng(0).normal(0, sigma, Y_grid.shape)

# Both partials from one smoothed surface
est = SurfaceDerivatives(smoothing="sigma").fit([x_axis, T_axis], Y_grid, sigma=sigma)
_, d = est.derivatives(est.coords_, order=[(1, 0), (0, 1)])
y_x, y_T = d[:, 0], d[:, 1]

# The ratio is ill-posed wherever the master curve is flat
keep = np.abs(y_x) > 0.15 * np.abs(y_x).max()
s_prime = y_T[keep] / y_x[keep]
T_keep = est.coords_[keep, 1]

E_eff = np.median(s_prime * gas_r * np.log(10.0) * T_keep**2)
print(f"E_eff = {E_eff:.2f} kJ/mol")   # E_eff = 55.93 kJ/mol  (true 55.85)
```

`s_prime` versus `T_keep` is also an ordinary symbolic regression problem in its own
right — fit it with a `SymbolicRegressor` over a library containing `1/T**2` instead of
assuming the Arrhenius form, and let the selection decide.

## Pitfalls

- **Boundaries.** Every smoother is weakest at the edge of its data. Drop a margin
  before reading derivatives, or expect the largest errors there.
- **Ratios of partials.** Mask points where the denominator is near zero, as above.
- **High orders.** Accuracy degrades with each order. Second partials need a well-sampled
  surface; third and higher rarely survive realistic noise.
- **GP size.** `method="gp"` is cubic in the number of points and raises above
  `max_points=800` rather than hanging. Subsample, or use `"tensor_spline"`.
- **Basis resolution.** The spline uses at most 12 basis functions per dimension by
  default, because GCV picks the penalty that is best for the *fit* and tends to
  under-smooth derivatives. Pass `n_basis=` when the surface genuinely has finer
  structure.
- **Irregular sampling.** Rectangular grids are the best-tested case. For strongly
  irregular sampling prefer `"gp"` (or `"local_poly"`), and check the reported
  uncertainty rather than assuming it.
