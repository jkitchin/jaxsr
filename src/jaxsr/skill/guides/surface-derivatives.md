# Multivariate Derivative Estimation (`SurfaceDerivatives`)

Estimate **partial derivatives of a surface** — several partials from one smoothed
fit — for problems where the regression target or the basis library contains
derivatives.

## When to use this

| Situation | Use |
|-----------|-----|
| One state trajectory over time, need `dX/dt` | `estimate_derivatives(X, t, ...)` (see `jaxsr.dynamics`) |
| Whole ODE system from time series | `discover_dynamics(X, t, ...)` |
| Data is a surface over 2+ coordinates, need `y_x` **and** `y_T` | `SurfaceDerivatives` |
| PDE-style discovery: `u_t = F(u, u_x, u_xx, ...)` | `SurfaceDerivatives` |
| Transform/shift laws, e.g. `y(x, T) = f(x + s(T))` ⟹ `y_T = s'(T)·y_x` | `SurfaceDerivatives` |

`estimate_derivatives` differentiates along a **single** axis (`X` is
`(n_times, n_states)`, `t` is 1-D). It cannot give you two partials of one surface.

## API

The snippets in this section assume `coords` (an `(n, d)` array), `values` (`(n,)`) and
`sigma` come from the user's data; the worked examples further down are self-contained.

```python
import numpy as np
from jaxsr import SurfaceDerivatives

est = SurfaceDerivatives(method="tensor_spline")   # or "local_poly", "gp"
est.fit(coords, values, sigma=0.01)                # coords (n, d), values (n,)

y, dy = est.derivatives(coords, order=[(1, 0), (0, 1)])
# y  -> (n,)      smoothed surface
# dy -> (n, 2)    column 0 = d/dx0, column 1 = d/dx1

y, dy, dy_se = est.derivatives(coords, order=[(1, 0), (0, 1)], return_std=True)
print(est.summary())        # method, smoothing level, effective dof, residual std
```

Gridded data can be passed as axes plus an N-D array — no meshgrid needed:

```python
est = SurfaceDerivatives().fit([x_axis, T_axis], Y_grid)   # Y_grid (len(x), len(T))
y, dy = est.derivatives(est.coords_, order=[(1, 0), (0, 1)])  # coords_ is the flat (n, 2) grid
```

One-call convenience wrapper:

```python
from jaxsr import estimate_partial_derivatives

y, dy = estimate_partial_derivatives(coords, values, order=[(1, 0), (0, 1)],
                                     method="tensor_spline", sigma=0.01)
```

**Signature notes** (common mistakes):

| API | Wrong | Right |
|-----|-------|-------|
| `derivatives()` | `dy = est.derivatives(...)` | returns a **tuple** `(y, dy)`, or `(y, dy, std)` with `return_std=True` |
| `order` | `order=1`, `order="x"` | a tuple per dimension: `(1, 0)`, or a list of them |
| single order | expecting shape `(n,)` | a single tuple still returns `(n, 1)` |
| query points | grid axes | `derivatives()` takes an `(n_query, d)` array — use `est.coords_` for the sample locations |
| `sigma` | a variance | a **standard deviation**, scalar or per point |

## Choosing a method

| Method | Data | Cost | Derivative uncertainty | Notes |
|--------|------|------|------------------------|-------|
| `"tensor_spline"` (default) | gridded or scattered, any `d` | fast | from the penalized-fit posterior | Best default. Penalty chosen by GCV. |
| `"local_poly"` | scattered, irregular | moderate (per-point fits) | sandwich variance of the local fit | Good on uneven sampling; degree bounds the total order. |
| `"gp"` | scattered, irregular, small `n` | `O(n³)`, capped by `max_points=800` | exact posterior, grows away from data | Best uncertainty; use when `n` is a few hundred. |

Derivative orders are always analytic partials of the fitted smoother — never finite
differences of noisy raw data.

## Choosing the smoothing level

**The smoothing hyperparameter must never be tuned against the downstream symbolic
score.** If the smoother is selected by which law the regression likes best, it can
manufacture that law; the fit looks excellent and the failure is silent.

| `smoothing=` | Meaning |
|--------------|---------|
| `"auto"` (default) | GCV (spline, local poly) or marginal likelihood (GP) |
| `"sigma"` | requires `sigma` at `fit()`; matches residual scatter to the known noise (the `s = n·σ²` rule) |
| float | use it verbatim: penalty `λ` (spline), bandwidth (local poly), noise variance (GP) |

`smoothing_scale=3.0` multiplies whatever was selected — the cheapest way to check how
much a discovered coefficient depends on the derivative stage.

## Reporting the smoothing actually used

Smoothing biases derivatives toward zero, and any coefficient read off them inherits
that bias. Make it visible:

```python
print(est.summary())
est.smoothing_          # λ / bandwidth / noise variance actually used
est.smoothing_source_   # "gcv", "marginal_likelihood", "sigma", or "fixed"
est.effective_dof_      # effective degrees of freedom of the smoother
est.residual_std_       # residual scatter of the fit
```

Sensitivity check — rerun the whole pipeline at several smoothing scales and report the
spread, not just one number:

```python
for scale in (1.0, 3.0, 10.0):
    est = SurfaceDerivatives(smoothing_scale=scale).fit(coords, values, sigma=sigma)
    ...  # refit the symbolic stage, record the coefficient
```

## Example: PDE-style discovery

```python
import numpy as np
from jaxsr import BasisLibrary, SurfaceDerivatives, SymbolicRegressor

# Heat equation data: u_t = 0.1 * u_xx
x = np.linspace(0, 2 * np.pi, 40)
t = np.linspace(0, 1.0, 30)
xx, tt = np.meshgrid(x, t, indexing="ij")
u = np.exp(-0.1 * tt) * np.sin(xx) + 0.5 * np.exp(-0.9 * tt) * np.sin(3 * xx)
noise = 0.002
u_obs = u + np.random.default_rng(0).normal(0, noise, u.shape)

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
print(model.expression_)   # y = 0.0958*u_xx  (true 0.1; the gap is smoothing bias)
```

## Example: shift law (`y_T = s'(T)·y_x`)

```python
import numpy as np
from jaxsr import SurfaceDerivatives

# Synthetic time-temperature superposition surface (Arrhenius shift, E = 55.85 kJ/mol)
gas_r, energy, t_ref = 8.314e-3, 55.85, 350.0
x_axis = np.linspace(-2.0, 4.0, 20)          # log frequency
T_axis = np.linspace(320.0, 400.0, 12)       # temperature, K
xx, TT = np.meshgrid(x_axis, T_axis, indexing="ij")
shift = -(energy / (gas_r * np.log(10.0))) * (1.0 / TT - 1.0 / t_ref)
sigma = 0.01
Y_grid = 2.0 + 1.5 * np.tanh(0.8 * (xx + shift - 1.0))
Y_grid = Y_grid + np.random.default_rng(0).normal(0, sigma, Y_grid.shape)

est = SurfaceDerivatives(smoothing="sigma").fit([x_axis, T_axis], Y_grid, sigma=sigma)
_, d = est.derivatives(est.coords_, order=[(1, 0), (0, 1)])
y_x, y_T = d[:, 0], d[:, 1]

keep = np.abs(y_x) > 0.15 * np.abs(y_x).max()   # the ratio is ill-posed where y_x ≈ 0
s_prime = y_T[keep] / y_x[keep]
T_keep = est.coords_[keep, 1]

print(np.median(s_prime * gas_r * np.log(10.0) * T_keep**2))   # ≈ 55.9 kJ/mol
```

Then regress `s_prime` against `T_keep` with a `SymbolicRegressor` rather than assuming
the Arrhenius form.

## Pitfalls

- **Boundaries.** Every smoother is weakest at the edge of the data. Drop a margin
  before reading derivatives, or expect the largest errors there.
- **Dividing partials.** Ratios like `y_T / y_x` blow up where the denominator crosses
  zero. Mask small denominators, as above.
- **High orders.** Each extra order costs accuracy. `degree` must be at least the
  highest order requested (`tensor_spline`: per dimension; `local_poly`: total).
- **GP size.** `method="gp"` is `O(n³)`; above `max_points=800` it raises rather than
  hanging. Subsample or switch to `"tensor_spline"`.
- **Basis resolution.** The spline defaults to at most 12 basis functions per
  dimension because GCV under-smooths derivatives; pass `n_basis=` for finer structure.
