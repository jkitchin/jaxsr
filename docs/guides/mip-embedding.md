# Embedding Models in MILPs (Pyomo and discopt)

A common reason to fit a surrogate is to optimize over it: pick the operating
conditions that minimize cost, or embed a unit model inside a larger design
problem. JAXSR models are a weighted sum of basis terms, so they can be written
directly into an optimization model. If every term is piecewise linear, that
embedding is an exact **mixed-integer linear program (MILP)**, which commercial
and open-source solvers (HiGHS, Gurobi, CPLEX, discopt) solve to proven global
optimality.

The workflow has two halves. First you fit with a basis library whose terms are
MILP-representable, most usefully **hinge** terms `max(0, x - k)` that give a
MARS-style piecewise-linear fit. Then you hand the fitted model to `to_pyomo` or
`to_discopt`, which writes the variables and constraints for you.

Install the optional dependencies with `pip install jaxsr[mip]` (Pyomo and the
HiGHS solver). For discopt, `pip install discopt`.

## Which terms are MILP-representable?

| Term | Builder | Formulation |
|------|---------|-------------|
| `1`, `x` | `add_constant()`, `add_linear()` | written directly |
| `max(0,x-k)`, `max(0,k-x)` | `add_hinges()` | 1 continuous + 1 binary, big-M |
| `abs(x)` | `add_transcendental(["abs"])` | 1 continuous + 1 binary, big-M |
| `I(c=v)` | `add_categorical_indicators()` | one-hot binary |
| `I(c=v)*x` | `add_categorical_interactions()` | exact McCormick product |

Everything else (polynomials, interactions, `log`, `exp`, ratios, power laws) is
nonlinear. You can still embed those with `allow_nonlinear=True`, which makes
the problem an MIQCP or MINLP that needs a nonlinear solver such as discopt,
BARON, SCIP or Gurobi's nonconvex mode.

## Step 1: fit a piecewise-linear model

`add_hinges` places knots for each continuous feature. Knots come from one of
three sources, in order of priority: an explicit `knots` dict, quantiles of the
training data `X`, or evenly spaced points inside `feature_bounds`. By default
each knot produces the mirrored pair `max(0, x - k)` and `max(0, k - x)`.

```python
import numpy as np
from jaxsr import BasisLibrary, SymbolicRegressor

rng = np.random.default_rng(0)
X = rng.uniform(0, 4, size=(300, 2))
cost = np.abs(X[:, 0] - 1.5) + 0.8 * np.maximum(0, 2.5 - X[:, 1]) + 0.3 * X[:, 1]

library = (
    BasisLibrary(n_features=2, feature_names=["T", "P"], feature_bounds=[(0, 4), (0, 4)])
    .add_constant()
    .add_linear()
    .add_hinges(X=X, n_knots=7)
    .add_transcendental(["abs"])
)
model = SymbolicRegressor(basis_library=library, max_terms=6, information_criterion="bic")
model.fit(X, cost)
print(model.expression_)
print(model.metrics_["r2"])
```

The knots in the term names are rounded to 6 significant digits so they survive
`to_callable()`, `to_sympy()` and `save()`/`load()` unchanged. Check that the
selected terms embed as a pure MILP before going further:

```python
from jaxsr import milp_incompatible_terms

print(milp_incompatible_terms(model))   # [] means a pure MILP
```

Knot placement matters more than knot count. The fit can only bend where a knot
exists, so if you know where the response changes behavior (a phase boundary,
a saturation point), pass those locations explicitly with
`knots={"T": [1.5], "P": [2.5]}`. Otherwise quantiles of the data are a good
default, and the information criterion keeps only the hinges that pay for
themselves.

## Step 2a: embed in Pyomo

`to_pyomo` populates a Pyomo `Block` (or returns a new `ConcreteModel` if you
pass none). Continuing from above:

```python
import pyomo.environ as pyo

m = pyo.ConcreteModel()
m.sr = pyo.Block()
model.to_pyomo(m.sr)                         # same as jaxsr.to_pyomo(model, m.sr)

m.obj = pyo.Objective(expr=m.sr.jaxsr_output, sense=pyo.minimize)
pyo.SolverFactory("appsi_highs").solve(m)

T_opt = pyo.value(m.sr.jaxsr_inputs["T"])
P_opt = pyo.value(m.sr.jaxsr_inputs["P"])
print(T_opt, P_opt, pyo.value(m.obj))
print(model.predict(np.array([[T_opt, P_opt]])))   # agrees with the objective
```

The block has these Pyomo components: `aux` (continuous variables, including
the inputs `aux["in_T"]` and the output `aux["out"]`), `binaries`, `integers`
and `cons`. The Python attributes `jaxsr_inputs`, `jaxsr_output` and
`jaxsr_onehot` are shortcuts to the variables you usually need.

### Connecting to your own variables

In a real design problem the inputs and output are already variables in your
model. Pass them through `inputs` and `output` and the block links them with
equality constraints:

```python
m = pyo.ConcreteModel()
m.T = pyo.Var(bounds=(0, 4))
m.P = pyo.Var(bounds=(0, 4))
m.cost = pyo.Var()
m.sr = pyo.Block()
model.to_pyomo(m.sr, inputs={"T": m.T, "P": m.P}, output=m.cost)

m.spec = pyo.Constraint(expr=m.T + m.P >= 5)         # your own constraints
m.obj = pyo.Objective(expr=m.cost)
pyo.SolverFactory("appsi_highs").solve(m)
print(pyo.value(m.T), pyo.value(m.P), pyo.value(m.cost))
```

## Step 2b: embed in discopt

`to_discopt` adds variables and constraints to a `discopt.Model` and returns a
`DiscoptEmbedding` with `inputs`, `output`, `onehot` and `binaries`. Every
variable name gets a `prefix` (default `"jaxsr"`), so to embed two models in one
problem give them different prefixes. Continuing from Step 1:

```python
import discopt

dm_model = discopt.Model("design")
emb = model.to_discopt(dm_model)             # same as jaxsr.to_discopt(model, dm_model)
dm_model.subject_to(emb.inputs["T"] + emb.inputs["P"] >= 5)
dm_model.minimize(emb.output)

result = dm_model.solve()
print(result.status, result.objective)
```

`to_discopt` accepts the same `inputs`, `output`, `bounds`, `output_sense` and
`allow_nonlinear` arguments as `to_pyomo`.

## Bounds are required and enforced

The big-M constants in the hinge and `abs` formulations come from the input
bounds, so every continuous input must have finite bounds. They are taken from,
in order: the `bounds` argument (`{"T": (0, 4)}` or a list in feature order),
the library's `feature_bounds`, or the min and max of the training data.

The embedding **enforces** those bounds on its inputs. This is deliberate: a
surrogate is only trustworthy where it was fitted, and an optimizer will
happily exploit extrapolation. If your problem needs a wider range, refit with
data that covers it rather than widening the bounds. Tight bounds also give
smaller big-M constants and faster solves.

## Making the MILP smaller with `output_sense`

The exact formulation of each hinge or `abs` term needs one binary. When a
piece is *convex in the direction you optimize*, the binary is unnecessary: if
the output is only ever minimized, a term with a positive coefficient on
`max(0, x - k)` is already tight with just `h >= 0` and `h >= x - k`.

```python
blk = model.to_pyomo(output_sense="min")   # drop binaries for convex pieces
```

Only use `output_sense` when the output appears *only* in a minimized objective
(for `"min"`) or *only* in a maximized one (for `"max"`). If the output also
appears in a constraint like `cost >= 2`, the relaxed pieces can float upward
and the answer is wrong. The default, `output_sense=None`, is exact in every
setting.

## Categorical inputs

Categorical features become one-hot binaries, plus an integer input variable
that uses JAXSR's `0..K-1` category encoding. `jaxsr_onehot[("catalyst", "Pd")]`
(Pyomo) or `emb.onehot[("catalyst", "Pd")]` (discopt) gives the binary for one
category, which is the variable you usually want to constrain or cost.

```python
import numpy as np
import pyomo.environ as pyo
from jaxsr import BasisLibrary, SymbolicRegressor

rng = np.random.default_rng(1)
cat = rng.integers(0, 3, size=200)
T = rng.uniform(300, 400, size=200)
y = 0.01 * T + 2.0 * (cat == 1) - 0.004 * T * (cat == 2)
X = np.column_stack([cat, T]).astype(float)

library = (
    BasisLibrary(
        n_features=2,
        feature_names=["catalyst", "T"],
        feature_types=["categorical", "continuous"],
        categories={0: ["Pt", "Pd", "Ni"]},
    )
    .add_constant()
    .add_linear()
    .add_categorical_indicators()
    .add_categorical_interactions()
)
model = SymbolicRegressor(basis_library=library, max_terms=5).fit(X, y)

blk = model.to_pyomo()
blk.obj = pyo.Objective(expr=blk.jaxsr_output, sense=pyo.maximize)
pyo.SolverFactory("appsi_highs").solve(blk)
print({c: pyo.value(b) for (_, c), b in blk.jaxsr_onehot.items()}, pyo.value(blk.obj))
```

## Nonlinear terms

If the best model needs a quadratic or a `log`, you can still embed it with
`allow_nonlinear=True`. The terms are written as Pyomo or discopt expressions
and the hinge, `abs` and categorical terms keep their MILP formulations. The
problem is then an MINLP, so use a solver that handles it; discopt solves these
to global optimality.

```python
import discopt
import numpy as np
from jaxsr import BasisLibrary, SymbolicRegressor

rng = np.random.default_rng(2)
X = rng.uniform(-1, 1, size=(100, 1))
y = (X[:, 0] - 0.3) ** 2 + 0.5 * np.maximum(0, X[:, 0])

library = (
    BasisLibrary(n_features=1, feature_names=["x"])
    .add_constant()
    .add_linear()
    .add_polynomials(max_degree=2)
    .add_hinges(knots={"x": [0.0]})
)
model = SymbolicRegressor(basis_library=library, max_terms=4).fit(X, y)

dm_model = discopt.Model("minlp")
emb = model.to_discopt(dm_model, allow_nonlinear=True)
dm_model.minimize(emb.output)
print(dm_model.solve().objective)
```

## Limitations

Models with parametric (`add_parametric`) or custom (`add_custom`) basis
functions cannot be embedded, because JAXSR cannot see inside them. Rational
forms, compositions and SISSO-style features are also not supported. The
embedding is built once from the fitted coefficients; if you refit, rebuild
the block. `MultiOutputSymbolicRegressor` is not supported directly, but you
can embed each `model.estimators_[i]` separately, giving each its own block
(Pyomo) or prefix (discopt).
