# Resampling and Selection Stability

Bootstrap-based stability answers a question worth asking: *if I had collected this data
again, would symbolic regression have picked the same terms?* But the answer is only as
honest as the **unit you resample**, and getting that wrong does not produce an obvious
error. It produces a confident, narrow, plausible-looking spread that is far too narrow.

| Your rows are… | Resample | Call |
|---|---|---|
| Independent measurements | rows | `bootstrap_model_selection(model, X, y)` |
| Grouped — several rows per condition, curve, subject, batch | groups | `bootstrap_model_selection(model, X, y, groups=...)` |
| Produced by an upstream fit — smoother, estimated derivative, simulation | the whole pipeline | `bootstrap_model_selection(model, None, None, resample_fn=...)` |

The default is rows, which is right only for the first case.

## Grouped rows

If every row of an isotherm shares a temperature, a row bootstrap leaks: a replicate
that drops half the isotherm still trains on the other half. The spread it reports
reflects within-curve noise, not the between-condition variability you actually care
about.

```python
import numpy as np
from jaxsr import bootstrap_model_selection

# One label per row saying which condition / curve / subject it came from
groups = np.repeat([250.0, 275.0, 300.0, 325.0], 40)

stability = bootstrap_model_selection(model, X, y, n_bootstrap=200, groups=groups, seed=0)
stability["resampling"]        # 'groups'
stability["stability_score"]
```

Whole groups are drawn with replacement, so a replicate never sees part of a group it
also trained on. At least two distinct groups are required.

The same concern applies to cross-validation, where a row-level split leaks a group
across the train/test boundary and reports an optimistic score:

```python
from jaxsr import cross_validate

scores = cross_validate(model, X, y, groups=groups, cv=4)
scores["mean_test_score"], scores["std_test_score"]
```

Passing `groups` **promotes** `strategy` to `"group-kfold"` automatically, so you cannot
silently get a row-level split on grouped data. The available strategies are `"kfold"`,
`"group-kfold"` and `"leave-one-group-out"`.

`cv` cannot exceed the number of distinct groups — with 4 conditions the default `cv=5`
raises rather than quietly producing an empty fold. Either lower `cv`, or use
`strategy="leave-one-group-out"` when groups are few:

```python
scores = cross_validate(model, X, y, groups=groups, strategy="leave-one-group-out")
```

## Rows that came out of an upstream fit

When the regression target is an estimated derivative, each row is a spline evaluation
— not a measurement. Resampling those rows perturbs nothing about the smoother that
produced them, so the dominant error source is *invisible* to a row-wise bootstrap.

`resample_fn` regenerates the data per replicate, including the upstream stage:

```python
import numpy as np
from jaxsr import bootstrap_model_selection

def replicate(rng):
    """Called once per replicate with the bootstrap's RandomState."""
    raw = simulate_experiment(seed=rng.randint(2**31))   # or resample raw measurements
    smoother = fit_smoother(raw)                          # re-run the upstream stage
    return smoother.features(), smoother.derivative()     # -> (X_b, y_b)

stability = bootstrap_model_selection(
    model, None, None, n_bootstrap=90, seed=0, resample_fn=replicate
)
stability["resampling"]              # 'pipeline'
stability["n_distinct_structures"]
```

`X` and `y` go unused — pass `None` — because `resample_fn` supplies every replicate.
Taking the bootstrap's `RandomState` as its only argument is what keeps the ensemble
reproducible under `seed`.

`groups` and `resample_fn` are mutually exclusive: `resample_fn` already decides how a
replicate is built.

## Reporting replicates you produced yourself

If you orchestrated the replicates — a loop over simulated datasets, a batch of fits
from a cluster job — hand them to the same reporting code:

```python
from jaxsr import summarize_selection_replicates

models = [fit_one_replicate(i) for i in range(90)]
summary = summarize_selection_replicates(models, reference=full_fit, resampling="pipeline")

summary["stability_score"]
summary["n_distinct_structures"]
summary["feature_frequencies"]
summary["parameter_distributions"]   # for parametric bases
```

It also accepts mappings (`{"features": [...], "expression": "..."}`) or plain lists of
selected term names, so replicates from outside JAXSR report identically. Without a
`reference`, `stability_score` is the frequency of the most common structure.

Features are keyed by **basis identity**, not by rendered name, so a parametric basis
appears once as the registered `"exp(-a*x)"` rather than once per refitted value. Before
this, a parametric library could score 0.0 stability while every replicate selected the
same basis.

## Reading the result honestly

A narrow coefficient interval is misleading when the bootstrap keeps selecting a
*different symbolic structure*; that is why `n_distinct_structures` and `structures` are
reported alongside `feature_frequencies`.

And stability is not validity. It measures whether the search is reproducible, not
whether the model is right. In at least one well-documented case the two point in
opposite directions: see the
[superposition guide](superposition.md#2-expression-stability-is-anti-correlated-with-validity),
where a material with no valid transform produced a *perfectly* stable law while the
genuine case disagreed with itself 38% of the time. Whenever a held-out test is
available, that is the verdict; stability is context.

## See also

- [Superposition](superposition.md) — a module where stability and validity diverge
- [Sample weights](sample-weights.md) — weights follow their rows into every fold and
  every group resample
