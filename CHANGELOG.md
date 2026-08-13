# Changelog

All notable changes to JAXSR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases prior to 0.2.2 predate this changelog; see the git history
for details.

## [Unreleased]

### Added
- **Superposition / master curves** (`jaxsr.superposition`) — discovers a *symbolic*
  transform that collapses a family of curves indexed by a condition
  (time–temperature superposition and its many relatives), rather than returning a
  table of per-curve shift factors
  ([#14](https://github.com/jkitchin/jaxsr/issues/14)):
  - `SuperpositionRegressor` — takes a tidy `(condition, abscissa, response[, channel])`
    table and learns `s(c)` (and optionally `v(c)`) as a sparse symbolic law. It uses
    the differential strategy: fit one smoothed surface per channel, use
    `y_q = σ·s'(q)·y_x + v'(q)` to eliminate the unknown master curve, sparse-regress
    `s'` and `v'` against structured `Θ(q) ⊙ y_x | Θ(q)` blocks, then integrate back
    analytically with the anchor `s(c_ref) = 0`.
  - Candidate families `"arrhenius"`, `"wlf"` (with the denominator constant fitted by
    profile likelihood) and `"polynomial"`, each carrying an exact antiderivative so
    the recovered law is symbolic rather than a quadrature table.
  - `validity_report_` — the module's verdict, graded `supported` /
    `weakly_supported` / `not_supported` from **leave-one-condition-out collapse**
    against a measured noise floor. A withheld condition takes no part in the
    smoothing or the discovery and is shifted by prediction alone. This is deliberately
    the headline rather than expression stability, which is *anti*-correlated with
    validity: a material with no valid shift factor produces a confident, stable,
    reproducible shift law and a fine in-sample collapse.
  - `shift_factors()`, `transform()`, `predict()`, `master_curve_` (with uncertainty
    band), and `effective_activation_energy()` — a structure-independent physical
    summary, since structurally different laws routinely agree on the transform to
    ~0.01 decades while disagreeing on the equation.
  - `stability_` — a pipeline-level ensemble (residual or whole-curve resampling) that
    re-runs the smoother per replicate, reported via `summarize_selection_replicates`.
  - `collapse_rmse` — public scorer for a collapse produced by hand or another tool.
  - Rows are weighted by the precision of the estimated target
    (`weighting="derivative_se"`, built on the `sample_weight` support below), since
    derivative standard errors blow up at the edges of the condition range — exactly
    where a shift law is most tempted to bend. On a synthetic Arrhenius benchmark at
    3% noise this halves the shift error (0.023 → 0.013 decades) and tightens `E_eff`
    from 52.9 ± 1.8 to 54.1 ± 1.2 kJ/mol against a true 55.9.
  - Conventions are checked rather than assumed: the domain sign convention, kelvin
    (non-positive rejected, Celsius-looking warned), and a dimensionless condition
    coordinate so design columns stay order-one.
- Documentation for superposition: user guide (`docs/guides/superposition.md`), API
  reference page, matching skill guide, and a worked notebook
  (`docs/examples/superposition_master_curves.ipynb`) that walks the whole story on
  synthetic rheology — recovery against a known answer, the negative control that no
  scalar shift can collapse, and the stability trap.
- New guides `docs/guides/structured-blocks.md` and `docs/guides/resampling.md`, giving
  published coverage to the structured-block and grouped/pipeline-resampling features
  above, which previously existed only in the skill guides.
- **Multivariate derivative estimation** (`jaxsr.derivatives`) — analytic partial
  derivatives of a smoothed N-D surface, for problems whose regression needs more
  than one partial (PDE-style discovery `u_t = F(u, u_x, u_xx, ...)`, or transform
  laws such as `y(x, T) = f(x + s(T))` where `y_T = s'(T)·y_x`):
  - `SurfaceDerivatives` — fits a smoother to scattered or gridded data and returns
    requested mixed partials with standard errors. Three smoothers: `"tensor_spline"`
    (penalized tensor-product B-splines, the default), `"local_poly"` (local
    polynomial regression), and `"gp"` (Gaussian process with derivative posterior).
  - `estimate_partial_derivatives` — one-call convenience wrapper.
  - Smoothing is selected only by criteria blind to the downstream symbolic score
    (GCV, log marginal likelihood, or a supplied noise level), and the level actually
    used is reported via `smoothing_`, `smoothing_source_`, `effective_dof_`,
    `residual_std_`, and `summary()`. `smoothing_scale` re-runs the estimate at a
    deliberately different smoothing level to expose smoothing-induced bias.
- Documentation for multivariate derivative estimation: user guide
  (`docs/guides/surface-derivatives.md`), API reference page, and skill guide.
- **Structured basis blocks** (`jaxsr.basis`) — design-matrix blocks of the form
  `Θ(a) ⊙ b`, where `Θ` is a basis over one variable and `b` is another column of the
  data, typically a measured or estimated derivative. A coefficient selected inside
  such a block is literally a term of the unknown coefficient *function* multiplying
  `b`, which is what makes shift laws and implicit dynamics discoverable at all
  ([#18](https://github.com/jkitchin/jaxsr/issues/18)):
  - `BasisLibrary.add_block(library, multiply_by=..., block_name=...)` — copies another
    library's functions, optionally times a data column. Parametric bases carry over as
    parametric, so profile-likelihood optimisation still applies inside the block.
  - `BasisLibrary.blocks` — block label mapped to the indices it owns.
  - `BasisLibrary.filter_by_block(include=..., exclude=...)` — select indices by block
    membership.
  - `BasisLibrary.without_blocks(*names)` — a copy with whole blocks dropped, with
    parametric bookkeeping re-indexed. Dropping a block and refitting is the first
    diagnostic for a structured library: it answers whether the block earned its place.
  - `BasisFunction.block` records the label, and is included in `to_dict()`.
- **Grouped and pipeline-level resampling** — a stability score computed at the wrong
  level always looks better than the truth, so the resampling unit is now explicit
  ([#17](https://github.com/jkitchin/jaxsr/issues/17)):
  - `bootstrap_model_selection(..., groups=...)` resamples whole groups rather than
    rows, for data where several rows share one experimental condition, one measured
    curve, or one subject. Row resampling there reports a spread far narrower than the
    real between-group variability.
  - `bootstrap_model_selection(..., resample_fn=...)` regenerates the data per
    replicate, for rows that are themselves outputs of an upstream step — a smoother,
    an estimated derivative, a simulation. Resampling such rows perturbs nothing about
    the step that produced them, so the dominant error source is invisible to a
    row-wise bootstrap. The two are mutually exclusive, and the result reports which
    was used via `"resampling"`.
  - `summarize_selection_replicates(replicates, reference=..., resampling=...)` is now
    public and exported, so replicates produced outside JAXSR can be reported the same
    way.
  - `cross_validate(..., groups=..., strategy=...)` with strategies `"kfold"`,
    `"group-kfold"` and `"leave-one-group-out"`. Passing `groups` promotes `strategy`
    to `"group-kfold"`, so a row-level split cannot silently leak a group across the
    train/test boundary.
  - `jaxsr.metrics.group_indices(groups)` — the shared group-labelling helper.
- `BasisLibrary.canonical_name(index)` / `BasisLibrary.canonical_names` —
  the name a basis function was registered with, which for a parametric
  basis is the template (`"exp(-a*x)"`) rather than the fitted rendering
  (`"exp(-0.4913*x)"`). This is the stable identity to key on when
  aggregating across refits.
- `BasisLibrary.copy()` — an independent copy of a library, so repeated
  refits cannot rewrite the caller's basis names or rebind its parametric
  evaluation closures.
- `summarize_selection_replicates` (and therefore
  `bootstrap_model_selection`) now returns `"parameter_distributions"`:
  mean/sd/q05/q95/n of each parametric basis's nonlinear parameters across
  replicates.
- **`sample_weight` is now implemented** (#19). It was previously accepted by
  `SymbolicRegressor.fit()` and silently ignored, so a user passing measurement
  variances got an unweighted fit that looked like a weighted one. Weighted least
  squares is now applied consistently across:
  - all four selection strategies, so weights steer *which terms are chosen*, not
    only their coefficients;
  - the reported MSE, R², and the AIC/BIC/AICc computed from them;
  - constraint refitting (`fit_constrained_ols`), term pruning, and parametric
    parameter optimisation;
  - `sigma_`, `covariance_matrix_`, `coefficient_intervals()`,
    `predict_interval()`, `confidence_band()`, `anova()`, the bootstrap
    functions, `cross_validate()`, and jackknife+ conformal prediction.

  Weights are normalised to average 1, so only their ratios matter and the fit is
  invariant to their overall scale. The effective sample size in the information
  criteria stays the nominal `n` — weighting does not manufacture observations,
  which is why duplicating rows is not an equivalent trick. New
  `SymbolicRegressor.effective_sample_size_` reports the Kish effective sample
  size as a diagnostic, and `sample_weight_` exposes the normalised weights.

  Invalid weights (wrong length, negative, non-finite, all-zero) now raise
  `ValueError` instead of being ignored.

  `sample_weight` was also added to `fit_symbolic()`,
  `MultiOutputSymbolicRegressor.fit()`, `SymbolicRegressor.score()`,
  `fit_ols()`, `fit_ridge()`, `select_features()`, `cross_validate()`,
  `compute_mse/rmse/mae/r2/adjusted_r2/mape/all_metrics()`, `compute_cv_score()`,
  `compute_loo_mse()`, `compute_press()`, and `bootstrap_model_selection()`;
  `SymbolicRegressor.update()` gained `sample_weight_new`. Weighting composes
  with the group-aware resampling above: weights follow their rows into every
  fold and every group resample. `bootstrap_model_selection` rejects
  `sample_weight` together with `resample_fn`, since a replicate that
  regenerates its own rows leaves a stored weight with no row to belong to.
- New guide `docs/guides/sample-weights.md` (and the matching skill guide)
  covering weight semantics, the effective-sample-size policy, recipes for
  variance-derived and replicate weights, and what is deliberately left
  unweighted.

### Fixed
- `bootstrap_model_selection` could not aggregate parametric basis
  functions: `feature_frequencies` was keyed by the rendered name, so each
  replicate's re-optimised parameter produced a distinct key and
  `stability_score` was 0.0 even when every replicate selected the same
  basis. Features are now keyed by basis identity
  ([#16](https://github.com/jkitchin/jaxsr/issues/16)).
- Cloning an estimator (`bootstrap_model_selection`,
  `MultiOutputSymbolicRegressor`) shared the template's basis library. Fitting
  a parametric library rewrites basis names and rebinds evaluation closures in
  place, so clones overwrote each other's fitted parameters and left the
  original model predicting with the last clone's values. Clones now get their
  own `BasisLibrary.copy()`.
- `compute_all_metrics()` no longer reports `max_error` over rows that carry no
  weight.

## [0.3.0] - 2026-07-02

### Added
- **Additive symbolic regression** (`jaxsr.additive`) — fits models of the
  form `f(x) = c + Σ ηₖ·gₖ(x)` where each term is a small symbolic
  expression discovered by the existing JAXSR machinery (boosting with
  interpretable weak learners):
  - `StagewiseSymbolicRegressor` — boosting-style regressor that fits each
    new symbolic term to the current residual, with save/load support.
  - `BackfittingSymbolicRegressor` — GAM-style regressor that revises
    terms in place across sweeps, warm-started from a stagewise fit.
  - `RecursiveSymbolicRegressor` (experimental) — residual-guided
    expansion of the basis library.
  - Loss functions for robust and quantile regression: `SquaredError`,
    `AbsoluteError`, `HuberLoss`, `QuantileLoss`, plus `Loss`/`get_loss`
    registry.
  - `bootstrap_additive` / `bootstrap_predict_additive` — bootstrap
    structural uncertainty (term inclusion probabilities and a predictive
    ensemble).
  - `refit_ols`, `AdditiveSymbolicModel`, `additive_predict`.
- Documentation and examples for additive symbolic regression: user guide
  (`docs/guides/additive-symbolic-regression.md`), API reference, example
  notebook and script, and skill guide/template.
- `RELEASING.md` — release checklist and troubleshooting guide covering
  version bumps, CHANGELOG.md promotion, manuscript currency audit,
  notebook execution, tagging, GitHub release, PyPI trusted publishing,
  and Zenodo archival.
- `CHANGELOG.md` — this file, following Keep a Changelog format.
- `ROADMAP.md` — forward-looking design document for discopt-based MIQP
  best-subset selection (Tier 1) and combined selection plus constraint
  enforcement (Tier 2). Deferred until discopt/ripopt APIs stabilize.

### Fixed
- `SymbolicRegressor` and `SymbolicClassifier` no longer produce NaN
  predictions when the basis library contains functions that are
  non-finite on the training data — such basis functions are now removed
  (with a warning) before fitting.
- `SymbolicClassifier` now prunes negligible terms from fitted models.
- Coefficient refits guard against float32 ill-conditioning.
- Repaired corrupted cells in seven example notebooks (source lines with
  stripped newlines, character-exploded cells, lost indentation, stray
  parentheses) that made some cells fail or silently execute as no-ops.
  All 22 example notebooks now run to completion.
- `sklearn_integration.ipynb` used `rng.randn(...)` on a
  `np.random.default_rng()` generator (no such method) — replaced with
  `rng.standard_normal(...)`.
- `manuscript/jaxsr-paper.org`: the active-learning loop now uses
  `learner.suggest(n_points=1).points` — `suggest()` returns an
  `AcquisitionResult`, not a coordinate array.

## [0.2.2] - 2026-04-12

### Added
- Automated release publishing to PyPI via GitHub release trigger,
  using OIDC trusted publishing (no long-lived API tokens in the repo).
- Zenodo archival integration — each GitHub release is now automatically
  archived on Zenodo and issued a citable DOI.
- `CITATION.cff` with linked DOI, so GitHub renders a "Cite this
  repository" button in the sidebar.
- Zenodo DOI badge in `README.md`.
- GitHub release badge in `README.md`.
- Manuscript source now tracked in the repo at
  `manuscript/jaxsr-paper.org` and `manuscript/references.bib`.
- `.zenodo.json` with metadata used by Zenodo on each release.

### Fixed
- `manuscript/jaxsr-paper.org`: the `ResponseSurface` code example now
  passes the required `bounds` argument (was a runtime crash
  as-written).
- `manuscript/jaxsr-paper.org`: architecture section updated to reflect
  20 modules and ~20,000 lines of Python.
- `manuscript/jaxsr-paper.org`: `MultiOutputSymbolicRegressor` moved
  from "future work" to shipped contributions, reflecting that it is
  already exported from `jaxsr.__init__`.
- `manuscript/jaxsr-paper.org`: Physical Constraints section now
  documents `constraint_selection_weight` for constraint-aware
  selection, not only post-selection refit.

[Unreleased]: https://github.com/jkitchin/jaxsr/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jkitchin/jaxsr/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/jkitchin/jaxsr/releases/tag/v0.2.2
