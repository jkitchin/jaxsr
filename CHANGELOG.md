# Changelog

All notable changes to JAXSR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases prior to 0.2.2 predate this changelog; see the git history
for details.

## [Unreleased]

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
