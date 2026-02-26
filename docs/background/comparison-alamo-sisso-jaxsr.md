# Comparison: ALAMO vs SISSO vs JAXSR

Symbolic regression from a predefined basis library requires solving a combinatorial
optimization problem: which subset of candidate terms best explains the data without
overfitting? ALAMO, SISSO, and JAXSR take different approaches to this problem.

## ALAMO (Automated Learning of Algebraic Models for Optimization)

ALAMO uses **mixed-integer quadratic programming (MIQP)** with branch-and-bound to
perform best subset selection.

- Introduces binary variables $z_j \in \{0,1\}$ for each candidate basis function
- Minimizes $\|y - \Phi w\|^2 + \lambda \sum z_j$ subject to $|w_j| \le M \cdot z_j$
  (big-M formulation)
- Solves exactly via commercial MIQP solvers (BARON, GAMS)
- **Guarantees global optimality** for a given basis library and complexity budget
- Uses adaptive sampling: fits a model, identifies regions of poor fit, samples new
  data there, and refits

The branch-and-bound explores a tree of subsets, pruning branches that provably cannot
beat the current best solution. This is exponential worst-case but fast in practice for
moderate library sizes (~50--100 terms).

**References:**

- Cozad, A., Sahinidis, N. V., & Miller, D. C. (2014). Learning surrogate models for
  simulation-based optimization. *AIChE Journal*, 60(6), 2211--2227.

## SISSO (Sure Independence Screening and Sparsifying Operator)

SISSO is a two-stage approach:

1. **Feature construction**: Recursively compose features through algebraic operations,
   generating millions of candidate descriptors.
2. **Sure Independence Screening (SIS)**: Rank features by correlation with the target
   and keep the top-$k$ (e.g., top 1000). This is a fast univariate filter.
3. **L0 regularization via exhaustive combinatorial search**: Among the screened subset,
   enumerate all combinations up to a given model size and pick the one minimizing
   residual error.

The L0 "compression" is brute-force enumeration over the screened features. SISSO's
innovation is that SIS makes this tractable---screen from millions down to ~1000, then
exhaustively search $\binom{1000}{2}$ or $\binom{1000}{3}$ combinations. For model
dimension > 3 or 4, even this becomes impractical, so SISSO uses a residual-based
iterative scheme: pick the best 1D descriptor, compute the residual, then pick the best
1D descriptor of the residual, and so on. This iterative scheme is a greedy
approximation, not truly L0 optimal for the joint problem.

**References:**

- Ouyang, R., Curtarolo, S., Ahmetcik, E., Scheffler, M., & Ghiringhelli, L. M. (2018).
  SISSO: A compressed-sensing method for identifying the best low-dimensional descriptor
  in an immensity of offered candidates. *Physical Review Materials*, 2(8), 083802.

## JAXSR

JAXSR offers four selection strategies with different optimality trade-offs:

| Strategy          | Algorithm                                         | Optimality                                      |
|-------------------|---------------------------------------------------|-------------------------------------------------|
| `exhaustive`      | Enumerate all subsets up to `max_terms`           | **Globally optimal** (for given library and IC) |
| `greedy_forward`  | Iteratively add the best-improving term           | Approximate (greedy)                            |
| `greedy_backward` | Iteratively remove the least-useful term          | Approximate (greedy)                            |
| `lasso_path`      | L1 screening followed by OLS refit on active sets | Heuristic                                       |

Key design decisions:

- **No MIQP solver needed.** JAXSR uses either exhaustive enumeration (small libraries)
  or greedy heuristics (large libraries), avoiding dependence on commercial solvers.
- **Information criteria as the sparsity penalty.** BIC $\approx n \log(\text{MSE}) + k \log(n)$
  naturally penalizes model complexity, replacing explicit L0 or big-M constraints.
- **LASSO for screening only.** When used, L1 identifies promising subsets, then OLS
  refits to avoid shrinkage bias (similar in spirit to SISSO's SIS screening).
- **Closed-form coefficient fitting** via `lstsq` (SVD-based). No iterative optimization
  for coefficients.
- **SISSO-style basis expansion is available** (`expand_sisso_style` in `basis.py`) but
  is optional and decoupled from selection.

## Side-by-Side Comparison

| Aspect                 | ALAMO                    | SISSO                        | JAXSR                                    |
|------------------------|--------------------------|------------------------------|------------------------------------------|
| Selection mechanism    | MIQP + branch-and-bound  | SIS + L0 exhaustive          | IC-based greedy / exhaustive / LASSO     |
| Global optimality      | Yes (via solver)         | Yes within screened set      | Yes only with `exhaustive` strategy      |
| Scalability            | ~50--100 basis functions | Millions (via SIS filtering) | ~20 exhaustive, ~1000+ with LASSO/greedy |
| Sparsity control       | Big-M + binary variables | L0 norm directly             | Information criteria (AIC/BIC/AICc)      |
| External solver needed | Yes (BARON/GAMS)         | No                           | No                                       |
| Adaptive sampling      | Yes (built-in)           | No                           | Yes (via `ActiveLearner`)                |
| Coefficient fitting    | QP (within MIQP)         | OLS on selected subset       | OLS or ridge (closed-form)               |

## Practical Implications

- **ALAMO** is the most theoretically rigorous. Branch-and-bound guarantees the globally
  optimal sparse model. However, it requires commercial solvers and scales poorly beyond
  ~100 candidates.
- **SISSO** scales to enormous feature spaces via SIS filtering, but the L0 search is
  still exhaustive over screened features, and the iterative residual scheme for models
  with more than 3--4 terms is a greedy approximation.
- **JAXSR** trades theoretical guarantees for accessibility and flexibility. For small
  problems, `exhaustive` gives the same global guarantee as ALAMO. For larger problems,
  greedy selection with BIC is a pragmatic approximation. The LASSO path approach is
  closest in spirit to SISSO's screening philosophy. No commercial solver dependencies
  are required, and JAX provides GPU acceleration for large datasets.
