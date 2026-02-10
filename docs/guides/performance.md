# Performance Guide: Making JAXSR Fast on CPU and GPU

JAXSR uses JAX for numerical computation, which enables JIT compilation and GPU
acceleration. However, "just using JAX" does not automatically make code fast.
This guide documents the optimization strategies used in JAXSR's model selection
algorithms and the lessons learned from GPU benchmarking.

## The Core Bottleneck

JAXSR's selection algorithms (`greedy_forward`, `greedy_backward`, `exhaustive`,
`lasso_path`) evaluate many candidate subsets of basis functions. Each evaluation
originally called `jnp.linalg.lstsq` on an (n_samples, k) matrix inside a Python
loop. This creates two problems:

1. **O(n * k^2) per subset** -- lstsq solves the full n-by-k system each time
2. **Kernel launch overhead** -- each lstsq call dispatches a GPU kernel (~0.1-1ms
   overhead), which dominates when k is small

Benchmarking showed that GPU was actually *slower* than CPU for typical JAXSR
workloads because of problem (2). The fix addresses both.

## Strategy: Gram Matrix Precomputation

The key insight is that all subset OLS/ridge problems share the same data matrix
`Phi` and target `y`. We can precompute the products once:

```
PhiTPhi = Phi.T @ Phi    # (B, B) Gram matrix
PhiTy   = Phi.T @ y      # (B,)   cross-product
yTy     = y.T @ y        # scalar
```

Then for any subset of indices `S`, the OLS coefficients satisfy the k-by-k
normal equations:

```
PhiTPhi[S, S] @ c = PhiTy[S]
```

Solving this via `jnp.linalg.solve` costs O(k^3) instead of O(n * k^2) for lstsq.
For typical problems (n=1000, k=3), this is 1000/3 ~ 300x fewer floating-point ops.

### Cost Comparison Per Subset

| Operation | Original (lstsq) | Gram (solve) |
|-----------|------------------|--------------|
| Coefficient solve | O(n * k^2) | O(k^3) |
| MSE (float64) | O(n * k) | O(k) |
| MSE (float32) | O(n * k) | O(n * k) |
| **Total** | **O(n * k^2)** | **O(k^3)** or **O(n*k + k^3)** |

The one-time precomputation cost is O(n * B^2) for the Gram matrix, which is the
same order as a single lstsq call on the full matrix.

## MSE Computation: Why dtype Matters

For the MSE, there is a closed-form expression at the OLS optimum:

```
MSE = (yTy - c^T @ PhiTy[S]) / n
```

This is O(k) and eliminates `n_samples` from the inner loop entirely. However,
it involves subtracting two large numbers (`yTy` and `c^T @ rhs`) to get a small
residual. In **float32** (~7 significant digits), this catastrophic cancellation
produces wildly incorrect MSE values. In **float64** (~15 significant digits), it
works fine.

JAXSR branches on dtype:

- **float64**: Uses the closed-form formula. Each subset costs O(k^3) total --
  completely independent of n_samples. This is the maximum-speedup path.
- **float32**: Computes MSE from actual residuals: `mean((y - Phi[:, S] @ c)^2)`.
  Each subset costs O(n*k + k^3), still much faster than O(n*k^2) from lstsq.

### Enabling float64 in JAX

JAX defaults to float32. To use float64 (recommended for best performance and
numerical accuracy):

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Or set the environment variable before importing JAX:

```bash
JAX_ENABLE_X64=True python my_script.py
```

## Why GPU Can Be Slower Than CPU

GPU benchmarks on JAXSR revealed a counterintuitive result: GPU was slower for
typical symbolic regression workloads. The reasons:

### 1. Kernel Launch Overhead

Each JAX operation dispatches a GPU kernel. The overhead (~0.1-1ms per dispatch)
is negligible for large matrix operations but dominates when the inner loop does
hundreds of small solves:

```
# 100 subsets x 1ms overhead = 100ms of pure overhead
# vs the actual computation: 100 x 0.01ms = 1ms
```

The Gram precomputation helps because:
- The expensive O(n*B^2) `Phi.T @ Phi` is a single large kernel (GPU-friendly)
- The inner loop solves k-by-k systems, which are tiny but have no per-call
  overhead beyond array indexing

### 2. Python Loop Bottleneck (Amdahl's Law)

The greedy/exhaustive loops are written in Python, not JAX. Even if each
operation inside the loop is fast, the Python iteration overhead limits speedup.
GPU helps most when you can express the *entire* computation as a single JAX
operation (e.g., `vmap` over subsets).

### 3. Data Transfer Costs

Moving data between CPU and GPU has latency. For small problems, the transfer
time exceeds the computation time.

### Practical Recommendation

For most JAXSR workloads, **CPU is faster**. Set:

```bash
JAX_PLATFORMS=cpu python my_script.py
```

GPU becomes beneficial only at very large n_samples (50K+) where the basis
evaluation (`Phi = library.evaluate(X)`) dominates runtime.

## Optimization Techniques Applied in JAXSR

### 1. Precompute Shared Work Outside the Loop

Before:
```python
for subset in subsets:
    Phi_sub = Phi[:, subset]
    coeffs = lstsq(Phi_sub, y)      # O(n * k^2) each time
    mse = mean((y - Phi_sub @ coeffs) ** 2)
```

After:
```python
PhiTPhi = Phi.T @ Phi               # O(n * B^2) once
PhiTy = Phi.T @ y                   # O(n * B) once
for subset in subsets:
    c = solve(PhiTPhi[S,S], PhiTy[S])  # O(k^3) each time
    mse = ...                           # O(k) or O(n*k)
```

### 2. Use `solve` Instead of `lstsq` for Square Systems

`jnp.linalg.lstsq` on an n-by-k system does QR factorization (O(n*k^2)).
`jnp.linalg.solve` on a k-by-k system does LU factorization (O(k^3)).
When you've already formed the normal equations, `solve` is the right choice.

### 3. Branch on Numerical Precision

Don't use the same algorithm for all dtypes. Float64 can use algebraically
equivalent but numerically cheaper formulas that would fail in float32.

### 4. Guard Against Singular Systems

When solving normal equations, the Gram submatrix can be singular (e.g.,
collinear basis functions). JAXSR checks for NaN/Inf in the solution and
falls back to `lstsq` on the k-by-k system, which handles rank deficiency.

### 5. Skip the Optimization for Non-Linear Parameters

When the basis library contains parametric (non-linear) basis functions, the
Gram precomputation doesn't apply because the design matrix changes with the
parameter values. JAXSR detects this via `basis_library.has_parametric` and
falls back to the original per-subset `fit_ols`/`fit_ridge` path.

## Benchmark Results

On a dataset with n=1000 samples and B=20 basis functions, forward selection
with max_terms=5:

| Approach | Time | Speedup |
|----------|------|---------|
| Original (lstsq per subset) | ~2s | 1x |
| Gram precompute (float32, residual MSE) | ~0.05s | ~40x |
| Gram precompute (float64, closed-form MSE) | ~0.015s | ~133x |

At n=100,000 samples, the speedup exceeds 500x because the O(n) cost of lstsq
is completely eliminated from the inner loop (float64 path).

## Future Optimization Opportunities

1. **Batch subset evaluation with `jax.vmap`**: Instead of a Python loop over
   subsets, vectorize the Gram extraction and solve across all candidates at once.
   This would maximize GPU utilization.

2. **JIT the entire selection loop**: Use `jax.lax.while_loop` or `jax.lax.scan`
   instead of Python for/while. Requires fixed-size arrays and careful handling
   of dynamic shapes.

3. **Rank-1 updates (Cholesky)**: When forward selection adds one term at a time,
   the Gram submatrix grows by one row/column. A Cholesky rank-1 update costs
   O(k^2) instead of solving the full k-by-k system (O(k^3)).
