# Structured Basis Blocks

Ordinary symbolic regression asks *which terms* explain `y`. Some problems instead ask
**what function multiplies a known column** — and that is a different question:

```
y_c = s'(c) * y_x + v'(c)
```

Here `y_x` and `y_c` are measured (or estimated) columns, and `s'` and `v'` are the
unknown *functions* you want back. You cannot write that as a flat basis over
`(c, y_x)`: you need every candidate term of `s'` multiplied by `y_x`, and every
candidate term of `v'` standing alone.

`BasisLibrary.add_block` builds exactly that — a design-matrix block of the form
`Θ(a) ⊙ b`, where `Θ` is a basis over one variable and `b` is another column of the
data. A coefficient selected inside such a block is literally a term of the unknown
coefficient function multiplying `b`.

## Quick start

```python
from jaxsr import BasisLibrary

# The candidate terms of the unknown coefficient function, over its own argument
theta = (BasisLibrary(n_features=1, feature_names=["c"])
         .add_constant()
         .add_linear()
         .add_polynomials(max_degree=2))

# One block per unknown function
library = (BasisLibrary(n_features=2, feature_names=["c", "y_x"])
           .add_block(theta, multiply_by="y_x", block_name="horizontal")
           .add_block(theta, block_name="vertical"))

library.names
# ['y_x', 'c*y_x', 'c^2*y_x', '1', 'c', 'c^2']

library.blocks
# {'horizontal': [0, 1, 2], 'vertical': [3, 4, 5]}
```

Fit as usual. A selected `c*y_x` means the term `c` appears in `s'(c)`; a selected `c`
means it appears in `v'(c)`.

## What `add_block` handles for you

- **Names** are generated as `<basis>*<column>`. The constant collapses to just the
  column name (`1*y_x` → `y_x`), and a source name that is a bare sum is parenthesised
  (`1+c` → `(1+c)*y_x`).
- **Complexity** is inherited from the source, plus 1 for the multiplication, plus any
  `complexity_offset`.
- **Feature indices are remapped.** The source library is written against its own
  columns; `add_block` re-expresses it on the target's feature space, matching features
  by name. Pass `feature_map={"source_name": "target_name"}` when the names differ.
- **Parametric terms pass through as parametric** — bounds, `log_scale` and the name
  template are preserved, so profile-likelihood optimisation still applies *inside* the
  block. This is what lets a block carry a genuinely nonlinear candidate such as
  `1/(c2+q)^2` with `c2` free.
- **The source library is copied, not shared**, so one `theta` can seed several blocks
  without them interfering.

## Working with blocks

```python
library.filter_by_block(include="horizontal")   # -> [0, 1, 2]
library.filter_by_block(exclude=["vertical"])   # -> [0, 1, 2]

reduced = library.without_blocks("vertical")    # new library; the original is untouched
```

`without_blocks` returns a copy with parametric bookkeeping re-indexed, so the reduced
library is immediately fittable.

## The first diagnostic: did the block earn its place?

Comparing a fit against `library.without_blocks(...)` is the fastest check on a
structured library, and it is worth doing every time you add a second block.

```python
from jaxsr import SymbolicRegressor

full = SymbolicRegressor(basis_library=library, max_terms=3).fit(X, y)
no_vertical = SymbolicRegressor(
    basis_library=library.without_blocks("vertical"), max_terms=3
).fit(X, y)

print(full.metrics_["bic"], no_vertical.metrics_["bic"])
```

If dropping a block barely moves the information criterion, that block is not earning
its terms — and worse, it may be *competing* with another block for the same variation.
A horizontal/vertical identifiability trade-off is a real hazard whenever two blocks can
explain the same structure, and the symptom is a fit that looks fine while the two
recovered functions are individually meaningless.

## Where this is used

`SuperpositionRegressor` (see [Superposition](superposition.md)) is built on this: it
puts one block on `y_x` for the horizontal shift and, optionally, one unmultiplied block
(or one per channel, multiplied by a channel indicator) for the vertical shift. That
module is a worked example of the whole pattern — structured blocks, a parametric term
inside a block, and a `without_blocks`-style question about whether the vertical block
belongs.

## Limitation

Block functions are not deserializable. Like `add_custom`, the library config saves, but
the block has to be re-added after `load()`.

## See also

- [Superposition](superposition.md) — the pattern applied end to end
- [Multivariate derivative estimation](surface-derivatives.md) — where `y_x` and `y_c`
  usually come from
