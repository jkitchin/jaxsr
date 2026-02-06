"""
Basis Function Library for JAXSR.

Provides a flexible, extensible system for defining candidate basis functions
for symbolic regression.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass
class BasisFunction:
    """
    A single basis function with metadata.

    Parameters
    ----------
    name : str
        Human-readable name for the basis function.
    func : Callable
        Function that takes X of shape (n_samples, n_features) and returns
        array of shape (n_samples,).
    complexity : int
        Complexity score for Pareto optimization (higher = more complex).
    feature_indices : tuple
        Indices of features used by this basis function.
    func_type : str
        Type of function (for serialization): "constant", "linear", "polynomial",
        "interaction", "transcendental", "ratio", "custom".
    func_config : dict
        Configuration for reconstructing the function (for serialization).
    """

    name: str
    func: Callable[[jnp.ndarray], jnp.ndarray]
    complexity: int = 1
    feature_indices: tuple[int, ...] = ()
    func_type: str = "custom"
    func_config: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, X: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the basis function on input data."""
        return self.func(X)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excluding func)."""
        return {
            "name": self.name,
            "complexity": self.complexity,
            "feature_indices": self.feature_indices,
            "func_type": self.func_type,
            "func_config": self.func_config,
        }


@dataclass
class ParametricBasisInfo:
    """Metadata for a parametric basis function with free nonlinear parameters.

    Parameters
    ----------
    basis_index : int
        Index of the corresponding BasisFunction in the library.
    name : str
        Name template with parameter symbols, e.g. ``"exp(-a*x)"``.
    func : Callable
        Function ``(X, **params) -> array`` of shape ``(n_samples,)``.
    param_bounds : dict
        ``{param_name: (lower, upper)}`` bounds for optimisation.
    initial_params : dict
        ``{param_name: initial_value}`` midpoint (or geometric mean) guesses.
    log_scale : bool
        If True, optimise in log-space (useful when the parameter spans
        orders of magnitude).
    resolved_params : dict or None
        Optimised parameter values, set after fitting.
    """

    basis_index: int
    name: str
    func: Callable
    param_bounds: dict[str, tuple[float, float]]
    initial_params: dict[str, float]
    log_scale: bool = False
    resolved_params: dict[str, float] | None = None


def _safe_log(x: jnp.ndarray) -> jnp.ndarray:
    """Safe logarithm that handles non-positive values."""
    return jnp.where(x > 0, jnp.log(jnp.maximum(x, 1e-10)), jnp.nan)


def _safe_sqrt(x: jnp.ndarray) -> jnp.ndarray:
    """Safe square root that handles negative values."""
    return jnp.where(x >= 0, jnp.sqrt(jnp.maximum(x, 0)), jnp.nan)


def _safe_inv(x: jnp.ndarray) -> jnp.ndarray:
    """Safe inverse that handles zero."""
    return jnp.where(jnp.abs(x) > 1e-10, 1.0 / x, jnp.nan)


def _safe_div(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Safe division that handles zero denominator."""
    return jnp.where(jnp.abs(y) > 1e-10, x / y, jnp.nan)


class BasisLibrary:
    """
    Library of candidate basis functions for symbolic regression.

    Generates basis functions up to specified complexity from input features.
    Supports method chaining for convenient library construction.

    Parameters
    ----------
    n_features : int
        Number of input features.
    feature_names : list of str, optional
        Names for each feature. Defaults to ["x0", "x1", ...].
    feature_bounds : list of tuple, optional
        Bounds (lower, upper) for each feature. Used for constraint-aware
        basis generation and adaptive sampling.

    Examples
    --------
    >>> library = (BasisLibrary(n_features=2, feature_names=["x", "y"])
    ...     .add_constant()
    ...     .add_linear()
    ...     .add_polynomials(max_degree=3)
    ...     .add_interactions(max_order=2)
    ...     .add_transcendental(["log", "exp"])
    ... )
    >>> Phi = library.evaluate(X)
    """

    _VALID_FEATURE_TYPES = {"continuous", "categorical"}

    def __init__(
        self,
        n_features: int,
        feature_names: list[str] | None = None,
        feature_bounds: list[tuple[float, float]] | None = None,
        feature_types: list[str] | None = None,
        categories: dict[int, list] | None = None,
    ):
        self.n_features = n_features
        self.feature_names = feature_names or [f"x{i}" for i in range(n_features)]
        self.feature_bounds = feature_bounds
        self.feature_types = feature_types or ["continuous"] * n_features
        self.categories: dict[int, list] = categories or {}
        self.basis_functions: list[BasisFunction] = []
        self._compiled_evaluate: Callable | None = None
        self._parametric_info: list[ParametricBasisInfo] = []

        if len(self.feature_names) != n_features:
            raise ValueError(
                f"Number of feature names ({len(self.feature_names)}) "
                f"must match n_features ({n_features})"
            )

        if len(self.feature_types) != n_features:
            raise ValueError(
                f"Number of feature_types ({len(self.feature_types)}) "
                f"must match n_features ({n_features})"
            )

        for ft in self.feature_types:
            if ft not in self._VALID_FEATURE_TYPES:
                raise ValueError(
                    f"Invalid feature type '{ft}'. Must be one of {self._VALID_FEATURE_TYPES}"
                )

        for idx in self.categorical_indices:
            if idx not in self.categories:
                raise ValueError(
                    f"Feature {idx} ('{self.feature_names[idx]}') is categorical but "
                    f"no categories provided. Pass categories={{idx: [val1, val2, ...]}}"
                )

    @property
    def continuous_indices(self) -> list[int]:
        """Indices of continuous features."""
        return [i for i, t in enumerate(self.feature_types) if t == "continuous"]

    @property
    def categorical_indices(self) -> list[int]:
        """Indices of categorical features."""
        return [i for i, t in enumerate(self.feature_types) if t == "categorical"]

    def add_constant(self) -> BasisLibrary:
        """Add constant (intercept) term."""
        self.basis_functions.append(
            BasisFunction(
                name="1",
                func=lambda X: jnp.ones(X.shape[0]),
                complexity=0,
                feature_indices=(),
                func_type="constant",
                func_config={},
            )
        )
        self._compiled_evaluate = None
        return self

    def add_linear(self) -> BasisLibrary:
        """Add linear terms: x_i for each continuous feature."""
        for i in self.continuous_indices:
            name = self.feature_names[i]
            # Use default argument to capture i correctly
            self.basis_functions.append(
                BasisFunction(
                    name=name,
                    func=partial(lambda X, idx: X[:, idx], idx=i),
                    complexity=1,
                    feature_indices=(i,),
                    func_type="linear",
                    func_config={"feature_index": i},
                )
            )
        self._compiled_evaluate = None
        return self

    def add_polynomials(self, max_degree: int = 2) -> BasisLibrary:
        """
        Add polynomial terms: x_i^d for d in 2..max_degree.

        Parameters
        ----------
        max_degree : int
            Maximum polynomial degree (default 2).
        """
        if max_degree < 2:
            return self

        for i in self.continuous_indices:
            for d in range(2, max_degree + 1):
                name = f"{self.feature_names[i]}^{d}"
                self.basis_functions.append(
                    BasisFunction(
                        name=name,
                        func=partial(lambda X, idx, deg: X[:, idx] ** deg, idx=i, deg=d),
                        complexity=d,
                        feature_indices=(i,),
                        func_type="polynomial",
                        func_config={"feature_index": i, "degree": d},
                    )
                )
        self._compiled_evaluate = None
        return self

    def add_interactions(self, max_order: int = 2) -> BasisLibrary:
        """
        Add interaction terms: products of distinct features.

        Parameters
        ----------
        max_order : int
            Maximum interaction order (default 2 for pairwise).
        """
        for order in range(2, max_order + 1):
            for combo in itertools.combinations(self.continuous_indices, order):
                name = "*".join(self.feature_names[i] for i in combo)
                combo_list = list(combo)
                self.basis_functions.append(
                    BasisFunction(
                        name=name,
                        func=partial(
                            lambda X, indices: jnp.prod(X[:, indices], axis=1),
                            indices=jnp.array(combo_list),
                        ),
                        complexity=order,
                        feature_indices=tuple(combo),
                        func_type="interaction",
                        func_config={"feature_indices": combo_list, "order": order},
                    )
                )
        self._compiled_evaluate = None
        return self

    def add_transcendental(
        self,
        funcs: list[str] | None = None,
        safe: bool = True,
    ) -> BasisLibrary:
        """
        Add transcendental terms: log, exp, sqrt, inv, sin, cos.

        Parameters
        ----------
        funcs : list of str, optional
            Which functions to include. Defaults to ["log", "exp", "sqrt", "inv"].
        safe : bool
            If True, use safe versions that return NaN for invalid inputs.
        """
        funcs = funcs or ["log", "exp", "sqrt", "inv"]

        # Define function mappings with safe versions
        transcendental_map = {
            "log": (_safe_log if safe else jnp.log, "log({})", 2),
            "exp": (jnp.exp, "exp({})", 2),
            "sqrt": (_safe_sqrt if safe else jnp.sqrt, "sqrt({})", 2),
            "inv": (_safe_inv if safe else (lambda x: 1.0 / x), "1/{}", 2),
            "sin": (jnp.sin, "sin({})", 2),
            "cos": (jnp.cos, "cos({})", 2),
            "tan": (jnp.tan, "tan({})", 3),
            "sinh": (jnp.sinh, "sinh({})", 3),
            "cosh": (jnp.cosh, "cosh({})", 3),
            "tanh": (jnp.tanh, "tanh({})", 2),
            "abs": (jnp.abs, "abs({})", 1),
            "square": (jnp.square, "({})**2", 2),
        }

        for i in self.continuous_indices:
            for func_name in funcs:
                if func_name not in transcendental_map:
                    raise ValueError(
                        f"Unknown transcendental function: {func_name}. "
                        f"Available: {list(transcendental_map.keys())}"
                    )
                fn, template, complexity = transcendental_map[func_name]
                name = template.format(self.feature_names[i])
                self.basis_functions.append(
                    BasisFunction(
                        name=name,
                        func=partial(lambda X, idx, f: f(X[:, idx]), idx=i, f=fn),
                        complexity=complexity,
                        feature_indices=(i,),
                        func_type="transcendental",
                        func_config={"feature_index": i, "func_name": func_name},
                    )
                )
        self._compiled_evaluate = None
        return self

    def add_ratios(self, safe: bool = True) -> BasisLibrary:
        """
        Add ratio terms: x_i / x_j for all distinct pairs.

        Parameters
        ----------
        safe : bool
            If True, use safe division that returns NaN for zero denominator.
        """
        for i in self.continuous_indices:
            for j in self.continuous_indices:
                if i != j:
                    name = f"{self.feature_names[i]}/{self.feature_names[j]}"
                    if safe:
                        self.basis_functions.append(
                            BasisFunction(
                                name=name,
                                func=partial(
                                    lambda X, num, den: _safe_div(X[:, num], X[:, den]),
                                    num=i,
                                    den=j,
                                ),
                                complexity=2,
                                feature_indices=(i, j),
                                func_type="ratio",
                                func_config={"numerator_index": i, "denominator_index": j},
                            )
                        )
                    else:
                        self.basis_functions.append(
                            BasisFunction(
                                name=name,
                                func=partial(
                                    lambda X, num, den: X[:, num] / X[:, den],
                                    num=i,
                                    den=j,
                                ),
                                complexity=2,
                                feature_indices=(i, j),
                                func_type="ratio",
                                func_config={"numerator_index": i, "denominator_index": j},
                            )
                        )
        self._compiled_evaluate = None
        return self

    def add_custom(
        self,
        name: str,
        func: Callable[[jnp.ndarray], jnp.ndarray],
        complexity: int = 3,
        feature_indices: tuple[int, ...] | None = None,
    ) -> BasisLibrary:
        """
        Add a custom basis function.

        Parameters
        ----------
        name : str
            Human-readable name.
        func : callable
            Function that takes X of shape (n_samples, n_features) and returns
            array of shape (n_samples,).
        complexity : int
            Complexity score for Pareto optimization.
        feature_indices : tuple of int, optional
            Indices of features used by this function.

        Notes
        -----
        Custom functions cannot be serialized automatically. The library
        can still be saved, but the custom function will need to be re-added
        after loading.
        """
        self.basis_functions.append(
            BasisFunction(
                name=name,
                func=func,
                complexity=complexity,
                feature_indices=feature_indices or (),
                func_type="custom",
                func_config={"name": name},
            )
        )
        self._compiled_evaluate = None
        return self

    # ------------------------------------------------------------------
    # Categorical basis functions
    # ------------------------------------------------------------------

    def add_categorical_indicators(
        self,
        features: list[int] | None = None,
    ) -> BasisLibrary:
        """
        Add indicator (dummy variable) basis functions for categorical features.

        For each categorical feature with K categories, adds K-1 indicator
        functions using reference encoding (first category dropped) to avoid
        multicollinearity with the intercept.

        Parameters
        ----------
        features : list of int, optional
            Indices of categorical features to encode. Defaults to all
            categorical features.

        Returns
        -------
        self : BasisLibrary
            For method chaining.

        Raises
        ------
        ValueError
            If a specified feature is not categorical.
        """
        features = features if features is not None else self.categorical_indices

        for i in features:
            if self.feature_types[i] != "categorical":
                raise ValueError(f"Feature {i} ('{self.feature_names[i]}') is not categorical")

            cats = self.categories[i]
            # Reference encoding: drop the first category
            for cat_val in cats[1:]:
                name = f"I({self.feature_names[i]}={cat_val})"
                self.basis_functions.append(
                    BasisFunction(
                        name=name,
                        func=partial(
                            lambda X, idx, val: (X[:, idx] == val).astype(jnp.float32),
                            idx=i,
                            val=cat_val,
                        ),
                        complexity=1,
                        feature_indices=(i,),
                        func_type="indicator",
                        func_config={"feature_index": i, "category_value": cat_val},
                    )
                )

        self._compiled_evaluate = None
        return self

    def add_categorical_interactions(
        self,
        cat_features: list[int] | None = None,
        cont_features: list[int] | None = None,
    ) -> BasisLibrary:
        """
        Add interactions between categorical indicators and continuous features.

        For each (categorical feature, continuous feature) pair, creates
        indicator * continuous terms. This allows the model to learn
        different slopes per category.

        Parameters
        ----------
        cat_features : list of int, optional
            Categorical feature indices. Defaults to all categorical.
        cont_features : list of int, optional
            Continuous feature indices. Defaults to all continuous.

        Returns
        -------
        self : BasisLibrary
            For method chaining.
        """
        cat_features = cat_features if cat_features is not None else self.categorical_indices
        cont_features = cont_features if cont_features is not None else self.continuous_indices

        for ci in cat_features:
            cats = self.categories[ci]
            # Reference encoding: drop the first category
            for cat_val in cats[1:]:
                for cj in cont_features:
                    name = f"I({self.feature_names[ci]}={cat_val})" f"*{self.feature_names[cj]}"
                    self.basis_functions.append(
                        BasisFunction(
                            name=name,
                            func=partial(
                                lambda X, ci, val, cj: (
                                    (X[:, ci] == val).astype(jnp.float32) * X[:, cj]
                                ),
                                ci=ci,
                                val=cat_val,
                                cj=cj,
                            ),
                            complexity=2,
                            feature_indices=(ci, cj),
                            func_type="categorical_interaction",
                            func_config={
                                "categorical_index": ci,
                                "category_value": cat_val,
                                "continuous_index": cj,
                            },
                        )
                    )

        self._compiled_evaluate = None
        return self

    # ------------------------------------------------------------------
    # Parametric basis functions
    # ------------------------------------------------------------------

    @property
    def has_parametric(self) -> bool:
        """Whether the library contains parametric basis functions."""
        return len(self._parametric_info) > 0

    def add_parametric(
        self,
        name: str,
        func: Callable,
        param_bounds: dict[str, tuple[float, float]],
        complexity: int = 3,
        feature_indices: tuple[int, ...] | None = None,
        log_scale: bool = False,
    ) -> BasisLibrary:
        """
        Add a parametric basis function with free nonlinear parameters.

        The nonlinear parameter(s) are optimised via profile likelihood during
        model selection: for each candidate value the linear coefficients are
        solved exactly via OLS.

        Parameters
        ----------
        name : str
            Name with parameter symbols, e.g. ``"exp(-a*x)"``.
        func : callable
            ``func(X, **params) -> array`` of shape ``(n_samples,)``.
        param_bounds : dict
            ``{param_name: (lower, upper)}`` search bounds for each parameter.
        complexity : int
            Complexity score for Pareto optimisation.
        feature_indices : tuple of int, optional
            Indices of input features used by this function.
        log_scale : bool
            If True, search in log-space (useful for parameters that span
            orders of magnitude).

        Returns
        -------
        self : BasisLibrary
            For method chaining.

        Examples
        --------
        >>> library.add_parametric(
        ...     name="exp(-a*x)",
        ...     func=lambda X, a: jnp.exp(-a * X[:, 0]),
        ...     param_bounds={"a": (0.01, 10.0)},
        ...     feature_indices=(0,),
        ... )
        """
        import re

        # Compute initial guesses
        initial_params: dict[str, float] = {}
        for pname, (lo, hi) in param_bounds.items():
            if log_scale and lo > 0 and hi > 0:
                initial_params[pname] = float(np.exp((np.log(lo) + np.log(hi)) / 2))
            else:
                initial_params[pname] = (lo + hi) / 2.0

        # Create readable initial name with values substituted
        initial_name = name
        for pname, val in initial_params.items():
            initial_name = re.sub(
                r"\b" + re.escape(pname) + r"\b",
                f"{val:.4g}",
                initial_name,
            )

        # Evaluation closure pinned to *initial_params*
        def _make_func(f, params):
            return lambda X: f(X, **params)

        bf = BasisFunction(
            name=initial_name,
            func=_make_func(func, initial_params),
            complexity=complexity,
            feature_indices=feature_indices or (),
            func_type="parametric",
            func_config={
                "name": name,
                "param_bounds": {k: list(v) for k, v in param_bounds.items()},
                "log_scale": log_scale,
            },
        )

        idx = len(self.basis_functions)
        self.basis_functions.append(bf)

        self._parametric_info.append(
            ParametricBasisInfo(
                basis_index=idx,
                name=name,
                func=func,
                param_bounds=param_bounds,
                initial_params=initial_params,
                log_scale=log_scale,
            )
        )

        self._compiled_evaluate = None
        return self

    @staticmethod
    def _resolve_parametric_name(name: str, params: dict[str, float]) -> str:
        """Substitute optimised values into a parametric basis-function name."""
        import re

        resolved = name
        for pname, val in params.items():
            resolved = re.sub(
                r"\b" + re.escape(pname) + r"\b",
                f"{val:.4g}",
                resolved,
            )
        return resolved

    def add_polynomial_interactions(
        self,
        max_total_degree: int = 3,
        max_individual_degree: int = 2,
    ) -> BasisLibrary:
        """
        Add polynomial terms with mixed powers: x_i^a * x_j^b * ...

        Parameters
        ----------
        max_total_degree : int
            Maximum sum of all exponents (default 3).
        max_individual_degree : int
            Maximum exponent for any single variable (default 2).
        """
        # Generate all combinations of exponents over continuous features
        cont_idx = self.continuous_indices
        n_cont = len(cont_idx)
        for total_deg in range(2, max_total_degree + 1):
            for cont_exponents in self._generate_exponent_combinations(
                n_cont, total_deg, max_individual_degree
            ):
                # Skip if only one variable (handled by add_polynomials)
                if sum(1 for e in cont_exponents if e > 0) < 2:
                    continue

                # Map back to full feature space
                exponents = [0] * self.n_features
                for ci, exp in zip(cont_idx, cont_exponents, strict=True):
                    exponents[ci] = exp

                # Build name and function
                terms = []
                indices = []
                for i, exp in enumerate(exponents):
                    if exp > 0:
                        indices.append(i)
                        if exp == 1:
                            terms.append(self.feature_names[i])
                        else:
                            terms.append(f"{self.feature_names[i]}^{exp}")

                name = "*".join(terms)
                exponents_array = jnp.array(exponents)

                self.basis_functions.append(
                    BasisFunction(
                        name=name,
                        func=partial(
                            lambda X, exp: jnp.prod(X**exp, axis=1),
                            exp=exponents_array,
                        ),
                        complexity=total_deg,
                        feature_indices=tuple(indices),
                        func_type="polynomial_interaction",
                        func_config={"exponents": list(exponents)},
                    )
                )

        self._compiled_evaluate = None
        return self

    def _generate_exponent_combinations(
        self,
        n_vars: int,
        total: int,
        max_individual: int,
    ):
        """Generate all valid exponent combinations."""
        if n_vars == 1:
            if total <= max_individual:
                yield (total,)
            return

        for exp in range(min(total, max_individual) + 1):
            for rest in self._generate_exponent_combinations(
                n_vars - 1, total - exp, max_individual
            ):
                yield (exp,) + rest

    def build_default(
        self,
        max_poly_degree: int = 3,
        include_transcendental: bool = True,
        include_ratios: bool = False,
    ) -> BasisLibrary:
        """
        Build a default library similar to ALAMO's standard set.

        Parameters
        ----------
        max_poly_degree : int
            Maximum polynomial degree.
        include_transcendental : bool
            Whether to include transcendental functions.
        include_ratios : bool
            Whether to include ratio terms.
        """
        self.add_constant()
        self.add_linear()
        self.add_polynomials(max_poly_degree)
        self.add_interactions(max_order=2)

        if include_transcendental:
            self.add_transcendental()
        if include_ratios:
            self.add_ratios()

        if self.categorical_indices:
            self.add_categorical_indicators()
            if self.continuous_indices:
                self.add_categorical_interactions()

        return self

    def add_compositions(
        self,
        outer_funcs: list[str] | None = None,
        inner_forms: list[str] | None = None,
    ) -> BasisLibrary:
        """
        Add compositions: outer_func(inner_form).

        Creates functions like log(x*y), exp(x/y), sqrt(x+y), etc.

        Parameters
        ----------
        outer_funcs : list of str, optional
            Outer functions to apply. Defaults to ["log", "exp", "sqrt"].
        inner_forms : list of str, optional
            Inner forms: "product", "ratio", "sum", "diff".
            Defaults to ["product", "ratio"].

        Examples
        --------
        >>> library.add_compositions(["log", "exp"], ["product", "ratio"])
        # Adds: log(x*y), log(x/y), exp(x*y), exp(x/y), etc.
        """
        outer_funcs = outer_funcs or ["log", "exp", "sqrt"]
        inner_forms = inner_forms or ["product", "ratio"]

        func_map = {
            "log": (_safe_log, "log"),
            "exp": (jnp.exp, "exp"),
            "sqrt": (_safe_sqrt, "sqrt"),
            "abs": (jnp.abs, "abs"),
            "tanh": (jnp.tanh, "tanh"),
        }

        for i in self.continuous_indices:
            for j in self.continuous_indices:
                if i == j:
                    continue

                for outer_name in outer_funcs:
                    if outer_name not in func_map:
                        continue
                    outer_fn, outer_str = func_map[outer_name]

                    for inner_form in inner_forms:
                        fi, fj = self.feature_names[i], self.feature_names[j]

                        # Skip duplicate pairs for symmetric operations
                        if inner_form == "product" and i > j:
                            continue

                        if inner_form == "product":
                            name = f"{outer_str}({fi}*{fj})"
                            func = partial(
                                lambda X, i, j, fn: fn(X[:, i] * X[:, j]), i=i, j=j, fn=outer_fn
                            )
                            complexity = 4
                        elif inner_form == "ratio":
                            name = f"{outer_str}({fi}/{fj})"
                            func = partial(
                                lambda X, i, j, fn: fn(_safe_div(X[:, i], X[:, j])),
                                i=i,
                                j=j,
                                fn=outer_fn,
                            )
                            complexity = 4
                        elif inner_form == "sum":
                            name = f"{outer_str}({fi}+{fj})"
                            func = partial(
                                lambda X, i, j, fn: fn(X[:, i] + X[:, j]), i=i, j=j, fn=outer_fn
                            )
                            complexity = 3
                        elif inner_form == "diff":
                            name = f"{outer_str}({fi}-{fj})"
                            func = partial(
                                lambda X, i, j, fn: fn(X[:, i] - X[:, j]), i=i, j=j, fn=outer_fn
                            )
                            complexity = 3
                        else:
                            continue

                        self.basis_functions.append(
                            BasisFunction(
                                name=name,
                                func=func,
                                complexity=complexity,
                                feature_indices=(i, j),
                                func_type="composition",
                                func_config={
                                    "outer": outer_name,
                                    "inner": inner_form,
                                    "i": i,
                                    "j": j,
                                },
                            )
                        )

        self._compiled_evaluate = None
        return self

    def add_rational_forms(
        self,
        numerator_degree: int = 1,
        denominator_degree: int = 1,
    ) -> BasisLibrary:
        """
        Add rational function templates: (a + b*x) / (1 + c*y).

        Common in chemical kinetics (Langmuir-Hinshelwood, Michaelis-Menten).

        Parameters
        ----------
        numerator_degree : int
            Max polynomial degree in numerator.
        denominator_degree : int
            Max polynomial degree in denominator.

        Examples
        --------
        >>> library.add_rational_forms()
        # Adds: x/(1+y), x*y/(1+x), x/(1+x+y), etc.
        """
        # Single variable rational forms: x/(1+x), x/(1+x^2)
        for i in self.continuous_indices:
            fi = self.feature_names[i]

            # x / (1 + x)
            self.basis_functions.append(
                BasisFunction(
                    name=f"{fi}/(1+{fi})",
                    func=partial(lambda X, i: X[:, i] / (1 + X[:, i]), i=i),
                    complexity=3,
                    feature_indices=(i,),
                    func_type="rational",
                    func_config={"form": "saturation", "i": i},
                )
            )

            # x / (1 + x)^2 - derivative of Langmuir
            self.basis_functions.append(
                BasisFunction(
                    name=f"{fi}/(1+{fi})^2",
                    func=partial(lambda X, i: X[:, i] / (1 + X[:, i]) ** 2, i=i),
                    complexity=4,
                    feature_indices=(i,),
                    func_type="rational",
                    func_config={"form": "saturation_deriv", "i": i},
                )
            )

        # Two-variable rational forms
        for i in self.continuous_indices:
            for j in self.continuous_indices:
                if i == j:
                    continue

                fi, fj = self.feature_names[i], self.feature_names[j]

                # x*y / (1 + x) - Langmuir-Hinshelwood
                self.basis_functions.append(
                    BasisFunction(
                        name=f"{fi}*{fj}/(1+{fi})",
                        func=partial(lambda X, i, j: X[:, i] * X[:, j] / (1 + X[:, i]), i=i, j=j),
                        complexity=4,
                        feature_indices=(i, j),
                        func_type="rational",
                        func_config={"form": "LH_single", "i": i, "j": j},
                    )
                )

                # x*y / (1 + x + y) - competitive adsorption
                self.basis_functions.append(
                    BasisFunction(
                        name=f"{fi}*{fj}/(1+{fi}+{fj})",
                        func=partial(
                            lambda X, i, j: X[:, i] * X[:, j] / (1 + X[:, i] + X[:, j]), i=i, j=j
                        ),
                        complexity=5,
                        feature_indices=(i, j),
                        func_type="rational",
                        func_config={"form": "competitive", "i": i, "j": j},
                    )
                )

                # x / (1 + y) - inhibition
                self.basis_functions.append(
                    BasisFunction(
                        name=f"{fi}/(1+{fj})",
                        func=partial(lambda X, i, j: X[:, i] / (1 + X[:, j]), i=i, j=j),
                        complexity=3,
                        feature_indices=(i, j),
                        func_type="rational",
                        func_config={"form": "inhibition", "i": i, "j": j},
                    )
                )

        self._compiled_evaluate = None
        return self

    def add_power_laws(
        self,
        exponents: list[float] | None = None,
    ) -> BasisLibrary:
        """
        Add power law terms with fractional exponents.

        Common in empirical correlations (Nu = Re^0.8 * Pr^0.33).

        Parameters
        ----------
        exponents : list of float, optional
            Exponents to use. Defaults to [0.25, 0.33, 0.5, 0.67, 0.75, 1.5, 2.0].
        """
        exponents = exponents or [0.25, 0.33, 0.5, 0.67, 0.75, 1.5, 2.0]

        for i in self.continuous_indices:
            fi = self.feature_names[i]

            for exp in exponents:
                # Skip integer exponents (handled by add_polynomials)
                if exp == int(exp):
                    continue

                # Format exponent nicely
                if exp == 0.5:
                    name = f"sqrt({fi})"
                elif exp == 0.33 or abs(exp - 1 / 3) < 0.01:
                    name = f"{fi}^(1/3)"
                elif exp == 0.25:
                    name = f"{fi}^(1/4)"
                elif exp == 0.67 or abs(exp - 2 / 3) < 0.01:
                    name = f"{fi}^(2/3)"
                elif exp == 0.75:
                    name = f"{fi}^(3/4)"
                else:
                    name = f"{fi}^{exp}"

                self.basis_functions.append(
                    BasisFunction(
                        name=name,
                        func=partial(
                            lambda X, i, e: jnp.power(jnp.maximum(X[:, i], 1e-10), e), i=i, e=exp
                        ),
                        complexity=2,
                        feature_indices=(i,),
                        func_type="power",
                        func_config={"feature_index": i, "exponent": exp},
                    )
                )

        self._compiled_evaluate = None
        return self

    def expand_sisso_style(
        self,
        operations: list[str] | None = None,
        max_depth: int = 2,
    ) -> BasisLibrary:
        """
        SISSO-style recursive feature expansion.

        Iteratively applies operations to existing features to create
        new composite features.

        Parameters
        ----------
        operations : list of str, optional
            Operations to apply: "add", "sub", "mul", "div", "exp", "log", "sqrt", "sq", "inv".
            Defaults to ["mul", "div", "sq", "sqrt"].
        max_depth : int
            Maximum recursion depth.

        Notes
        -----
        This can create a very large library. Use with caution.
        """
        operations = operations or ["mul", "div", "sq", "sqrt"]

        # Start with continuous features only
        current_features = []
        for i in self.continuous_indices:
            current_features.append(
                {
                    "name": self.feature_names[i],
                    "func": partial(lambda X, i: X[:, i], i=i),
                    "complexity": 1,
                    "indices": (i,),
                }
            )

        # Iteratively expand
        for _depth in range(max_depth):
            new_features = []

            # Unary operations on current features
            for feat in current_features:
                if "sq" in operations:
                    new_features.append(
                        {
                            "name": f"({feat['name']})^2",
                            "func": partial(lambda X, f: f(X) ** 2, f=feat["func"]),
                            "complexity": feat["complexity"] + 1,
                            "indices": feat["indices"],
                        }
                    )

                if "sqrt" in operations:
                    new_features.append(
                        {
                            "name": f"sqrt({feat['name']})",
                            "func": partial(lambda X, f: _safe_sqrt(f(X)), f=feat["func"]),
                            "complexity": feat["complexity"] + 1,
                            "indices": feat["indices"],
                        }
                    )

                if "inv" in operations:
                    new_features.append(
                        {
                            "name": f"1/({feat['name']})",
                            "func": partial(lambda X, f: _safe_inv(f(X)), f=feat["func"]),
                            "complexity": feat["complexity"] + 1,
                            "indices": feat["indices"],
                        }
                    )

                if "exp" in operations:
                    new_features.append(
                        {
                            "name": f"exp({feat['name']})",
                            "func": partial(lambda X, f: jnp.exp(f(X)), f=feat["func"]),
                            "complexity": feat["complexity"] + 2,
                            "indices": feat["indices"],
                        }
                    )

                if "log" in operations:
                    new_features.append(
                        {
                            "name": f"log({feat['name']})",
                            "func": partial(lambda X, f: _safe_log(f(X)), f=feat["func"]),
                            "complexity": feat["complexity"] + 2,
                            "indices": feat["indices"],
                        }
                    )

            # Binary operations between features
            for fi, feat_i in enumerate(current_features):
                for fj, feat_j in enumerate(current_features):
                    if fi >= fj:
                        continue

                    if "mul" in operations:
                        new_features.append(
                            {
                                "name": f"({feat_i['name']})*({feat_j['name']})",
                                "func": partial(
                                    lambda X, f1, f2: f1(X) * f2(X),
                                    f1=feat_i["func"],
                                    f2=feat_j["func"],
                                ),
                                "complexity": feat_i["complexity"] + feat_j["complexity"],
                                "indices": tuple(set(feat_i["indices"]) | set(feat_j["indices"])),
                            }
                        )

                    if "div" in operations:
                        new_features.append(
                            {
                                "name": f"({feat_i['name']})/({feat_j['name']})",
                                "func": partial(
                                    lambda X, f1, f2: _safe_div(f1(X), f2(X)),
                                    f1=feat_i["func"],
                                    f2=feat_j["func"],
                                ),
                                "complexity": feat_i["complexity"] + feat_j["complexity"] + 1,
                                "indices": tuple(set(feat_i["indices"]) | set(feat_j["indices"])),
                            }
                        )

                    if "add" in operations:
                        new_features.append(
                            {
                                "name": f"({feat_i['name']})+({feat_j['name']})",
                                "func": partial(
                                    lambda X, f1, f2: f1(X) + f2(X),
                                    f1=feat_i["func"],
                                    f2=feat_j["func"],
                                ),
                                "complexity": feat_i["complexity"] + feat_j["complexity"],
                                "indices": tuple(set(feat_i["indices"]) | set(feat_j["indices"])),
                            }
                        )

                    if "sub" in operations:
                        new_features.append(
                            {
                                "name": f"({feat_i['name']})-({feat_j['name']})",
                                "func": partial(
                                    lambda X, f1, f2: f1(X) - f2(X),
                                    f1=feat_i["func"],
                                    f2=feat_j["func"],
                                ),
                                "complexity": feat_i["complexity"] + feat_j["complexity"],
                                "indices": tuple(set(feat_i["indices"]) | set(feat_j["indices"])),
                            }
                        )

            current_features = new_features

        # Add all generated features to library
        seen_names = set(self.names)
        for feat in current_features:
            if feat["name"] not in seen_names:
                self.basis_functions.append(
                    BasisFunction(
                        name=feat["name"],
                        func=feat["func"],
                        complexity=feat["complexity"],
                        feature_indices=feat["indices"],
                        func_type="sisso",
                        func_config={},
                    )
                )
                seen_names.add(feat["name"])

        self._compiled_evaluate = None
        return self

    def evaluate(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate all basis functions on input data.

        Parameters
        ----------
        X : jnp.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        Phi : jnp.ndarray
            Design matrix of shape (n_samples, n_basis).
        """
        if len(self.basis_functions) == 0:
            raise ValueError("No basis functions defined. Add some first.")

        X = jnp.atleast_2d(X)
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        columns = [bf.func(X) for bf in self.basis_functions]
        return jnp.column_stack(columns)

    def evaluate_subset(
        self,
        X: jnp.ndarray,
        indices: list[int] | jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Evaluate a subset of basis functions.

        Parameters
        ----------
        X : jnp.ndarray
            Input array of shape (n_samples, n_features).
        indices : array-like
            Indices of basis functions to evaluate.

        Returns
        -------
        Phi : jnp.ndarray
            Design matrix of shape (n_samples, len(indices)).
        """
        X = jnp.atleast_2d(X)
        columns = [self.basis_functions[i].func(X) for i in indices]
        return jnp.column_stack(columns)

    @property
    def names(self) -> list[str]:
        """List of basis function names."""
        return [bf.name for bf in self.basis_functions]

    @property
    def complexities(self) -> jnp.ndarray:
        """Array of complexity scores."""
        return jnp.array([bf.complexity for bf in self.basis_functions])

    def __len__(self) -> int:
        """Number of basis functions in the library."""
        return len(self.basis_functions)

    def __repr__(self) -> str:
        return (
            f"BasisLibrary(n_features={self.n_features}, "
            f"n_basis={len(self)}, "
            f"feature_names={self.feature_names})"
        )

    def summary(self) -> str:
        """Return a summary of the library contents."""
        lines = [
            f"BasisLibrary with {len(self)} basis functions:",
            f"  Features: {self.feature_names}",
            "",
            "  Basis functions:",
        ]
        for i, bf in enumerate(self.basis_functions):
            lines.append(f"    [{i:3d}] {bf.name} (complexity={bf.complexity})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize library configuration to dictionary.

        Notes
        -----
        Custom functions are not fully serializable. The library config
        is saved, but custom functions will need to be re-added manually
        after loading.
        """
        d = {
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "feature_bounds": self.feature_bounds,
            "basis_functions": [bf.to_dict() for bf in self.basis_functions],
        }
        if any(t != "continuous" for t in self.feature_types):
            d["feature_types"] = self.feature_types
            d["categories"] = {str(k): v for k, v in self.categories.items()}
        return d

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> BasisLibrary:
        """
        Reconstruct library from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration from to_dict().

        Returns
        -------
        library : BasisLibrary
            Reconstructed library. Custom functions will raise an error
            and need to be re-added manually.
        """
        categories_raw = config.get("categories", {})
        categories = {int(k): v for k, v in categories_raw.items()} if categories_raw else {}
        library = cls(
            n_features=config["n_features"],
            feature_names=config["feature_names"],
            feature_bounds=config.get("feature_bounds"),
            feature_types=config.get("feature_types"),
            categories=categories or None,
        )

        for bf_config in config["basis_functions"]:
            func_type = bf_config["func_type"]
            fc = bf_config["func_config"]

            if func_type == "constant":
                library.add_constant()
            elif func_type == "linear":
                # Add specific linear term
                i = fc["feature_index"]
                library.basis_functions.append(
                    BasisFunction(
                        name=bf_config["name"],
                        func=partial(lambda X, idx: X[:, idx], idx=i),
                        complexity=bf_config["complexity"],
                        feature_indices=(i,),
                        func_type="linear",
                        func_config=fc,
                    )
                )
            elif func_type == "polynomial":
                i = fc["feature_index"]
                d = fc["degree"]
                library.basis_functions.append(
                    BasisFunction(
                        name=bf_config["name"],
                        func=partial(lambda X, idx, deg: X[:, idx] ** deg, idx=i, deg=d),
                        complexity=bf_config["complexity"],
                        feature_indices=(i,),
                        func_type="polynomial",
                        func_config=fc,
                    )
                )
            elif func_type == "interaction":
                indices = jnp.array(fc["feature_indices"])
                library.basis_functions.append(
                    BasisFunction(
                        name=bf_config["name"],
                        func=partial(
                            lambda X, idx: jnp.prod(X[:, idx], axis=1),
                            idx=indices,
                        ),
                        complexity=bf_config["complexity"],
                        feature_indices=tuple(fc["feature_indices"]),
                        func_type="interaction",
                        func_config=fc,
                    )
                )
            elif func_type == "transcendental":
                library._add_transcendental_from_config(bf_config, fc)
            elif func_type == "ratio":
                num, den = fc["numerator_index"], fc["denominator_index"]
                library.basis_functions.append(
                    BasisFunction(
                        name=bf_config["name"],
                        func=partial(
                            lambda X, n, d: _safe_div(X[:, n], X[:, d]),
                            n=num,
                            d=den,
                        ),
                        complexity=bf_config["complexity"],
                        feature_indices=(num, den),
                        func_type="ratio",
                        func_config=fc,
                    )
                )
            elif func_type == "indicator":
                i = fc["feature_index"]
                cat_val = fc["category_value"]
                library.basis_functions.append(
                    BasisFunction(
                        name=bf_config["name"],
                        func=partial(
                            lambda X, idx, val: (X[:, idx] == val).astype(jnp.float32),
                            idx=i,
                            val=cat_val,
                        ),
                        complexity=bf_config["complexity"],
                        feature_indices=(i,),
                        func_type="indicator",
                        func_config=fc,
                    )
                )
            elif func_type == "categorical_interaction":
                ci = fc["categorical_index"]
                cat_val = fc["category_value"]
                cj = fc["continuous_index"]
                library.basis_functions.append(
                    BasisFunction(
                        name=bf_config["name"],
                        func=partial(
                            lambda X, ci, val, cj: (
                                (X[:, ci] == val).astype(jnp.float32) * X[:, cj]
                            ),
                            ci=ci,
                            val=cat_val,
                            cj=cj,
                        ),
                        complexity=bf_config["complexity"],
                        feature_indices=(ci, cj),
                        func_type="categorical_interaction",
                        func_config=fc,
                    )
                )
            elif func_type == "parametric":
                raise ValueError(
                    f"Cannot deserialize parametric function '{bf_config['name']}'. "
                    "Re-add it manually using add_parametric()."
                )
            elif func_type == "custom":
                raise ValueError(
                    f"Cannot deserialize custom function '{bf_config['name']}'. "
                    "Re-add it manually using add_custom()."
                )
            else:
                raise ValueError(f"Unknown function type: {func_type}")

        return library

    def _add_transcendental_from_config(
        self,
        bf_config: dict[str, Any],
        fc: dict[str, Any],
    ) -> None:
        """Helper to add transcendental function from config."""
        i = fc["feature_index"]
        func_name = fc["func_name"]

        transcendental_map = {
            "log": _safe_log,
            "exp": jnp.exp,
            "sqrt": _safe_sqrt,
            "inv": _safe_inv,
            "sin": jnp.sin,
            "cos": jnp.cos,
            "tan": jnp.tan,
            "sinh": jnp.sinh,
            "cosh": jnp.cosh,
            "tanh": jnp.tanh,
            "abs": jnp.abs,
            "square": jnp.square,
        }

        if func_name not in transcendental_map:
            raise ValueError(f"Unknown transcendental function: {func_name}")

        fn = transcendental_map[func_name]
        self.basis_functions.append(
            BasisFunction(
                name=bf_config["name"],
                func=partial(lambda X, idx, f: f(X[:, idx]), idx=i, f=fn),
                complexity=bf_config["complexity"],
                feature_indices=(i,),
                func_type="transcendental",
                func_config=fc,
            )
        )

    def save(self, filepath: str) -> None:
        """
        Save library configuration to JSON file.

        Parameters
        ----------
        filepath : str
            Path to save the configuration.
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> BasisLibrary:
        """
        Load library from JSON file.

        Parameters
        ----------
        filepath : str
            Path to the configuration file.
        """
        with open(filepath) as f:
            config = json.load(f)
        return cls.from_dict(config)

    def filter_by_complexity(
        self,
        max_complexity: int | None = None,
        min_complexity: int | None = None,
    ) -> list[int]:
        """
        Get indices of basis functions within complexity bounds.

        Parameters
        ----------
        max_complexity : int, optional
            Maximum allowed complexity.
        min_complexity : int, optional
            Minimum required complexity.

        Returns
        -------
        indices : list of int
            Indices of basis functions meeting the criteria.
        """
        indices = []
        for i, bf in enumerate(self.basis_functions):
            if max_complexity is not None and bf.complexity > max_complexity:
                continue
            if min_complexity is not None and bf.complexity < min_complexity:
                continue
            indices.append(i)
        return indices

    def filter_by_features(
        self,
        required_features: list[int] | None = None,
        excluded_features: list[int] | None = None,
    ) -> list[int]:
        """
        Get indices of basis functions using specified features.

        Parameters
        ----------
        required_features : list of int, optional
            Only include basis functions using these features.
        excluded_features : list of int, optional
            Exclude basis functions using these features.

        Returns
        -------
        indices : list of int
            Indices of basis functions meeting the criteria.
        """
        indices = []
        required = set(required_features) if required_features else None
        excluded = set(excluded_features) if excluded_features else set()

        for i, bf in enumerate(self.basis_functions):
            features_used = set(bf.feature_indices)

            if excluded and features_used & excluded:
                continue
            if required is not None and not features_used.issubset(required):
                continue
            indices.append(i)

        return indices
