"""
Symbolic Simplification for JAXSR.

Post-processes discovered expressions to simplify and clean up the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np


@dataclass
class SimplificationResult:
    """
    Result of expression simplification.

    Parameters
    ----------
    coefficients : jnp.ndarray
        Simplified coefficients.
    names : list of str
        Simplified basis function names.
    expression : str
        Human-readable simplified expression.
    sympy_expr : optional
        SymPy expression if use_sympy was True.
    n_terms_original : int
        Number of terms before simplification.
    n_terms_simplified : int
        Number of terms after simplification.
    """
    coefficients: jnp.ndarray
    names: List[str]
    expression: str
    sympy_expr: Any = None
    n_terms_original: int = 0
    n_terms_simplified: int = 0


def simplify_expression(
    coefficients: jnp.ndarray,
    basis_names: List[str],
    tolerance: float = 1e-6,
    use_sympy: bool = True,
    feature_names: Optional[List[str]] = None,
) -> SimplificationResult:
    """
    Simplify a discovered expression.

    Operations performed:
    - Remove near-zero coefficients
    - Combine like terms
    - Apply algebraic identities (via SymPy if enabled)

    Parameters
    ----------
    coefficients : jnp.ndarray
        Coefficient values.
    basis_names : list of str
        Names of basis functions.
    tolerance : float
        Threshold for removing small coefficients.
    use_sympy : bool
        Whether to use SymPy for algebraic simplification.
    feature_names : list of str, optional
        Original feature names for SymPy conversion.

    Returns
    -------
    result : SimplificationResult
        Simplified expression.

    Examples
    --------
    >>> coefficients = jnp.array([2.5, 0.0001, 1.2, -0.8])
    >>> names = ["x", "y", "x^2", "y^2"]
    >>> result = simplify_expression(coefficients, names, tolerance=1e-3)
    >>> print(result.expression)
    """
    n_original = len(coefficients)

    # Step 1: Remove near-zero coefficients
    coefficients, basis_names = _remove_small_coefficients(
        coefficients, basis_names, tolerance
    )

    # Step 2: Try SymPy simplification if requested
    sympy_expr = None
    if use_sympy and len(coefficients) > 0:
        try:
            sympy_expr = _sympy_simplify(coefficients, basis_names, feature_names)
            # Extract simplified form back
            simplified_coeffs, simplified_names = _extract_from_sympy(
                sympy_expr, feature_names
            )
            if simplified_coeffs is not None:
                coefficients = simplified_coeffs
                basis_names = simplified_names
        except Exception:
            # Fall back to basic simplification if SymPy fails
            pass

    # Step 3: Build expression string
    expression = _build_expression(coefficients, basis_names)

    return SimplificationResult(
        coefficients=jnp.array(coefficients) if len(coefficients) > 0 else jnp.array([]),
        names=basis_names,
        expression=expression,
        sympy_expr=sympy_expr,
        n_terms_original=n_original,
        n_terms_simplified=len(coefficients),
    )


def _remove_small_coefficients(
    coefficients: jnp.ndarray,
    names: List[str],
    tolerance: float,
) -> Tuple[List[float], List[str]]:
    """Remove coefficients below tolerance."""
    filtered_coeffs = []
    filtered_names = []

    for coef, name in zip(coefficients, names):
        if abs(float(coef)) >= tolerance:
            filtered_coeffs.append(float(coef))
            filtered_names.append(name)

    return filtered_coeffs, filtered_names


def _sympy_simplify(
    coefficients: List[float],
    names: List[str],
    feature_names: Optional[List[str]] = None,
):
    """Use SymPy for algebraic simplification."""
    import sympy

    # Create symbols
    if feature_names is None:
        # Extract feature names from basis names
        feature_names = _extract_feature_names(names)

    symbols = {name: sympy.Symbol(name) for name in feature_names}

    # Build expression
    expr = sympy.Integer(0)
    for coef, name in zip(coefficients, names):
        term = _parse_term_to_sympy(name, symbols)
        expr = expr + coef * term

    # Simplify
    simplified = sympy.simplify(expr)
    return simplified


def _extract_feature_names(basis_names: List[str]) -> List[str]:
    """Extract feature names from basis function names."""
    features = set()

    for name in basis_names:
        if name == "1":
            continue

        # Handle simple feature names
        if name.isidentifier():
            features.add(name)
            continue

        # Handle powers: x^2
        if "^" in name and "*" not in name:
            base = name.split("^")[0]
            if base.isidentifier():
                features.add(base)
            continue

        # Handle interactions: x*y
        if "*" in name:
            parts = name.split("*")
            for part in parts:
                part = part.strip()
                # Handle powers within interactions: x^2*y
                if "^" in part:
                    part = part.split("^")[0]
                if part.isidentifier():
                    features.add(part)
            continue

        # Handle transcendental: log(x), exp(x)
        for func in ["log(", "exp(", "sqrt(", "sin(", "cos("]:
            if name.startswith(func) and name.endswith(")"):
                inner = name[len(func):-1]
                if inner.isidentifier():
                    features.add(inner)
                break

        # Handle inverse: 1/x
        if name.startswith("1/"):
            inner = name[2:]
            if inner.isidentifier():
                features.add(inner)
            continue

        # Handle ratios: x/y
        if "/" in name and not name.startswith("1/"):
            parts = name.split("/")
            for part in parts:
                if part.isidentifier():
                    features.add(part)

    return list(sorted(features))


def _parse_term_to_sympy(name: str, symbols: dict):
    """Parse a basis function name to SymPy expression."""
    import sympy

    # Constant
    if name == "1":
        return sympy.Integer(1)

    # Linear term
    if name in symbols:
        return symbols[name]

    # Power: x^2
    if "^" in name and "*" not in name and "/" not in name:
        parts = name.split("^")
        if len(parts) == 2 and parts[0] in symbols:
            return symbols[parts[0]] ** int(parts[1])

    # Interaction: x*y
    if "*" in name and "/" not in name:
        result = sympy.Integer(1)
        parts = name.split("*")
        for part in parts:
            part = part.strip()
            if "^" in part:
                base, power = part.split("^")
                if base in symbols:
                    result = result * symbols[base] ** int(power)
            elif part in symbols:
                result = result * symbols[part]
        return result

    # Transcendental functions
    transcendental_map = {
        "log": sympy.log,
        "exp": sympy.exp,
        "sqrt": sympy.sqrt,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
    }

    for func_name, sympy_func in transcendental_map.items():
        prefix = f"{func_name}("
        if name.startswith(prefix) and name.endswith(")"):
            inner = name[len(prefix):-1]
            if inner in symbols:
                return sympy_func(symbols[inner])

    # Inverse: 1/x
    if name.startswith("1/"):
        inner = name[2:]
        if inner in symbols:
            return 1 / symbols[inner]

    # Ratio: x/y
    if "/" in name and not name.startswith("1/"):
        parts = name.split("/")
        if len(parts) == 2 and parts[0] in symbols and parts[1] in symbols:
            return symbols[parts[0]] / symbols[parts[1]]

    # Fallback: treat as symbol
    return sympy.Symbol(name)


def _extract_from_sympy(
    expr,
    feature_names: Optional[List[str]] = None,
) -> Tuple[Optional[List[float]], Optional[List[str]]]:
    """Extract coefficients and basis names from SymPy expression."""
    import sympy

    # Try to expand and collect terms
    try:
        expr = sympy.expand(expr)

        # Get all terms
        if isinstance(expr, sympy.Add):
            terms = expr.args
        else:
            terms = [expr]

        coefficients = []
        names = []

        for term in terms:
            coef, rest = term.as_coeff_Mul()
            coefficients.append(float(coef))

            if rest == sympy.Integer(1):
                names.append("1")
            else:
                names.append(str(rest))

        return coefficients, names

    except Exception:
        return None, None


def _build_expression(
    coefficients: List[float],
    names: List[str],
    precision: int = 4,
) -> str:
    """Build human-readable expression string."""
    if not coefficients:
        return "y = 0"

    terms = []
    for coef, name in zip(coefficients, names):
        if abs(coef) < 1e-10:
            continue

        # Format coefficient
        abs_coef = abs(coef)
        if abs_coef >= 1e4 or abs_coef < 1e-4:
            coef_str = f"{abs_coef:.{precision}e}"
        else:
            coef_str = f"{abs_coef:.{precision}g}"

        # Build term
        if name == "1":
            term = coef_str
        elif coef_str == "1":
            term = name
        else:
            term = f"{coef_str}*{name}"

        # Add sign
        if coef < 0:
            terms.append(f"- {term}")
        elif terms:
            terms.append(f"+ {term}")
        else:
            terms.append(term)

    if not terms:
        return "y = 0"

    return "y = " + " ".join(terms)


# =============================================================================
# Expression Manipulation
# =============================================================================


def round_coefficients(
    coefficients: jnp.ndarray,
    decimals: int = 4,
) -> jnp.ndarray:
    """Round coefficients to specified decimal places."""
    return jnp.round(coefficients, decimals)


def threshold_coefficients(
    coefficients: jnp.ndarray,
    threshold: float = 1e-6,
) -> jnp.ndarray:
    """Set coefficients below threshold to zero."""
    return jnp.where(jnp.abs(coefficients) < threshold, 0.0, coefficients)


def normalize_expression(
    coefficients: jnp.ndarray,
    names: List[str],
) -> Tuple[jnp.ndarray, List[str]]:
    """
    Normalize expression for comparison.

    Sorts terms by name and handles sign conventions.
    """
    # Sort by name
    pairs = list(zip(coefficients, names))
    pairs.sort(key=lambda x: x[1])

    coefficients = jnp.array([p[0] for p in pairs])
    names = [p[1] for p in pairs]

    return coefficients, names


def expressions_equivalent(
    expr1: Tuple[jnp.ndarray, List[str]],
    expr2: Tuple[jnp.ndarray, List[str]],
    tolerance: float = 1e-6,
) -> bool:
    """
    Check if two expressions are mathematically equivalent.

    Parameters
    ----------
    expr1 : tuple of (coefficients, names)
        First expression.
    expr2 : tuple of (coefficients, names)
        Second expression.
    tolerance : float
        Tolerance for coefficient comparison.

    Returns
    -------
    equivalent : bool
        True if expressions are equivalent.
    """
    # Normalize both expressions
    coeffs1, names1 = normalize_expression(*expr1)
    coeffs2, names2 = normalize_expression(*expr2)

    # Remove zero coefficients
    mask1 = jnp.abs(coeffs1) >= tolerance
    mask2 = jnp.abs(coeffs2) >= tolerance

    coeffs1 = coeffs1[mask1]
    names1 = [n for n, m in zip(names1, mask1) if m]

    coeffs2 = coeffs2[mask2]
    names2 = [n for n, m in zip(names2, mask2) if m]

    # Compare
    if len(coeffs1) != len(coeffs2):
        return False

    if names1 != names2:
        return False

    return bool(jnp.allclose(coeffs1, coeffs2, atol=tolerance))


# =============================================================================
# Complexity Analysis
# =============================================================================


def compute_expression_complexity(
    names: List[str],
    complexity_weights: Optional[Dict[str, int]] = None,
) -> int:
    """
    Compute total complexity of an expression.

    Parameters
    ----------
    names : list of str
        Basis function names.
    complexity_weights : dict, optional
        Custom complexity weights for different term types.

    Returns
    -------
    complexity : int
        Total complexity score.
    """
    if complexity_weights is None:
        complexity_weights = {
            "constant": 0,
            "linear": 1,
            "polynomial": 2,
            "interaction": 2,
            "transcendental": 3,
            "ratio": 3,
        }

    total = 0

    for name in names:
        term_type = _classify_term(name)
        weight = complexity_weights.get(term_type, 2)
        total += weight

    return total


def _classify_term(name: str) -> str:
    """Classify a term by type."""
    if name == "1":
        return "constant"

    if name.isidentifier():
        return "linear"

    if "^" in name and "*" not in name and "/" not in name:
        return "polynomial"

    if "*" in name and "/" not in name:
        return "interaction"

    for func in ["log", "exp", "sqrt", "sin", "cos"]:
        if name.startswith(f"{func}("):
            return "transcendental"

    if "/" in name:
        return "ratio"

    return "unknown"


def suggest_simpler_forms(
    coefficients: jnp.ndarray,
    names: List[str],
    feature_names: List[str],
    max_suggestions: int = 3,
) -> List[str]:
    """
    Suggest potentially simpler equivalent forms.

    Parameters
    ----------
    coefficients : jnp.ndarray
        Current coefficients.
    names : list of str
        Current basis function names.
    feature_names : list of str
        Original feature names.
    max_suggestions : int
        Maximum number of suggestions.

    Returns
    -------
    suggestions : list of str
        Suggested simpler forms.
    """
    suggestions = []

    try:
        import sympy

        # Build SymPy expression
        symbols = {name: sympy.Symbol(name) for name in feature_names}

        expr = sympy.Integer(0)
        for coef, name in zip(coefficients, names):
            term = _parse_term_to_sympy(name, symbols)
            expr = expr + float(coef) * term

        # Try different simplification strategies
        strategies = [
            ("factor", lambda e: sympy.factor(e)),
            ("collect", lambda e: sympy.collect(e, list(symbols.values()))),
            ("horner", lambda e: sympy.horner(e) if len(symbols) == 1 else e),
            ("trigsimp", lambda e: sympy.trigsimp(e)),
            ("ratsimp", lambda e: sympy.ratsimp(e)),
        ]

        original_str = str(sympy.simplify(expr))

        for name, strategy in strategies:
            try:
                result = strategy(expr)
                result_str = str(result)

                # Check if it's actually simpler
                if (
                    result_str != original_str
                    and len(result_str) < len(original_str) * 1.2
                    and result_str not in suggestions
                ):
                    suggestions.append(f"y = {result_str}")

                    if len(suggestions) >= max_suggestions:
                        break

            except Exception:
                continue

    except Exception:
        pass

    return suggestions
