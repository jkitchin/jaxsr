#!/usr/bin/env python3
"""
Validate JAXSR notebooks for API correctness.

Checks notebook code cells against known API signatures and patterns from CLAUDE.md.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any


# API validation patterns from CLAUDE.md lines 97-133
API_CHECKS = [
    {
        "name": "coefficient_intervals",
        "pattern": r"coefficient_intervals\(\)",
        "antipattern": r"\(lo,\s*hi\)",
        "expected": "Returns dict[str, (est, lo, hi, se)] - 4-tuple, not 2-tuple",
    },
    {
        "name": "bootstrap_coefficients",
        "pattern": r"bootstrap_coefficients\(",
        "antipattern": r"result\.(intervals|names|lower|upper)",
        "expected": 'Returns plain dict - use result["names"], result["lower"], etc.',
    },
    {
        "name": "bootstrap_predict",
        "pattern": r"bootstrap_predict\(",
        "antipattern": r"result\.(upper|lower|mean)",
        "expected": 'Returns plain dict - use result["upper"], result["lower"], etc.',
    },
    {
        "name": "BayesianModelAverage",
        "pattern": r"BayesianModelAverage\(",
        "antipattern": r"\.(weights_|models_)",
        "expected": "Use .weights and .expressions (no trailing underscore)",
    },
    {
        "name": "CanonicalAnalysis",
        "pattern": r"CanonicalAnalysis\(",
        "antipattern": r"\.predicted_response",
        "expected": "Use .stationary_response (not .predicted_response)",
    },
    {
        "name": "ActiveLearner_args",
        "pattern": r"ActiveLearner\(",
        "antipattern": r"ActiveLearner\([^,]+,\s*[^,]*acq",
        "expected": "ActiveLearner(model, bounds, acquisition) - bounds is 2nd arg",
    },
    {
        "name": "ResponseSurface_n_factors",
        "pattern": r"ResponseSurface\(",
        "antipattern": r"ResponseSurface\(\s*(?!n_factors)",
        "expected": "ResponseSurface(n_factors=...) - n_factors is required first arg",
    },
    {
        "name": "information_criterion_cv",
        "pattern": r'information_criterion\s*=\s*["\']cv["\']',
        "antipattern": r'information_criterion\s*=\s*["\']cv["\']',
        "expected": 'Only "aic", "aicc", "bic" supported - not "cv"',
    },
    {
        "name": "add_transcendental_defaults",
        "pattern": r"add_transcendental\(\)",
        "antipattern": r"#.*7 default",
        "expected": 'Defaults: ["log", "exp", "sqrt", "inv"] - 4 functions, not 7',
    },
]


def extract_code_cells(notebook_path: Path) -> list[tuple[str, str, int]]:
    """Extract code cells from notebook with cell IDs and indices."""
    with open(notebook_path) as f:
        nb = json.load(f)

    code_cells = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            cell_id = cell.get("id", f"cell-{idx}")
            code_cells.append((cell_id, source, idx))

    return code_cells


def check_api_usage(cell_id: str, source: str, cell_idx: int) -> list[dict[str, Any]]:
    """Check a code cell for API misuse patterns."""
    issues = []

    for check in API_CHECKS:
        # Check if API is used
        if re.search(check["pattern"], source):
            # Check for antipattern
            if "antipattern" in check and re.search(check["antipattern"], source):
                issues.append(
                    {
                        "check": check["name"],
                        "cell_id": cell_id,
                        "cell_index": cell_idx,
                        "pattern": check["pattern"],
                        "antipattern": check["antipattern"],
                        "expected": check["expected"],
                        "severity": "high",
                    }
                )

    return issues


def validate_notebook(notebook_path: Path) -> dict[str, Any]:
    """Validate a single notebook and return issues found."""
    code_cells = extract_code_cells(notebook_path)
    all_issues = []

    for cell_id, source, idx in code_cells:
        issues = check_api_usage(cell_id, source, idx)
        all_issues.extend(issues)

    return {
        "notebook": notebook_path.name,
        "path": str(notebook_path),
        "total_cells": len(code_cells),
        "issues": all_issues,
        "issue_count": len(all_issues),
    }


def main():
    """Validate all notebooks passed as arguments."""
    if len(sys.argv) < 2:
        print("Usage: validate_notebooks.py <notebook1.ipynb> [notebook2.ipynb ...]")
        sys.exit(1)

    all_results = []
    total_issues = 0

    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue

        result = validate_notebook(path)
        all_results.append(result)
        total_issues += result["issue_count"]

        # Print summary for this notebook
        if result["issue_count"] > 0:
            print(f"\n{result['notebook']}: {result['issue_count']} issues found")
            for issue in result["issues"]:
                print(f"  - Cell {issue['cell_index']} ({issue['cell_id']}): {issue['check']}")
                print(f"    Expected: {issue['expected']}")

    # Save detailed results
    output_path = Path("NOTEBOOK_API_VALIDATION.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total notebooks checked: {len(all_results)}")
    print(f"Total issues found: {total_issues}")
    print(f"Detailed results saved to: {output_path}")

    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()
