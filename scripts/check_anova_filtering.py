#!/usr/bin/env python3
"""
Check JAXSR notebooks for correct ANOVA filtering patterns.

Validates that ANOVA summary rows are filtered before computing percentages.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any


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


def check_anova_patterns(cell_id: str, source: str, cell_idx: int) -> list[dict[str, Any]]:
    """Check for ANOVA usage patterns and potential issues."""
    issues = []

    # Pattern 1: Using anova_result.rows without filtering
    if re.search(r"anova.*\.rows", source, re.IGNORECASE):
        # Check if filtering for summary rows is present
        has_filter = bool(
            re.search(r'not in.*["\']Model["\'].*["\']Residual["\'].*["\']Total["\']', source)
            or re.search(r'!=.*["\']Model["\']', source)
            or re.search(r'!=.*["\']Residual["\']', source)
            or re.search(r'!=.*["\']Total["\']', source)
        )

        # Check if computing percentages
        has_percentage = bool(
            re.search(r"[*/%]\s*100", source) or re.search(r"contribution|percent", source, re.IGNORECASE)
        )

        if has_percentage and not has_filter:
            issues.append(
                {
                    "type": "anova_filtering",
                    "cell_id": cell_id,
                    "cell_index": cell_idx,
                    "finding": "ANOVA rows used for percentage without filtering summary rows",
                    "expected": 'Filter {"Model", "Residual", "Total"} before computing %',
                    "severity": "high",
                    "fix": "rows = [r for r in anova.rows if r.source not in {'Model', 'Residual', 'Total'}]",
                }
            )

    # Pattern 2: Using p_value without null guard
    if re.search(r"\.p_value", source):
        has_null_guard = bool(
            re.search(r"p_value is not None", source) or re.search(r"p_value.*!=.*None", source)
        )

        if not has_null_guard and "p_value" in source:
            # Check if it's being formatted or used in computation
            uses_p_value = bool(
                re.search(r"p_value\s*[<>=]", source)
                or re.search(r"format.*p_value", source)
                or re.search(r"f['\"].*p_value", source)
            )

            if uses_p_value:
                issues.append(
                    {
                        "type": "anova_p_value_null",
                        "cell_id": cell_id,
                        "cell_index": cell_idx,
                        "finding": "p_value used without null guard (can be None for summary rows)",
                        "expected": "Check 'if p_value is not None' before using",
                        "severity": "medium",
                        "fix": "if row.p_value is not None: ...",
                    }
                )

    return issues


def validate_notebook(notebook_path: Path) -> dict[str, Any]:
    """Validate a single notebook for ANOVA patterns."""
    code_cells = extract_code_cells(notebook_path)
    all_issues = []

    for cell_id, source, idx in code_cells:
        issues = check_anova_patterns(cell_id, source, idx)
        all_issues.extend(issues)

    return {
        "notebook": notebook_path.name,
        "path": str(notebook_path),
        "total_cells": len(code_cells),
        "issues": all_issues,
        "issue_count": len(all_issues),
    }


def main():
    """Validate all notebooks for ANOVA patterns."""
    if len(sys.argv) < 2:
        print("Usage: check_anova_filtering.py <notebook1.ipynb> [notebook2.ipynb ...]")
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
            print(f"\n{result['notebook']}: {result['issue_count']} ANOVA issues found")
            for issue in result["issues"]:
                print(f"  - Cell {issue['cell_index']} ({issue['cell_id']}): {issue['type']}")
                print(f"    Finding: {issue['finding']}")
                print(f"    Fix: {issue['fix']}")

    # Save detailed results
    output_path = Path("NOTEBOOK_ANOVA_VALIDATION.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total notebooks checked: {len(all_results)}")
    print(f"Total ANOVA issues found: {total_issues}")
    print(f"Detailed results saved to: {output_path}")

    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()
