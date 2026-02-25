#!/usr/bin/env python3
"""
Validate JAXSR notebook imports against public API.

Checks that all imports from jaxsr exist in src/jaxsr/__init__.py __all__.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any


def get_public_api() -> set[str]:
    """Extract __all__ exports from jaxsr __init__.py."""
    init_path = Path(__file__).parent.parent / "src" / "jaxsr" / "__init__.py"

    if not init_path.exists():
        print(f"Error: Cannot find {init_path}")
        sys.exit(1)

    with open(init_path) as f:
        content = f.read()

    # Extract __all__ list
    all_match = re.search(r"__all__\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not all_match:
        print("Error: Cannot parse __all__ from __init__.py")
        sys.exit(1)

    # Parse quoted strings from __all__
    all_str = all_match.group(1)
    exports = set(re.findall(r'["\']([^"\']+)["\']', all_str))

    return exports


def extract_code_cells(notebook_path: Path) -> list[tuple[str, str, int]]:
    """Extract code cells from notebook."""
    with open(notebook_path) as f:
        nb = json.load(f)

    code_cells = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            cell_id = cell.get("id", f"cell-{idx}")
            code_cells.append((cell_id, source, idx))

    return code_cells


def check_imports(cell_id: str, source: str, cell_idx: int, public_api: set[str]) -> list[dict[str, Any]]:
    """Check imports in a code cell against public API."""
    issues = []

    # Pattern 1: from jaxsr import X, Y, Z
    from_imports = re.finditer(r"from\s+jaxsr\s+import\s+([^\n#]+)", source)
    for match in from_imports:
        import_str = match.group(1)
        # Split on comma and clean up
        imported = [name.strip().split(" as ")[0] for name in import_str.split(",")]

        for name in imported:
            name = name.strip()
            if name and name not in public_api:
                issues.append(
                    {
                        "type": "invalid_import",
                        "cell_id": cell_id,
                        "cell_index": cell_idx,
                        "symbol": name,
                        "finding": f"'{name}' imported but not in jaxsr.__all__",
                        "severity": "high",
                        "fix": f"Check if '{name}' is exported or use private import",
                    }
                )

    # Pattern 2: from jaxsr.module import X (allowed for private APIs)
    submodule_imports = re.finditer(r"from\s+jaxsr\.(\w+)\s+import", source)
    for match in submodule_imports:
        # Just flag for review - these might be intentional private imports
        pass

    return issues


def validate_notebook(notebook_path: Path, public_api: set[str]) -> dict[str, Any]:
    """Validate imports in a notebook."""
    code_cells = extract_code_cells(notebook_path)
    all_issues = []

    for cell_id, source, idx in code_cells:
        issues = check_imports(cell_id, source, idx, public_api)
        all_issues.extend(issues)

    return {
        "notebook": notebook_path.name,
        "path": str(notebook_path),
        "total_cells": len(code_cells),
        "issues": all_issues,
        "issue_count": len(all_issues),
    }


def main():
    """Validate notebook imports."""
    if len(sys.argv) < 2:
        print("Usage: check_imports.py <notebook1.ipynb> [notebook2.ipynb ...]")
        sys.exit(1)

    # Load public API
    public_api = get_public_api()
    print(f"Loaded {len(public_api)} public symbols from jaxsr.__all__")

    all_results = []
    total_issues = 0

    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue

        result = validate_notebook(path, public_api)
        all_results.append(result)
        total_issues += result["issue_count"]

        # Print summary
        if result["issue_count"] > 0:
            print(f"\n{result['notebook']}: {result['issue_count']} import issues found")
            for issue in result["issues"]:
                print(f"  - Cell {issue['cell_index']}: {issue['symbol']}")
                print(f"    {issue['finding']}")

    # Save results
    output_path = Path("NOTEBOOK_IMPORT_VALIDATION.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total notebooks checked: {len(all_results)}")
    print(f"Total import issues found: {total_issues}")
    print(f"Detailed results saved to: {output_path}")

    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()
