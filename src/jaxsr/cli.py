"""
Command-Line Interface for JAXSR DOE workflows.

Provides the ``jaxsr`` command with subcommands for creating studies,
generating designs, importing results, fitting models, and generating
reports — all operating on ``.jaxsr`` study files.

Requires the ``cli`` optional dependency group::

    pip install jaxsr[cli]

Usage::

    jaxsr init "my_study" --factors "temperature:300:500" "pressure:1:10"
    jaxsr design my_study.jaxsr --method latin_hypercube --n-points 20
    jaxsr add my_study.jaxsr results.xlsx
    jaxsr fit my_study.jaxsr --max-terms 5
    jaxsr report my_study.jaxsr -o report.xlsx
    jaxsr status my_study.jaxsr
"""

from __future__ import annotations

import sys

try:
    import click
except ImportError:
    print(
        "Error: click is required for the jaxsr CLI. " "Install it with: pip install jaxsr[cli]",
        file=sys.stderr,
    )
    sys.exit(1)


def _parse_factor(spec: str) -> dict:
    """
    Parse a factor specification string.

    Formats:
        "name:low:high"           → continuous factor
        "name:level1,level2,..."  → categorical factor

    Parameters
    ----------
    spec : str
        Factor specification string.

    Returns
    -------
    factor : dict
        Dictionary with keys: name, type, bounds (continuous) or levels (categorical).
    """
    parts = spec.split(":", maxsplit=2)
    if len(parts) < 2:
        raise click.BadParameter(
            f"Factor spec '{spec}' must be 'name:low:high' or 'name:level1,level2,...'"
        )

    name = parts[0].strip()

    if len(parts) == 3:
        # Continuous: name:low:high
        try:
            low = float(parts[1])
            high = float(parts[2])
        except ValueError:
            raise click.BadParameter(
                f"Could not parse bounds for '{name}': '{parts[1]}' and '{parts[2]}' "
                f"must be numbers"
            ) from None
        return {"name": name, "type": "continuous", "bounds": (low, high)}
    else:
        # Could be categorical (name:a,b,c) or error
        value_str = parts[1].strip()
        if "," in value_str:
            levels = [v.strip() for v in value_str.split(",")]
            return {"name": name, "type": "categorical", "levels": levels}
        else:
            raise click.BadParameter(
                f"Factor spec '{spec}' is ambiguous. Use 'name:low:high' for continuous "
                f"or 'name:a,b,c' for categorical."
            )


@click.group()
@click.version_option(package_name="jaxsr")
def main():
    """JAXSR: JAX-based Symbolic Regression for Design of Experiments."""
    pass


# =============================================================================
# init
# =============================================================================


@main.command()
@click.argument("name")
@click.option(
    "--factors",
    "-f",
    multiple=True,
    required=True,
    help='Factor spec: "name:low:high" (continuous) or "name:a,b,c" (categorical).',
)
@click.option("--description", "-d", default="", help="Study description.")
@click.option("--output", "-o", default=None, help="Output file path (default: <name>.jaxsr).")
def init(name, factors, description, output):
    """Create a new DOE study.

    Example:

        jaxsr init catalyst_opt -f "temperature:300:500" -f "pressure:1:10" -f "catalyst:A,B,C"
    """
    from .study import DOEStudy

    parsed = [_parse_factor(f) for f in factors]

    factor_names = [p["name"] for p in parsed]
    bounds = []
    feature_types = []
    categories = {}

    for i, p in enumerate(parsed):
        feature_types.append(p["type"])
        if p["type"] == "continuous":
            bounds.append(p["bounds"])
        else:
            levels = p["levels"]
            bounds.append((0, len(levels) - 1))
            categories[i] = levels

    study = DOEStudy(
        name=name,
        factor_names=factor_names,
        bounds=bounds,
        feature_types=feature_types if any(ft == "categorical" for ft in feature_types) else None,
        categories=categories if categories else None,
        description=description,
    )

    filepath = output or f"{name}.jaxsr"
    study.save(filepath)
    click.echo(f"Created study '{name}' with {len(factor_names)} factors.")
    click.echo(f"Saved to: {filepath}")


# =============================================================================
# design
# =============================================================================


@main.command()
@click.argument("study_file", type=click.Path(exists=True))
@click.option("--method", "-m", default="latin_hypercube", help="Design method.")
@click.option("--n-points", "-n", default=20, type=int, help="Number of design points.")
@click.option("--seed", "-s", default=None, type=int, help="Random seed.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "csv", "xlsx"]),
    default="table",
    help="Output format.",
)
@click.option("--output", "-o", default=None, help="Output file (for csv/xlsx formats).")
def design(study_file, method, n_points, seed, fmt, output):
    """Generate an experimental design.

    Example:

        jaxsr design my_study.jaxsr -m latin_hypercube -n 20 --format xlsx -o experiments.xlsx
    """
    from .study import DOEStudy

    study = DOEStudy.load(study_file)
    X = study.create_design(method=method, n_points=n_points, random_state=seed)
    study.save(study_file)

    n_runs = len(X)
    click.echo(f"Generated {n_runs} design points using {method}.")

    if fmt == "table":
        _print_design_table(study, X)
    elif fmt == "csv":
        text = _design_to_csv(study, X)
        if output:
            with open(output, "w") as f:
                f.write(text)
            click.echo(f"Written to: {output}")
        else:
            click.echo(text)
    elif fmt == "xlsx":
        from .excel import generate_template

        xlsx_path = output or study_file.replace(".jaxsr", "_template.xlsx")
        generate_template(study, xlsx_path)
        click.echo(f"Excel template written to: {xlsx_path}")


# =============================================================================
# add
# =============================================================================


@main.command()
@click.argument("study_file", type=click.Path(exists=True))
@click.argument("data_file", type=click.Path(exists=True))
@click.option("--notes", default="", help="Notes for this batch of observations.")
@click.option("--skip-validation", is_flag=True, help="Skip template validation.")
def add(study_file, data_file, notes, skip_validation):
    """Import experiment results from an Excel template or CSV.

    Example:

        jaxsr add my_study.jaxsr completed_experiments.xlsx
    """
    from .study import DOEStudy

    study = DOEStudy.load(study_file)

    if data_file.endswith(".xlsx") or data_file.endswith(".xls"):
        from .excel import TemplateValidationError, read_completed_template

        try:
            X, y = read_completed_template(study, data_file, skip_validation=skip_validation)
        except TemplateValidationError as e:
            click.echo(f"Validation error: {e}", err=True)
            raise SystemExit(1) from None
    elif data_file.endswith(".csv"):
        import numpy as np

        data = np.genfromtxt(data_file, delimiter=",", skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        X = data[:, : study.n_factors]
        y = data[:, study.n_factors]
    else:
        click.echo(f"Unsupported file format: {data_file}", err=True)
        click.echo("Supported formats: .xlsx, .csv", err=True)
        raise SystemExit(1)

    study.add_observations(X, y, notes=notes)
    study.save(study_file)

    click.echo(f"Added {len(y)} observations. Total: {study.n_observations}")
    if study.design_points is not None:
        n_pending = len(study.pending_points)
        if n_pending > 0:
            click.echo(f"{n_pending} design points still pending.")
        else:
            click.echo("All design points completed!")


# =============================================================================
# fit
# =============================================================================


@main.command()
@click.argument("study_file", type=click.Path(exists=True))
@click.option("--max-terms", "-t", default=10, type=int, help="Maximum model terms.")
@click.option("--strategy", default="greedy_forward", help="Selection strategy.")
@click.option("--criterion", default="aicc", help="Information criterion (aic, aicc, bic).")
def fit(study_file, max_terms, strategy, criterion):
    """Fit a symbolic regression model.

    Example:

        jaxsr fit my_study.jaxsr --max-terms 5 --strategy greedy_forward
    """
    from .study import DOEStudy

    study = DOEStudy.load(study_file)

    try:
        model = study.fit(
            max_terms=max_terms,
            strategy=strategy,
            information_criterion=criterion,
        )
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from None

    study.save(study_file)

    click.echo(f"Model: {model.expression_}")
    click.echo(f"  MSE:   {model._result.mse:.6g}")
    click.echo(f"  AIC:   {model._result.aic:.4f}")
    click.echo(f"  BIC:   {model._result.bic:.4f}")
    click.echo(f"  Terms: {model._result.n_terms}")


# =============================================================================
# suggest
# =============================================================================


@main.command()
@click.argument("study_file", type=click.Path(exists=True))
@click.option("--n-points", "-n", default=5, type=int, help="Number of points to suggest.")
@click.option("--strategy", default="space_filling", help="Suggestion strategy.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "csv"]),
    default="table",
    help="Output format.",
)
def suggest(study_file, n_points, strategy, fmt):
    """Suggest next experiments.

    Example:

        jaxsr suggest my_study.jaxsr -n 5 --strategy uncertainty
    """
    from .study import DOEStudy

    study = DOEStudy.load(study_file)

    try:
        next_pts = study.suggest_next(n_points=n_points, strategy=strategy)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from None

    click.echo(f"Suggested {len(next_pts)} next experiments:")
    if fmt == "table":
        _print_design_table(study, next_pts)
    else:
        click.echo(_design_to_csv(study, next_pts))


# =============================================================================
# report
# =============================================================================


@main.command()
@click.argument("study_file", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output file (.xlsx or .docx).")
def report(study_file, output):
    """Generate a report (Excel or Word).

    Example:

        jaxsr report my_study.jaxsr -o report.xlsx
        jaxsr report my_study.jaxsr -o report.docx
    """
    from .study import DOEStudy

    study = DOEStudy.load(study_file)

    if not study.is_fitted:
        click.echo("Error: Study has no fitted model. Run 'jaxsr fit' first.", err=True)
        raise SystemExit(1)

    if output.endswith(".xlsx"):
        from .excel import add_report_sheets

        add_report_sheets(study, output)
        click.echo(f"Excel report written to: {output}")
    elif output.endswith(".docx"):
        from .reporting import generate_word_report

        generate_word_report(study, output)
        click.echo(f"Word report written to: {output}")
    else:
        click.echo(f"Unsupported format: {output}. Use .xlsx or .docx", err=True)
        raise SystemExit(1)


# =============================================================================
# status
# =============================================================================


@main.command()
@click.argument("study_file", type=click.Path(exists=True))
def status(study_file):
    """Show study status summary.

    Example:

        jaxsr status my_study.jaxsr
    """
    from .study import DOEStudy

    study = DOEStudy.load(study_file)
    click.echo(study.summary())


# =============================================================================
# install-skill
# =============================================================================


@main.command()
@click.option("--port", "-p", default=8501, type=int, help="Port for the Streamlit server.")
@click.option("--study", "-s", default=None, type=click.Path(), help="Pre-load a .jaxsr study.")
def app(port, study):
    """Launch the interactive JAXSR DOE app (Streamlit).

    Example:

        jaxsr app
        jaxsr app --port 8502 --study my_study.jaxsr
    """
    try:
        from .app import launch_app
    except ImportError:
        click.echo(
            "Error: streamlit is required for the JAXSR app. "
            "Install it with: pip install jaxsr[app]",
            err=True,
        )
        raise SystemExit(1) from None

    launch_app(port=port, study_path=study)


@main.command("install-skill")
@click.option(
    "--target",
    "-t",
    default=None,
    help="Target directory (default: .claude/skills/jaxsr in current directory).",
)
def install_skill(target):
    """Install the JAXSR Claude Code skill files.

    Copies the JAXSR skill (SKILL.md, guides, and templates) into a
    Claude Code skills directory so that Claude can assist with JAXSR
    setup, analysis, and reporting.

    Example:

        jaxsr install-skill
        jaxsr install-skill --target ~/.claude/skills/jaxsr
    """
    import shutil
    from pathlib import Path

    # Locate the bundled skill files relative to this package
    skill_source = Path(__file__).parent / "skill"

    if not skill_source.exists():
        # Fallback: check the repo root .claude/skills/jaxsr
        repo_root = Path(__file__).parent.parent.parent
        skill_source = repo_root / ".claude" / "skills" / "jaxsr"

    if not skill_source.exists():
        pkg_path = Path(__file__).parent / "skill"
        repo_path = Path(__file__).parent.parent.parent / ".claude" / "skills" / "jaxsr"
        click.echo(
            "Error: Could not locate JAXSR skill files.\n"
            f"  Searched: {pkg_path}\n"
            f"  Searched: {repo_path}\n"
            "Reinstall with: pip install -e '.[cli]'",
            err=True,
        )
        raise SystemExit(1)

    # Determine target directory
    if target is None:
        target_dir = Path.cwd() / ".claude" / "skills" / "jaxsr"
    else:
        target_dir = Path(target)

    # Safety: refuse to delete system or home directories
    resolved = target_dir.resolve()
    dangerous = {Path("/"), Path("/home"), Path("/usr"), Path("/etc"), Path("/var"), Path("/tmp")}
    if resolved in dangerous or resolved == Path.home():
        click.echo(
            f"Error: Refusing to use '{resolved}' as target — " "it is a system or home directory.",
            err=True,
        )
        raise SystemExit(1)

    # Ensure parent directories exist
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # Copy skill files
    if target_dir.exists():
        if not target_dir.is_dir():
            click.echo(f"Error: '{target_dir}' exists but is not a directory.", err=True)
            raise SystemExit(1)
        click.echo(f"Updating existing skill at: {target_dir}")
        shutil.rmtree(target_dir)
    else:
        click.echo(f"Installing skill to: {target_dir}")

    shutil.copytree(skill_source, target_dir)

    # Count installed files
    n_files = sum(1 for _ in target_dir.rglob("*") if _.is_file())
    click.echo(f"Installed {n_files} skill files.")
    click.echo(f"\nSkill location: {target_dir}")
    click.echo("Claude Code will now have access to JAXSR guidance when working in this project.")


# =============================================================================
# Helpers
# =============================================================================


def _print_design_table(study, X):
    """Print a design matrix as a formatted table."""

    n_runs, n_factors = X.shape
    names = study.factor_names

    # Determine column widths
    col_widths = [max(5, len("Run"))]
    for name in names:
        col_widths.append(max(10, len(name) + 2))

    # Header
    header = f"{'Run':>{col_widths[0]}}"
    for i, name in enumerate(names):
        header += f"  {name:>{col_widths[i + 1]}}"
    click.echo(header)
    click.echo("-" * len(header))

    # Determine categorical factors
    cat_indices = set()
    if study.feature_types:
        cat_indices = {i for i, ft in enumerate(study.feature_types) if ft == "categorical"}

    # Rows
    for row_idx in range(n_runs):
        row_str = f"{row_idx + 1:>{col_widths[0]}}"
        for col_idx in range(n_factors):
            val = X[row_idx, col_idx]
            if col_idx in cat_indices and study.categories and col_idx in study.categories:
                cat_val = str(study.categories[col_idx][int(val)])
                row_str += f"  {cat_val:>{col_widths[col_idx + 1]}}"
            else:
                row_str += f"  {val:>{col_widths[col_idx + 1]}.3f}"
        click.echo(row_str)


def _design_to_csv(study, X):
    """Convert a design matrix to CSV format."""

    lines = [",".join(study.factor_names + ["Response"])]

    cat_indices = set()
    if study.feature_types:
        cat_indices = {i for i, ft in enumerate(study.feature_types) if ft == "categorical"}

    for row in X:
        parts = []
        for col_idx, val in enumerate(row):
            if col_idx in cat_indices and study.categories and col_idx in study.categories:
                parts.append(str(study.categories[col_idx][int(val)]))
            else:
                parts.append(f"{val:.6f}")
        parts.append("")  # empty response column
        lines.append(",".join(parts))

    return "\n".join(lines)
