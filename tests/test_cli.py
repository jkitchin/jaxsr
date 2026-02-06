"""Tests for the JAXSR Click CLI."""

from __future__ import annotations

import os

import numpy as np
import pytest
from click.testing import CliRunner

from jaxsr.cli import main
from jaxsr.study import DOEStudy


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def study_file(tmp_path):
    """Create a study file for testing."""
    study = DOEStudy(
        name="cli_test",
        factor_names=["x1", "x2"],
        bounds=[(0, 1), (0, 1)],
    )
    filepath = str(tmp_path / "test.jaxsr")
    study.save(filepath)
    return filepath


@pytest.fixture
def study_with_design(tmp_path):
    """Create a study file with a design."""
    study = DOEStudy(
        name="cli_test",
        factor_names=["x1", "x2"],
        bounds=[(0, 1), (0, 1)],
    )
    study.create_design(method="latin_hypercube", n_points=10, random_state=42)
    filepath = str(tmp_path / "test.jaxsr")
    study.save(filepath)
    return filepath


@pytest.fixture
def fitted_study_file(tmp_path):
    """Create a fitted study file."""
    np.random.seed(42)
    study = DOEStudy(
        name="cli_test",
        factor_names=["x1", "x2"],
        bounds=[(0, 1), (0, 1)],
    )
    X = np.random.rand(30, 2)
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.01 * np.random.randn(30)
    study.add_observations(X, y, record_iteration=False)
    study.fit(max_terms=3)
    filepath = str(tmp_path / "fitted.jaxsr")
    study.save(filepath)
    return filepath


# =============================================================================
# init command
# =============================================================================


class TestInitCommand:
    """Test the 'init' subcommand."""

    def test_init_basic(self, runner, tmp_path):
        output_file = str(tmp_path / "my_study.jaxsr")
        result = runner.invoke(
            main,
            ["init", "my_study", "-f", "x1:0:10", "-f", "x2:5:20", "-o", output_file],
        )
        assert result.exit_code == 0, result.output
        assert "Created study" in result.output
        assert os.path.exists(output_file)

        study = DOEStudy.load(output_file)
        assert study.name == "my_study"
        assert study.factor_names == ["x1", "x2"]
        assert study.bounds == [(0.0, 10.0), (5.0, 20.0)]

    def test_init_with_categorical(self, runner, tmp_path):
        output_file = str(tmp_path / "cat_study.jaxsr")
        result = runner.invoke(
            main,
            [
                "init",
                "cat_study",
                "-f",
                "temp:300:500",
                "-f",
                "catalyst:A,B,C",
                "-d",
                "Test study",
                "-o",
                output_file,
            ],
        )
        assert result.exit_code == 0, result.output

        study = DOEStudy.load(output_file)
        assert study.feature_types == ["continuous", "categorical"]
        assert study.categories == {1: ["A", "B", "C"]}
        assert study.description == "Test study"

    def test_init_bad_factor_spec(self, runner, tmp_path):
        result = runner.invoke(
            main,
            ["init", "bad", "-f", "x1", "-o", str(tmp_path / "bad.jaxsr")],
        )
        assert result.exit_code != 0


# =============================================================================
# design command
# =============================================================================


class TestDesignCommand:
    """Test the 'design' subcommand."""

    def test_design_table(self, runner, study_file):
        result = runner.invoke(
            main,
            ["design", study_file, "-m", "latin_hypercube", "-n", "5", "-s", "42"],
        )
        assert result.exit_code == 0, result.output
        assert "Generated 5 design points" in result.output

    def test_design_csv(self, runner, study_file):
        result = runner.invoke(
            main,
            [
                "design",
                study_file,
                "-m",
                "latin_hypercube",
                "-n",
                "5",
                "-s",
                "42",
                "--format",
                "csv",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "x1,x2,Response" in result.output

    def test_design_xlsx(self, runner, study_file, tmp_path):
        xlsx_path = str(tmp_path / "design.xlsx")
        result = runner.invoke(
            main,
            [
                "design",
                study_file,
                "-m",
                "latin_hypercube",
                "-n",
                "5",
                "-s",
                "42",
                "--format",
                "xlsx",
                "-o",
                xlsx_path,
            ],
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(xlsx_path)


# =============================================================================
# add command
# =============================================================================


class TestAddCommand:
    """Test the 'add' subcommand."""

    def test_add_csv(self, runner, study_with_design, tmp_path):
        # Create a CSV with fake results
        csv_path = str(tmp_path / "results.csv")
        study = DOEStudy.load(study_with_design)
        X = study.design_points[:3]
        with open(csv_path, "w") as f:
            f.write("x1,x2,response\n")
            for row in X:
                f.write(f"{row[0]},{row[1]},{row[0] + row[1]}\n")

        result = runner.invoke(main, ["add", study_with_design, csv_path])
        assert result.exit_code == 0, result.output
        assert "Added 3 observations" in result.output

    def test_add_xlsx(self, runner, study_with_design, tmp_path):
        from jaxsr.excel import generate_template

        study = DOEStudy.load(study_with_design)
        xlsx_path = str(tmp_path / "template.xlsx")
        generate_template(study, xlsx_path)

        # Fill in responses
        import openpyxl

        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb["Design"]
        for row in range(2, 6):  # First 4 rows
            ws.cell(row=row, column=4, value=float(row * 10))
        wb.save(xlsx_path)
        wb.close()

        result = runner.invoke(main, ["add", study_with_design, xlsx_path])
        assert result.exit_code == 0, result.output
        assert "Added 4 observations" in result.output


# =============================================================================
# fit command
# =============================================================================


class TestFitCommand:
    """Test the 'fit' subcommand."""

    def test_fit_basic(self, runner, tmp_path):
        # Create a study with observations
        np.random.seed(42)
        study = DOEStudy(
            name="fit_test",
            factor_names=["x1", "x2"],
            bounds=[(0, 1), (0, 1)],
        )
        X = np.random.rand(30, 2)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1]
        study.add_observations(X, y, record_iteration=False)
        filepath = str(tmp_path / "fit.jaxsr")
        study.save(filepath)

        result = runner.invoke(main, ["fit", filepath, "--max-terms", "3"])
        assert result.exit_code == 0, result.output
        assert "Model:" in result.output
        assert "MSE:" in result.output

    def test_fit_no_data(self, runner, study_file):
        result = runner.invoke(main, ["fit", study_file])
        assert result.exit_code != 0


# =============================================================================
# status command
# =============================================================================


class TestStatusCommand:
    """Test the 'status' subcommand."""

    def test_status(self, runner, study_file):
        result = runner.invoke(main, ["status", study_file])
        assert result.exit_code == 0, result.output
        assert "cli_test" in result.output

    def test_status_fitted(self, runner, fitted_study_file):
        result = runner.invoke(main, ["status", fitted_study_file])
        assert result.exit_code == 0, result.output
        assert "Model:" in result.output


# =============================================================================
# report command
# =============================================================================


class TestReportCommand:
    """Test the 'report' subcommand."""

    def test_report_xlsx(self, runner, fitted_study_file, tmp_path):
        output = str(tmp_path / "report.xlsx")
        result = runner.invoke(main, ["report", fitted_study_file, "-o", output])
        assert result.exit_code == 0, result.output
        assert os.path.exists(output)

    def test_report_docx(self, runner, fitted_study_file, tmp_path):
        output = str(tmp_path / "report.docx")
        result = runner.invoke(main, ["report", fitted_study_file, "-o", output])
        assert result.exit_code == 0, result.output
        assert os.path.exists(output)

    def test_report_not_fitted(self, runner, study_file, tmp_path):
        output = str(tmp_path / "report.xlsx")
        result = runner.invoke(main, ["report", study_file, "-o", output])
        assert result.exit_code != 0

    def test_report_bad_format(self, runner, fitted_study_file, tmp_path):
        output = str(tmp_path / "report.pdf")
        result = runner.invoke(main, ["report", fitted_study_file, "-o", output])
        assert result.exit_code != 0


# =============================================================================
# suggest command
# =============================================================================


class TestSuggestCommand:
    """Test the 'suggest' subcommand."""

    def test_suggest_basic(self, runner, fitted_study_file):
        result = runner.invoke(
            main,
            ["suggest", fitted_study_file, "-n", "3", "--strategy", "space_filling"],
        )
        assert result.exit_code == 0, result.output
        assert "Suggested 3" in result.output

    def test_suggest_not_fitted(self, runner, study_file):
        result = runner.invoke(main, ["suggest", study_file])
        assert result.exit_code != 0
