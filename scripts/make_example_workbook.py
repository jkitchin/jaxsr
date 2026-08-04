#!/usr/bin/env python
"""
Generate the example workbook offered by the JAXSR browser app.

The workbook doubles as a worked example and as a template: it shows the shape
of file the app expects (one row per run, one column per variable, a header
row) and it deliberately includes the messy bits real data has -- an ID column,
a text column, a variable that does not belong in the model, and a few failed
runs with a blank response.

The data is generated from equations recorded on the "Answer key" sheet, so a
newcomer can check whether the app found the right thing.

Usage
-----
    python scripts/make_example_workbook.py [output.xlsx]

Called automatically by ``scripts/build_webapp.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xlsxwriter

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "webapp" / "example" / "jaxsr-example.xlsx"

# The reactor sheet is generated from this equation.  Four terms, all
# recoverable from the app's default basis (constant + linear + polynomials to
# degree 3 + interactions to order 2).
REACTOR_TRUTH = (
    "conversion = 0.0040*temp_K + 0.055*pressure_bar - 2.9e-6*temp_K^2 + 1.8e-5*temp_K*pressure_bar"
)
CALIBRATION_TRUTH = "concentration = 0.4348*signal - 0.1739"

INSTRUCTIONS = [
    ("title", "JAXSR example workbook"),
    (
        "body",
        "This workbook is both a worked example and a template. Open the app at "
        "https://kitchingroup.cheme.cmu.edu/jaxsr/app/ and upload this file to follow along. "
        "Nothing is uploaded to a server: the app runs entirely inside your browser.",
    ),
    ("head", "What is in here"),
    (
        "body",
        "• reactor — 120 runs of a fictional catalytic reactor. This is the sheet to start with.\n"
        "• calibration — a simple 40-point instrument calibration, for a one-feature example.\n"
        "• Answer key — the equations the data was generated from, so you can check the result.",
    ),
    ("head", "Try it"),
    (
        "body",
        "1. Drop this file onto the app. Choose the 'reactor' sheet if it is not already selected.\n\n"
        "2. Check the column roles. The app guesses them, and for this sheet it should land on:\n"
        "       run_id → Ignore        (a label, not a measurement)\n"
        "       catalyst → Ignore      (text; see 'What is not covered yet' below)\n"
        "       temp_K → Feature\n"
        "       pressure_bar → Feature\n"
        "       stir_rpm → Feature\n"
        "       conversion → Target    (the quantity you want an equation for)\n\n"
        "3. Note the message under the table: three runs are dropped. Two have a blank conversion "
        "(the analysis failed) and one has 'n/a' typed where a pressure should be. The app uses only "
        "rows that are numeric in every selected column, and tells you how many it discarded.\n\n"
        "4. Leave the candidate functions at their defaults: constant, linear, polynomials up to "
        "degree 3, and interactions up to order 2. The readout should say 13 candidate terms.\n\n"
        "5. Press Fit.",
    ),
    ("head", "What you should see"),
    (
        "body",
        "The top-ranked model has five terms and R² of about 0.999. Four of them are the equation on "
        "the 'Answer key' sheet: a linear temperature term, a linear pressure term, a negative "
        "temperature-squared term, and a small temperature×pressure interaction. The fifth is the "
        "intercept.\n\n"
        "Three things are worth noticing.\n\n"
        "First, stir_rpm was offered to the search as a feature but does not appear in the answer. It "
        "was generated independently of conversion. A method that fits noise would have picked it up; "
        "that it did not is the check that the selection is doing its job.\n\n"
        "Second, the intercept is selected but its confidence interval straddles zero. The data was "
        "generated with no constant term, and the fit says so — the estimate is not distinguishable "
        "from zero. This is what a term that does not belong looks like when it does get selected.\n\n"
        "Third, look at the ranked table, not only the winner. AICc falls by more than a hundred at "
        "every step from one term up to five, and the last of those steps is the temperature-squared "
        "term entering — so there is no ambiguity that curvature is real in this data. That is what a "
        "clear-cut case looks like. On real measurements the last one or two steps are often worth "
        "only a few units of AICc, and then the choice between a simpler and a slightly more accurate "
        "model is yours to make rather than the criterion's. Click any row to see its coefficients, "
        "intervals and residual plots.",
    ),
    ("head", "Reading the coefficient table"),
    (
        "body",
        "Each coefficient comes with a confidence interval. A checkmark means the interval excludes "
        "zero, so that term is distinguishable from noise. Compare the four checked intervals against "
        "the equation on the 'Answer key' sheet: each one should contain the true value. That, rather "
        "than exact agreement, is the right test — the data has noise in it.\n\n"
        "The ANOVA table underneath apportions the variation among the terms. A term with a large "
        "coefficient but a tiny percentage contribution is not doing much work.",
    ),
    ("head", "Then try"),
    (
        "body",
        "• Switch the sheet to 'calibration' and fit concentration against signal. One feature, one "
        "straight line — the simplest case, and a good sanity check.\n\n"
        "• On the reactor sheet, set Max terms to 2 and refit. Watch R² fall and the ranked table "
        "shorten. This is the accuracy-versus-simplicity trade-off the Pareto chart draws.\n\n"
        "• Turn on Transcendental or Ratios and refit. The library grows, the search gets slower, and "
        "the risk of a term fitting noise rather than physics goes up. Only enable families you have a "
        "physical reason to expect.\n\n"
        "• Export the Python script and the cleaned data. Running that script with `pip install jaxsr` "
        "reproduces the fit outside the browser, which is where you would go to script it over many "
        "datasets.",
    ),
    ("head", "Using your own data"),
    (
        "body",
        "Lay it out like the 'reactor' sheet: one header row of column names, one row per observation, "
        "one column per variable. Extra columns are fine — mark them Ignore. Blank cells are fine — "
        "those rows are dropped and counted. .xlsx, .xls and .csv all work.",
    ),
    ("head", "What is not covered yet"),
    (
        "body",
        "Text columns such as 'catalyst' cannot be used as features in this version, so the app marks "
        "them Ignore. The jaxsr library itself does handle categorical variables "
        "(BasisLibrary.add_categorical_indicators); it is the browser front end that does not expose "
        "them yet.",
    ),
]


def _write_instructions(workbook: xlsxwriter.Workbook) -> None:
    """Write the 'Start here' sheet."""
    sheet = workbook.add_worksheet("Start here")
    sheet.hide_gridlines(2)
    sheet.set_column("A:A", 2)
    sheet.set_column("B:B", 100)

    title = workbook.add_format({"bold": True, "font_size": 18, "font_color": "#1F2328"})
    head = workbook.add_format(
        {"bold": True, "font_size": 12, "font_color": "#2563EB", "align": "left"}
    )
    body = workbook.add_format({"text_wrap": True, "valign": "top", "font_size": 11})

    row = 1
    for kind, text in INSTRUCTIONS:
        if kind == "title":
            sheet.write(row, 1, text, title)
            row += 2
        elif kind == "head":
            sheet.write(row, 1, text, head)
            row += 1
        else:
            lines = text.count("\n") + 1
            # Roughly 95 characters fit on a line at this column width.
            wrapped = sum(max(1, len(part) // 95 + 1) for part in text.split("\n"))
            sheet.set_row(row, 15 * max(lines, wrapped))
            sheet.write(row, 1, text, body)
            row += 2


def _write_reactor(workbook: xlsxwriter.Workbook, rng: np.random.Generator) -> None:
    """Write the main dataset: three usable features, one of them irrelevant."""
    n = 120
    temp = rng.uniform(300.0, 500.0, n)
    pressure = rng.uniform(1.0, 10.0, n)
    stir = rng.uniform(200.0, 900.0, n)  # deliberately unrelated to the response
    conversion = (
        0.0040 * temp
        + 0.055 * pressure
        - 2.9e-6 * temp**2
        + 1.8e-5 * temp * pressure
        + rng.normal(0.0, 0.008, n)
    )

    header = workbook.add_format(
        {"bold": True, "bg_color": "#EAF0FE", "bottom": 1, "border_color": "#B7C7EE"}
    )
    num3 = workbook.add_format({"num_format": "0.000"})
    num1 = workbook.add_format({"num_format": "0.0"})
    num4 = workbook.add_format({"num_format": "0.0000"})

    sheet = workbook.add_worksheet("reactor")
    sheet.freeze_panes(1, 0)
    for col, (name, width) in enumerate(
        [
            ("run_id", 10),
            ("catalyst", 10),
            ("temp_K", 10),
            ("pressure_bar", 14),
            ("stir_rpm", 10),
            ("conversion", 12),
        ]
    ):
        sheet.write(0, col, name, header)
        sheet.set_column(col, col, width)

    catalysts = ["Pt/Al2O3", "Pd/C", "Ru/TiO2"]
    # A few runs that did not produce a usable number, so the app has something
    # to drop and report.
    blank_response = {17, 88}
    bad_pressure = {54}

    for i in range(n):
        sheet.write(i + 1, 0, f"R-{i + 1:03d}")
        sheet.write(i + 1, 1, catalysts[i % 3])
        sheet.write_number(i + 1, 2, round(float(temp[i]), 1), num1)
        if i in bad_pressure:
            sheet.write(i + 1, 3, "n/a")
        else:
            sheet.write_number(i + 1, 3, round(float(pressure[i]), 3), num3)
        sheet.write_number(i + 1, 4, round(float(stir[i])), num1)
        if i in blank_response:
            sheet.write_blank(i + 1, 5, None)
        else:
            sheet.write_number(i + 1, 5, round(float(conversion[i]), 4), num4)


def _write_calibration(workbook: xlsxwriter.Workbook, rng: np.random.Generator) -> None:
    """Write a one-feature calibration dataset."""
    n = 40
    concentration = np.linspace(0.5, 20.0, n)
    signal = 2.3 * concentration + 0.4 + rng.normal(0.0, 0.05, n)

    header = workbook.add_format(
        {"bold": True, "bg_color": "#EAF0FE", "bottom": 1, "border_color": "#B7C7EE"}
    )
    num3 = workbook.add_format({"num_format": "0.000"})

    sheet = workbook.add_worksheet("calibration")
    sheet.freeze_panes(1, 0)
    for col, (name, width) in enumerate([("standard", 10), ("signal", 12), ("concentration", 15)]):
        sheet.write(0, col, name, header)
        sheet.set_column(col, col, width)
    for i in range(n):
        sheet.write(i + 1, 0, f"STD-{i + 1:02d}")
        sheet.write_number(i + 1, 1, round(float(signal[i]), 3), num3)
        sheet.write_number(i + 1, 2, round(float(concentration[i]), 3), num3)


def _write_answers(workbook: xlsxwriter.Workbook) -> None:
    """Write the sheet recording how the data was generated."""
    sheet = workbook.add_worksheet("Answer key")
    sheet.hide_gridlines(2)
    sheet.set_column("A:A", 2)
    sheet.set_column("B:B", 100)

    head = workbook.add_format({"bold": True, "font_size": 12, "font_color": "#2563EB"})
    body = workbook.add_format({"text_wrap": True, "valign": "top"})
    mono = workbook.add_format({"font_name": "Menlo", "font_size": 10, "text_wrap": True})

    sheet.write(1, 1, "How this data was generated", head)
    sheet.set_row(2, 30)
    sheet.write(
        2,
        1,
        "Both sheets are synthetic, so there is a right answer. Gaussian noise was added on top "
        "of the equations below.",
        body,
    )

    sheet.write(4, 1, "reactor", head)
    sheet.set_row(5, 30)
    sheet.write(5, 1, REACTOR_TRUTH, mono)
    sheet.set_row(6, 45)
    sheet.write(
        6,
        1,
        "Noise: normal, standard deviation 0.008. stir_rpm was drawn independently and does not "
        "appear in the equation — a good model leaves it out.",
        body,
    )

    sheet.write(8, 1, "calibration", head)
    sheet.set_row(9, 15)
    sheet.write(9, 1, CALIBRATION_TRUTH, mono)
    sheet.set_row(10, 30)
    sheet.write(
        10,
        1,
        "Generated as signal = 2.3*concentration + 0.4 with noise of standard deviation 0.05, then "
        "inverted so that concentration is the response.",
        body,
    )

    sheet.write(12, 1, "Why the recovered coefficients will not match exactly", head)
    sheet.set_row(13, 45)
    sheet.write(
        13,
        1,
        "The noise means the fitted coefficients are estimates. The confidence intervals in the app "
        "should contain the values above; that is the check to make, not exact agreement.",
        body,
    )


def main(argv: list[str]) -> int:
    """
    Build the example workbook.

    Parameters
    ----------
    argv : list of str
        Optional single argument: the output path.

    Returns
    -------
    int
        Process exit code.
    """
    output = Path(argv[0]) if argv else DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    # Fixed seed so the workbook is byte-stable across rebuilds.
    rng = np.random.default_rng(20260804)

    workbook = xlsxwriter.Workbook(str(output))
    workbook.set_properties(
        {
            "title": "JAXSR example workbook",
            "subject": "Worked example and template for the JAXSR browser app",
            "comments": "Synthetic data; see the Answer key sheet.",
        }
    )
    _write_instructions(workbook)
    _write_reactor(workbook, rng)
    _write_calibration(workbook, rng)
    _write_answers(workbook)
    workbook.close()

    print(f"  example   {output.relative_to(REPO_ROOT)} ({output.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
