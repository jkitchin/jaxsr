/*
 * Rendering for the ranked candidate table and the per-model detail pane.
 */

import { coefficientChart, paretoChart, scatter, toStandaloneSVG } from "./plots.js";

const NUM = (v, digits = 4) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : Number(v).toPrecision(digits);

const FIXED = (v, digits = 2) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : Number(v).toFixed(digits);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Render the ranked candidate table.
 *
 * @param {HTMLElement} host
 * @param {object} fitResult
 * @param {(row: object) => void} onSelect
 */
export function renderCandidates(host, fitResult, onSelect) {
  host.replaceChildren();
  const { candidates, information_criterion: ic, path_semantics: semantics, strategy } = fitResult;

  const caption = element(
    "p",
    "muted",
    `${candidates.total} candidate model${candidates.total === 1 ? "" : "s"} from ${strategy} — ` +
      `the path records ${semantics}. Ranked by ${ic.toUpperCase()} (lower is better).`
  );
  host.append(caption);

  if (candidates.truncated) {
    host.append(
      element("p", "muted", `Showing the top ${candidates.rows.length}; ${candidates.truncated} more not shown.`)
    );
  }

  const table = element("table", "ranked");
  const head = element("thead");
  const headRow = element("tr");
  for (const label of ["#", "Terms", "Expression", "R²", "RMSE", "AIC", "AICc", "BIC"]) {
    headRow.append(element("th", null, label));
  }
  head.append(headRow);
  table.append(head);

  const body = element("tbody");
  for (const row of candidates.rows) {
    const tr = element("tr", row.is_best ? "is-best" : null);
    tr.tabIndex = 0;
    tr.dataset.indices = JSON.stringify(row.indices);

    const rank = element("td", "rank");
    rank.append(document.createTextNode(String(row.rank)));
    if (row.is_best) rank.append(element("span", "badge best", "best"));
    else if (row.is_pareto) rank.append(element("span", "badge pareto", "pareto"));
    if (row.pruned) {
      const tag = element("span", "badge muted-badge", "pruned");
      tag.title =
        "The search picked a larger model, but terms contributing less than prune_tol " +
        "were dropped afterwards. This row shows what jaxsr returns.";
      rank.append(tag);
    }
    tr.append(rank);

    tr.append(element("td", "num", String(row.n_terms)));
    tr.append(element("td", "expr", row.expression));
    tr.append(element("td", "num", NUM(row.r2, 5)));
    tr.append(element("td", "num", NUM(row.rmse, 3)));
    tr.append(element("td", "num", FIXED(row.aic)));
    tr.append(element("td", "num", FIXED(row.aicc)));
    tr.append(element("td", "num", FIXED(row.bic)));

    const select = () => {
      body.querySelectorAll("tr").forEach((r) => r.classList.remove("is-selected"));
      tr.classList.add("is-selected");
      onSelect(row);
    };
    tr.addEventListener("click", select);
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        select();
      }
    });
    body.append(tr);
  }
  table.append(body);
  host.append(table);

  host.append(chartCard("Accuracy versus complexity", paretoChart({ rows: candidates.rows })));

  const first = body.querySelector("tr");
  if (first) first.click();
}

/**
 * Render diagnostics for one candidate model.
 *
 * @param {HTMLElement} host
 * @param {object} row - the ranked-table row that was clicked
 * @param {object} diag - kernel diagnostics payload
 */
export function renderDetail(host, row, diag) {
  host.replaceChildren();

  const header = element("div", "detail-header");
  header.append(element("h3", null, `${row.n_terms}-term model`));
  header.append(element("code", "expression", row.expression));
  host.append(header);

  const stats = element("div", "stat-row");
  for (const [label, value] of [
    ["R²", NUM(row.r2, 5)],
    ["RMSE", NUM(row.rmse, 3)],
    ["MSE", NUM(row.mse, 3)],
    ["AICc", FIXED(row.aicc)],
    ["BIC", FIXED(row.bic)],
    ["Complexity", String(row.complexity)],
  ]) {
    const card = element("div", "stat");
    card.append(element("span", "stat-label", label));
    card.append(element("span", "stat-value", value));
    stats.append(card);
  }
  host.append(stats);

  host.append(coefficientTable(diag.intervals, diag.alpha));
  const coefCard = chartCard(
    `Coefficients on a common scale, ${Math.round((1 - diag.alpha) * 100)}% intervals`,
    coefficientChart({ intervals: diag.intervals })
  );
  coefCard.append(
    element(
      "p",
      "muted",
      "Each estimate is divided by its own standard error so terms of very different " +
        "magnitude can be compared. A bar clear of the zero line is a term the data supports."
    )
  );
  host.append(coefCard);

  const charts = element("div", "chart-grid");
  charts.append(
    chartCard(
      "Predicted versus observed",
      scatter({
        x: diag.parity.observed,
        y: diag.parity.predicted,
        xLabel: "observed",
        yLabel: "predicted",
        title: "Predicted versus observed",
        reference: "identity",
        square: true,
      })
    )
  );
  charts.append(
    chartCard(
      "Residuals",
      scatter({
        x: diag.residuals.predicted,
        y: diag.residuals.residual,
        xLabel: "predicted",
        yLabel: "residual",
        title: "Residuals versus fitted values",
        reference: "zero",
      })
    )
  );
  if (diag.qq.theoretical.length) {
    charts.append(
      chartCard(
        "Normal Q-Q",
        scatter({
          x: diag.qq.theoretical,
          y: diag.qq.sample,
          xLabel: "theoretical quantile",
          yLabel: "standardised residual",
          title: "Normal quantile-quantile plot",
          reference: "identity",
          square: true,
        })
      )
    );
  }
  host.append(charts);

  if (diag.anova) {
    host.append(anovaTable(diag.anova));
  } else {
    host.append(
      element(
        "p",
        "muted",
        "ANOVA is shown for the model the information criterion selected. Select the best-ranked row to see it."
      )
    );
  }
}

function coefficientTable(intervals, alpha) {
  const section = element("section", "panel");
  section.append(element("h4", null, "Coefficients"));
  const table = element("table", "data");
  const head = element("tr");
  const pct = Math.round((1 - alpha) * 100);
  for (const label of ["Term", "Estimate", "Std. error", `Lower ${pct}%`, `Upper ${pct}%`, ""]) {
    head.append(element("th", null, label));
  }
  table.append(head);
  for (const row of intervals) {
    const tr = element("tr");
    tr.append(element("td", "expr", row.name));
    tr.append(element("td", "num", NUM(row.estimate, 5)));
    tr.append(element("td", "num", NUM(row.se, 3)));
    tr.append(element("td", "num", NUM(row.lower, 5)));
    tr.append(element("td", "num", NUM(row.upper, 5)));
    tr.append(element("td", "num", row.significant ? "✓" : ""));
    table.append(tr);
  }
  section.append(table);
  section.append(
    element(
      "p",
      "muted",
      "✓ marks a term whose interval excludes zero, so it is distinguishable from noise at this level."
    )
  );
  return section;
}

function anovaTable(anova) {
  const section = element("section", "panel");
  section.append(element("h4", null, "ANOVA"));
  if (anova.note) section.append(element("p", "warning", anova.note));

  const table = element("table", "data");
  const head = element("tr");
  for (const label of ["Source", "DF", "Sum sq.", "% contribution", "Mean sq.", "F", "p"]) {
    head.append(element("th", null, label));
  }
  table.append(head);

  const addRow = (row, className) => {
    const tr = element("tr", className);
    tr.append(element("td", "expr", row.source));
    tr.append(element("td", "num", String(row.df)));
    tr.append(element("td", "num", NUM(row.sum_sq, 4)));
    tr.append(element("td", "num", row.pct_contribution === null ? "" : `${FIXED(row.pct_contribution)}%`));
    tr.append(element("td", "num", NUM(row.mean_sq, 4)));
    tr.append(element("td", "num", NUM(row.f_value, 4)));
    tr.append(element("td", "num", row.p_value === null ? "—" : NUM(row.p_value, 3)));
    table.append(tr);
  };
  anova.terms.forEach((row) => addRow(row, null));
  anova.summary.forEach((row) => addRow(row, "summary-row"));
  section.append(table);
  return section;
}

function chartCard(title, svg) {
  const card = element("figure", "chart-card");
  const caption = element("figcaption");
  caption.append(element("span", null, title));
  const save = element("button", "link-button", "save .svg");
  save.type = "button";
  save.addEventListener("click", () => {
    const blob = new Blob([toStandaloneSVG(svg)], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.svg`;
    link.click();
    URL.revokeObjectURL(url);
  });
  caption.append(save);
  card.append(caption);
  card.append(svg);
  return card;
}
