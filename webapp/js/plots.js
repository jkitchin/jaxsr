/*
 * Small inline-SVG chart helpers.
 *
 * Deliberately not matplotlib: keeping the plots on the JavaScript side saves
 * ~10 MB of download, they inherit the page's light/dark theme through
 * currentColor and CSS variables, and each one can be saved as a real .svg.
 */

const NS = "http://www.w3.org/2000/svg";
const PAD = { top: 14, right: 16, bottom: 40, left: 58 };

function el(name, attrs = {}) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  return node;
}

/** Nice round tick values covering [lo, hi]. */
function ticks(lo, hi, count = 5) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0];
  if (lo === hi) return [lo];
  const span = hi - lo;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  }
  return out.length ? out : [lo, hi];
}

function fmt(value) {
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1e5 || abs < 1e-3) return value.toExponential(1);
  return String(Number(value.toPrecision(4)));
}

function extent(values, padFraction = 0.05) {
  const finite = values.filter((v) => Number.isFinite(v));
  if (!finite.length) return [0, 1];
  let lo = Math.min(...finite);
  let hi = Math.max(...finite);
  if (lo === hi) {
    const bump = Math.abs(lo) || 1;
    return [lo - bump * 0.1, hi + bump * 0.1];
  }
  const pad = (hi - lo) * padFraction;
  return [lo - pad, hi + pad];
}

/**
 * Build an empty chart frame with axes, returning helpers to draw into it.
 *
 * @returns {{svg: SVGElement, plot: SVGElement, sx: (v:number)=>number, sy: (v:number)=>number}}
 */
function frame({ width, height, xDomain, yDomain, xLabel, yLabel, title }) {
  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "chart",
    role: "img",
    "aria-label": title ?? "",
    preserveAspectRatio: "xMidYMid meet",
  });

  const innerW = width - PAD.left - PAD.right;
  const innerH = height - PAD.top - PAD.bottom;
  const [x0, x1] = xDomain;
  const [y0, y1] = yDomain;
  const sx = (v) => PAD.left + ((v - x0) / (x1 - x0 || 1)) * innerW;
  const sy = (v) => PAD.top + innerH - ((v - y0) / (y1 - y0 || 1)) * innerH;

  const grid = el("g", { class: "grid" });
  for (const t of ticks(x0, x1)) {
    grid.append(el("line", { x1: sx(t), y1: PAD.top, x2: sx(t), y2: PAD.top + innerH }));
    const label = el("text", { x: sx(t), y: PAD.top + innerH + 16, class: "tick" });
    label.textContent = fmt(t);
    svg.append(label);
  }
  for (const t of ticks(y0, y1)) {
    grid.append(el("line", { x1: PAD.left, y1: sy(t), x2: PAD.left + innerW, y2: sy(t) }));
    const label = el("text", { x: PAD.left - 8, y: sy(t) + 4, class: "tick tick-y" });
    label.textContent = fmt(t);
    svg.append(label);
  }
  svg.prepend(grid);

  const axis = el("g", { class: "axis" });
  axis.append(el("line", { x1: PAD.left, y1: PAD.top + innerH, x2: PAD.left + innerW, y2: PAD.top + innerH }));
  axis.append(el("line", { x1: PAD.left, y1: PAD.top, x2: PAD.left, y2: PAD.top + innerH }));
  svg.append(axis);

  if (xLabel) {
    const label = el("text", { x: PAD.left + innerW / 2, y: height - 6, class: "axis-label" });
    label.textContent = xLabel;
    svg.append(label);
  }
  if (yLabel) {
    const label = el("text", {
      x: 14,
      y: PAD.top + innerH / 2,
      class: "axis-label",
      transform: `rotate(-90 14 ${PAD.top + innerH / 2})`,
    });
    label.textContent = yLabel;
    svg.append(label);
  }

  const plot = el("g", { class: "marks" });
  svg.append(plot);
  return { svg, plot, sx, sy, innerW, innerH };
}

function pairs(xs, ys) {
  const out = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) out.push([xs[i], ys[i]]);
  }
  return out;
}

/**
 * Scatter plot, optionally with a reference line.
 *
 * @param {object} options
 * @returns {SVGElement}
 */
export function scatter({
  x,
  y,
  xLabel,
  yLabel,
  title,
  width = 460,
  height = 320,
  reference = null,
  square = false,
}) {
  const points = pairs(x, y);
  let xDomain = extent(points.map((p) => p[0]));
  let yDomain = extent(points.map((p) => p[1]));
  if (square) {
    const lo = Math.min(xDomain[0], yDomain[0]);
    const hi = Math.max(xDomain[1], yDomain[1]);
    xDomain = [lo, hi];
    yDomain = [lo, hi];
  }

  const { svg, plot, sx, sy } = frame({ width, height, xDomain, yDomain, xLabel, yLabel, title });

  if (reference === "identity") {
    plot.append(
      el("line", {
        x1: sx(xDomain[0]),
        y1: sy(xDomain[0]),
        x2: sx(xDomain[1]),
        y2: sy(xDomain[1]),
        class: "reference",
      })
    );
  } else if (reference === "zero") {
    plot.append(
      el("line", { x1: sx(xDomain[0]), y1: sy(0), x2: sx(xDomain[1]), y2: sy(0), class: "reference" })
    );
  }

  const radius = points.length > 2000 ? 1.3 : points.length > 500 ? 2 : 2.8;
  for (const [px, py] of points) {
    plot.append(el("circle", { cx: sx(px), cy: sy(py), r: radius, class: "point" }));
  }
  return svg;
}

/**
 * The accuracy/complexity trade-off curve, with Pareto members emphasised.
 *
 * @param {{rows: object[]}} options
 * @returns {SVGElement}
 */
export function paretoChart({ rows, width = 460, height = 320 }) {
  const usable = rows.filter((r) => Number.isFinite(r.complexity) && Number.isFinite(r.mse) && r.mse > 0);
  if (!usable.length) return frame({ width, height, xDomain: [0, 1], yDomain: [0, 1] }).svg;

  const xs = usable.map((r) => r.complexity);
  const ys = usable.map((r) => Math.log10(r.mse));
  const { svg, plot, sx, sy } = frame({
    width,
    height,
    xDomain: extent(xs, 0.12),
    yDomain: extent(ys, 0.12),
    xLabel: "complexity",
    yLabel: "log₁₀ MSE",
    title: "Accuracy versus complexity",
  });

  const front = usable
    .filter((r) => r.is_pareto)
    .sort((a, b) => a.complexity - b.complexity);
  if (front.length > 1) {
    const d = front
      .map((r, i) => `${i === 0 ? "M" : "L"}${sx(r.complexity)},${sy(Math.log10(r.mse))}`)
      .join(" ");
    plot.append(el("path", { d, class: "pareto-line" }));
  }

  for (const row of usable) {
    const classes = ["point"];
    if (row.is_pareto) classes.push("pareto");
    if (row.is_best) classes.push("best");
    const marker = el("circle", {
      cx: sx(row.complexity),
      cy: sy(Math.log10(row.mse)),
      r: row.is_best ? 6 : 4,
      class: classes.join(" "),
    });
    const tip = el("title");
    tip.textContent = `${row.n_terms} terms · ${row.expression}`;
    marker.append(tip);
    plot.append(marker);
  }
  return svg;
}

/**
 * Coefficient estimates and intervals, divided through by their standard error.
 *
 * Raw coefficients routinely span several orders of magnitude — a temperature
 * term at 4e-3 next to a squared term at -3e-6 — so on one shared axis every
 * small term collapses onto zero and the chart says nothing. Dividing by the
 * standard error puts every term on the same scale, keeps the zero line
 * meaningful, and makes "does this interval cross zero?" the thing you can
 * actually see. The unscaled numbers are in the table directly above.
 *
 * @param {{intervals: object[]}} options
 * @returns {SVGElement}
 */
export function coefficientChart({ intervals, width = 460 }) {
  const rows = intervals
    .filter((r) => Number.isFinite(r.estimate))
    .map((r) => {
      const se = Number.isFinite(r.se) && r.se > 0 ? r.se : null;
      return {
        ...r,
        t: se ? r.estimate / se : null,
        tLower: se && Number.isFinite(r.lower) ? r.lower / se : null,
        tUpper: se && Number.isFinite(r.upper) ? r.upper / se : null,
      };
    });

  const height = Math.max(150, 56 + rows.length * 34);
  // Give the labels room for the longest term, since interaction names get long.
  const longest = rows.reduce((max, r) => Math.max(max, r.name.length), 0);
  const innerLeft = Math.min(220, Math.max(96, longest * 6.6 + 16));

  const values = rows.flatMap((r) => [r.tLower, r.tUpper, r.t].filter(Number.isFinite));
  const [lo, hi] = values.length ? extent(values, 0.12) : [-1, 1];
  const domain = [Math.min(0, lo), Math.max(0, hi)];

  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "chart",
    role: "img",
    "aria-label": "Coefficient estimates divided by their standard error",
    preserveAspectRatio: "xMidYMid meet",
  });
  const innerW = width - innerLeft - 16;
  const sx = (v) => innerLeft + ((v - domain[0]) / (domain[1] - domain[0] || 1)) * innerW;

  svg.append(el("line", { x1: sx(0), y1: 8, x2: sx(0), y2: height - 34, class: "reference" }));

  rows.forEach((row, i) => {
    const y = 22 + i * 34;
    const label = el("text", { x: innerLeft - 10, y: y + 4, class: "tick tick-y" });
    label.textContent = row.name;
    const labelTip = el("title");
    labelTip.textContent = `${row.name} = ${row.estimate}`;
    label.append(labelTip);
    svg.append(label);

    if (row.t === null) return;

    if (Number.isFinite(row.tLower) && Number.isFinite(row.tUpper)) {
      svg.append(
        el("line", {
          x1: sx(row.tLower),
          y1: y,
          x2: sx(row.tUpper),
          y2: y,
          class: `interval${row.significant ? " significant" : ""}`,
        })
      );
      for (const bound of [row.tLower, row.tUpper]) {
        svg.append(el("line", { x1: sx(bound), y1: y - 5, x2: sx(bound), y2: y + 5, class: "interval-cap" }));
      }
    }
    const marker = el("circle", {
      cx: sx(row.t),
      cy: y,
      r: 4,
      class: `point${row.significant ? " best" : ""}`,
    });
    const tip = el("title");
    tip.textContent = `${row.name}: ${row.estimate} ± ${row.se}`;
    marker.append(tip);
    svg.append(marker);
  });

  for (const t of ticks(domain[0], domain[1], 4)) {
    const label = el("text", { x: sx(t), y: height - 14, class: "tick" });
    label.textContent = fmt(t);
    svg.append(label);
  }
  const caption = el("text", { x: innerLeft + innerW / 2, y: height - 2, class: "axis-label" });
  caption.textContent = "estimate ÷ standard error";
  svg.append(caption);
  return svg;
}

/**
 * Serialise a rendered chart for download.
 *
 * @param {SVGElement} svg
 * @returns {string}
 */
export function toStandaloneSVG(svg) {
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", NS);
  const style = document.createElementNS(NS, "style");
  // Inline enough styling that the file looks right outside the page.
  style.textContent = `
    .grid line { stroke: #d5d8dd; stroke-width: 1; }
    .axis line { stroke: #6b7280; stroke-width: 1; }
    text { font-family: system-ui, sans-serif; fill: #374151; }
    .tick { font-size: 11px; text-anchor: middle; }
    .tick-y { text-anchor: end; }
    .axis-label { font-size: 12px; text-anchor: middle; }
    .point { fill: #2563eb; fill-opacity: 0.6; }
    .point.best { fill: #dc2626; fill-opacity: 1; }
    .point.pareto { fill: #059669; fill-opacity: 0.9; }
    .reference { stroke: #9ca3af; stroke-dasharray: 4 3; }
    .pareto-line { fill: none; stroke: #059669; stroke-width: 1.5; }
    .interval { stroke: #2563eb; stroke-width: 2; }
    .interval.significant { stroke: #dc2626; }
    .interval-cap { stroke: #2563eb; stroke-width: 1.5; }
  `;
  clone.prepend(style);
  return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`;
}
