/*
 * JAXSR web app — UI state machine.
 *
 * Owns the DOM and talks to the Pyodide worker. All numerical work happens in
 * the worker; nothing here does maths beyond formatting.
 */

import { guessRoles, profileColumns, readWorkbook, validateRoles } from "./dataio.js";
import { renderCandidates, renderDetail } from "./report.js";

const $ = (id) => document.getElementById(id);

const EXAMPLE_URL = "example/jaxsr-example.xlsx";

const state = {
  workbook: null,
  sheet: null,
  columns: [],
  rows: [],
  profiles: [],
  roles: [],
  dataLoaded: false,
  fitResult: null,
  ready: false,
};

// ---------------------------------------------------------------------------
// Worker plumbing
// ---------------------------------------------------------------------------

const worker = new Worker("js/worker.js");
const pending = new Map();
let nextId = 0;

worker.onmessage = (event) => {
  const { type, id, ok, data, error, stage, detail } = event.data;
  if (type === "progress") {
    setStatus(stage === "ready" ? "ready" : "busy", detail);
    return;
  }
  const entry = pending.get(id);
  if (!entry) return;
  pending.delete(id);
  ok ? entry.resolve(data) : entry.reject(new Error(error));
};

worker.onerror = (event) => {
  setStatus("error", `Worker error: ${event.message}`);
};

/** Send a request to the worker and await its reply. */
function call(action, payload) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, action, payload });
  });
}

function setStatus(kind, message) {
  const node = $("status");
  node.dataset.kind = kind;
  node.textContent = message;
}

function showError(where, err) {
  const node = $(where);
  node.textContent = err.message ?? String(err);
  node.hidden = false;
}

function clearError(where) {
  const node = $(where);
  node.hidden = true;
  node.textContent = "";
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

setStatus("busy", "Starting Python…");
call("boot")
  .then((version) => {
    state.ready = true;
    setStatus("ready", `jaxsr ${version.jaxsr} · NumPy ${version.numpy}`);
    refreshRunnable();
  })
  .catch((err) => setStatus("error", `Could not start Python: ${err.message}`));

// ---------------------------------------------------------------------------
// 1. Data
// ---------------------------------------------------------------------------

const drop = $("drop-zone");
const fileInput = $("file-input");

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("dragging");
});
drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("dragging");
  if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) loadFile(fileInput.files[0]);
});

$("load-example").addEventListener("click", loadExample);

async function loadFile(file) {
  clearError("data-error");
  try {
    state.workbook = await readWorkbook(file);
    $("file-name").textContent = file.name;
    const picker = $("sheet-picker");
    picker.replaceChildren();
    for (const name of state.workbook.sheets) {
      picker.append(new Option(name, name));
    }
    $("sheet-row").hidden = state.workbook.sheets.length < 2;
    selectSheet(state.workbook.sheets[0]);
  } catch (err) {
    showError("data-error", err);
  }
}

$("sheet-picker").addEventListener("change", (e) => selectSheet(e.target.value));

function selectSheet(name) {
  state.sheet = name;
  const { columns, rows } = state.workbook.read(name);
  state.columns = columns;
  state.rows = rows;
  state.profiles = profileColumns(columns, rows);
  state.roles = guessRoles(state.profiles);
  renderRoleTable();
  renderPreview();
  $("data-config").hidden = false;
}

function renderRoleTable() {
  const host = $("role-table");
  host.replaceChildren();

  const header = document.createElement("tr");
  for (const label of ["Column", "Role", "First values", "Missing"]) {
    const th = document.createElement("th");
    th.textContent = label;
    header.append(th);
  }
  host.append(header);

  state.profiles.forEach((profile, i) => {
    const tr = document.createElement("tr");

    const name = document.createElement("td");
    name.className = "expr";
    name.textContent = profile.name;
    if (!profile.numeric) {
      const tag = document.createElement("span");
      tag.className = "badge muted-badge";
      tag.textContent = "text";
      tag.title = "Not numeric, so it cannot be a feature or target yet.";
      name.append(tag);
    }
    tr.append(name);

    const roleCell = document.createElement("td");
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Role for ${profile.name}`);
    for (const role of ["feature", "target", "ignore"]) {
      const option = new Option(role[0].toUpperCase() + role.slice(1), role);
      option.selected = state.roles[i] === role;
      select.append(option);
    }
    select.addEventListener("change", () => {
      // Only one target makes sense; adopting a new one releases the old.
      if (select.value === "target") {
        state.roles = state.roles.map((r, j) => (j !== i && r === "target" ? "feature" : r));
      }
      state.roles[i] = select.value;
      renderRoleTable();
      pushData();
    });
    roleCell.append(select);
    tr.append(roleCell);

    const preview = document.createElement("td");
    preview.className = "muted";
    preview.textContent = profile.preview;
    tr.append(preview);

    const missing = document.createElement("td");
    missing.className = "num";
    missing.textContent = profile.missing ? String(profile.missing) : "";
    tr.append(missing);

    host.append(tr);
  });

  pushData();
}

function renderPreview() {
  const host = $("data-preview");
  host.replaceChildren();
  const header = document.createElement("tr");
  for (const name of state.columns) {
    const th = document.createElement("th");
    th.textContent = name;
    header.append(th);
  }
  host.append(header);
  for (const row of state.rows.slice(0, 8)) {
    const tr = document.createElement("tr");
    for (const cell of row) {
      const td = document.createElement("td");
      td.textContent = cell === null ? "" : String(cell);
      tr.append(td);
    }
    host.append(tr);
  }
  $("preview-caption").textContent = `${state.rows.length} rows × ${state.columns.length} columns`;
}

let dataToken = 0;
async function pushData() {
  const problem = validateRoles(state.roles, state.profiles);
  const summary = $("data-summary");
  state.dataLoaded = false;
  refreshRunnable();

  if (problem) {
    summary.className = "notice warning";
    summary.textContent = problem;
    summary.hidden = false;
    return;
  }
  if (!state.ready) {
    summary.className = "notice";
    summary.textContent = "Waiting for Python to finish loading…";
    summary.hidden = false;
    return;
  }

  const token = ++dataToken;
  try {
    const info = await call("setData", {
      columns: state.columns,
      rows: state.rows,
      roles: state.roles,
    });
    if (token !== dataToken) return;
    state.dataLoaded = true;
    clearError("data-error");
    summary.className = "notice ok";
    const notes = info.warnings.length ? ` ${info.warnings.join(" ")}` : "";
    summary.textContent =
      `${info.n_samples} rows · ${info.n_features} feature(s): ${info.feature_names.join(", ")}` +
      ` · target: ${info.target_name}.${notes}`;
    summary.hidden = false;
    $("model-section").hidden = false;
    await refreshLibrary();
  } catch (err) {
    if (token !== dataToken) return;
    summary.hidden = true;
    showError("data-error", err);
  }
  refreshRunnable();
}

/**
 * Load the bundled example workbook through exactly the same path as an
 * uploaded file, so what the user sees is what their own spreadsheet will do.
 * The workbook also carries its own instructions, which is why it is a real
 * file rather than data synthesised here.
 */
async function loadExample() {
  clearError("data-error");
  try {
    const response = await fetch(EXAMPLE_URL);
    if (!response.ok) throw new Error(`Could not fetch the example (${response.status}).`);
    const blob = await response.blob();
    await loadFile(new File([blob], "jaxsr-example.xlsx"));
    // Skip past the instructions sheet to the data.
    if (state.workbook.sheets.includes("reactor")) {
      $("sheet-picker").value = "reactor";
      selectSheet("reactor");
    }
  } catch (err) {
    showError("data-error", err);
  }
}

// ---------------------------------------------------------------------------
// 2. Model configuration
// ---------------------------------------------------------------------------

const CONFIG_INPUTS = [
  "basis-constant",
  "basis-linear",
  "basis-poly",
  "poly-degree",
  "basis-inter",
  "inter-order",
  "basis-trans",
  "basis-ratios",
  "basis-comp",
  "basis-power",
  "max-terms",
];

for (const id of CONFIG_INPUTS) {
  $(id).addEventListener("change", refreshLibrary);
}
for (const box of document.querySelectorAll("input[name=trans-func]")) {
  box.addEventListener("change", refreshLibrary);
}
$("strategy").addEventListener("change", refreshRunnable);

function readConfig() {
  return {
    constant: $("basis-constant").checked,
    linear: $("basis-linear").checked,
    polynomials: { enabled: $("basis-poly").checked, max_degree: Number($("poly-degree").value) },
    interactions: { enabled: $("basis-inter").checked, max_order: Number($("inter-order").value) },
    transcendental: {
      enabled: $("basis-trans").checked,
      funcs: [...document.querySelectorAll("input[name=trans-func]:checked")].map((b) => b.value),
    },
    ratios: { enabled: $("basis-ratios").checked },
    compositions: { enabled: $("basis-comp").checked },
    power_laws: { enabled: $("basis-power").checked },
  };
}

let libraryToken = 0;
async function refreshLibrary() {
  if (!state.dataLoaded || !state.ready) return;
  const token = ++libraryToken;
  clearError("model-error");
  try {
    const info = await call("buildLibrary", {
      config: readConfig(),
      max_terms: Number($("max-terms").value),
    });
    if (token !== libraryToken) return;
    $("term-count").textContent = String(info.n_terms);
    $("term-names").textContent = info.names.join(" · ");

    const option = $("strategy").querySelector("option[value=exhaustive]");
    const { combinations, feasible } = info.exhaustive;
    option.disabled = !feasible;
    option.textContent = feasible
      ? `exhaustive (${combinations.toLocaleString()} subsets)`
      : `exhaustive — too large (${combinations.toLocaleString()} subsets)`;
    if (!feasible && $("strategy").value === "exhaustive") $("strategy").value = "greedy_forward";
  } catch (err) {
    if (token !== libraryToken) return;
    showError("model-error", err);
  }
  refreshRunnable();
}

function refreshRunnable() {
  $("fit-button").disabled = !(state.ready && state.dataLoaded);
}

// ---------------------------------------------------------------------------
// 3. Fit
// ---------------------------------------------------------------------------

$("fit-button").addEventListener("click", runFit);

async function runFit() {
  clearError("model-error");
  const button = $("fit-button");
  button.disabled = true;
  button.textContent = "Fitting…";
  setStatus("busy", "Fitting…");
  const started = performance.now();

  try {
    const result = await call("fit", {
      max_terms: Number($("max-terms").value),
      strategy: $("strategy").value,
      information_criterion: $("criterion").value,
      regularization: $("ridge").value ? Number($("ridge").value) : null,
    });
    state.fitResult = result;
    const elapsed = Math.round(performance.now() - started);
    $("results-section").hidden = false;
    $("fit-timing").textContent = `Fitted in ${elapsed} ms.`;
    renderCandidates($("candidates"), result, showDetail);
    $("export-section").hidden = false;
    setStatus("ready", `Fitted in ${elapsed} ms`);
    $("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError("model-error", err);
    setStatus("ready", "Fit failed");
  } finally {
    button.disabled = false;
    button.textContent = "Fit";
    refreshRunnable();
  }
}

async function showDetail(row) {
  const host = $("detail");
  host.replaceChildren();
  host.append(Object.assign(document.createElement("p"), { className: "muted", textContent: "Computing…" }));
  try {
    const diag = await call("diagnostics", {
      indices: row.indices,
      alpha: Number($("alpha").value),
    });
    renderDetail(host, row, diag);
  } catch (err) {
    host.replaceChildren();
    showError("model-error", err);
  }
}

$("alpha").addEventListener("change", () => {
  const selected = document.querySelector("#candidates tr.is-selected");
  if (selected) selected.click();
});

$("cv-button").addEventListener("click", async () => {
  const out = $("cv-result");
  out.textContent = "Running…";
  try {
    const result = await call("crossValidate", { folds: Number($("cv-folds").value) });
    // cross_validate reports negative MSE, so higher is better.
    out.textContent =
      `${result.folds}-fold CV — mean test score ${result.mean_test_score.toPrecision(4)} ` +
      `± ${result.std_test_score.toPrecision(3)} (negative MSE; closer to zero is better).`;
  } catch (err) {
    out.textContent = err.message;
  }
});

// ---------------------------------------------------------------------------
// 4. Export
// ---------------------------------------------------------------------------

for (const button of document.querySelectorAll("button[data-export]")) {
  button.addEventListener("click", async () => {
    const kind = button.dataset.export;
    const original = button.textContent;
    button.disabled = true;
    button.textContent = "Preparing…";
    try {
      const result = await call("export", { kind });
      download(result.filename, result.content);
      $("export-preview").textContent = result.content.slice(0, 4000);
      $("export-preview").hidden = false;
    } catch (err) {
      showError("model-error", err);
    } finally {
      button.disabled = false;
      button.textContent = original;
    }
  });
}

function download(filename, content) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
