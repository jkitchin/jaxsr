/*
 * Spreadsheet parsing and the column-role table.
 *
 * Parsing runs in JavaScript rather than Python for two reasons: it works the
 * instant the file is dropped, so the user can assign column roles while
 * Pyodide is still booting, and it keeps pandas and openpyxl out of the
 * download entirely.
 */

export const ROLES = ["feature", "target", "ignore"];

/**
 * Read a spreadsheet or CSV file.
 *
 * @param {File} file
 * @returns {Promise<{sheets: string[], read: (name: string) => object}>}
 */
export async function readWorkbook(file) {
  const buffer = await file.arrayBuffer();
  const workbook = XLSX.read(buffer, { type: "array", cellDates: false });
  if (!workbook.SheetNames.length) throw new Error("That file has no sheets.");
  return {
    sheets: workbook.SheetNames,
    read: (name) => readSheet(workbook.Sheets[name ?? workbook.SheetNames[0]]),
  };
}

/**
 * Turn a worksheet into headers plus rows, dropping fully blank rows.
 *
 * @param {object} sheet
 * @returns {{columns: string[], rows: any[][]}}
 */
function readSheet(sheet) {
  const grid = XLSX.utils.sheet_to_json(sheet, { header: 1, blankrows: false, defval: null });
  if (!grid.length) return { columns: [], rows: [] };

  const header = grid[0].map((cell, i) =>
    cell === null || cell === "" ? `column_${i + 1}` : String(cell).trim()
  );
  const columns = dedupe(header);
  const width = columns.length;
  const rows = grid
    .slice(1)
    .map((row) => Array.from({ length: width }, (_, i) => (i < row.length ? row[i] : null)))
    .filter((row) => row.some((cell) => cell !== null && cell !== ""));

  return { columns, rows };
}

/** Make column names unique so role lookups are unambiguous. */
function dedupe(names) {
  const seen = new Map();
  return names.map((name) => {
    const count = seen.get(name) ?? 0;
    seen.set(name, count + 1);
    return count === 0 ? name : `${name}_${count + 1}`;
  });
}

/**
 * Describe each column: is it numeric, and what does it look like?
 *
 * @param {string[]} columns
 * @param {any[][]} rows
 * @returns {{name: string, numeric: boolean, numericFraction: number, preview: string}[]}
 */
export function profileColumns(columns, rows) {
  return columns.map((name, i) => {
    const values = rows.map((row) => row[i]);
    const present = values.filter((v) => v !== null && v !== "");
    const numeric = present.filter((v) => Number.isFinite(toNumber(v)));
    const fraction = present.length ? numeric.length / present.length : 0;
    return {
      name,
      numeric: fraction >= 0.9 && present.length > 0,
      numericFraction: fraction,
      preview: present.slice(0, 3).map(formatCell).join(", ") || "(empty)",
      missing: values.length - present.length,
    };
  });
}

/** Coerce a cell to a number, or NaN. */
function toNumber(value) {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "string") {
    const trimmed = value.trim().replace(/,/g, "");
    if (trimmed === "") return NaN;
    return Number(trimmed);
  }
  return NaN;
}

function formatCell(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toPrecision(4).replace(/\.?0+$/, "");
  }
  return String(value);
}

/**
 * Guess roles: the last numeric column is the target, other numeric columns
 * are features, and anything non-numeric is ignored.
 *
 * @param {{numeric: boolean}[]} profiles
 * @returns {string[]}
 */
export function guessRoles(profiles) {
  const numericIdx = profiles.map((p, i) => (p.numeric ? i : -1)).filter((i) => i >= 0);
  if (numericIdx.length === 0) return profiles.map(() => "ignore");
  const target = numericIdx[numericIdx.length - 1];
  return profiles.map((p, i) => {
    if (i === target) return "target";
    return p.numeric ? "feature" : "ignore";
  });
}

/**
 * Explain why the current role assignment cannot be fitted, if it cannot.
 *
 * @param {string[]} roles
 * @param {{name: string, numeric: boolean}[]} profiles
 * @returns {string|null}
 */
export function validateRoles(roles, profiles) {
  const targets = roles.filter((r) => r === "target").length;
  const features = roles.filter((r) => r === "feature").length;
  if (targets === 0) return "Choose a target column — the quantity you want to predict.";
  if (targets > 1) return `Choose exactly one target column (${targets} are selected).`;
  if (features === 0) return "Choose at least one feature column.";

  const nonNumeric = roles
    .map((role, i) => (role !== "ignore" && !profiles[i].numeric ? profiles[i].name : null))
    .filter(Boolean);
  if (nonNumeric.length) {
    return `These columns are not numeric and cannot be used yet: ${nonNumeric.join(", ")}. Set them to Ignore.`;
  }
  return null;
}
