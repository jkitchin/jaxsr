/*
 * Pyodide worker for the JAXSR app.
 *
 * Everything Python happens here, off the main thread, so a long fit never
 * freezes the page. The protocol is deliberately dull: the page posts
 * {id, action, payload} and gets back {id, ok, data} or {id, ok:false, error}.
 * Progress during the (slow, one-time) boot is pushed as {type:"progress"}.
 *
 * The Python side returns JSON strings, so no PyProxy objects ever cross this
 * boundary and there is nothing to destroy.
 */

let pyodide = null;
let manifest = null;
let sympyLoaded = false;

// Relative fetches inside a worker resolve against the worker script, not the
// page, so build an explicit app root by stepping out of js/. This keeps the
// app working under any deploy prefix (locally at /, published at /jaxsr/app/).
const ROOT = new URL("../", self.location.href).href;

function progress(stage, detail) {
  self.postMessage({ type: "progress", stage, detail });
}

/** Boot Pyodide, install jaxsr, and load the shim + kernel. */
async function boot() {
  progress("manifest", "Reading build manifest");
  manifest = await (await fetch(`${ROOT}manifest.json`)).json();

  progress("pyodide", `Loading Pyodide ${manifest.pyodideVersion}`);
  importScripts(`${manifest.pyodideIndexURL}pyodide.js`);
  pyodide = await self.loadPyodide({ indexURL: manifest.pyodideIndexURL });

  progress("packages", "Loading NumPy and SciPy");
  await pyodide.loadPackage(manifest.corePackages);

  progress("jaxsr", `Installing jaxsr ${manifest.jaxsrVersion}`);
  // deps=False is essential: jaxsr declares jax, jaxlib, jupyter, nbconvert and
  // matplotlib as runtime dependencies. jaxlib has no WebAssembly build at all,
  // and the rest are unused here. Driven from Python rather than through
  // micropip's JS proxy, where `deps` would land on the `keep_going` positional.
  pyodide.globals.set("_wheel_url", new URL(manifest.wheel, ROOT).href);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(_wheel_url, deps=False)
`);

  progress("shim", "Installing the NumPy backend");
  for (const path of manifest.pythonModules) {
    const source = await (await fetch(`${ROOT}${path}`)).text();
    pyodide.FS.writeFile(`/home/pyodide/${path.split("/").pop()}`, source);
  }
  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/home/pyodide")
import jax_shim
jax_shim.install(force=True)
import kernel
`);

  progress("ready", "Ready");
  const version = JSON.parse(pyodide.runPython("kernel.get_version('{}')"));
  return version.data;
}

/**
 * sympy is several MB and only LaTeX export needs it, so it is fetched the
 * first time the user actually asks for an equation.
 */
async function ensureSympy() {
  if (sympyLoaded) return;
  progress("sympy", "Loading SymPy for equation rendering");
  await pyodide.loadPackage("sympy");
  sympyLoaded = true;
  progress("ready", "Ready");
}

/** Call a kernel function by name with a JSON-serialisable payload. */
function callKernel(name, payload) {
  pyodide.globals.set("_payload", JSON.stringify(payload ?? {}));
  const raw = pyodide.runPython(`kernel.${name}(_payload)`);
  return JSON.parse(raw);
}

const ACTIONS = {
  boot,
  setData: (p) => callKernel("set_data", p),
  buildLibrary: (p) => callKernel("build_library", p),
  fit: (p) => callKernel("fit", p),
  diagnostics: (p) => callKernel("diagnostics", p),
  crossValidate: (p) => callKernel("run_cross_validation", p),
  predictionBand: (p) => callKernel("prediction_band", p),
  export: async (p) => {
    if (p.kind === "latex") await ensureSympy();
    return callKernel("export_model", p);
  },
};

self.onmessage = async (event) => {
  const { id, action, payload } = event.data;
  const handler = ACTIONS[action];
  if (!handler) {
    self.postMessage({ id, ok: false, error: `Unknown action: ${action}` });
    return;
  }
  try {
    const result = await handler(payload);
    // Kernel handlers already answer in {ok, data} form; boot returns raw data.
    if (result && typeof result === "object" && "ok" in result) {
      self.postMessage({ id, ...result });
    } else {
      self.postMessage({ id, ok: true, data: result });
    }
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message ? err.message : err) });
  }
};
