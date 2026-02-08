# Claude Code Skills

JAXSR includes a set of [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill
files that enable an AI assistant to guide you through symbolic regression workflows
interactively. Instead of reading through documentation to find the right API calls, you can
describe your problem in natural language and get tailored recommendations for basis libraries,
selection strategies, uncertainty quantification, and more.

## What Are Claude Code Skills?

Claude Code is Anthropic's CLI tool for AI-assisted software development. **Skills** are
structured knowledge files that give the assistant domain-specific expertise. When you use
Claude Code in a project that contains skill files, the assistant automatically loads them
and can provide context-aware guidance.

JAXSR's skill files teach the assistant:

- How to recommend basis libraries based on your problem domain
- Which selection strategy and information criterion to choose
- When and how to apply physical constraints
- Which UQ method fits your needs
- How to set up DOE studies and active learning loops
- How to use the CLI for code-free workflows

## File Layout

The skill files live in two mirrored locations:

```
.claude/skills/jaxsr/          # Primary location (used by Claude Code)
├── SKILL.md                   # Main skill definition
├── guides/                    # Detailed topic guides
│   ├── active-learning.md
│   ├── basis-library.md
│   ├── cli.md
│   ├── constraints.md
│   ├── doe-workflow.md
│   ├── known-model-fitting.md
│   ├── model-fitting.md
│   ├── rsm.md
│   └── uncertainty.md
└── templates/                 # Ready-to-run starter scripts
    ├── active-learning-loop.py
    ├── basic-regression.py
    ├── constrained-model.py
    ├── doe-study.py
    ├── langmuir-isotherm.py
    ├── notebook-starter.py
    └── uncertainty-analysis.py

src/jaxsr/skill/               # Mirror (included in the package)
└── ...                        # Same contents as above
```

These two directories must stay in sync. After editing any file under `.claude/skills/jaxsr/`,
run:

```bash
rm -rf src/jaxsr/skill && cp -r .claude/skills/jaxsr src/jaxsr/skill
```

## Skill Components

### SKILL.md — Main Skill Definition

The entry point for the assistant. It contains:

- **Activation triggers** — when the skill should engage (e.g., "discover algebraic
  expressions", "set up a DOE study", "choose between UQ methods")
- **Assistant mode** — a step-by-step diagnostic flow that walks users through problem
  characterization, basis library selection, strategy choice, constraint setup, UQ method
  selection, and reporting format
- **Decision tables** — quick-lookup tables mapping scenarios to recommended configurations
- **Quick-reference API and CLI** — concise code snippets for common workflows
- **Common pitfalls** — mistakes the assistant should warn users about

### Guides

Each guide covers one topic in depth with code examples, comparison tables, and
decision flowcharts:

| Guide | Topic |
|-------|-------|
| `basis-library.md` | Building and configuring basis function libraries. Covers all `add_*()` methods, domain-specific recipes (DOE, engineering, kinetics), and library sizing guidelines. |
| `model-fitting.md` | Selection strategies (`greedy_forward`, `greedy_backward`, `exhaustive`, `lasso_path`) and information criteria (`aic`, `aicc`, `bic`). Includes a strategy comparison table and criterion selection flowchart. |
| `uncertainty.md` | UQ methods: OLS prediction intervals, coefficient intervals, Bayesian Model Averaging, conformal prediction, bootstrap, ensemble predictions, and ANOVA. |
| `constraints.md` | Physical constraints: bounds, monotonicity, convexity/concavity, sign constraints, and known coefficients. Hard vs. soft enforcement. |
| `doe-workflow.md` | Full DOE lifecycle: study creation, experimental design (Latin hypercube, factorial, CCD, Box-Behnken), data collection, model fitting, adaptive sampling, and reporting. |
| `active-learning.md` | Acquisition functions (uncertainty, error, leverage, gradient) and the adaptive sampling loop. |
| `rsm.md` | Response Surface Methodology: CCD and Box-Behnken designs, canonical analysis, ridge analysis, and optimization. |
| `known-model-fitting.md` | Fitting known model forms (Langmuir, Arrhenius, Michaelis-Menten) using parametric basis functions, including ANOVA and uncertainty analysis. |
| `cli.md` | Complete CLI reference: `jaxsr init`, `design`, `add`, `fit`, `suggest`, `report`, `status`. |

### Templates

Ready-to-run Python scripts the assistant can offer as starting points. Users can copy and
customize them:

| Template | Description |
|----------|-------------|
| `basic-regression.py` | Minimal workflow: load data, build library, fit, inspect, export. |
| `constrained-model.py` | Adds monotonicity and bound constraints to a regression model. |
| `doe-study.py` | End-to-end DOE: create study, generate design, add data, fit, suggest next experiments, report. |
| `uncertainty-analysis.py` | Runs all UQ methods (OLS, BMA, conformal, bootstrap, ensemble) and compares results. |
| `active-learning-loop.py` | Iterative loop: fit model, pick acquisition function, suggest points, collect data, repeat. |
| `langmuir-isotherm.py` | Known-model fitting for the Langmuir adsorption isotherm with parametric basis functions. |
| `notebook-starter.py` | Jupyter notebook cell structure with markdown headers and code cells for a complete analysis. |

## Using the Skills

### With Claude Code CLI

If you have [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed, simply
open a terminal in the JAXSR project directory and start a session:

```bash
claude
```

The assistant will automatically detect the skill files. You can then ask questions like:

- *"I have temperature and pressure data for a catalyst. Help me find an equation."*
- *"Which UQ method should I use for 30 data points?"*
- *"Set up a DOE study for three continuous factors."*
- *"I know the model is Langmuir — help me estimate the parameters."*

The assistant will walk you through the decision process step by step, recommend
configurations, and generate ready-to-run code.

### Without Claude Code

The skill files are still useful as standalone documentation. You can read them directly:

- Browse the guides in `.claude/skills/jaxsr/guides/` for in-depth topic coverage
- Copy templates from `.claude/skills/jaxsr/templates/` as starting points for your scripts
- Reference `SKILL.md` for quick-lookup decision tables and API cheat sheets

## Contributing to the Skills

When adding or modifying skill files:

1. Edit files in `.claude/skills/jaxsr/` (the primary location)
2. Sync to the package mirror:
   ```bash
   rm -rf src/jaxsr/skill && cp -r .claude/skills/jaxsr src/jaxsr/skill
   ```
3. Verify code examples against the actual JAXSR API (see the
   [Red-Team Review checklist](https://github.com/jkitchin/jaxsr/blob/main/CLAUDE.md)
   in CLAUDE.md)
4. Run linting:
   ```bash
   black src/ tests/
   ruff check --fix src/ tests/
   ```
