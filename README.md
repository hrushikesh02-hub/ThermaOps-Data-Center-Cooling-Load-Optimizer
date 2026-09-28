# ThermaOps — Data Center Cooling Optimizer

ThermaOps helps operators review past data-center conditions, estimate cooling
power and temperatures, and find cooling settings within saved operating limits.
It addresses the trade-off between cooling power use and temperature limits.

This is a local research dashboard and simulation. It does not control equipment.

## Features

- Replay 1,500 records from five data centers, with power trends and temperature checks.
- Predict cooling power, cooling load, server inlet temperature and rack hot spots.
- Search with Genetic Algorithm (GA) or Differential Evolution (DE).
- Set temperature limits, a power cap, extra temperature allowance and control ranges.
- Preview manual settings, apply recommendations to simulation, and export saved history.
- Check data quality, model accuracy and whether conditions are covered by training data.
- Use the same functionality through 17 documented API operations.

## Project structure

```text
ThermaOps/
├── datasets/
│   ├── raw/                 # Original observations and reference data
│   ├── processed/           # Required train, validation and test CSVs
│   └── correction_log.json
├── ml/
│   ├── preprocessing/       # Data checks and preprocessing pipelines
│   ├── models/              # Training, prediction, profile and training settings
│   │   └── artifacts/       # Four active model files; excluded from Git
│   ├── optimization/        # Shared GA/DE search and operating-region checks
│   ├── evaluation/          # Accuracy, plots and optional research comparisons
│   ├── config.py            # Features, project paths and tuning settings
│   └── utils.py
├── api/                     # Flask routes and application services
├── frontend/                # HTML templates, JavaScript, CSS and API reference
├── tests/                   # Model, training, API and recommendation regressions
├── outputs/                 # Generated reports and local operator database; ignored
├── docs/                    # Modeling notes and cleanup record
├── .gitattributes
├── .gitignore
├── requirements.txt
└── main.py                  # Main entry point
```

## Install

Validated on Windows with **Python 3.14.3**. Use the exact versions in
`requirements.txt`; saved ML models depend on matching library versions.

Open PowerShell in this folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

The frontend uses plain HTML, CSS and JavaScript served by Flask. It has no npm
dependencies, separate build step, CDN, API key or external service.

## Run the dashboard and API

The delivered local folder includes the four trained models and processed data:

```powershell
.venv\Scripts\python main.py
```

Open **http://127.0.0.1:8501**. API documentation is at **/api/docs**.
Use `main.py --port 8502` if the default port is occupied.
Use `main.py --state-dir PATH` to store operator history elsewhere.
The default database is `outputs/runtime/operator.sqlite3`.

After a **Git clone**, train the ignored model bundle once before starting:

```powershell
.venv\Scripts\python main.py prepare
.venv\Scripts\python main.py train --output-dir ml/models/artifacts
.venv\Scripts\python main.py
```

The training output directory must not already exist. The delivered local folder
already has this bundle, so use the candidate-training command below for retraining.

## Run the ML pipeline

```powershell
# Validate data, split separation and excluded model inputs
.venv\Scripts\python main.py prepare

# Reproduce training in a NEW candidate directory without changing active models
.venv\Scripts\python main.py train --output-dir outputs/candidate_models

# Evaluate the currently active models and create accuracy/importance plots
.venv\Scripts\python main.py evaluate

# Run API, prediction, training and recommendation tests
.venv\Scripts\python main.py test
```

Training compares five families for each of four targets and selects the lowest
validation mean absolute error (MAE). Only the four selected pipelines are saved.
`ml/models/training_config.json` preserves the original candidate settings, so
retraining needs no obsolete model files. Add `--tune` for training-only
time-series cross-validation; this may take substantially longer and can change
the selected models.

Candidate bundles include `training_report.json` with validation and held-out
test accuracy. After reviewing a candidate, stop the dashboard, back up the
active `ml/models/artifacts/` outside the repository, replace that entire folder
with the candidate bundle, then restart. Deploy its two JSON manifests together
with its four models. Model version checks prevent applying outdated saved runs.

Optional `main.py research` regenerates data exploration, accuracy plots,
GA/DE comparisons, constraint ablations and multi-seed robustness reports.
Research searches use 50 candidates and 100 generations and can take time.
All generated files go to `outputs/`.

## Algorithms and data

The active prediction models are Random Forest for cooling power, XGBoost for
cooling load and inlet temperature, and LightGBM for hot spot temperature.
Dummy mean prediction and linear regression are retained as training benchmarks;
their fitted artifacts are not shipped. Preprocessing uses median/mode
imputation, one-hot encoding and scaling where configured.

GA and DE share the same constraints and final checks. A training-fitted,
regularized Mahalanobis-distance check identifies unsupported operating conditions.
See [modeling notes](docs/MODELING.md) for limits and interpretation.

The supplied research data contains 10,000 observations. Source instrumentation
and the data-generation process were not supplied. Raw observations and the
optimization reference remain in `datasets/raw/`; reference recommendations are
never prediction inputs. Three processed CSVs retain the original 7,000/1,500/1,500
chronological partitions. The active seasonal profile uses 7,000 training rows
plus the first 1,000 validation rows, leaving **8,000 training / 500 validation /
1,500 held-out test** records. Test records are not used to fit or select models.

Only two derived margin columns were corrected: 27°C minus inlet temperature
and 35°C minus hot spot temperature. Measured values are unchanged. The full
corrected dataset is reconstructed in memory during checks instead of storing
another duplicate CSV. Small, required CSVs are versioned; `.gitattributes`
preserves their exact bytes for model fingerprint checks.

## GitHub preparation

- Source, tests, documentation, dependencies, training settings and required data are included.
- Trained model weights, operator history, logs, caches, environments and generated reports are ignored.
- No credentials or API keys are needed by this application.
- A fresh clone can train its own four-model bundle using the command above.
- Review `git status --short` before committing. Do not force-add ignored runtime files.
- Confirm redistribution rights for the supplied research data before making a repository public; no dataset license was supplied.

See [cleanup details](docs/CLEANUP.md) for moved and excluded files and verification results.
