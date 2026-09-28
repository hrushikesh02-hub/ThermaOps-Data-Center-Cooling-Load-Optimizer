# Project cleanup record

The complete original project was inventoried before the clean folder was built.
The source contained 15,044 files and about 1.01 GB, most of it from the local
virtual environment, caches, duplicate models, reports and runtime backups. The
original folder remains unchanged as a recovery copy.

## What moved

- `dashboard/app.py` and `dashboard/service.py` → `api/`
- dashboard templates and static assets → `frontend/templates/` and `frontend/static/`
- `data/train.csv`, `validation.csv`, `test.csv` → `datasets/processed/`
- the one complete original dataset → `datasets/raw/cooling_observations.csv`
- the optimization reference → `datasets/raw/optimization_reference.csv`
- preprocessing and data validation → `ml/preprocessing/`
- active model lineage plus four selected model files → `ml/models/artifacts/`
- model training/prediction code → `ml/models/`
- GA, DE, constraints and operating-region code → `ml/optimization/`
- evaluation, exploration and ablation code → `ml/evaluation/`
- five test files → `tests/`
- the live operator SQLite database → `outputs/runtime/operator.sqlite3`

Exact per-file source/destination mappings are in `moves.json`. The machine-
readable `cleanup_manifest.json` records each inspected non-environment source
file, its size, disposition and reason.

## What was excluded from the clean folder

- `.venv/`, `__pycache__/` and `.pyc`: machine-specific or generated files.
- `runtime/code_backups/` and logs: temporary rollback copies and server output.
- `results/` including `results/legacy/`: generated tables, plots and archived reports.
- 21 older top-level model artifacts and 16 unselected seasonal artifacts:
  superseded or duplicate fitted models. Their readable estimator settings are
  preserved in `ml/models/training_config.json`; the training command evaluates
  all five families and saves only four selected pipelines.
- `data/original/train.csv`, `validation.csv`, `test.csv`: duplicates of the one
  retained raw dataset; equality was verified before exclusion.
- `data/research_ready_dataset.csv`: duplicate full processed data; validation
  reconstructs the two corrected derived margins from the retained raw source.
- blocked legacy training commands (`run_training.py`, `run_power_training.py`),
  separate power-only training/evaluation, and `verify_research.py`: replaced by
  the working `main.py` commands and organized ML modules.
- old Markdown/PDF reports, `create_pdf.py`, `FILE_MANIFEST.json`, and fragmented
  optimization Markdown: historical or duplicate documentation consolidated into
  `README.md`, `MODELING.md` and this record.
- `start_dashboard.cmd`: replaced by the portable `python main.py` command.

No measured rows, targets, dashboard features, API operations, prediction models,
recommendation constraints, tests, saved limits, run history or simulated apply
events were removed from the delivered local application.

## Import and path updates

- `dashboard.*` → `api.*`
- `src.preprocessing` → `ml.preprocessing.pipeline`
- `src.data_integrity` → `ml.preprocessing.validate`
- `src.model_profile` → `ml.models.profile`
- model loading → `ml.models.predict`
- `optimization.*` → `ml.optimization.*`
- project configuration → `ml.config`
- `data/` → `datasets/raw/` or `datasets/processed/`
- `models/` → `ml/models/artifacts/`
- generated `results/` and runtime state → ignored `outputs/`
- Flask template/static roots → `frontend/templates/` and `frontend/static/`

All paths resolve from the project root and are independent of the current shell
directory. The web routes and frontend request paths remain unchanged.

## Verification

- Data validation passed for 10,000 records, 34 model inputs and five sites.
- All 40 regression tests passed after reorganization.
- All 17 documented API operations and static assets responded as expected.
- 1,500 snapshot requests produced 6,000 model outputs identical to the original
  active models (maximum absolute difference: 0 for every target).
- A full 20-candidate retraining run selected the same four algorithms and
  reproduced all 6,000 predictions within `1e-10` (observed maximum: 0).
- The model revision remains `7459ee4a6088d564`.
- Coverage remains 1,444/1,500 baseline supported and 1,448/1,500 reachable.
- Saved operator state was copied consistently: five policies, 15 recommendation
  runs and five events at verification time.
- The complete research command passed a reduced-budget smoke run, including data
  exploration, model evaluation, GA/DE comparison, ablation and robustness.
- JavaScript syntax validation passed.

Detailed evidence is written to ignored local file
`outputs/verification/cleanup_verification.json`.
