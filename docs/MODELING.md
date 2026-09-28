# Model and optimization behavior

All four models use the same 34 independent state/control inputs defined in
`ml/config.py`. Outcomes, optimization reference outputs and target-derived
columns are excluded. Each saved model contains its fitted preprocessing pipeline.

## Data and model lineage

The seasonal profile fits 8,000 chronological records, selects/calibrates on the
next 500, and evaluates on the final 1,500. The processed CSVs keep their original
7,000/1,500/1,500 partitions; `training_validation_prefix_rows: 1000` defines the
active training boundary. The original raw source is retained separately.

`ml/models/artifacts/model_profile.json` verifies SHA-256 fingerprints for all
three processed CSVs and four selected models. `selected_models.json` must agree.
The same profile drives prediction, calibration, operating-region fitting and
dashboard metrics. Move or copy a complete bundle; do not mix versions.

The cleaned delivery preserves the active bundle byte for byte and its model
revision. Moving directories therefore does not invalidate existing saved runs.
A later model change does invalidate them, as before.

## Recommendations

Decision order: cooling setpoint, supply-air temperature, fan speed, pump speed.
Temperature settings stay within 18–27°C; fan/pump stay within 30–100%. Operators
can tighten those bounds. Both GA and DE minimize predicted cooling power.

Every recommendation must meet:

- Predicted inlet temperature + validation allowance + operator reserve <= saved inlet limit (maximum 27°C).
- Predicted hot spot temperature + validation allowance + reserve <= saved hot spot limit (maximum 35°C).
- Predicted cooling power <= optional saved power cap.
- Physical operating-region and saved control-range checks.
- The same checks after controls are rounded to 0.1 units.

Allowances use the one-sided 95th-percentile validation residual. They are
empirical checks, not guaranteed future confidence bounds or equipment safety.
The optional research ablations use point predictions to isolate constraints.

Operating-region checks scale 17 independent physical features and fit a
Ledoit-Wolf covariance model to active training data. The cutoff remains its
99th-percentile training distance. Calendar fields do not determine this check.
The exact closest-control calculation checks all 81 bound/free combinations
before deciding that a fixed state cannot reach the supported region.

The unchanged active model supports 1,444 of 1,500 recorded baselines; 1,448 can
reach the supported region within default control bounds. Reachability alone
does not establish temperature/power feasibility. Searches retain candidates
across generations and seed the original and nearest supported settings.

Operator searches use 40 candidates, 35 generations and a default seed of 42.
Research defaults are 50 candidates and 100 generations. Infeasible runs retain
diagnostics but withhold recommendations and savings and cannot be applied.

## Interpretation

The dashboard compares estimated instantaneous power in kW. It does not measure
energy saved in kWh. Observed airflow and other state variables remain fixed
during search. These models capture associations; changing fan speed does not
simulate a physical airflow response. Results need independent equipment
validation before automatic control. Apply only records simulated settings.
