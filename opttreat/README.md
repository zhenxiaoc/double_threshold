# OptTreat Core Package

OptTreat is a simulation toolkit for optimal treatment parameter estimation.
The reusable workflow is:

```text
model.generate_data(n) -> split_treated_control(...) -> estimator.fit(...)
-> parameter.evaluate(...) -> optional variance.fit(...) -> summary/draw tables
```

## Module Map

| Path | Purpose |
| --- | --- |
| `config.py` | Dataclasses for estimator, parameter, and variance configuration. |
| `data.py` | Parses DataFrames/dicts into treated and control samples; normalizes array shapes. |
| `models/` | DGP families: `CCGFormulaModel` for CCG M1-M15 and `TaylorExpansionModel` for Taylor-style designs. |
| `estimation/` | First-stage estimators and their random-feature/spline basis builders. |
| `parameters/` | Welfare and value functionals for known and unknown target distributions. |
| `variance/` | Sieve-style variance estimators, including CCG SieveVar formulas. |
| `simulations/simulation_engine.py` | Shared Monte Carlo engine used by runnable simulation files. |
| `simulations/ccg2025/` | Chen, Chen, and Gao (2025) SieveVar workflow. |
| `simulations/TaylorModel/` | Random-feature simulations for Taylor expansions. |
| `simulations/high_D_tan2/` | Estimation-only high-dimensional tan2 Taylor simulations. |
| `tests/` | Pytest coverage for models, estimators, parameters, variance, and simulations. |

## Model Families

`CCGFormulaModel` contains `Model1` through `Model15` through one formula table
in `models/ccg_formula_model.py`. `get_model("Model1")` through
`get_model("Model15")` return `CCGFormulaModel` instances.

`TaylorExpansionModel` contains Taylor-style expansion designs. High-D tan2
simulations use explicit Taylor specs with `K=p=3`, `K=p=7`, and `K=p=10`.
`Model99`, `Model100`, `Model101`, and `Model102` are not active model names.

## Running Simulations

Simulation files are plain Python modules with explicit configuration variables
near the top. From the repository root:

```bash
python -m opttreat.simulations.ccg2025.run_ccg2025_sievevar
python -m opttreat.simulations.TaylorModel.run_taylor_rf
python -m opttreat.simulations.high_D_tan2.run_high_d_tan2_rf
```

The defaults are smoke-size settings. For research runs, edit the top-level
configuration variables in the corresponding file.

Simulation outputs use the shared naming convention:

```text
<simulation>_summary_n<sample-sizes>_rep<replications>_<settings>.csv
<simulation>_draws_n<sample-sizes>_rep<replications>_<settings>.csv
<simulation>_results_n<sample-sizes>_rep<replications>_<settings>.md
```

For example, `n1500_3000_6000_rep1500_nf100_K4_7_10` means sample sizes
`1500, 3000, 6000`, 1500 Monte Carlo replications per cell, 100 random
features, and `K` values `4, 7, 10`.

## Supported Components

Estimators:
- `rf_ridge`: ridge regression on random features.
- `sieve`: B-spline basis functions with `solver="ridge"` or `solver="pinv"`.

Parameters:
- `welfare_known`: `E[max(h(X), 0)]` under a known target distribution.
- `welfare_unknown`: empirical/common-support average of `max(h(X_i), 0)`.
- `value_known`: `E[1{h(X)>0}v(X)]` under a known target distribution.
- `value_unknown`: empirical average of `1{h(X_i)>0}v(X_i)`.

Variance:
- `sieve`: generic sieve-style variance.
- `ccg_sieve_var`: Chen, Chen, and Gao (2025) SieveVar formulas.

## Running Tests

```bash
python -m pytest opttreat/tests
```

## Estimator Output Contract

Every estimator exposes `fit(parsed_data)` and `get_output()`. The output dict
contains `h_hat`, fitted nuisance functions, design matrices, residuals,
feature maps, `X_all`, `alpha`, and `solver` where applicable.
