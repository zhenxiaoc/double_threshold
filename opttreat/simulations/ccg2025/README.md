# CCG 2025 SieveVar Simulations

This folder contains the OptTreat simulation workflow for the main Chen, Chen,
and Gao (2025) SieveVar Monte Carlo designs. The goal is to keep the paper
setup readable while reusing the package components:

```text
CCGFormulaModel -> simulated data -> sieve estimator -> target parameter -> CCG SieveVar -> output tables
```

The active runnable file is:

```bash
python -m opttreat.simulations.ccg2025.run_ccg2025_sievevar
```

The file is intentionally not a command-line application. To change the run,
open `run_ccg2025_sievevar.py` and edit the explicit top-level settings:
`MODEL_NAMES`, `REPLICATIONS`, `N_VALUES`, Sobol counts, spline settings, and
the output path.

## What Lives Where

| Location | Role |
| --- | --- |
| `opttreat/models/ccg_formula_model.py` | Defines the DGP formulas for Model1-Model15 through one CCG formula table. |
| `run_ccg2025_sievevar.py` | Defines the CCG paper simulation choices: model set, target type, spline dimensions, Sobol counts, value scale, and variance settings. |
| `opttreat/simulations/simulation_engine.py` | Runs the shared Monte Carlo loop: generate data, fit estimator, evaluate parameter, optionally estimate variance, and summarize draws. |
| `opttreat/estimation/` | Provides the sieve estimator used as the first-stage estimator. |
| `opttreat/parameters/` | Provides welfare and value target parameter classes. |
| `opttreat/variance/ccg_sieve_var.py` | Provides the CCG SieveVar variance estimator used for standard errors and coverage. |
| `results/` | Stores summary CSVs, draw-level CSVs, and Markdown reports. |

Models only define data-generating processes. The paper-specific tuning lives
in this simulation folder, not inside the model classes.

## Model Groups

| Models | Target | Dimension | Target support | Observed support | Paper role |
| --- | --- | ---: | --- | --- | --- |
| Model1-Model3 | Known-distribution welfare | 1 | `[0, 1]` | `[-0.2, 1.2]` | Theorem 1 |
| Model4-Model7 | Known-distribution welfare | 2 | `[0, 1]^2` | `[-0.2, 1.2]^2` | Theorem 1 |
| Model8-Model10 | Unknown/common-support welfare | 1 | observed common support | `[0, 1]` | Theorem 2 |
| Model11-Model14 | Unknown/common-support welfare | 2 | observed common support | `[0, 1]^2` | Theorem 2 |
| Model15 | Known-distribution value functional | 2 | `[-1.5, 1.5]^2` | `[-2, 2]^2` | Theorem 3 |

The main CCG paper set has 15 numbered models. The value design is represented
as `Model15` in OptTreat even though one local R script labels that theorem's
example internally as "Model 1".

## Explicit Model Formulas

Let `expit(z) = 1 / (1 + exp(-z))`. Each simulation draw uses:

```text
D | X ~ Bernoulli(p0(X))
Y = mu0(X, D) + epsilon
epsilon ~ N(0, 1)
h0(X) = mu0(X, 1) - mu0(X, 0)
```

For Model1-Model14:

```text
mu0(x, d) = mu0(x, 0) + d*h0(x)
```

For Model15, `mu0(x,0)=0`, so `mu0(x,d)=d*h0(x)`.

### One-Dimensional Models

| Model | Observed support | Target support | `p0(x)` | `mu0(x,0)` | `h0(x)` |
| --- | --- | --- | --- | --- | --- |
| Model1 | `[-0.2, 1.2]` | `[0, 1]` | `expit(1 - 2*x)` | `5*sin(2*pi*x)*cos(2*pi*x)` | `-0.4 + 2*x^2` |
| Model2 | `[-0.2, 1.2]` | `[0, 1]` | `expit(-0.5 + x)` | `0.5*abs(x)` | `0.5 - x^2` |
| Model3 | `[-0.2, 1.2]` | `[0, 1]` | `expit(0.5 - x)` | `x^2` | `1 - x` |
| Model8 | `[0, 1]` | `[0, 1]` | `expit(1 - 2*x)` | `5*sin(2*pi*x)*cos(2*pi*x)` | `-0.4 + 2*x^2` |
| Model9 | `[0, 1]` | `[0, 1]` | `expit(-0.5 + x)` | `0.5*abs(x)` | `0.5 - x^2` |
| Model10 | `[0, 1]` | `[0, 1]` | `expit(0.5 - x)` | `x^2` | `1 - x` |

Model1-Model3 and Model8-Model10 share the same formulas by pairs. The
difference is the observed covariate support: Model1-Model3 draw `X` from the
larger interval `[-0.2, 1.2]`, while Model8-Model10 draw `X` from `[0, 1]`.

### Two-Dimensional Welfare Models

For these models, write `x = (x1, x2)`.

| Model | Observed support | Target support | `p0(x)` | `mu0(x,0)` | `h0(x)` |
| --- | --- | --- | --- | --- | --- |
| Model4 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `(1 - x1^2 - x2^2) * (4 + sin(x1)*x2 + cos(x2))` | `0.5*x1 - 0.4*x2` |
| Model5 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `(1 - x1*x2) * (3 + sin(pi*x1)*cos(pi*x2))` | `0.3*x1 - 0.3*x2` |
| Model6 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(1.5*x1 - 0.5*x2)` | `log(1 + x1 + x2)` | `x1 - 0.7*x2` |
| Model7 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(-0.5 + x1 + 2*x2)` | `(x1^2 + x2^2) * exp(-x1 - x2)` | `0.5 - x2` |
| Model11 | `[0, 1]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `(1 - x1^2 - x2^2) * (4 + sin(x1)*x2 + cos(x2))` | `0.5*x1 - 0.4*x2` |
| Model12 | `[0, 1]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `(1 - x1*x2) * (3 + sin(pi*x1)*cos(pi*x2))` | `0.3*x1 - 0.3*x2` |
| Model13 | `[0, 1]^2` | `[0, 1]^2` | `expit(1.5*x1 - 0.5*x2)` | `log(1 + x1 + x2)` | `x1 - 0.7*x2` |
| Model14 | `[0, 1]^2` | `[0, 1]^2` | `expit(-0.5 + x1 + 2*x2)` | `(x1^2 + x2^2) * exp(-x1 - x2)` | `0.5 - x2` |

Model4-Model7 and Model11-Model14 share the same formulas by pairs. The
difference is again the observed covariate support: Model4-Model7 use
`[-0.2, 1.2]^2`, while Model11-Model14 use `[0, 1]^2`.

### Value Model

Model15 is the CCG value-functional design:

| Model | Observed support | Target support | `p0(x)` | `mu0(x,0)` | `h0(x)` |
| --- | --- | --- | --- | --- | --- |
| Model15 | `[-2, 2]^2` | `[-1.5, 1.5]^2` | `expit(x1 - x2)` | `0` | `(1 - x1^2 - x2^2) * (4 + sin(x1)*x2 + cos(x2))` |

The CCG value target used in this folder is:

```text
E[1{h0(X) > 0} * 9] = pi
```

## Target Parameters

The simulation file assigns one parameter type to each model:

| Parameter type | Models | Formula computed by OptTreat |
| --- | --- | --- |
| Known welfare | Model1-Model7 | `E[max(h(X), 0)]` over the target uniform distribution. |
| Unknown/common-support welfare | Model8-Model14 | empirical `mean(max(h(X_i), 0))` over the treated/control common-support sample. |
| Known value | Model15 | `E[1{h(X)>0} v(X)]` over `[-1.5, 1.5]^2`, with `v(X)=9`. |

For Model15 the target is the area of the positive treatment-effect region
scaled by `v(X)=9`; the true value is set to `pi`.

Known-distribution integrals use Sobol points with `sobol_scramble=False`, so
the integration rule mirrors the deterministic Sobol style used by the CCG
R workflow.

## Sieve First Stage

All CCG simulations in this folder use the sieve estimator:

```text
mu_hat_1(x) = fitted conditional mean for treated observations
mu_hat_0(x) = fitted conditional mean for control observations
h_hat(x)   = mu_hat_1(x) - mu_hat_0(x)
```

The treated and control regressions use separate spline bases
(`share_features=False`). The current CCG settings are:

| Setting | Value |
| --- | --- |
| spline degree | `3` |
| basis | tensor-product B-spline basis |
| knots | uniform |
| solver | Moore-Penrose pseudo-inverse, `pinv` |
| `pinv_rcond` | `sqrt(machine epsilon)`, about `1.49e-8` |

The pseudo-inverse path is used because the R simulations use `MASS::ginv(...)`
rather than ridge regularization. Small singular directions below the tolerance
are treated as numerically zero instead of being inverted.

## Spline Segment Choices

The CCG tuning table in `run_ccg2025_sievevar.py` specifies separate spline
segment counts for the control and treated first-stage regressions:

| Models | Control segments | Treated segments |
| --- | ---: | ---: |
| Model1 | 16 | 16 |
| Model2-Model5 | 1 | 1 |
| Model6-Model7 | 4 | 1 |
| Model8 | 8 | 8 |
| Model9 | 1 | 4 |
| Model10 | 4 | 1 |
| Model11-Model14 | 1 | 1 |
| Model15 | 1 | 4 |

For two-dimensional designs these are tensor-product spline segments, so the
number of columns grows with dimension.

## Variance And Inference

Every CCG spec in this folder supplies a `VarianceConfig`, so the summary table
includes standard errors and coverage.

| Target | SieveVar evaluation |
| --- | --- |
| Known welfare | Sobol integration over the target support using derivative `1{h_hat(x) >= 0}`. |
| Unknown/common-support welfare | empirical common-support evaluation plus the empirical variance component for `max(h_hat(X_i), 0)`. |
| Known value | level-set approximation using `-eps < h_hat(x) < eps`, with `eps=0.005` and value scale `9`. |

Paper-style settings use many more Sobol points for variance than the smoke
defaults. Model15 uses chunking because the paper-size value variance integral
uses `1,000,000` Sobol points.

## Run Settings

The script defaults to a small smoke run so that import and workflow errors are
easy to catch:

| Setting | Smoke default |
| --- | ---: |
| `MODEL_NAMES` | `("Model1", "Model8", "Model15")` |
| `REPLICATIONS` | `1` |
| `N_VALUES` | `(90,)` |
| `THETA_SOBOL` | `512` |
| `VARIANCE_SOBOL` | `32` |
| `VALUE_VARIANCE_SOBOL` | `64` |
| `CHUNK_SIZE` | `32` |
| `PROGRESS_EVERY` | `100` |

The paper-style constants are listed near the top of the same file:

| Setting | Paper-style value |
| --- | ---: |
| `PAPER_REPLICATIONS` | `2000` |
| `PAPER_N_VALUES` | `(1500, 3000, 6000)` |
| `PAPER_THETA_SOBOL` | `5000` |
| `PAPER_VARIANCE_SOBOL` | `40000` |
| `PAPER_VALUE_VARIANCE_SOBOL` | `1000000` |
| `PAPER_CHUNK_SIZE` | `50000` |

To run the full CCG exercise, set:

```python
MODEL_NAMES = tuple(f"Model{i}" for i in range(1, 16))
REPLICATIONS = PAPER_REPLICATIONS
N_VALUES = PAPER_N_VALUES
THETA_SOBOL = PAPER_THETA_SOBOL
VARIANCE_SOBOL = PAPER_VARIANCE_SOBOL
VALUE_VARIANCE_SOBOL = PAPER_VALUE_VARIANCE_SOBOL
CHUNK_SIZE = PAPER_CHUNK_SIZE
```

Then run:

```bash
python -m opttreat.simulations.ccg2025.run_ccg2025_sievevar
```

This is an expensive run: 15 models, 3 sample sizes, and 2000 replications gives
90,000 draw-level simulations.

## Output Files

Outputs are written to `opttreat/simulations/ccg2025/results/` using the shared
simulation naming convention:

```text
ccg2025_sievevar_summary_n<sample-sizes>_rep<replications>_<models>.csv
ccg2025_sievevar_draws_n<sample-sizes>_rep<replications>_<models>.csv
ccg2025_sievevar_results_n<sample-sizes>_rep<replications>_<models>.md
```

For example, the full all-model paper-style run writes:

```text
ccg2025_sievevar_summary_n1500_3000_6000_rep2000_M1_15.csv
ccg2025_sievevar_draws_n1500_3000_6000_rep2000_M1_15.csv
ccg2025_sievevar_results_n1500_3000_6000_rep2000_M1_15.md
```

The summary CSV contains one row per model and sample size:

| Column | Meaning |
| --- | --- |
| `spec` | simulation label, such as `Model8` |
| `model` | same model label, kept for easy grouping |
| `theorem` | CCG theorem group |
| `n` | sample size |
| `W_true` | target truth from the model and parameter object |
| `W_mean` | Monte Carlo mean of `W_hat` |
| `bias` | `W_mean - W_true` |
| `sd` | Monte Carlo standard deviation of `W_hat` |
| `replications` | number of Monte Carlo replications |
| `se` | average SieveVar standard error |
| `sd_se` | Monte Carlo standard deviation of the standard errors |
| `coverage` | empirical coverage of nominal 95 percent intervals |

The draw-level CSV contains one row per replication:

| Column | Meaning |
| --- | --- |
| `spec`, `model`, `theorem`, `n` | design identifiers |
| `rep` | replication index |
| `W_hat` | estimated welfare or value target |
| `W_true` | target truth |
| `se` | SieveVar standard error for that draw |

The Markdown report records the same settings and summary table in a readable
format for quick inspection.

## Reading The Results

For welfare models, lower absolute bias and an empirical `sd` close to average
`se` indicate that the estimator and SieveVar are behaving similarly to the CCG
reported tables. Coverage should be close to the nominal 0.95 level in the
large paper-style runs, with normal Monte Carlo fluctuation.

Python results are intended to be statistically comparable to the CCG R results,
not bit-for-bit identical. Random-number streams, Sobol implementations, spline
matrix construction, and pseudo-inverse details differ across Python and R.

## Notes

- `Model4` is included because OptTreat treats the main CCG design set as
  Model1-Model15, even though one local two-dimensional R workflow commented it
  out.
- `Model15` is the canonical OptTreat name for the value/area design.
- Changing `REPLICATIONS` changes the number of Monte Carlo repetitions. It is
  separate from the sample size `n`, which is controlled by `N_VALUES`.
- Historical CSVs may exist in `results/`; active output names identify the
  exact sample sizes, replication count, and model set used for the run.
