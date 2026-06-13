# TaylorModel Random-Feature Simulations

This folder runs estimation-only Monte Carlo simulations for
`TaylorExpansionModel` using random-feature ridge regression. The workflow is:

```text
TaylorExpansionModel -> generated data -> RF ridge first stage
-> known-distribution welfare estimate -> summary/draw CSVs
```

No inference is computed here: no standard errors, confidence intervals, or
coverage columns are produced.

Run from the project root:

```bash
python -m opttreat.simulations.TaylorModel.run_taylor_rf
```

## Current Simulation Settings

The runnable file is `run_taylor_rf.py`. The main settings are explicit at the
top of the file:

| setting | current value | meaning |
| --- | --- | --- |
| `EXPANSIONS` | `tan2`, `sinh2`, `rational`, `hyperbolic`, `exp_pm` | Formula families to simulate. |
| `K_VALUES` | `(4, 7, 10)` | Taylor truncation orders. |
| `p` | `p = K` | Covariate dimension equals truncation order in this run. |
| `N_VALUES` | `(1500, 3000, 6000)` | Sample sizes. |
| `REPLICATIONS` | `1500` | Monte Carlo repetitions per design cell. |
| `THETA_SOBOL` | `5000` | Sobol points used to evaluate welfare. |
| `N_FEATURES` | `100` | Number of random features used by RF ridge. |
| `ACTIVATION` | `exp` | Activation applied to random projections. |
| `RFG_TYPE` | `iid_sphere` | Random-feature direction generator. |
| `ALPHA` | `1e-5` | Ridge penalty. |

Each design cell is identified by:

```text
<expansion>_K<K>_p<p>
```

For example, `tan2_K7_p7` means the `tan2` formula with truncation order
`K=7` and covariate dimension `p=7`.

## Data-Generating Process

For all TaylorModel designs:

- observed covariates are drawn from `[-0.2, 1.2]^p`;
- target covariates for welfare are drawn from `[0, 1]^p`;
- treatment is randomized with `p0(x)=0.5`;
- outcomes satisfy `Y = baseline(X) + D * h0(X) + noise`;
- the noise standard deviation is inherited from `ModelBase` and defaults to 1.

Here `baseline(x)` is the untreated conditional mean:

```text
baseline(x) = mu0(x,0)
h0(x) = mu0(x,1) - mu0(x,0)
mu0(x,d) = baseline(x) + d * h0(x)
```

So `baseline(x)` is not a separate estimand. It is only the code's name for
the control-state regression function, while `h0(x)` is the treatment-effect
function.

The known-distribution welfare target is:

```text
W0 = E[max(h0(X), 0)],    X ~ Uniform([0,1]^p)
```

The estimator reports:

```text
W_hat = E[max(h_hat(X), 0)]
```

using `THETA_SOBOL` unscrambled Sobol points on `[0,1]^p`.

## Model Specifications

The model formulas live in `opttreat/models/taylor_expansion_model.py`.

### `tan2`

`tan2` is a constant-gap multivariate design. It uses the first `K` covariates
and even powers:

```text
baseline(x) = sum_{k=1}^K a_k x_k^{2k}
h0(x) = 1
```

The coefficients `a_k` are the first `K` nonconstant Taylor coefficients of
`tan(z)^2`. Because `h0(x)=1`, the true welfare is `W0=1`.

### `sinh2`

`sinh2` is also a constant-gap multivariate design:

```text
baseline(x) = sum_{k=1}^K a_k x_k^{2k}
h0(x) = 1
```

The coefficients `a_k` come from the Taylor expansion of `sinh(z)^2`, where
the nonconstant even-power coefficient is `2^(2k-1)/(2k)!`. The true welfare
is `W0=1`.

### `rational`

`rational` is a constant-gap multivariate polynomial design:

```text
baseline(x) = sum_{k=1}^K (k+1) x_k^k
h0(x) = 1
```

This design is intentionally harder as `K=p` grows because the polynomial terms
increase in degree and coefficient size across coordinates. The true welfare is
still `W0=1`.

### `hyperbolic`

`hyperbolic` is a varying-gap univariate-in-first-coordinate design. Only
`x_1` enters the baseline and treatment effect:

```text
baseline(x) = Taylor_K[sinh(x_1)]
mu0(x,1) = Taylor_K[cosh(x_1)]
h0(x) = Taylor_K[cosh(x_1)] - Taylor_K[sinh(x_1)]
```

Extra coordinates are present when `p>1`, but the formula uses only `x_1`.
The target welfare is evaluated by Sobol integration rather than a hard-coded
closed form.

### `exp_pm`

`exp_pm` is another varying-gap first-coordinate design:

```text
baseline(x) = Taylor_K[exp(-x_1)]
mu0(x,1) = Taylor_K[exp(x_1)]
h0(x) = Taylor_K[exp(x_1)] - Taylor_K[exp(-x_1)]
```

As with `hyperbolic`, only `x_1` enters the signal, while the simulation may
include additional nuisance dimensions through `p=K`.

## RF Ridge Estimator

The estimator is `method="rf_ridge"`. It fits treated and control regression
functions separately:

```text
mu_t(x) = E[Y | X=x, D=1]
mu_c(x) = E[Y | X=x, D=0]
h_hat(x) = mu_t_hat(x) - mu_c_hat(x)
```

The random-feature map is built before the ridge fits. With the current
settings:

- `rfg_type="iid_sphere"` draws random directions on the unit sphere;
- `activation="exp"` applies an exponential activation to projected covariates;
- `share_features=True` uses the same random-feature map for treated and
  control samples;
- `n_features=100` gives both regressions a 100-column random-feature design;
- `alpha=1e-5` is the ridge penalty used by `sklearn.linear_model.Ridge` with
  `fit_intercept=False`.

The fitted first stage stores design matrices, residuals, feature maps, and the
callable `h_hat`. The simulation then evaluates `h_hat` on Sobol target points
and averages `max(h_hat(X),0)`.

## Output Files

Outputs are written under `opttreat/simulations/TaylorModel/results/` with
explicit sample-size and replication naming:

- `TaylorModel_rf_summary_n<sample-sizes>_rep<replications>_nf<features>_K<values>.csv`
- `TaylorModel_rf_draws_n<sample-sizes>_rep<replications>_nf<features>_K<values>.csv`
- `TaylorModel_rf_results_n<sample-sizes>_rep<replications>_nf<features>_K<values>.md`

For example:

```text
TaylorModel_rf_summary_n1500_3000_6000_rep1500_nf100_K4_7_10.csv
```

means sample sizes `1500, 3000, 6000`, 1500 replications per design cell,
100 random features, and `K` values `4, 7, 10`.
