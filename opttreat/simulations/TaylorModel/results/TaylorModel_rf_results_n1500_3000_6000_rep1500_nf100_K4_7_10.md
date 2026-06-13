# TaylorModel RF Results

This report records the TaylorExpansionModel random-feature welfare simulations
run from `opttreat/simulations/TaylorModel/run_taylor_rf.py`.

## Run Settings

| setting | value |
|:--|:--|
| expansions | `tan2`, `sinh2`, `rational`, `hyperbolic`, `exp_pm` |
| K values | `4`, `7`, `10` |
| p | `p = K` |
| sample sizes | `1500`, `3000`, `6000` |
| replications | `1500` |
| random features | `100` |
| random feature generator | `iid_sphere` |
| activation | `exp` |
| ridge alpha | `1e-5` |
| parameter | known-distribution welfare |
| inference | none |

CSV outputs:

- Summary: `TaylorModel_rf_summary_n1500_3000_6000_rep1500_nf100_K4_7_10.csv`
- Draws: `TaylorModel_rf_draws_n1500_3000_6000_rep1500_nf100_K4_7_10.csv`

## Summary Results

| expansion   |   K |   p |    n |   W_true |   W_mean |      bias |       sd |   replications | rfg_type   | activation   |   n_features |   alpha |
|:------------|----:|----:|-----:|---------:|---------:|----------:|---------:|---------------:|:-----------|:-------------|-------------:|--------:|
| tan2        |   4 |   4 | 1500 | 1        | 0.997757 | -0.002243 | 0.087169 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| tan2        |   4 |   4 | 3000 | 1        | 0.996848 | -0.003152 | 0.0604   |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| tan2        |   4 |   4 | 6000 | 1        | 1.00139  |  0.001393 | 0.043272 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| sinh2       |   4 |   4 | 1500 | 1        | 1.00475  |  0.004748 | 0.087717 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| sinh2       |   4 |   4 | 3000 | 1        | 1.00063  |  0.000635 | 0.060412 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| sinh2       |   4 |   4 | 6000 | 1        | 1.00014  |  0.000137 | 0.042034 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| rational    |   4 |   4 | 1500 | 1        | 0.9997   | -0.0003   | 0.086039 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| rational    |   4 |   4 | 3000 | 1        | 1.00034  |  0.000344 | 0.06097  |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| rational    |   4 |   4 | 6000 | 1        | 0.998715 | -0.001285 | 0.042538 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| hyperbolic  |   4 |   4 | 1500 | 0.633425 | 0.633515 |  9e-05    | 0.085611 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| hyperbolic  |   4 |   4 | 3000 | 0.633425 | 0.634797 |  0.001372 | 0.061347 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| hyperbolic  |   4 |   4 | 6000 | 0.633425 | 0.635095 |  0.00167  | 0.042309 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| exp_pm      |   4 |   4 | 1500 | 1.083    | 1.08991  |  0.006916 | 0.084599 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| exp_pm      |   4 |   4 | 3000 | 1.083    | 1.0871   |  0.004104 | 0.059446 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| exp_pm      |   4 |   4 | 6000 | 1.083    | 1.08453  |  0.001531 | 0.042144 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| tan2        |   7 |   7 | 1500 | 1        | 0.99667  | -0.00333  | 0.094366 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| tan2        |   7 |   7 | 3000 | 1        | 0.999069 | -0.000931 | 0.065301 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| tan2        |   7 |   7 | 6000 | 1        | 0.998736 | -0.001264 | 0.045109 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| sinh2       |   7 |   7 | 1500 | 1        | 0.998537 | -0.001463 | 0.094059 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| sinh2       |   7 |   7 | 3000 | 1        | 0.998628 | -0.001372 | 0.063989 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| sinh2       |   7 |   7 | 6000 | 1        | 0.998391 | -0.001609 | 0.046342 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| rational    |   7 |   7 | 1500 | 1        | 1.04992  |  0.04992  | 0.226953 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| rational    |   7 |   7 | 3000 | 1        | 1.01659  |  0.016591 | 0.168142 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| rational    |   7 |   7 | 6000 | 1        | 1.00212  |  0.002119 | 0.120919 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| hyperbolic  |   7 |   7 | 1500 | 0.63221  | 0.636159 |  0.003948 | 0.091098 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| hyperbolic  |   7 |   7 | 3000 | 0.63221  | 0.634882 |  0.002671 | 0.064529 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| hyperbolic  |   7 |   7 | 6000 | 0.63221  | 0.633482 |  0.001272 | 0.046492 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| exp_pm      |   7 |   7 | 1500 | 1.08582  | 1.09776  |  0.01194  | 0.088585 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| exp_pm      |   7 |   7 | 3000 | 1.08582  | 1.0887   |  0.002881 | 0.062145 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| exp_pm      |   7 |   7 | 6000 | 1.08582  | 1.08902  |  0.003197 | 0.043866 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| tan2        |  10 |  10 | 1500 | 1        | 0.99041  | -0.00959  | 0.096988 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| tan2        |  10 |  10 | 3000 | 1        | 0.989299 | -0.010701 | 0.065954 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| tan2        |  10 |  10 | 6000 | 1        | 0.99162  | -0.00838  | 0.045335 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| sinh2       |  10 |  10 | 1500 | 1        | 0.990847 | -0.009153 | 0.094213 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| sinh2       |  10 |  10 | 3000 | 1        | 0.989296 | -0.010704 | 0.065134 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| sinh2       |  10 |  10 | 6000 | 1        | 0.989292 | -0.010708 | 0.045761 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| rational    |  10 |  10 | 1500 | 1        | 1.83761  |  0.837612 | 0.638492 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| rational    |  10 |  10 | 3000 | 1        | 1.46979  |  0.469794 | 0.461883 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| rational    |  10 |  10 | 6000 | 1        | 1.2381   |  0.238098 | 0.371889 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| hyperbolic  |  10 |  10 | 1500 | 0.632213 | 0.636545 |  0.004332 | 0.092626 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| hyperbolic  |  10 |  10 | 3000 | 0.632213 | 0.633592 |  0.001379 | 0.063832 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| hyperbolic  |  10 |  10 | 6000 | 0.632213 | 0.633313 |  0.001101 | 0.044618 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| exp_pm      |  10 |  10 | 1500 | 1.08582  | 1.11071  |  0.024884 | 0.089216 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| exp_pm      |  10 |  10 | 3000 | 1.08582  | 1.10698  |  0.021161 | 0.063638 |           1500 | iid_sphere | exp          |          100 |   1e-05 |
| exp_pm      |  10 |  10 | 6000 | 1.08582  | 1.10306  |  0.017239 | 0.043473 |           1500 | iid_sphere | exp          |          100 |   1e-05 |

## Maximum Absolute Bias Across Sample Sizes

|   K | expansion   |   abs_bias |
|----:|:------------|-----------:|
|   4 | exp_pm      |   0.006916 |
|   4 | hyperbolic  |   0.00167  |
|   4 | rational    |   0.001285 |
|   4 | sinh2       |   0.004748 |
|   4 | tan2        |   0.003152 |
|   7 | exp_pm      |   0.01194  |
|   7 | hyperbolic  |   0.003948 |
|   7 | rational    |   0.04992  |
|   7 | sinh2       |   0.001609 |
|   7 | tan2        |   0.00333  |
|  10 | exp_pm      |   0.024884 |
|  10 | hyperbolic  |   0.004332 |
|  10 | rational    |   0.837612 |
|  10 | sinh2       |   0.010708 |
|  10 | tan2        |   0.010701 |

## Notes

- `tan2`, `sinh2`, `hyperbolic`, and the lower-dimensional `rational`
  designs have small bias with 100 random features.
- `rational` becomes difficult as `K=p` increases. At `K=p=10`, the bias is
  still large even at `n=6000`.
- This workflow is estimation-only. The table intentionally has no standard
  errors, confidence intervals, or coverage columns.
