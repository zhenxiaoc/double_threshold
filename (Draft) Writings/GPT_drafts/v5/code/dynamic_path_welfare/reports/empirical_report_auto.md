# Empirical Report (auto-assembled)

Dataset: **simulation**  |  config hash: `6f263c444d239016`

### table02_sample_construction

| step                         |    n |
|:-----------------------------|-----:|
| simulated draws              | 2000 |
| valid schema O=(S,T1,X,T2,Y) | 2000 |
| non-missing Y                | 2000 |
| analysis sample              | 2000 |

### table03_variable_defs

| symbol   | definition                           | timing   | type       |
|:---------|:-------------------------------------|:---------|:-----------|
| S        | baseline state (pre-T1)              | t0       | continuous |
| T1       | first-stage randomized treatment     | t1       | binary     |
| X        | intermediate state (post-T1, pre-T2) | t2       | continuous |
| T2       | second-stage randomized treatment    | t3       | binary     |
| Y        | final outcome (post-T2)              | t4       | continuous |

### table04_continuous_state

| variable   |   n_unique |   max_point_mass |    q05 |    q50 |   q95 | effectively_continuous   | genuinely_continuous   |
|:-----------|-----------:|-----------------:|-------:|-------:|------:|:-------------------------|:-----------------------|
| S          |       2000 |           0.0005 | -1.9   | -0.01  | 1.966 | True                     | True                   |
| X          |       2000 |           0.0005 | -2.524 |  0.042 | 2.57  | True                     | True                   |

### table05_path_counts

|   path |   count |   P(T1=1)_emp |   P(T2=1)_emp |   P(T1=1)_design |   P(T2=1)_design |
|-------:|--------:|--------------:|--------------:|-----------------:|-----------------:|
|     00 |     503 |          0.49 |         0.485 |              0.5 |              0.5 |
|     01 |     516 |          0.49 |         0.485 |              0.5 |              0.5 |
|     10 |     527 |          0.49 |         0.485 |              0.5 |              0.5 |
|     11 |     454 |          0.49 |         0.485 |              0.5 |              0.5 |

### table06_balance

| check          |   value |
|:---------------|--------:|
| AUC(T1|S)      |   0.509 |
| AUC(T2|S,T1,X) |   0.521 |

### table07_attrition

| quantity          |   value |
|:------------------|--------:|
| missing_S         |       0 |
| missing_X         |       0 |
| missing_Y         |       0 |
| missing_Y_path_00 |       0 |
| missing_Y_path_01 |       0 |
| missing_Y_path_10 |       0 |
| missing_Y_path_11 |       0 |

### table08_markov

| quantity                  |   value |
|:--------------------------|--------:|
| MSE restricted Y~f(X,T2)  |  0.258  |
| MSE rich Y~f(S,T1,X,T2)   |  0.2682 |
| incremental MSE reduction | -0.0395 |
| markov_questionable       |  0      |

### table09_boundary_roots

| boundary   |   location |   quantile |   derivative |   local_n | regular   | flags   |
|:-----------|-----------:|-----------:|-------------:|----------:|:----------|:--------|
| delta      |    -0.0668 |      0.467 |       0.579  |       159 | True      |         |
| kappa      |     0.0748 |      0.521 |       0.2916 |       140 | True      |         |

### table10_path_components

|   path |   V_ab (orig) |   V_ab (SD units) |   truth |
|-------:|--------------:|------------------:|--------:|
|     00 |        0.0382 |            0.033  |  0.0602 |
|     01 |        0.0882 |            0.0762 |  0.0623 |
|     10 |        0.013  |            0.0112 |  0.0196 |
|     11 |        0.8854 |            0.7643 |  0.889  |

### table11_total_check

| quantity               |   value |
|:-----------------------|--------:|
| sum of components      | 1.0248  |
| direct total (E[A*])   | 1.0248  |
| component-sum residual | 0       |
| true total             | 1.03107 |

### table12_method_comparison

| method          |   estimate |         lo |         hi |
|:----------------|-----------:|-----------:|-----------:|
| plug-in(direct) |   0.885359 | nan        | nan        |
| sieve-Riesz     |   0.897309 |   0.874809 |   0.919809 |
| IPW             |   0.737171 |   0.622989 |   0.851352 |
| AIPW            |   0.884467 |   0.822702 |   0.946233 |

### table13_intervals

| method                             |   estimate |       lo |       hi |        se | note                                                                              |
|:-----------------------------------|-----------:|---------:|---------:|----------:|:----------------------------------------------------------------------------------|
| sieve-Riesz (K=8, conditional)     |   0.897309 | 0.874809 | 0.919809 | 0.0114797 | sieve-Riesz variance CONDITIONAL on estimated densities (m, p_a)                  |
| participant bootstrap (full-refit) |   0.886296 | 0.808996 | 0.973536 | 0.0427393 | percentile CI; not guaranteed valid for the hard-threshold target -- see MC study |
| multiplier bootstrap               |   0.897309 | 0.874891 | 0.919933 | 0.0117788 | conditional on estimated densities m, p_a                                         |

### table14_cost

|   cost |      V11 |
|-------:|---------:|
|  0     | 0.885553 |
|  0.025 | 0.857954 |
|  0.05  | 0.830354 |
|  0.1   | 0.775155 |

### table15_spline_sensitivity

|   K |   estimate |        se |       lo |       hi |
|----:|-----------:|----------:|---------:|---------:|
|   5 |   0.899315 | 0.0109572 | 0.877839 | 0.920791 |
|   6 |   0.89717  | 0.0112753 | 0.875071 | 0.919269 |
|   8 |   0.897309 | 0.0114797 | 0.874809 | 0.919809 |
|  10 |   0.897718 | 0.0118258 | 0.87454  | 0.920896 |

### table16_monte_carlo

| dgp   |    n |   n_rep |   V11_true |   bias_sieve |   rmse_sieve |   mc_sd_sieve |   mean_se |   se_ratio |   cov90 |   cov95 |   median_len95_sd |   delta_root_err |   kappa_root_err |   fail_rate |
|:------|-----:|--------:|-----------:|-------------:|-------------:|--------------:|----------:|-----------:|--------:|--------:|------------------:|-----------------:|-----------------:|------------:|
| dgp1  |  750 |     150 |     0.889  |       0.0075 |       0.0553 |        0.055  |    0.0204 |     0.3716 |  0.4667 |  0.58   |            0.156  |           0.0725 |           0.1366 |           0 |
| dgp1  | 1500 |     150 |     0.889  |       0.0093 |       0.0398 |        0.0388 |    0.0143 |     0.3686 |  0.4267 |  0.5    |            0.1109 |           0.0503 |           0.1248 |           0 |
| dgp2  |  750 |     150 |     0.5349 |      -0.0192 |       0.0603 |        0.0574 |    0.0696 |     1.2129 |  0.76   |  0.8333 |            0.285  |           0.3176 |           0.1169 |           0 |
| dgp2  | 1500 |     150 |     0.5349 |      -0.0045 |       0.0394 |        0.0393 |    0.0366 |     0.9331 |  0.6933 |  0.76   |            0.1806 |           0.2043 |           0.0917 |           0 |
| dgp3  |  750 |     150 |     0.7054 |      -0.1168 |       0.2028 |        0.1663 |    0.0219 |     0.1316 |  0.2667 |  0.3133 |            0.1425 |           0.0744 |           0.529  |           0 |
| dgp3  | 1500 |     150 |     0.7054 |      -0.0439 |       0.1159 |        0.1076 |    0.0157 |     0.1456 |  0.3533 |  0.3867 |            0.1231 |           0.0511 |           0.4422 |           0 |
| dgp4  |  750 |     150 |     0.7167 |      -0.0012 |       0.0588 |        0.059  |    0.022  |     0.373  |  0.5533 |  0.5933 |            0.1455 |         nan      |           0.1291 |           0 |
| dgp4  | 1500 |     150 |     0.7167 |       0.002  |       0.0406 |        0.0407 |    0.0138 |     0.3402 |  0.44   |  0.5    |            0.1028 |         nan      |           0.0811 |           0 |
| dgp5  |  750 |     150 |     1.2868 |      -0.0231 |       0.0554 |        0.0506 |    0.0314 |     0.62   |  0.6533 |  0.7267 |            0.2383 |           0.0724 |         nan      |           0 |
| dgp5  | 1500 |     150 |     1.2868 |      -0.019  |       0.0427 |        0.0384 |    0.0223 |     0.5818 |  0.5933 |  0.6733 |            0.1742 |           0.0516 |         nan      |           0 |
| dgp6  |  750 |     150 |     1.7206 |      -0.0514 |       0.126  |        0.1154 |    0.026  |     0.2249 |  0.2133 |  0.2733 |            0.2005 |           0.0491 |           0.1785 |           0 |
| dgp6  | 1500 |     150 |     1.7206 |      -0.0526 |       0.0912 |        0.0748 |    0.0182 |     0.2431 |  0.2467 |  0.3067 |            0.1423 |           0.0319 |           0.1623 |           0 |
| dgp7  |  750 |     150 |     1.2868 |       0.0097 |       0.0822 |        0.0819 |    0.027  |     0.3301 |  0.3933 |  0.4933 |            0.1994 |           0.1028 |           0.1087 |           0 |
| dgp7  | 1500 |     150 |     1.2868 |       0.0107 |       0.0573 |        0.0565 |    0.0183 |     0.3239 |  0.34   |  0.4467 |            0.1389 |           0.0645 |           0.0957 |           0 |

### table17_go_no_go

| criterion                 | value                                                 |
|:--------------------------|:------------------------------------------------------|
| coverage95                | 0.9                                                   |
| coverage95>=0.90          | True                                                  |
| fail_rate<0.05            | True                                                  |
| median_len_sd<=1.0        | True                                                  |
| se_ratio_within_20pct     | True                                                  |
| roots_stable              | True                                                  |
| support_ok                | True                                                  |
| USABLE                    | True                                                  |
| INFORMATIVE               | True                                                  |
| verdict                   | informative                                           |
| interval_method           | participant full-refit bootstrap                      |
| conditional_sieve_cov95   | 0.5                                                   |
| conditional_sieve_verdict | NOT usable (SE conditional on densities under-covers) |
