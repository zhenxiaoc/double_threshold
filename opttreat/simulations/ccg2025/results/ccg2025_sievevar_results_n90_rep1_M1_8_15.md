# ccg2025_sievevar Results

## Run Settings

| setting                | value                   |
|:-----------------------|:------------------------|
| models                 | Model1, Model8, Model15 |
| sample sizes           | 90                      |
| replications           | 1                       |
| progress every         | 100                     |
| theta Sobol            | 512                     |
| welfare variance Sobol | 32                      |
| value variance Sobol   | 64                      |
| spline degree          | 3                       |
| spline basis           | tensor                  |
| spline knots           | uniform                 |
| solver                 | pinv                    |
| pinv rcond             | 1.4901161193847656e-08  |

## Output Files

- Summary: `ccg2025_sievevar_summary_n90_rep1_M1_8_15.csv`
- Draws: `ccg2025_sievevar_draws_n90_rep1_M1_8_15.csv`

## Summary Results

| spec    | model   | theorem   |   n |   W_true |   W_mean |     bias |   sd |   replications |       se |   sd_se |   coverage |
|:--------|:--------|:----------|----:|---------:|---------:|---------:|-----:|---------------:|---------:|--------:|-----------:|
| Model1  | Model1  | Theorem 1 |  90 | 0.384362 | 1.22397  | 0.839606 |    0 |              1 | 0.236457 |       0 |          0 |
| Model8  | Model8  | Theorem 2 |  90 | 0.384362 | 0.648138 | 0.263776 |    0 |              1 | 0.184873 |       0 |          1 |
| Model15 | Model15 | Theorem 3 |  90 | 3.14159  | 4.02539  | 0.883798 |    0 |              1 | 0        |       0 |          0 |

## Maximum Absolute Bias

| model   | spec    |   abs_bias |
|:--------|:--------|-----------:|
| Model1  | Model1  |   0.839606 |
| Model15 | Model15 |   0.883798 |
| Model8  | Model8  |   0.263776 |

## Notes

- CCG paper-specific tuning lives in this simulation layer, not in the model classes.
- Coverage columns are reported only because variance_config is provided.
