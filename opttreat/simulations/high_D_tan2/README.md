# High-D Tan2 Random-Feature Simulations

This folder runs estimation-only random-feature welfare simulations for explicit
tan2 Taylor designs:

```python
TaylorExpansionModel(expansion="tan2", K=3, p=3)
TaylorExpansionModel(expansion="tan2", K=7, p=7)
TaylorExpansionModel(expansion="tan2", K=10, p=10)
```

Run:

```bash
python -m opttreat.simulations.high_D_tan2.run_high_d_tan2_rf
```

The script is not CLI-heavy. Edit the top-level variables in
`run_high_d_tan2_rf.py` for sample sizes, replications, Sobol points, random
features, activation, and ridge penalty.

This workflow computes estimates only:

- no variance;
- no standard errors;
- no confidence intervals;
- no coverage.

`Model99`, `Model100`, `Model101`, and `Model102` are not active model names.
Use explicit `TaylorExpansionModel` specs instead.

Outputs are written with explicit sample-size and replication naming:

- `high_D_tan2_rf_summary_n<sample-sizes>_rep<replications>_nf<features>_K<values>.csv`
- `high_D_tan2_rf_draws_n<sample-sizes>_rep<replications>_nf<features>_K<values>.csv`
- `high_D_tan2_rf_results_n<sample-sizes>_rep<replications>_nf<features>_K<values>.md`
