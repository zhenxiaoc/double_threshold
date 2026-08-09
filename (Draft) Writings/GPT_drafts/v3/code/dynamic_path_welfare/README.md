# dynamic_path_welfare

Plug-in estimation and **irregular inference** for the path-specific welfare component
`V_11^*` of an *estimated optimal two-stage dynamic treatment regime*.

Target parameter (see [`docs/theory_summary.md`](docs/theory_summary.md)):

```
V_11^* = E[ Y^(1,1) · 1{delta(X^(1)) >= 0} · 1{kappa(S) >= 0} ]
       = ∫_{D1+} ∫_{D2+} mu_1(x) p_1(x|s) m(s) dx ds
```

the welfare contributed by the **(1,1) treatment path** under the optimal regime — **not**
the value of a fixed policy and **not** the total optimal value `V^*`. Theory follows
`may_2026.tex` (Zhenxiao Chen).

## Status of the empirical application (read this first)

A July-2026 access audit found **no dataset that is downloadable without a data-use
agreement passes the hard gates** (n ≥ 1000 independent units, both stages randomized,
four `(T1,T2)` paths, genuinely/effectively continuous scalar `S`,`X`, known propensities).
The only perfectly-matched design — the Ida et al. energy-rebate 2×2 experiment (n=2,400)
— has **restricted, non-redistributable** microdata. See
[`docs/dataset_search.md`](docs/dataset_search.md) and
[`docs/data_access_blockers.md`](docs/data_access_blockers.md).

Therefore the software, boundary diagnostics, sieve-Riesz + bootstrap inference, and the
Monte Carlo study are **complete and run on calibrated simulated data with known truth**.
**This project does not claim a completed valid main empirical application.**

## Key result (simulated, honest)

| interval | 95% coverage | median length | verdict |
|---|---|---|---|
| sieve-Riesz (conditional on densities) | **under-covers** (0.27–0.83) | 0.10–0.28 SD | SE omits transition-law uncertainty (`se_ratio < 1`) |
| **participant bootstrap (full-refit)** | ≈ nominal | ≈ 0.45 SD | **recommended / usable & informative** |
| AIPW (fixed learned policy) | ≈ 0.92–0.95 | ≈ 0.35 SD | benchmark for a *different, regular* object |

The scalar-case plug-in rate is `n^{-s/(2s+1)}` — **irregular, slower than root-`n`**
(Chen & Gao 2026), in contrast to the *total* welfare `V^*` which is root-`n` regular.

## Install

```bash
python -m pip install -e .           # or: pip install -e ".[dev,io]"
```

Requires Python ≥ 3.11 (tested on 3.14) with numpy, pandas, scipy, scikit-learn,
statsmodels, matplotlib, joblib, pydantic, pyyaml.

## Pipeline (CLI)

```bash
python -m path_welfare.cli search-data --config configs/dataset_search.yaml
python -m path_welfare.cli audit       --config configs/public_fallback.yaml
python -m path_welfare.cli estimate    --config configs/public_fallback.yaml
python -m path_welfare.cli boundaries  --config configs/public_fallback.yaml
python -m path_welfare.cli infer       --config configs/public_fallback.yaml --nboot 499 --jobs 6
python -m path_welfare.cli simulate    --config configs/public_fallback.yaml --nrep 1000 --jobs 6
python -m path_welfare.cli robustness  --config configs/public_fallback.yaml
python -m path_welfare.cli report      --config configs/public_fallback.yaml
make test        # 44 tests
make all         # full pipeline + report
```

For a **real** dataset, fill in the matching config (`configs/energy.yaml`,
`intern_health.yaml`, or `smart.yaml`), place the unit-level file under `data/raw/`
(git-ignored — restricted microdata is never committed), and run the same commands.

## Layout

```
src/path_welfare/   estimator, riesz, aipw, densities, bootstrap, boundaries,
                    diagnostics, simulation, plotting, reporting, cli, schemas, ...
docs/               theory_summary, estimand, identification, inference_derivation,
                    aipw_derivation, dataset_search, data_access_blockers
tests/              44 tests (no-leakage, path-sum, spline & functional derivatives vs
                    finite differences, AIPW mean-zero, roots, reproducibility, ...)
configs/            dataset_search, energy, intern_health, smart, public_fallback
results/            tables/, figures/, simulations/, logs/
reports/            empirical_report.md / .html
```

## What is design-based vs. modelling-based

* **Design** (given randomization): sequential ignorability, positivity, known propensities.
* **Maintained modelling restriction**: scalar Markov sufficiency `E[Y|S,T1,X,T2]=E[Y|X,T2]`
  (testable; a diagnostic flags it), spline/smoothness, regular-margin conditions.
* **Diagnostic-supported**: continuity of `S,X`, balance, attrition.
* Nothing untestable is asserted "satisfied." See [`docs/identification.md`](docs/identification.md).
