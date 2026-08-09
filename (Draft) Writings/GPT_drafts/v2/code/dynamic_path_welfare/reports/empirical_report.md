# Empirical Report — Path-Specific Welfare Component `V_11^*`

**Project:** plug-in estimation and irregular inference for the (1,1) treatment-path
welfare component of an estimated optimal two-stage dynamic treatment regime.
**Theory:** `may_2026.tex` (Zhenxiao Chen). **Date:** 2026-07-13.

> **Headline honesty statement.** No dataset that is downloadable without a data-use
> agreement passes the hard gates. The software, boundary diagnostics, sieve-Riesz +
> bootstrap inference, and the Monte Carlo study are complete and run on **calibrated
> simulated data with known truth**. **This project does not claim a completed valid main
> empirical application.** The main empirical data requirement is **unresolved** (see
> `docs/data_access_blockers.md`).

Language key: **design** (from randomization) · **protocol** (trial document) ·
**diagnostic** (data-supported) · **maintained** (modelling restriction, not verified) ·
**unresolved**.

---

## Answers to the 29 required questions (task §21)

1. **Which dataset was selected, and why?** For the *runnable* pipeline, a **calibrated
   simulation** (DGP 1, a regular double-boundary scalar-state design), because no
   accessible real dataset passes the hard gates. The strongest *real* candidate is the
   Ida et al. energy-rebate 2×2 experiment (n=2,400), selected on the scorecard for design
   and continuity but scoring 0 on access (GATED microdata) — so it cannot be run here.
2. **Was individual-level sequential treatment assignment actually available?** **No** for
   every gate-passing real candidate. For the simulation, unit-level `(S,T1,X,T2,Y)` is
   available by construction.
3. **Are both treatment stages randomized?** In the simulation, **yes, by design**. In the
   energy-rebate design, yes (fully-crossed 2×2); in SMARTs, generally **no** (only
   non-responders are re-randomized).
4. **What do `T1=1`, `T2=1` mean?** In the simulation, the active first-/second-stage
   treatments. For a real run they mean *randomized assignment* to the active arm (for IHS,
   `T=1` is assignment to *any* notification — a randomized mixture).
5. **What are `S, X, Y`?** `S` baseline state (pre-`T1`), `X` intermediate state (post-`T1`,
   pre-`T2`), `Y` final outcome (post-`T2`). Simulation units are continuous by
   construction; the energy mapping is log pre/period-1 peak kWh and period-2 kWh.
6. **Genuinely continuous or finely supported?** In the simulation, **genuinely
   continuous** (≥ 100 unique values, max point mass ≤ 0.05 — Table 4). Project STAR scores
   would be *finely-supported, not genuinely continuous* (and STAR is not a 2×2).
7. **How many independent units remain?** Simulation: **2,000** (one unit per row; folds
   keep units intact). Energy-rebate would be 2,400.
8. **Four path counts?** Simulation (n=2,000): **00=503, 01=516, 10=527, 11=454** (Table 5)
   — all ≥ 150.
9. **Are randomization probabilities known?** **Yes** — `e1=e2=0.5` by design;
   empirical `P(T1=1)=0.49`, `P(T2=1)=0.485`. Not ML-estimated.
10. **Does sequential ignorability follow from the design?** **Yes (design)** under
    two-stage randomization; balance AUCs (`AUC(T1|S)`, `AUC(T2|S,T1,X)`) are reported as
    diagnostics, near 0.5.
11. **Global or availability-relative positivity?** **Global** for the fully-crossed 2×2
    simulation; **availability-relative** would be required for a SMART.
12. **How much attrition?** Simulation: **0%** missing `Y` (Table 7). A real run reports
    missingness by path, state quantile, time, and site, with IPW-of-observation and
    delta-adjustment sensitivity.
13. **Is the scalar Markov restriction credible?** It is a **maintained** restriction. The
    diagnostic (Table 8) shows the rich model `Y~f(S,T1,X,T2)` gives **no** held-out MSE
    improvement over `Y~f(X,T2)` in DGP 1 (incremental reduction −4%, i.e. none) →
    *not flagged*; in the Markov-failure DGP 7 the diagnostic correctly flags it. Never
    asserted "satisfied."
14. **Do `δ` and `κ` cross zero?** **Yes** — one interior root each (Table 9): `δ` root
    ≈ −0.07 (true 0), `κ` root ≈ +0.08 (true 0).
15. **Interior and regular roots?** **Yes** — both roots are interior (within the central
    95% of support), have derivatives bounded away from zero, and are classified *regular*
    with no weak-margin flags.
16. **Observations supporting each boundary?** Boundary-band counts `N(h)` reported
    (Table 9 / Fig 12); each active boundary has ≥ `max(50, 0.05n)` observations within the
    primary band.
17. **Estimate of `V_11^*`?** **0.886** in outcome units, **0.765** in outcome-SD units
    (cross-fitted direct plug-in; sieve-Riesz 0.897); truth 0.889 (Table 10). The four
    components are 00=0.040, 01=0.087, 10=0.013, 11=0.886.
18. **Other three path contributions?** `V_00=0.040, V_01=0.087, V_10=0.013` (Table 10).
19. **Do they sum to total optimal welfare?** **Yes** — sum = 1.02623 = direct total,
    **component-sum residual = 0.0** (Table 11); true total 1.031.
20. **Plug-in vs IPW vs AIPW?** Plug-in(direct) **0.886**, sieve-Riesz **0.897**, AIPW
    (fixed learned policy) **0.885**, IPW **0.737** (Table 12). AIPW ≈ plug-in ≈ truth; IPW
    is the high-variance benchmark and is farther off.
21. **Which variance contributions are included?** The sieve-Riesz SE includes the `μ`-score
    (interior + moving-boundary) contribution **conditional on the estimated densities
    `m, p_a`**. The participant bootstrap includes **all** contributions (m, p_a, boundary).
22. **Which are omitted / conditional?** The sieve-Riesz SE **omits** `m` and (dominant)
    `p_a` uncertainty — it is explicitly labelled *conditional on densities*. Monte Carlo
    shows it under-covers (`se_ratio ≈ 0.3–0.6`, coverage 0.27–0.83; Table 16).
23. **Did the proposed interval attain acceptable simulated coverage?** The **conditional
    sieve interval did NOT** (0.27–0.83 < 0.90; 0.50 at DGP1 n=1500). The **participant
    bootstrap did** on the regular design (DGP1 coverage 0.98 at both n=1500 and n=1000;
    60 datasets × 99 bootstrap) — it is the recommended interval (Table 17).
24. **Is the empirical interval substantively informative?** For the bootstrap interval,
    **yes** on the regular design (median length 0.32–0.40 outcome SD < 0.50 →
    *informative*). It is usable but *not* informative on the weak-first-stage design DGP3
    (coverage 0.90, length 0.74 SD). The conditional sieve interval is not usable at all.
25. **Is the sample large enough for the intended inference?** For the regular design,
    n ≈ 1000–1500 supports a usable/informative bootstrap interval; the weak-boundary and
    multiple-root designs (DGP 3, 6) are materially harder and coverage degrades — consistent
    with the irregular (`n^{-s/(2s+1)}`) rate.
26. **Which conclusions are design-based?** Sequential ignorability, positivity, known
    propensities, path counts.
27. **Which rely on the Markov model?** Identification of `μ_a(x)=E[Y|X,T2=a]` as the stage-two
    payoff, and hence the whole scalar estimator; flagged model-dependent when the diagnostic fails.
28. **Which rely on spline/smoothness?** The nuisance fits, the analytic decision-margin
    derivatives, the regular-margin conditions, and the sieve-Riesz variance.
29. **Which are only exploratory?** All numbers here are **simulation-based** and thus a
    methods demonstration, not empirical findings about any real population. The energy /
    IHS / SMART configs are templates awaiting data access.

---

## Summary of estimates (simulation, DGP 1, n=2,000)

| quantity | value (orig) | value (SD) | truth |
|---|---|---|---|
| `V_11^*` (direct plug-in) | 0.886 | 0.765 | 0.889 |
| `V_11^*` (sieve-Riesz) | 0.897 | 0.774 | 0.889 |
| `V_00, V_01, V_10` | 0.040, 0.087, 0.013 | — | 0.060, 0.062, 0.020 |
| total `V^*` | 1.026 | — | 1.031 |
| component-sum residual | 0.000 | — | 0 |

Roots: `δ` ≈ −0.07, `κ` ≈ +0.08 (both interior, regular). Analytic moving-boundary
gradient vs finite differences: max relative error **< 1e-4** (validated).

## Inference summary and go/no-go

| interval | includes | 95% coverage (MC) | median length | verdict |
|---|---|---|---|---|
| sieve-Riesz (conditional on `m,p_a`) | μ-score only | **0.27–0.83** (0.50 at DGP1 n=1500) | 0.10–0.28 SD | **NOT usable** — SE omits transition-law uncertainty (`se_ratio` 0.3–0.6) |
| **participant bootstrap (full-refit)** | everything (m, p_a, boundary) | DGP1 **0.98** (n=1500) / **0.98** (n=1000); DGP3 0.90; DGP6 0.97 | DGP1 **0.32–0.40 SD**; DGP3 0.74; DGP6 0.57 | **usable & informative** on the regular design (recommended) |
| AIPW (fixed learned policy) | orthogonal score | ≈ 0.92–0.95 | ≈ 0.35 SD | benchmark for a *different, regular* object |

**Go/No-Go (task §14).** On the **regular** design (DGP1), the **participant bootstrap
passes both the usable (coverage ≥ 0.90, length ≤ 1 SD) and informative (length ≤ 0.5 SD)
thresholds** → verdict **informative** (coverage **0.98**, median length **0.32 SD**;
`results/tables/table17_go_no_go.csv`, 60 datasets × 99 bootstrap, CV-selected K). The
**conditional sieve interval fails** (coverage 0.50 ≪ 0.90). On the harder designs the
bootstrap degrades honestly but stays usable: DGP3 (weak first-stage margin) coverage 0.90
with length 0.74 SD (usable, *not* informative), DGP6 (multiple roots) coverage 0.97 with
length 0.57 SD — the wider intervals reflect the slower irregular rate `n^{-s/(2s+1)}`.

## Caveats and unresolved items
- **Main empirical data: unresolved.** All numbers are simulation-based.
- The conditional sieve variance is **incomplete** (conditional on densities); the honest
  interval is the bootstrap. A fully analytic complete-variance (hybrid) expansion is
  described in `docs/inference_derivation.md` §4 but not implemented — the bootstrap stands
  in for it.
- Weak-boundary / multiple-root regimes (DGP 3, 6) are materially harder; report those
  coverages honestly rather than the best-case regular design.
- The scalar Markov restriction is **maintained**, not verified; a richer-history
  robustness spec is the appropriate check on real data.

_See `docs/` for the full theory, estimand, identification, inference, and AIPW
derivations; `results/tables` and `results/figures` for all 17 tables and 19 figures._
