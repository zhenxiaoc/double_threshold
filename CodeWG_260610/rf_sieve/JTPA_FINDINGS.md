# JTPA Application with the RF-Sieve Pipeline

*Script `jtpa_rf_application.py`; results `results/JTPA_rf_results.csv`. Data:
`KT_Data1.csv` (n = 9,223; the Kitagawa-Tetenov sample used in the paper's
Section 6: 30-month earnings, pre-program earnings, education).*

**Data note.** The local archives (KT_Data1.csv and the KT ECMA supplement
jtpa2.dta) contain only the two covariates used in the paper. The richer JTPA
demographics (age, race, sex, marital status, ...) require the full NJS/Upjohn
archive — flagged as a data acquisition item for the "all covariates" showcase.
Everything below is therefore the paper's own setting, re-examined with the
validated random-feature pipeline.

## Headline table (trimmed sample, n = 9,220; CV-selected K, gamma = 1.5)

| | Share treated | Welfare gain |
|---|---|---|
| **RF sieve, no cost** | **0.914 (0.795, 1.034)** | **$1,384 ($777, $1,990)** |
| paper (B-spline) | 0.89 (0.73, 1.05) | $1,519 ($764, $2,274) |
| KT (2018), no CI | 0.91 | $1,693 |
| **RF sieve, $774 cost** | **0.823 (0.717, 0.929)** | **$703 ($145, $1,260)** |
| paper (B-spline) | 0.80 (0.53, 1.07) | $858 ($152, $1,564) |
| KT (2018), no CI | 0.78 | $996 |

## Findings

1. **Cross-sieve validation of the paper's empirics.** A completely different
   first stage (random cos features + OLS, K = 25-50 chosen by CV) reproduces
   the B-spline point estimates within ~0.02 (share) and ~$150 (welfare).
   The paper's empirical conclusions are not an artifact of the spline basis.

2. **Tighter confidence intervals, no KDE needed.** The empirical-measure band
   SE (sample average over {|h_hat(X_i)| < eps} — valid in the unknown-F case,
   Theorem 6) gives share-CI widths of 0.24 / 0.21 vs the paper's 0.32 / 0.54,
   *without* the Gaussian-KDE + bandwidth-scaling machinery of the paper's
   implementation (Appendix C.1-C.2 tuning sensitivity becomes moot). This is
   a concrete methodological simplification to consider for the paper: in the
   F = F_0 case the empirical measure integrates against f_0 automatically.

3. **Debiasing diagnostics point in economically sensible directions.**
   - LOO-W (the validated welfare debiasing): $1,384 -> $1,281 (no cost),
     $703 -> $594 (cost). The plug-in welfare gain carries an upward
     Jensen-bias of roughly $100 (7-15%) — the paper's (and KT's) plug-in
     numbers are likely biased up by a similar order.
   - LOO-V with delta0 = 0.2 (the extreme-share correction from the sweep —
     the JTPA share ~0.9 is squarely in that regime): 0.914 -> 0.965 (no cost),
     0.823 -> 0.882 (cost). The one-sided sign-flip bias *underestimates* the
     share when it is high; the corrected share is ~0.05 higher. Note the JTPA
     first stage is very noisy (CV-RMSE ~ $15.6k vs sd(h_hat) ~ $1.6k), exactly
     the regime where the simulations showed this bias binds.

4. **Stability.** Across 20 independent feature draws, the trimmed estimates
   move by only 0.003-0.007 (share) and $5-26 (welfare) — the RF sieve's
   randomness is immaterial after trimming. Untrimmed, instability is 5-7x
   larger (sd(h_hat) doubles from extrapolation), reaffirming the paper's
   trimming practice; min-max common-support trimming removes only 3 of 9,223
   observations here.

## Suggested use in the paper

- A short empirical-robustness subsection (or appendix table): the RF-sieve
  column next to the existing B-spline column, plus the LOO-debiased rows as
  bias diagnostics.
- Replace (or complement) the KDE-based sigma_V computation with the
  empirical-measure band derivative — simpler, tuning-free apart from iota,
  and it produced tighter, internally consistent CIs here.
- The debiased numbers suggest framing: "the optimal program treats ~90% of
  the eligible population and raises average 30-month earnings by roughly
  $600-$1,300 net of costs" — robust across sieves and debiasing.

## Caveats

- LOO corrections are reported as point diagnostics with the plug-in SE
  (per Theorem 5 the same studentization is asymptotically valid; the sweep
  showed it is adequate for W and for V with delta0 ~ 0.2).
- CV-RMSE >> sd(h_hat): the first-stage signal-to-noise in JTPA is low, so all
  CATE-based estimands here lean on the asymptotics; the simulation
  gatekeeper rule (h_rmse vs sd(tau)) cannot be verified directly on real data
  — worth a remark rather than silence.
- Untrimmed rows use extrapolated fits; reported only for comparability with
  the paper's Appendix C.3.
