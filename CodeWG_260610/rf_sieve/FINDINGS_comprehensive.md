# Comprehensive Simulation Evidence: RF-Sieve Welfare/Value Inference in High Dimensions

*Final consolidation of the systematic sweep (rounds logged in
`INVESTIGATION_LOG.md`; scripts `sweep_S1`-`sweep_S7b` + `check_loo_v_delta.py`;
all summaries/draws in `results/`). Builds on `FINDINGS_v_highdim.md` (base
results), `FINDINGS_directions.md` (D1-D4), `THEORY_NOTE_effective_dim.md`.*

## Evidence table by claim

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| C1 | Plug-in V + band SE is valid broadly | S1: coverage 1.00 across cos/relu/tanh features, heteroskedastic errors, d_x in {10,50}, dense/sparse DGPs; D2: 0.91-0.97 well-calibrated with low-dim sieve; D4: valid post-CV-selection (0.98) | **Supported**, with the extreme-share caveat (A2, below) |
| C2 | W needs debiasing; LOO-W works | D3: LOO-W bias ~0 and coverage 0.85-0.93 where plug-in W covered 0.00-0.19; S3: screening x LOO-W = 0.89-0.95 at every signal strength; S7b: LOO-W bias 0.0016 | **Supported** (residual 0.90-0.95 gap = smoothing bias from extra coords / K) |
| C3 | Debiased V matters at marginal smoothness | S2 (kink DGP, Holder 1.6 vs 2.5, d_x=2): the theoretical threshold is NOT visible at n <= 16000 — plug-in V covers 0.89-0.96 even when its sufficient condition fails; cross-p differences are K-rule artifacts | **Honest null**: the smoothness side conditions are conservative in practice; validity tracks realized first-stage fit (h_rmse), not the threshold |
| C4 | Effective dimension governs, not d_x | D2: invariance over d_x in {10,50,100}; rate exponents -0.45 to -0.49; flat sieve variance | **Supported** (theory sketch + formalization plan in THEORY_NOTE) |
| C5 | Practical pipeline valid end-to-end | S3: lasso screening recovers support 100% even at tau_scale=1; V 0.94-0.98; D4: CV tuning of (K, gamma) preserves V at 0.983 | **Supported** |
| C6 | Failures are visible/fixable | S4+S6A+S7: see failure-mode map below | **Mapped** |
| C7 | Appendix-D estimator repairs ML first stages | S5+S6B: GBM bias 0.086 -> 0.022; coverage 0.98 with 5 folds + augmented SE (0.36 naive) | **Supported**, with the augmented-SE refinement |

## The failure-mode map (C6)

| Violation | Behavior | Fix / guidance |
|---|---|---|
| Vanishing boundary gradient (Assn 2(c)) | **Graceful**: V covers 0.99, SE/SD 1.56 — the band widens automatically | none needed; document |
| Extreme share (V_true ~ 0.95) | **Transient**: one-sided flip bias, -0.06 at n=4000 -> covers 0.93 at n=16000; or LOO-V with delta0 ~ 0.2 removes it at n=4000 (bias -0.002, sd +30%) | larger n / better first stage, or tuned LOO-V; JTPA-relevant |
| Moderate overlap violation (p_min ~ 0.02) | V still 0.95; W degrades | monitor p_hat |
| Extreme overlap (p in (0.001, 0.999)) | **Silent catastrophic** (V coverage 0.00, SE does not warn) | common-support trimming restores V to 0.93 (S6A) — exactly the paper's empirical practice |
| Screening false negative | bias the SE cannot see; coverage 0.84 | union-conservative screening; report V across nested supports |

## Methodological refinements discovered by the sweep

1. **First-stage RMSE is the universal gatekeeper.** Across all ~40 cells, V
   coverage >= 0.93 whenever the (cross-validatable) first-stage RMSE is
   comfortably below sd(tau); every failure traces to first-stage bias. This is
   the practical diagnostic to recommend.
2. **Debias corrections need either negligible or accounted-for noise.**
   - W's LOO correction is smooth and low-noise: plug-in SE suffices.
   - The cross-fitted IF correction (Appendix D): use >= 5 folds and
     se_aug^2 = se_plug^2 + var_hat(correction terms)/n  (S6B: 0.79 -> 0.98).
   - V's LOO correction via central differences: the step delta0 matters
     enormously for indicator functionals — delta0 = 0.05 gives sd 0.10 (useless),
     delta0 = 0.2-0.5 gives sd 0.019-0.022 with full bias removal
     (`check_loo_v_delta.py`). Recommend delta0 ~ 0.2 x SD(h_hat).
   - The independence-heuristic variance for LOO-V fails (terms correlated
     through the common first stage) — do not use it; with delta0 tuned it is
     unnecessary.
3. **Band width iota in [0.01, 0.02]** cuts the V interval's conservativeness
   ~25-30% at no coverage cost (D4); scrambled Sobol is unnecessary.
4. **K-selection differs by functional**: CV-optimal K serves V; W prefers
   smaller K (or debiasing). Do not reuse one K blindly for both.

## The recommended procedure (as validated)

1. Tune (K, gamma) of the RF sieve by split-half CV of first-stage MSE.
2. If sparsity is plausible, lasso-screen coordinates (honest split optional
   under strong signals; mandatory caution under weak ones) and build features
   on the screened support.
3. Trim to common support when estimated propensities are extreme.
4. Report **plug-in V** with the eps-band sieve SE (iota ~ 0.01-0.02); switch to
   **LOO-V with delta0 ~ 0.2** when the estimated share is extreme (> ~0.9).
5. Report **LOO-debiased W** with the plug-in sieve SE.
6. With a generic ML first stage, use the Appendix-D cross-fitted IF estimator
   with >= 5 folds and the augmented SE.

## Limitations / open items

- Coverage MC error at 150 reps is ~ +/- 1.8%; headline cells could be rerun at
  500+ reps for publication tables.
- The Theorem-5 smoothness threshold for V could not be exhibited numerically
  at n <= 16000 (S2): an honest remark for the paper rather than a deliverable
  table. A sharper design (higher d_x with matched K paths, larger n) might
  show it; low priority.
- Oracle-propensity trimming in S6A isolates the mechanism; replicate with
  estimated p_hat.
- All evidence is for the RF sieve with uniform covariates and known target F;
  the unknown-F (sample-average) variants were validated only in the original
  B-spline simulations.
- Augmented-SE for the IF estimator is conservative by construction (ignores
  the negative covariance); a sharper variance is a small theory question.

## Verdict

The convergence criteria of `INVESTIGATION_LOG.md` are met: every claim is
supported or honestly bounded by dedicated experiments, both anomalies (A1
correction-noise, A2 extreme-share) are resolved with concrete fixes, and the
failure-mode map is complete. The simulation evidence is, in my assessment,
substantially convincing for: (i) a high-dimensional simulation section built
on the RF linear sieve (V valid to d_x = 100 under low effective dimension;
LOO-W; screening pipeline), and (ii) first numerical support for the
Appendix-D estimator. The natural next consumer is the JTPA application with
the full covariate set.
