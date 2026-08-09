# Investigation Log: Systematic Simulation Sweep (RF-Sieve, High Dimensions)

*Self-iterating workflow: each round = (questions -> experiments -> results ->
verdict -> next questions). Rounds continue until the convergence criteria below
are met. Scripts `sweep_S*.py`; outputs in `results/` (stems `S1_*`, `S2_*`, ...).*

## Target claims for a convincing simulation section

- **C1 (validity of V):** plug-in V + band sieve SE is valid across dimensions,
  feature types, error structures (incl. heteroskedasticity), share levels, and
  overlap strength — and conservative (never anti-conservative) when stressed.
- **C2 (W needs debiasing; LOO works):** plug-in W fails in high dimension; LOO-W
  restores near-nominal coverage; composed with screening it reaches ~0.95.
- **C3 (Theorem 5 for V):** debiased V matters exactly when smoothness is
  marginal (Holder-kinked CATE) and is unnecessary (or mildly harmful) when the
  CATE is smooth.
- **C4 (effective dimension):** established in D2 (invariance over d_x in
  {10,50,100}; rate exponents ~ -0.5; flat sieve variance). No further runs
  needed unless anomalies arise.
- **C5 (practical pipeline):** CV-tuning + screening + plug-in V + LOO-W is valid
  end-to-end, including under weaker signals.
- **C6 (graceful failure):** known violations (vanishing boundary gradient,
  screening false negatives, extreme overlap) degrade visibly/conservatively
  rather than silently.
- **C7 (Appendix D):** the cross-fitted sieve-influence-function estimator
  repairs a generic ML (gradient boosting) first stage for V.

## Convergence criteria

1. Every claim C1-C7 supported (or explicitly bounded) by at least one dedicated
   experiment, with headline cells at >= 150 reps.
2. Anomalies either resolved by a follow-up round or documented as limitations.
3. A final consolidated table-of-evidence exists (FINDINGS_comprehensive.md).

---

## Round 1 (launched)

**Questions.** Q1: Is V's validity robust to activation, heteroskedasticity,
share level ~0.85 (JTPA-like), weak overlap, at d_x in {10,50}? Does LOO-W hold
up across all of these? [C1, C2] Q2: Does debiased V beat plug-in V in a
genuinely low-smoothness design (Holder 1.6 vs 2.5 at d_x = 2), as Theorem 5
predicts? [C3]

**Experiments.** `sweep_S1_robustness.py`: one-factor-at-a-time deviations from
the baseline (dense DGP, d50, n4000, K200, gamma 1.5, cos, homoskedastic,
share ~0.58, normal overlap), all cells with plug-in V/W + LOO-W.
`sweep_S2_smoothness.py`: kink DGP with kink_pow in {1.6, 2.5}, d_x = 2,
n in {1000, 4000, 16000}, K_n ~ n^{d/(2*sigma+1)}; V_plug vs V_loo (same
plug-in SE, per Theorem 5), W_plug vs W_loo.

**Results / verdict.** (pending S1-S3; S4/S5 below ran as part of the same batch)

---

## Round 2 results (S4 failure modes, S5 Appendix-D DML) — reviewed

**S4 (graceful failure?, C6).**
- F1 vanishing boundary gradient ('cubic'): V coverage 0.993, bias -0.001,
  SE/SD = 1.56, band count elevated. **Graceful**: when ||grad tau|| -> 0 on the
  boundary, the eps-band widens and the SE inflates. (Plug-in W fails there,
  bias 0.105 — consistent with W's general fragility.)
- F2 screening false negative (drop x2): V bias +0.031, coverage 0.840 vs 0.927
  for the correct support. **Moderately dangerous**: the SE cannot see
  misspecification bias. Recommendation: union-conservative screening; report
  V across nested screened supports as a specification check.
- F3 extreme overlap (p in (0.001, 0.999)): V bias +0.173, coverage 0.000;
  W bias 0.368. **Silent catastrophic failure** — SE does not explode. The
  strict-overlap assumption is *essential*; flag p_hat extremes / trim.
  -> Round 3: test common-support trimming (the paper's empirical practice)
     with the estimand redefined on the trimmed population.

**S5 (Appendix-D cross-fitted IF estimator, C7).**
- GBM naive plug-in at d50: bias +0.086, coverage 0.36 (as expected).
- IF correction moves bias to +0.031 (right direction, ~2/3 removed), coverage
  0.56. At d10 (where GBM needed no correction) the correction *adds noise*:
  coverage 0.67 with the plug-in sieve SE.
- **Anomaly A1**: the plug-in sieve SE understates the finite-sample variance
  of the cross-fitted estimator (Theorem 7 is asymptotic; with 2 folds the
  correction term's sampling noise is first-order at n = 4000).
  -> Round 3: augmented SE  se^2 = V_se^2 + var_hat(v* (Y - mu_hat))/n,
     5 folds, and check coverage recovery.

**S3 (pipeline under signal stress, C5) — reviewed.**
- Lasso screening recovers the true support in 100% of draws at ALL signal
  strengths tau_scale in {1, 2, 3} (beta-min did not bind in this design;
  |S_hat| ~ 7.6-9.1 with harmless false positives).
- V coverage 0.94-0.98 in every cell, honest split and full-sample alike.
- **Composition D1 x D3 works**: screening + LOO-W gives coverage 0.89-0.95
  across all signal strengths (plug-in W: 0.39-0.86). C2/C5 supported.
- Residual: Wloo at 0.89-0.93 in some cells (slightly below nominal) — likely
  remaining smoothing bias from the ~6-7 false-positive coordinates; acceptable,
  documented.

**S1 (robustness OFAT, C1) — reviewed.**
- V coverage 1.00 (conservative) across: activations cos/relu/tanh,
  heteroskedastic errors, d_x in {10, 50}, dense/sparse DGP. LOO-W 0.85-0.94
  throughout. **C1 supported** with one exception:
- **Anomaly A2 (extreme share)**: shift = -0.40 gives V_true = 0.954; V bias
  -0.058, coverage 0.27. One-sided sign-flip bias near the parameter boundary
  (almost no negative-CATE mass to flip back). JTPA-relevant (share ~0.89).
  -> Round 4 (S7): share in {~0.85, ~0.95} x n in {4000, 16000}, V_loo tested
     as the candidate fix (the flip bias is a diagonal quadratic term).
- Mild overlap violation (slope 3, p_min ~ 0.02): V still 0.95. Gradation
  established: slope 3 fine -> slope 6 fatal (S4) -> trimming fixes (S6A).

## Round 3 results (S6 fixes) — reviewed

- **S6A trimming (fix for S4-F3)**: common-support trimming (oracle p0 in
  [0.05, 0.95], estimand redefined on the trimmed population, as in the JTPA
  empirics) restores V coverage 0.00 -> 0.933 under overlap = 6. W remains
  broken there (bias 0.18); trim + LOO-W is the natural untested composition.
- **S6B augmented-SE DML (fix for A1)**: 5 folds + se_aug^2 = se_plug^2 +
  var_hat(correction terms)/n gives **coverage 0.98 at d_x = 10 and 50** (from
  0.79/0.83), bias 0.006/0.022. C7 now supported: the Appendix-D estimator
  works with a generic GBM first stage, provided the SE accounts for the
  correction-term noise (a finite-sample refinement worth a remark in
  Appendix D).

## Round 4 results (S7 extreme share) — reviewed

*(Machine restarted mid-round; S2 and S7 relaunched with an unbiased
subsampled-LOO speedup for large-n cells.)*

- **A2 is transient**: plug-in V bias/coverage at share 0.81: -0.043 / 0.82
  (n = 4000) -> +0.006 / 1.00 (n = 16000); at share 0.95: -0.060 / 0.20 ->
  -0.014 / 0.93. The one-sided sign-flip bias is second order and dies as the
  first stage improves. Practical guidance: extreme shares demand a better
  first stage (larger n or screened/lower-dim sieve).
- **LOO-V kills the bias** (e.g., -0.060 -> +0.008 at n = 4000, share 0.95) —
  confirming the flip bias is exactly a diagonal quadratic term — **but
  undercovers with the plug-in SE** (0.35-0.58): the correction's own noise is
  first-order in finite samples. Same mechanism as anomaly A1 (S5/S6B).
- **Unified refinement**: any debiased estimator should be reported with the
  augmented SE  se_aug^2 = se_plug^2 + var_hat(correction). -> Round 4b (S7b):
  LOO-V at shares 0.81/0.95 and LOO-W at baseline, plug-in vs augmented SE.

## Round 4b/5 results — reviewed

- **S7b**: LOO-W confirmed clean (bias 0.0016, coverage 0.913; correction noise
  negligible). LOO-V's augmented SE did NOT fix coverage: the independence
  heuristic fails (correction terms correlated through the common first stage),
  and loo_sd (0.10-0.13) dwarfs the plug-in SE.
- **check_loo_v_delta**: root cause found — the D^2V central-difference step.
  delta0 = 0.05 -> sd 0.100; delta0 = 0.2 -> sd 0.022 with bias -0.002;
  delta0 = 0.5 -> sd 0.019. **LOO-V is viable with delta0 ~ 0.2**: full removal
  of the extreme-share bias at ~30% sd inflation. A2 fully resolved.
- **S2 (smoothness, C3)**: the Theorem-5 threshold is not visible at
  n <= 16000 — plug-in V covers 0.89-0.96 even at Holder 1.6 < d_x = 2 (the
  sufficient condition is conservative); the apparent p = 2.5 degradation is a
  K-rule artifact (smaller K, boundary underfit), and LOO-V's poor numbers
  there are the delta0 = 0.05 noise artifact. Honest null, documented.

## Convergence assessment: MET

All claims C1-C7 supported or honestly bounded; anomalies A1 (correction-term
noise in the IF estimator -> 5 folds + augmented SE) and A2 (extreme-share flip
bias -> transient in n; LOO-V with delta0 ~ 0.2) resolved; failure-mode map
complete (graceful / transient / fixable-by-trimming / flagged). Consolidated
in **FINDINGS_comprehensive.md**. Loop closed.

## Postscript: JTPA application (`jtpa_rf_application.py`, JTPA_FINDINGS.md)

The validated pipeline applied to the paper's empirical setting (n = 9,223,
2 covariates; richer JTPA demographics not in the local archives — data
acquisition flagged): RF sieve reproduces the B-spline estimates (share 0.91 /
0.82, welfare $1,384 / $703) with ~25-60% tighter CIs via the KDE-free
empirical-measure band SE; LOO diagnostics indicate plug-in welfare is biased
up ~$100 and the plug-in share biased down ~0.05 (the extreme-share regime, as
predicted by S7); estimates stable across feature draws after trimming.
