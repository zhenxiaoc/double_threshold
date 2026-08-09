# Note for Zhenxiao: Simulation Issues, Sweep Results, and JTPA Findings

*Updated June 10, 2026. Original note (June 9) covered Part 1 (code-verification
items) and a list of proposed simulations; the proposals have since been
**implemented and run** — Parts 2-4 below summarize what was found. All scripts,
results, and detailed write-ups are in `ClaudeWS/rf_sieve/` (see Part 5 for the
file map). The revised draft is `ClaudeWS/main.tex`.*

---

## Part 1: Verification items on the EXISTING code (still pending, unchanged)

### 1.1 Welfare variance formula (HIGH PRIORITY — affects JTPA empirics)

Theorem 1's variance was corrected in the draft: the old formula

    sigma_W^2 = E[ 1{h0>=0} lambda^2 sigma_eps^2(X) / (p0(1-p0)) ]

is valid only if E[eps^2|X,D] does not depend on D. The general form (now in
Theorem 1) weights the two arms separately:

    sigma_W^2 = E[ 1{h0>=0} lambda^2 ( sigma_eps^2(X,1)/p0 + sigma_eps^2(X,0)/(1-p0) ) ].

- Existing simulation tables remain valid (homoskedastic designs).
- **JTPA welfare SEs should be recomputed with the general formula** (earnings
  data is almost certainly heteroskedastic across arms).
- Confirm the code uses residuals `Y_i - mu_hat(X_i, D_i)` (an earlier draft
  wrote `Y_i - h_hat(X_i)`).

### 1.2 DML appendix description vs table mismatch

Appendix B.4's text previously described averaging the *indicator*; the table
reports *welfare* values. I changed the text to `[.]_+` to match the table —
**please verify against the DML code** (if the code averaged the indicator, the
table is mislabeled instead).

### 1.3 Sieve score bootstrap formula

Fixed three typos in the displayed Z_n^* (missing inverse on (B'B/n), argument
[nu] -> [psi_bar], a double-counted sqrt(n)). **Check the code that produced the
1.866 critical value** against the corrected formula.

### 1.4 Smaller table fixes (keep if regenerating)

V_true headers in the Model-15 sensitivity tables; "Models 8-14 (GAM)" caption;
nu_0 -> v_0; W_0 -> W_true; folds notation K -> \mathcal{K}; covariate
dimension renamed d_x throughout the draft.

---

## Part 2: What the new simulations found (headline results)

Full details: `FINDINGS_v_highdim.md`, `FINDINGS_directions.md`, and the final
consolidation `FINDINGS_comprehensive.md` (evidence table by claim +
failure-mode map). The investigation was run as logged rounds
(`INVESTIGATION_LOG.md`) until each claim was supported or bounded.

1. **V inference works in high dimension with a random-feature (RF) linear
   sieve.** Per-arm OLS on random cos features + the paper's eps-band sieve SE:
   coverage 0.93-1.00 at d_x = 50 (even d_x = 100 with structure), where tensor
   B-splines cannot be built. Robust to relu/tanh features, heteroskedastic
   errors, CV-selected tuning.
2. **W and V are fundamentally different in practice** (as we suspected):
   plug-in W fails in essentially every high-dim cell (Jensen/ReLU bias under a
   sqrt(n)-scaled SE), while V's band SE self-normalizes. **LOO debiasing
   rescues W** almost completely (e.g., bias 0.17 -> -0.003; coverage 0.00 ->
   0.85-0.93), and composed with screening reaches 0.89-0.95.
3. **Effective dimension, not ambient dimension, governs V.** With an
   s-supported sieve, RMSE/SE/coverage are *invariant* over d_x in {10,50,100}
   and the empirical rate exponent is ~ -0.5. Theory sketch + formalization
   plan in `THEORY_NOTE_effective_dim.md` (cylinder factorization of the
   boundary integral; sparsity relaxes the smoothness side conditions from d_x
   to s — the rate exponent never depended on d_x).
4. **Screen-then-sieve works.** Lasso screening (Y on [X, D*X, D]) recovered
   the true support in 100% of draws at all signal strengths; post-screening V
   is nominal; honest split ~ full-sample under strong signals.
5. **Appendix-D cross-fitted IF estimator works with a GBM first stage** —
   bias 0.086 -> 0.022 — but needs >= 5 folds and an **augmented SE**
   (se_aug^2 = se_plug^2 + var_hat(correction)/n): coverage 0.36 (naive) ->
   0.98. Worth a finite-sample remark in Appendix D.
6. **Failure-mode map** (for the paper's discussion): vanishing boundary
   gradient = graceful (SE widens); extreme share (V ~ 0.95) = transient
   one-sided bias, gone by n = 16000 or via tuned LOO-V; extreme overlap =
   SILENT failure, fixed by common-support trimming (coverage 0.00 -> 0.93);
   screening false negatives = the one quiet danger (coverage 0.84) — use
   union-conservative screening.
7. **Tuning guidance** (validated): CV on first-stage MSE for (K, gamma) is a
   valid surrogate for V (post-selection coverage 0.98); band width iota in
   [0.01, 0.02]; **delta0 ~ 0.2 x SD(h_hat)** for LOO-V central differences
   (0.05 makes the correction uselessly noisy — sd 0.10 vs 0.022); scrambled
   Sobol unnecessary.
8. **One honest null (S2)**: the Theorem-5 smoothness threshold for V (sigma
   between (d_x+1)/2 and d_x, kinked CATE at d_x = 2) is NOT visible at
   n <= 16000 — plug-in V covers ~0.9-0.96 even when its sufficient condition
   fails. Recommend a remark ("side conditions are conservative; validity
   tracks realized first-stage fit") rather than a table.

**The recommended procedure** (boxed in `FINDINGS_comprehensive.md`): CV-tune
(K, gamma) -> screen if sparsity plausible -> trim if propensities extreme ->
plug-in V with band SE (LOO-V with delta0 ~ 0.2 if share > ~0.9) -> LOO-W
always -> Appendix-D IF estimator with augmented SE for generic ML first stages.

---

## Part 3: JTPA application (new — `JTPA_FINDINGS.md`)

Applied the validated pipeline to KT_Data1.csv (the paper's exact setting;
note: neither local JTPA file has covariates beyond prevearn/edu — getting the
full NJS demographics is a **data acquisition item** if we want the "all
covariates" showcase).

| trimmed, n = 9,220 | Share | Welfare gain |
|---|---|---|
| RF sieve, no cost | 0.914 (0.795, 1.034) | $1,384 ($777, $1,990) |
| paper B-spline | 0.89 (0.73, 1.05) | $1,519 ($764, $2,274) |
| RF sieve, $774 cost | 0.823 (0.717, 0.929) | $703 ($145, $1,260) |
| paper B-spline | 0.80 (0.53, 1.07) | $858 ($152, $1,564) |

- **Cross-sieve validation**: a completely different sieve reproduces our
  B-spline numbers (within 0.02 / ~$150). Strong robustness story.
- **Tighter CIs without KDE**: the band SE computed on the *empirical measure*
  (valid in the F = F_0 case, Theorem 6) needs no kernel density estimation —
  CI widths 0.24/0.21 vs 0.32/0.54 — and would let us drop the Hscv/bandwidth
  machinery (and its sensitivity appendix) from the implementation.
- **Debiasing diagnostics**: LOO-W says plug-in welfare is biased UP ~$100
  (debiased: $1,281 / $594); LOO-V (delta0 = 0.2) says the share is biased
  DOWN ~0.05 (debiased: 0.965 / 0.882) — JTPA's share sits exactly in the
  extreme-share regime, and its first stage is very noisy (CV-RMSE ~$15.6k vs
  sd(h_hat) ~$1.6k), so these corrections are economically meaningful.
- Estimates are stable across RF feature draws (<= 0.007 / $26 after trimming).

Suggested paper use: an RF-sieve robustness column next to the B-spline
results + the LOO rows as bias diagnostics; consider switching sigma_V to the
empirical-measure band derivative.

---

## Part 4: Suggested division of labor / next steps

1. **(You) Part 1 verifications** against the R code — unchanged, still the
   top priority since 1.1 affects the published JTPA SEs.
2. **(You) Port the pipeline into `opttreat`**: add `solver="pinv"` to
   `rf_ridge` (OLS-on-RF); `CCGSieveVariance` works unchanged with RF feature
   maps; add the empirical-measure bun for the unknown-F case; LOO debias as a
   post-estimation option (closed-form leverage `e_i/(1-H_ii)`, central
   differences with delta0 = 0.05 for W / 0.2 for V).
3. **(You) Production reruns** of the headline cells at 500+ reps for paper
   tables (current sweeps: 150-200 reps, MC error ~ +/-1.8% on coverage).
4. **(Either) JTPA with the corrected sigma_W** (item 1.1) and, if we get the
   NJS demographics, the full-covariate screening showcase.
5. **(Wayne/XC) Theory**: the effective-dimension formalization
   (`THEORY_NOTE_effective_dim.md` has the lemma/theorem plan) and the
   augmented-SE refinement for the Appendix-D estimator.
6. Replicate S6A trimming with *estimated* propensities (oracle p0 was used to
   isolate the mechanism).

## Part 5: File map (`ClaudeWS/rf_sieve/`)

- `rf_sieve_lib.py` — shared library (DGPs, features, OLS, band inference,
  SS/LOO debias incl. subsampled LOO + variance, lasso screening).
- `rf_sieve_highd_sim.py`, `rf_sieve_v_highdim_explore.py` — base explorations.
- `explore_D1..D4_*.py` — screening / effective dim / W-debias / tuning.
- `sweep_S1..S7b_*.py`, `check_loo_v_delta.py` — the systematic sweep.
- `jtpa_rf_application.py` — the empirical application.
- Write-ups: `FINDINGS_v_highdim.md`, `FINDINGS_directions.md`,
  `FINDINGS_comprehensive.md` (final consolidation), `THEORY_NOTE_effective_dim.md`,
  `JTPA_FINDINGS.md`, `INVESTIGATION_LOG.md` (round-by-round audit trail).
- `results/` — all summary/draws CSVs (stems D1-D4, S1-S7b, JTPA_rf_results).

— prepared by Claude (Wayne's session), June 10, 2026
