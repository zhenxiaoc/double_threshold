# Findings: Four Directions for High-Dimensional Estimation and Inference

*Follow-up to `FINDINGS_v_highdim.md`. Scripts `explore_D1_screening.py` ...
`explore_D4_tuning.py` + shared `rf_sieve_lib.py`; all summaries/draws in `results/`
(stems `D1_*` ... `D4_*`). Theory sketch in `THEORY_NOTE_effective_dim.md`.
150 reps (D1-D3) / 120 reps (D4) per cell; sparse DGP = signal in (x1, x2) with
d_x - 2 pure-noise covariates; dense DGP = two dense linear indices.*

## D1. Screen-then-sieve: it works, including honestly

Sparse DGP, d_x = 50. Lasso screening (Y on [X, D*X, D], coordinate selected if its
main or interaction coefficient is nonzero, capped at 10), then K = 50 random
features supported on the screened set:

| variant | n | S recovered | mean |S_hat| | h_rmse | V bias | V cover | W bias | W cover |
|---|---|---|---|---|---|---|---|---|
| dense features, no screen | 1000 | -- | (50) | 1.42 | -0.014 | 0.980 | 0.367 | 0.000 |
| screen, full sample | 1000 | 100% | 8.6 | 0.57 | +0.007 | 0.953 | 0.077 | 0.620 |
| screen on half, fit on half | 1000 | 100% | 8.1 | 0.84 | -0.001 | 0.967 | 0.152 | 0.467 |
| oracle support | 1000 | -- | 2 | 0.33 | +0.008 | 0.960 | 0.025 | 0.880 |
| dense features, no screen | 4000 | -- | (50) | 0.61 | +0.021 | 0.993 | 0.093 | 0.073 |
| screen, full sample | 4000 | 100% | 8.9 | 0.30 | +0.014 | 0.967 | 0.024 | 0.813 |
| screen on half, fit on half | 4000 | 100% | 8.6 | 0.39 | +0.012 | 0.967 | 0.036 | 0.813 |
| oracle support | 4000 | -- | 2 | 0.16 | +0.002 | 0.960 | 0.005 | 0.940 |

Takeaways: (i) the lasso recovers the true support in **every** draw at both n (strong
beta-min in this design); (ii) post-screening **V coverage is essentially nominal
(0.95-0.97)**, and -- notably -- full-sample screening is as good as the honest split
here, so the post-selection distortion is negligible under strong signals;
(iii) screening moves W from hopeless (0.00-0.07) to decent (0.81) at n = 4000;
the remaining gap to oracle (0.94) comes from the ~7 false-positive coordinates
keeping some first-stage noise. Combining screening with the D3 LOO debiasing of W
is the obvious next composition.

## D2. Effective dimension: the cylinder conjecture confirmed numerically

Sparse DGP, oracle 2-coordinate features, K = 50, d_x in {10, 50, 100},
n in {1000, 4000, 16000}:

- **Ambient-dimension invariance**: at each n, h_rmse, RMSE(V_hat), SE, and coverage
  are statistically indistinguishable across d_x = 10, 50, 100 (e.g., at n = 4000:
  V_rmse 0.0278 / 0.0274 / 0.0301; V coverage 0.91 / 0.95 / 0.93). The sieve
  variance n*Var_hat(V) is flat (~3.5-4.3) in both d_x and n.
- **Rate exponents**: log RMSE(V) ~ log n slopes are -0.470 / -0.492 / -0.452 for
  d_x = 10 / 50 / 100 -- effectively the n^{-1/2}-like rate expected for an analytic
  CATE, independent of ambient dimension.
- **W also works in every cell** (coverage 0.91-0.98), confirming that sparsity
  rescues the welfare functional through the first-stage channel.
- Coverage with the s-dimensional sieve is **well-calibrated** (0.91-0.97), unlike
  the conservative intervals under dense features -- the band SE's conservativeness
  is a high-dimensional-sieve phenomenon, not intrinsic.

These are exactly implications (i)-(iii) of the cylinder factorization in
`THEORY_NOTE_effective_dim.md`: the pathwise derivative of V only sees the
S-marginal average of the first-stage error, so the problem is effectively
s-dimensional. The note lays out the formalization plan (factorization lemma,
s-dimensional upper bound, embedded lower bound, index-sparsity version).

## D3. SS/LOO debiasing rescues W -- LOO almost completely

Dense DGP cells where plug-in W failed; all coverage with the same plug-in sieve SE:

| cell | W plug-in bias / cover | W SS bias / cover | W LOO bias / cover |
|---|---|---|---|
| d_x=10, n=4000, K=200 | +0.073 / 0.187 | +0.011 / 0.853 | **-0.0004 / 0.927** |
| d_x=50, n=4000, K=200 | +0.101 / 0.080 | +0.024 / 0.840 | **+0.010 / 0.900** |
| d_x=50, n=4000, K=400 | +0.172 / 0.000 | +0.048 / 0.553 | **-0.003 / 0.853** |

The LOO correction (exact OLS leverage residuals + central-difference D^2W) removes
the Jensen/ReLU bias almost entirely -- even in the worst cell (bias 0.172 -> -0.003)
-- and lifts coverage from 0.00-0.19 to 0.85-0.93. SS removes most but not all of the
bias (its half-sample fits are noisier). This numerically vindicates the Theorem-5
mechanism *applied to W* and makes "LOO-debiased welfare + RF sieve" a viable
high-dimensional procedure. Residual undercoverage (~0.85-0.93 vs 0.95) plausibly
reflects remaining smoothing bias and the unadjusted SE; combining with screening
(D1) or mildly undersmoothed K should close it.

**Caution (V):** applying SS debiasing to V *hurt* it here (coverage 0.57-0.63 vs
plug-in 0.99-1.00). Plug-in V was already fine -- its band SE self-normalizes -- so
the correction only adds half-sample noise, and pairing the SS point estimate with
the full-sample plug-in SE is internally inconsistent. Practical rule suggested by
the data: **debias W, plug-in V** (debiasing V matters only when smoothness is
genuinely marginal, which these analytic designs are not).

## D4. Data-driven tuning: CV on the first stage is a valid surrogate

Menu K in {50, 100, 200, 400} x gamma in {1.5, 3, 6}, selected per draw by
split-half CV of the first-stage MSE, then refit on the full sample (d_x = 50,
n = 4000):

- CV picks gamma = 1.5 in ~99-100% of draws (never gamma = 6, which was the broken
  config) and median K = 100-200.
- **Post-selection V coverage: 0.983** in both DGPs, bias ~0.01. So tuning the RF
  sieve by ordinary first-stage cross-validation -- exactly what a practitioner
  would do -- preserves V inference. (W stays poor under CV tuning, 0.28-0.45:
  CV-optimal smoothing is not what W's bias needs; W needs debiasing (D3) or
  screening (D1), not better MSE.)
- **iota sweep** (band width eps = iota * SD(h_hat)): SE/SD falls monotonically,
  2.7 -> 1.3 (dense) and 2.4 -> 1.16 (sparse) as iota goes 0.005 -> 0.05; coverage
  holds at 1.00 through iota = 0.02 but drops to 0.86 (sparse) at iota = 0.05.
  Recommendation: iota in [0.01, 0.02] -- cuts the conservativeness ~25-30% at no
  coverage cost; do not push to 0.05.
- **Scrambled vs plain Sobol**: indistinguishable (SE 0.0379 vs 0.0374, same
  coverage) -- a clean null; plain Sobol is fine even at d_x = 50.

## Synthesis: a coherent recipe for high-dimensional welfare/value inference

1. **Tune the RF sieve by first-stage CV** over (K, gamma) -- valid for V (D4).
2. **Screen coordinates when sparsity is plausible** -- lasso screening recovers
   support and restores near-oracle behavior; full-sample screening was as good as
   honest splitting under strong signals, but the split (or the Appendix-D
   cross-fitted sieve-influence-function estimator) is the theoretically safe
   default (D1).
3. **Report plug-in V with iota ~ 0.01-0.02 band SE** (D4); the interval is valid
   whenever the first stage tracks the boundary, and conservative otherwise.
4. **Report LOO-debiased W** with the plug-in SE (D3); never trust plug-in W in
   high dimensions without debiasing.
5. The theory target that organizes all of this: the **cylinder/effective-dimension
   framework** (D2 + theory note) -- V's rate exponent was never dimension-dependent;
   sparsity (coordinate- or index-) relaxes the smoothness side conditions from
   d_x to s, and the numerical evidence matches its testable implications exactly.

## Suggested next steps

- Compose D1 x D3: screening + LOO-debiased W (expect ~0.95 W coverage at d_x = 50).
- V_loo / V_ss with an SS-consistent variance (half-sample patty) to give debiased-V
  a fair test in a *low-smoothness* design (kinked CATE), where Theorem 5 predicts
  it should matter.
- Weak-signal screening: redo D1 with TAU_SCALE = 1 to stress beta-min; quantify
  when honest splitting starts to dominate full-sample screening.
- Port the pipeline into `opttreat` (estimator `rf_ols` + `screen` option;
  `CCGSieveVariance` unchanged) and replicate on the JTPA data with all available
  covariates (current empirics use 2 of them -- the natural showcase).
- Formalize the theory note: factorization lemma + s-dimensional Theorem 3/5
  analogs + embedded lower bound; index-sparsity version connecting to the
  Barron/RF approximation literature.
