# Random-Feature (Shallow NN) Linear Sieve: High-Dimensional Exploration

*Exploration scripts drafted in `ClaudeWS/rf_sieve/` (June 2026). Companion note to the
revised draft `ClaudeWS/main.tex` and the existing code base in `OptTreat/`.*

## 1. The idea

A shallow neural network with **random first-layer weights** and a trained linear output
layer is a **linear series (sieve) estimator**: conditional on the random draw
`(w_k, b_k)_{k=1..K}`, the features

    psi_k(x) = act( gamma * (w_k' x + b_k) ),    (w_k, b_k) iid Uniform(S^{d_x}),

are a fixed K-dimensional basis, and the second stage is OLS. Consequently the paper's
entire inference apparatus applies **verbatim**, with `psi` replacing B-splines:

- stacked treated/control regression (eq. `Sieve_OLS` in the draft);
- sieve variance `sigma_V_hat^2 = Bun' * Patty * Bun`, where `Patty` is the OLS sandwich
  `(B'B)^- B' diag(e^2) B (B'B)^-` and `Bun` is the pathwise-derivative vector — the
  indicator average for the welfare functional W, the eps-band average for the value
  functional V (Section 4.1 of the draft);
- LOO debiasing for V (Theorem 5), since the leverage formula `e_i/(1-H_ii)` is exact
  for OLS.

The payoff is **dimension scalability**: tensor-product B-splines need K ~ (basis per
coordinate)^{d_x}, which dies around d_x = 3-4, while random features live on a K-budget
chosen freely, and approximate ridge/single-index ("Barron-class") functions at rates
that are much less sensitive to d_x. The natural target DGPs are therefore CATEs built
from a few ridge functions — exactly what `HighDimDGP` in the script implements.

## 2. What is already in the code base (and what was missing)

`OptTreat/Python Codes/opttreat` already contains:

- random-feature builders (`estimation/features/`: iid sphere, quasi sphere, activations)
  and an `rf_ridge` first-stage estimator;
- a feature-map-agnostic sieve variance class `variance/ccg_sieve_var.py`
  (`CCGSieveVariance`) implementing exactly the Patty/Bun formulas;
- high-D RF simulations (`simulations/high_D_tan2/`, `simulations/TaylorModel/`) — but
  these are **estimation-only** (no SEs/CIs/coverage) and use **ridge** (alpha > 0).

What was missing for the "RF = linear sieve" story:
1. an **OLS** (pinv) second stage — the sieve theory is about least squares, and the
   `Patty` sandwich is the OLS covariance; with ridge the two are mismatched;
2. the **value functional + eps-band inference in high d_x** (the high-D runs covered
   welfare point estimation only);
3. **coverage** experiments, and the LOO debiased V.

`rf_sieve_highd_sim.py` in this folder fills these three gaps in one self-contained
script (numpy/scipy/pandas only; no opttreat import, so it runs anywhere).

## 3. What the script does

- **DGP** (`HighDimDGP`, any d_x): X ~ U[0,1]^{d_x}; CATE = `TAU_SCALE` x (sum of two
  sigmoid ridge functions − const), so the boundary {tau = 0} is a smooth curved
  (d_x−1)-manifold with nonvanishing gradient (Assumption 2(c)); logistic propensity in
  two indices with comfortable overlap; optional treatment-arm heteroskedasticity
  (`HETEROSKEDASTIC = True`) — the sieve sandwich variance is automatically robust to it
  (cf. the corrected sigma_W^2 discussion after Theorem 1 of the draft).
- **First stage**: per-arm OLS (pinv) on shared random features + intercept.
- **Functionals**: W and V (= treated share, v0 = 1) under known F = U[0,1]^{d_x},
  evaluated by Sobol quasi-MC; truth by a larger Sobol run.
- **Inference**: sieve SEs as above; 95% CI coverage recorded per cell.
- **Optional** (`RUN_LOO_DEBIAS = True`): LOO debiased V with D^2V computed by
  second-order central differences on Sobol points (Remark "Numerical Implementation of
  D^2V" in the draft) and exact OLS leave-one-out residuals.
- **Diagnostic**: `n_varV_mean` = n * Var_hat(V_hat) per cell. For B-splines the theory
  gives sigma_{V,n}^2 ~ K^{1/d_x}; tracking this quantity across (n, K, d_x) gives an
  *empirical* read on the sieve-Riesz-norm growth for the RF basis (see open question
  Q1 below).

Defaults are smoke-sized (runs in seconds); `PAPER`-scale suggestions are in the
comments of the CONFIG block.

## 4. Findings from the smoke run (worth knowing before scaling up)

(50 reps, d_x in {3,5}, n in {500, 1000}, K in {25, 50}, cos features, gamma = 3)

1. **V inference works out of the box**: coverage 0.92-1.00 in every cell, mild
   conservativeness (SE somewhat above SD). This is the headline: the eps-band sieve
   t-statistic for the irregular functional appears valid with a random-feature basis.
2. **W is the delicate one**: `W(h_hat)` has an upward Jensen/ReLU bias of order
   E[(h_hat - tau)^2] concentrated near the boundary. With weak signal (TAU_SCALE = 1,
   i.e., |tau| <~ 0.5 vs sd(eps) = 1) and K/n_arm ~ 0.15, this bias dominated and W
   coverage collapsed (down to 0.04!). With TAU_SCALE = 3 it is moderate and shrinking
   in n, but still visibly worse for K = 50 than K = 25.
   - Practical implication: **K-selection should differ by functional** — W wants a
     smaller K (its bias is second-order in the first-stage error), while V tolerates
     larger K (its band-based SE grows along with the noise). This mirrors the
     undersmoothing discussions in the draft and is itself a presentable finding.
3. The LOO correction for V runs but is noisy at tiny (n, M_SOBOL); it is a
   second-order correction and needs paper-scale M_SOBOL (>= 32768) to be meaningful.

## 5. Open theory questions to discuss with Xiaohong / Wayne

- **Q1 (sieve Riesz growth)**: the K^{(d_x-m)/d_x} growth of ||v*_K||^2 in Chen & Gao
  is derived for *local* bases (B-splines) via the partition-of-unity/decomposition
  argument. Random features are **global** basis functions; does the same growth rate
  hold (conditional on the feature draw)? The `n_varV_mean` diagnostic estimates this
  empirically: regress log(n_varV) on log(K) by dimension.
- **Q2 (conditioning)**: the cleanest framing treats inference as **conditional on the
  realized features** — then the basis is deterministic and the stacked-OLS theory
  applies as-is; unconditionally, the basis is exchangeable-random and K plays the role
  of the sieve dimension. Worth one remark in any write-up.
- **Q3 (approximation rates)**: for CATEs in a Barron-type class, RF approximation
  error decays like K^{-1/2} *independent of d_x* (Barron/random-feature literature;
  e.g., Rahimi-Recht). Plugging this into the bias-variance trade-off suggests
  rate-optimal K and attainable rates for V that escape the d_x-exponent — potentially
  a new theoretical section. The minimax lower bound (Theorem 4) is for Holder balls;
  for a Barron-ball parameter space the lower bound would also change.
- **Q4 (ridge vs OLS)**: `rf_ridge` with small alpha ~ OLS, but the Patty sandwich is
  the OLS covariance. Either set alpha -> 0 (use pinv, as here) or derive the
  ridge-adjusted sandwich `(B'B + aI)^{-1} B' diag(e^2) B (B'B + aI)^{-1}` — a one-line
  change in `CCGSieveVariance` worth making in the package.

## 6. Porting into `opttreat` (suggested, NOT done here — outside ClaudeWS)

1. `estimation/rf_ridge.py`: add `solver="pinv"` option (mirroring `sieve.py`) so the
   estimator is genuinely OLS-on-random-features; keep `alpha` for comparison runs.
2. `variance/ccg_sieve_var.py`: works unchanged for welfare/value with RF feature maps
   (it already consumes `feature_map_t/c` from the estimator output). Optionally add the
   ridge-adjusted sandwich (Q4).
3. New runnable: `simulations/high_D_rf_sieve/run_high_d_rf_sieve.py` — port of this
   script into the `SimulationSpec`/`simulation_engine` framework, with
   `variance_config=ccg_sieve_var` instead of `None`.
4. Models: add the ridge-function DGP here as a `ModelBase` subclass (or extend
   `TaylorExpansionModel`) so it is reusable; the existing tan2 Taylor designs can be
   rerun with inference turned on for continuity with the earlier estimation-only runs.

## 7. Files

- `rf_sieve_highd_sim.py` — the simulation script (self-contained; smoke defaults).
- `rf_sieve_v_highdim_explore.py` — follow-up: V-functional inference at d_x = 10 and 50,
  dense vs sparse DGPs, dense vs support-sparse vs oracle features, gamma sensitivity.
- `FINDINGS_v_highdim.md` — write-up of the d_x = 50 results and proposed directions
  (screen-then-sieve, cylinder-boundary effective-dimension theory, SS/LOO for W).
- `rf_sieve_lib.py` — shared library (DGPs, feature maps, OLS, inference, SS/LOO
  debiasing, lasso screening) used by the four direction-exploration scripts.
- `explore_D1_screening.py` ... `explore_D4_tuning.py` — the four directions:
  screening, effective dimension, W debiasing, data-driven tuning.
- `FINDINGS_directions.md` — consolidated results of D1–D4 and a practical recipe.
- `THEORY_NOTE_effective_dim.md` — cylinder-factorization theory sketch and
  formalization plan for the effective-dimension conjecture.
- `sweep_S1`–`sweep_S7b` + `check_loo_v_delta.py` — the systematic sweep
  (robustness, smoothness, pipeline stress, failure modes, Appendix-D DML,
  fixes and follow-ups), orchestrated via `INVESTIGATION_LOG.md`.
- `FINDINGS_comprehensive.md` — final consolidated evidence table, failure-mode
  map, methodological refinements, and the validated procedure.
- `results/` — outputs (`*_summary_*.csv`, `*_draws_*.csv`, `*_results_*.md`) following
  the opttreat naming convention.
