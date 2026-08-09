# Simulation Study: The Surrogate-Induced Harm Share (a double-threshold value functional)

**Author engine:** Zhenxiao Chen JMP — companion simulation to *"Inference on Policy
Value under Admissibility-Constrained Optimal Treatment Rules."*
**Status:** self-contained, runnable (`run_study.py`, `make_figures.py`), calibrated to real data.

This study is a **second worked example** for the paper's theory, distinct from the dynamic
two-stage path-welfare example in `../dynamic_path_welfare`. Where that example has *nested*
thresholds in *different* spaces (`δ` on the intermediate state `X`, then `κ` on the baseline
state `S`), this one has **two thresholds on the same covariate vector**, producing a decision
region whose boundary has **two codimension-1 pieces meeting at a codimension-2 corner**. It is
the cleanest possible illustration of the double-threshold geometry and of the paper's
regular-vs-irregular distinction.

---

## 1. Parameter and interpretation

Let `X ∈ ℝ^d` be pre-treatment covariates, `S` a short-run (2-year) outcome, `Y` a long-run
(3-year) outcome, and `W ∈ {0,1}` a randomized treatment. Define the two conditional average
treatment effects (CATEs)

```
tau_S(x) = E[S(1) − S(0) | X = x]      (short-run effect)
tau_Y(x) = E[Y(1) − Y(0) | X = x]      (long-run effect)
```

A policymaker who only sees short-run outcomes uses the **short-run-optimal rule**
`pi*_S(x) = 1{tau_S(x) ≥ 0}`. The **surrogate-induced harm share** is

```
theta  =  Pr( tau_S(X) ≥ 0 , tau_Y(X) ≤ 0 )
       =  ∫ 1{tau_S(x) ≥ 0} 1{tau_Y(x) ≤ 0} f(x) dx
```

— the population share that is **treated because treatment looks good in the short run but is
actually harmed in the long run**. It is the generic double-threshold functional
`∫ 1{h0 ≥ 0} 1{g0 ≥ 0} φ f` with `h0 = tau_S`, `g0 = −tau_Y`, `φ ≡ 1`.

Reported alongside `theta` are the **four sign quadrants** (a policy confusion matrix)

| | `tau_Y ≥ 0` | `tau_Y < 0` |
|---|---|---|
| **`tau_S ≥ 0`** | `θ_{++}` correctly treated | `θ_{+−}` = **θ** harmed long-run |
| **`tau_S < 0`** | `θ_{−+}` withheld despite gain | `θ_{−−}` correctly untreated |

and the **conditional harm ratio** `ρ = θ_{+−}/(θ_{++}+θ_{+−}) = P(tau_Y<0 | tau_S≥0)` — the
fraction of the short-run-treated who are harmed long-run. Its denominator is the ordinary
single-threshold treat-share `Pr(tau_S≥0)`, which the paper's framework already studies; the
numerator is the new double-threshold object.

---

## 2. Why this illustrates the paper's theory

**Geometry.** The boundary of the harm region is `M_S ∪ M_Y` with

```
M_S = {tau_S = 0, tau_Y < 0}   (short-run threshold, restricted to long-run losers)
M_Y = {tau_Y = 0, tau_S > 0}   (long-run threshold, restricted to short-run winners)
```

meeting at the corner `C = {tau_S = 0, tau_Y = 0}` (codimension 2). This is the distinguishing
feature relative to the single-threshold treat-share `Pr(tau_S≥0)`, whose boundary is a single
surface `{tau_S=0}`.

**Moving-boundary derivative.** Perturbing `tau_S → tau_S + t·δ_S`, `tau_Y → tau_Y + t·δ_Y`, the
same coarea/moving-boundary formula used for `V_11^*` in `may_2026.tex` §2.4 gives **two
non-cancelling boundary integrals**

```
Dθ[δ_S,δ_Y] =  ∫_{M_S} δ_S f / ‖∇tau_S‖ dH^{d-1}   −   ∫_{M_Y} δ_Y f / ‖∇tau_Y‖ dH^{d-1}.
```

The `+` on `M_S` (raising `tau_S` **expands** `{tau_S≥0}`) and the `−` on `M_Y` (raising `tau_Y`
**shrinks** `{tau_Y≤0}`) are verified numerically (see §5). Under transversality
(`∇tau_S, ∇tau_Y` linearly independent on `C`) the corner is codimension 2 and contributes only
at second order, so it drops out of the first-order derivative.

**Regular vs irregular.** Because the boundary weights here are `f` (not the contrast `δ` that
*vanishes* on its own margin), the boundary terms do **not** cancel: `theta` is an **irregular**
(thin-set) functional with no √n influence function, converging at the Chen & Gao (2026)
codimension-1 rate `n^{−s/(2s+1)}`, exactly like the value functional `V(h)=∫1{h≥0}m` and unlike
total welfare. The study contrasts `theta` with a **regular companion**

```
W_Y = E[ max(tau_Y(X), 0) ]     (value of the self-optimal long-run rule),
```

whose derivative's boundary weight is `tau_Y`, which **vanishes** on `{tau_Y=0}` (the envelope
cancellation), so `W_Y` is √n-regular. Same DGP, same samples: `theta` irregular, `W_Y` regular.

**A finite-sample caveat (borne out by the study).** The regular/irregular distinction is
DGP-independent *at the level of the pathwise derivative* — the two boundary terms do not cancel
for `theta` but do for `W_Y`. The sub-√n *rate* `n^{−s/(2s+1)}`, however, is a **nonparametric,
minimax** statement: it is the rate of the boundary integral `∫_M (τ̂−τ)` and only governs the
plug-in once the sieve dimension `K` grows with `n`. On a *smooth* calibrated oracle estimated with
a *fixed* moderate `K`, the plug-in is effectively parametric — `thetâ` is ≈ √n with near-nominal
coverage — and the irregularity surfaces only in the growing-`K` / finite-smoothness regime (§5).
The study reports both regimes honestly rather than forcing a slow rate.

---

## 3. Calibrated DGP (the "graduation" oracle)

**Data.** Banerjee et al. (2015) poverty-graduation RCT, Pakistan/Sindh subset, shipped with the
Chen & Ritzwoller `longterm` R package as `data/graduation.rda` (854 households, treatment
randomized, 52% treated). We take `S =` 2-year per-capita monthly total consumption
(`ctotal_pcmonth_end`), `Y =` 3-year (`ctotal_pcmonth_fup`) — the same economic outcome at two
horizons — and `d=2` baseline covariates: baseline consumption and baseline asset index. The
program's average effect **fades** from +12.3 (2-yr) to +6.0 (3-yr), and the CATEs are only
weakly correlated (≈0.3–0.5) across horizons, so a genuine harm quadrant exists.

**Oracle.** All covariates are pushed through a Gaussian quantile transform (outlier-robust,
clean support), clipped to `[−3,3]^2`. We fit smooth kernel-ridge conditional means
`μ_{S,w}(x), μ_{Y,w}(x)` per arm and define the oracle CATE surfaces
`tau_S = μ_{S,1}−μ_{S,0}`, `tau_Y = μ_{Y,1}−μ_{Y,0}`. The covariate density `f` is a Gaussian KDE
on the transformed covariates. Because `theta` is a functional of these surfaces and `f`, the
**population truth is computed, not estimated**, by 1.5M-draw Monte Carlo and by 500×500 grid
quadrature.

**Relationship to Chen & Ritzwoller's GAN calibration (their App. D.2/D.3).** CR fit three
cascaded WGANs — `X`, then `S|X,W`, then `Y|S,X,W` — with the `ds-wgan` package and take 10^7
draws as the population (a job they report at ≈60 CPU-years on a cluster). Our oracle reproduces
the **same cascade structure** `X → (S(0),S(1)) → (Y(0),Y(1))` with smooth conditional means and
nonparametric residual resampling in place of the WGANs; the short→long coupling is preserved
without altering `tau_Y(x)`. This runs in CPU-seconds and gives an *exact* truth, because
`theta` depends only on the conditional means and `f` — the GAN's role (realistic conditional
*shape*) does not move the truth. A GPU WGAN backend is a drop-in future upgrade; it would change
finite-sample noise, not the estimand.

**Noise knob.** Consumption has a low CATE signal-to-noise ratio (ATE≈11 on residual SD≈53). The
oracle exposes `noise_scale`: the **surfaces and truth are fixed**, and `noise_scale` multiplies
the resampled residuals. The primary DGP uses `noise_scale=0.34` (SNR≈1, the CCG regime, for a
clean methods demonstration); `noise_scale=1.0` (the realistic low-SNR case) is reported as
robustness. This is a standard, clearly-labelled design choice, not a change to the target.

---

## 4. Estimator and inference

**Sieve nuisances (CCG-aligned).** We reuse the Chen–Chen–Gao `opttreat` `SieveEstimator`
(tensor B-spline basis, degree 3, `pinv` solver, separate treated/control fits with analytic
gradients) to fit **two** CATE surfaces, then plug in

```
theta_hat = P_n[ 1{tau_hat_S(X) ≥ 0} 1{tau_hat_Y(X) < 0} ]   (tie → treated, per the paper).
```

Using a sieve (rather than a black-box learner) keeps the estimator inside the paper's theory and
is far cheaper than tree-based DML — the whole MC is dense linear algebra, hence GPU-batchable.

**Two-band sieve-Riesz variance (NEW — this project's derivation).** `opttreat`'s `SieveVariance`
covers only the *single*-threshold value functional. We derive the double-threshold analogue: the
Riesz derivative vector `Bun` is a sum of **two boundary bands**,

```
Bun_S = +(1/2ε_S)·mean[ 1{|tau_hat_S|<ε_S, tau_hat_Y<0} · b_S ]      (S surface, + sign)
Bun_Y = −(1/2ε_Y)·mean[ 1{|tau_hat_Y|<ε_Y, tau_hat_S≥0} · b_Y ]      (Y surface, − sign)
```

with `b = [ψ_t, −ψ_c]`. Per-arm Riesz weights `w = G^{-1}Bun`, influence `e·(ψ w)/n_a`, and the
S- and Y-contributions are **summed per unit before squaring**, capturing the within-unit
`(S,Y)` covariance. **This extension is not validated by prior code; we validate it by Monte
Carlo** (the `se_ratio = mean(SE)/MC-SD` and its coverage). The `may_2026.tex` §5 that would host
the analytic double-threshold variance is currently a stub — so this is genuinely new territory.

**Primary interval = full-refit nonparametric bootstrap** (re-estimate both CATE surfaces on each
resample). It reflects the boundary-location uncertainty a delta-method SE misses and does not
rely on the unverified analytic derivation. The two-band sieve SE is reported alongside with an
honest `se_ratio`.

---

## 5. What the study runs (`run_study.py`)

1. **Population truth** — `theta`, the four quadrants, `ρ`, treat-share, ATEs (1.5M MC + grid).
2. **Geometry diagnostics** — `‖∇tau_S‖` on `M_S`, `‖∇tau_Y‖` on `M_Y` (both bounded away from
   0 → regular margins), the corner transversality `|cos∠(∇tau_S,∇tau_Y)|` (< 1 → transversal),
   and the `H^{d-1}` lengths of `M_S, M_Y`.
3. **Two-boundary derivative verification** — analytic narrow-band (`+D_MS`, `−D_MY`) vs central
   finite differences of `theta(t)`, for `tau_S`-only, `tau_Y`-only, and combined perturbations.
4. **Rate experiment** — `n ∈ {1000,2000,4000,8000,16000}`, 500 reps: bias, MC-SD, RMSE, and
   log-log slopes for `theta` (irregular), the treat-share (irregular), and `W_Y` (regular).
5. **Sieve-dimension (K) sensitivity** — `segments ∈ {1,2,3}` at `n=4000`.
6. **Bootstrap coverage** — full-refit bootstrap at `n ∈ {2000,8000}`.
7. **Robustness** — the realistic low-SNR DGP (`noise_scale=1.0`).

Figures (`make_figures.py`): (1) the two surfaces with `M_S`, `M_Y`, harm region, corner;
(2) the sign-quadrant scatter + confusion matrix; (3) the convergence-rate log-log plot;
(4) the two-band-SE validation and coverage.

---

## 6. Guard-rails (honest reporting)

- The truth is computed from the oracle, never from an estimate.
- `theta` is never given a naïve √n SE; its irregularity is stated and the interval is a
  full-refit bootstrap, with the analytic two-band SE clearly labelled as a **new, MC-validated**
  extension.
- The `noise_scale` design choice is stated explicitly; both SNR regimes are reported.
- Any undercoverage is decomposed into bias vs variance (`se_ratio`, `|bias|/SD`) rather than
  hidden — undercoverage from plug-in bias is the expected irregular-functional behaviour and
  motivates bias correction (undersmoothing / a leave-one-out debias, as in `opttreat`).
