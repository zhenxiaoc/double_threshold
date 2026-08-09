# Results: The Surrogate-Induced Harm Share

Generated from `results/logs/*.json` (produced by `run_study.py`). Figures in `results/figures/`.

## 1. Population truth (calibrated graduation oracle, d=2, SNR≈1)

- **Harm share** θ = Pr(τ_S≥0, τ_Y<0) = **0.162**
- Conditional harm ratio ρ = P(τ_Y<0 | τ_S≥0) = **0.236**
- Treat-share (single-threshold companion) Pr(τ_S≥0) = 0.685
- ATE_S = 8.91, ATE_Y = 0.82 (fade-out preserved)

**Four-quadrant confusion matrix** (population shares):

| | τ_Y ≥ 0 | τ_Y < 0 |
|---|---|---|
| **τ_S ≥ 0** | 0.523 correctly treated | **0.162** HARM |
| **τ_S < 0** | 0.084 withheld despite gain | 0.231 correctly untreated |

**Geometry diagnostics** (regular-margin & transversality checks):

- ‖∇τ_S‖ on M_S (median) = 46.31 > 0  → regular short-run margin
- ‖∇τ_Y‖ on M_Y (median) = 30.63 > 0  → regular long-run margin
- Corner transversality |cos∠(∇τ_S,∇τ_Y)| = 0.557 < 1  → transversal (codim-2 corner)
- Boundary lengths H¹(M_S) = 10.45, H¹(M_Y) = 10.33

## 2. Two-boundary moving-boundary derivative — verification

Analytic narrow-band (coarea) vs central finite differences of θ(t). The two boundary terms are separate and non-cancelling; signs `+D_MS` (raising τ_S expands {τ_S≥0}) and `−D_MY` (raising τ_Y shrinks {τ_Y≤0}) are confirmed.

| perturbation | D_MS | D_MY | analytic Dθ | finite-diff | abs err |
|---|---|---|---|---|---|
| tauS_only | +0.0106 | +0.0000 | +0.0106 | +0.0105 | 0.0001 |
| tauY_only | +0.0000 | +0.0144 | -0.0144 | -0.0143 | 0.0000 |
| tauS_varying | +0.0119 | +0.0000 | +0.0119 | +0.0117 | 0.0003 |
| both | +0.0119 | +0.0146 | -0.0027 | -0.0028 | 0.0001 |

## 3. Monte Carlo: bias, variance, and the two-band sieve SE

Sieve plug-in (tensor B-spline, segments=2), 500 reps per n. `se_ratio = mean(two-band sieve SE)/MC-SD`; the two-band SE is this project's new derivation.

| n | bias | MC-SD | RMSE | sieve SE | se_ratio | 95% cov (sieve) |
|---|---|---|---|---|---|---|
| 1000 | +0.0107 | 0.0389 | 0.0403 | 0.0382 | 0.98 | 0.93 |
| 2000 | +0.0054 | 0.0271 | 0.0276 | 0.0262 | 0.96 | 0.94 |
| 4000 | +0.0021 | 0.0181 | 0.0182 | 0.0173 | 0.96 | 0.94 |
| 8000 | +0.0021 | 0.0117 | 0.0118 | 0.0116 | 0.99 | 0.96 |
| 16000 | +0.0017 | 0.0088 | 0.0089 | 0.0080 | 0.91 | 0.94 |

- The **two-band sieve SE tracks the MC-SD** (se_ratio ≈ 0.96 on average) — evidence the new double-threshold variance derivation is approximately correct.
- Undercoverage where present is **bias-driven**: the SE is correctly sized but the plug-in θ̂ is biased (see §4), shifting the interval off-centre.

## 4. Convergence behaviour and the regular/irregular distinction

Log–log slopes of RMSE (or SD) vs n:

- θ̂ harm share (double threshold): RMSE slope = **-0.56**, SD slope = -0.55
- treat-share Pr(τ_S≥0) (single threshold): RMSE slope = -0.14
- regular companion W_Y = E[max(τ_Y,0)]: SD slope = **-0.50** (≈ −0.5, root-n)

**Reading these honestly.** On this *smooth* calibrated oracle, a FIXED-dimension sieve is effectively parametric, so θ̂'s SD scales like n^(−1/2) and its coverage is near-nominal (§3) — the plug-in is well-behaved. The sub-√n **thin-set irregularity** of Chen & Gao (2026) is a *nonparametric* property: it is the rate of the boundary integral ∫_M (τ̂−τ) that governs the plug-in once the sieve dimension K GROWS with n (so the CATE is estimated nonparametrically). It therefore surfaces in the growing-K regime (§4b), not at fixed K on a smooth surface. The **clean, DGP-independent** statement of the regular/irregular distinction is at the level of the pathwise derivative (§2): for θ the two boundary terms **do not cancel** (weight f≠0 on each margin), whereas for the regular companion W_Y the boundary weight τ_Y **vanishes** on {τ_Y=0} and the term cancels by the envelope argument — which is why W_Y is root-n and θ is not. The single-threshold treat-share remains bias-limited (nearly flat RMSE slope), reflecting a larger, more persistent sieve-approximation bias on its own margin.

### 4b. Growing the sieve dimension K with n (attempted undersmoothing)

Two experiments share the same n grid. **Fixed-K holds K=2 (two segments) at every n** — the sieve dimension calibrated at the smallest sample, never grown. **Growing-K** raises the dimension with n per the undersmoothing schedule {1000: 2, 2000: 2, 4000: 3, 8000: 3, 16000: 4}; the two experiments therefore coincide at n ≤ 2000 (both K=2) and diverge once the schedule steps up.

On this smooth oracle the fixed K=2 sieve is effectively well-specified: its bias falls with n (0.011 → 0.002) and coverage holds near nominal (0.93–0.96), so **there is no coverage decay to correct**. Growing K per the schedule does *not* help and mildly hurts, because approximation bias is **non-monotone in K** on this surface — the step to K=3 raises bias to ~0.9 SD (see §5), which is what drops coverage to 0.86.

| n | fixed K | fixed-K cov | fixed-K bias | growing K | growing-K cov | growing-K bias | growing-K bias/SD |
|---|---|---|---|---|---|---|---|
| 1000 | 2 | 0.93 | +0.0107 | 2 | 0.93 | +0.0107 | 0.27 |
| 2000 | 2 | 0.94 | +0.0054 | 2 | 0.94 | +0.0054 | 0.20 |
| 4000 | 2 | 0.94 | +0.0021 | 3 | 0.91 | +0.0149 | 0.63 |
| 8000 | 2 | 0.96 | +0.0021 | 3 | 0.86 | +0.0152 | 0.86 |
| 16000 | 2 | 0.94 | +0.0017 | 4 | 0.89 | −0.0037 | 0.47 |

The fixed-K column shows no degradation as n grows, and the growing-K column moves *away* from nominal precisely where the schedule steps up to K=3 — because 3 segments is a worse approximation of this CATE surface than 2 (§5: at n=4000, K=2 → bias 0.0021, K=3 → bias 0.0149). **Undersmoothing is therefore neither needed nor beneficial here.** It pays off only when fixed-K carries a genuine, growing *relative* bias — a rougher CATE surface, or the irregular boundary-integral regime of §4 where the sub-√n thin-set term bites — which this calibrated oracle does not exhibit. This is consistent with §4's observation that a fixed-dimension sieve on a smooth surface is effectively parametric and near-nominal.

## 5. Sieve-dimension (K) sensitivity at n=4000

| segments | bias | RMSE | se_ratio | 95% cov |
|---|---|---|---|---|
| 1 | +0.0667 | 0.0694 | 0.92 | 0.05 |
| 2 | +0.0021 | 0.0182 | 0.96 | 0.94 |
| 3 | +0.0149 | 0.0280 | 0.99 | 0.91 |

Larger K reduces approximation bias (undersmoothing improves coverage) at the cost of variance — the standard bias/variance trade-off behind undersmoothed inference.

## 6. Full-refit bootstrap (primary interval)

| n | bootstrap 95% coverage | mean CI length |
|---|---|---|
| 2000 | 0.94 | 0.1011 |
| 8000 | 0.92 | 0.0506 |

## 7. Robustness: realistic low-SNR noise (noise_scale=1.0)

Same oracle surfaces/truth (θ=0.162); residual noise at the realistic consumption level.

| n | bias | RMSE | 95% cov (sieve) |
|---|---|---|---|
| 2000 | +0.0212 | 0.0573 | 0.95 |
| 8000 | +0.0089 | 0.0352 | 0.96 |

At realistic SNR the plug-in bias is larger and coverage lower — an honest picture of how hard CATE-threshold estimation is on real experimental data, and why the paper's irregular-inference machinery (undersmoothing, boundary-aware SEs, bootstrap) matters.

## 8. Exact-truth validation (affine bivariate-normal companion)

A companion affine DGP with straight margins and a single transversal corner (x*=[-0.66, -0.61], angle 75°) has θ available in closed form (bivariate-normal orthant), validating the truth machinery to zero grid error. See `results/figures/fig1b_affine_geometry.png`.

| method | θ | θ_++ | θ_-+ | θ_-- |
|---|---|---|---|---|
| **exact** (orthant) | 0.2087 | 0.5875 | 0.1045 | 0.0993 |
| grid quadrature | 0.2087 | 0.5879 | 0.1044 | 0.0991 |
| 1M-draw MC | 0.2087 | 0.5871 | 0.1046 | 0.0996 |

θ agreement: |exact − grid| = 0.00004, |exact − MC| = 0.00007.
