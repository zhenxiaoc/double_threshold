# Sieve-DML inference for the double-threshold harm share

`src/harm_share/sieve_dml.py` implements debiased inference for

```
theta = Pr( tau_S(X) >= 0 , tau_Y(X) < 0 )
```

with a machine-learning first stage — a **random-feature ridge (RF)** or **gradient
boosting (GBR, an XGBoost equivalent)** for the CATE nuisances — and the paper's
**two-band sieve-Riesz** variance for the debiasing. The estimand is a *double*
threshold, so the Riesz representer has **two bands**, one per decision boundary
(M_S = {tau_S=0, tau_Y<0} and M_Y = {tau_Y=0, tau_S>=0}).

## Why the ordinary DML correction fails here

theta is a **thin-set (level-set) functional** — it has no root-n influence
function (Chen & Gao 2026). The first thing we tried, the textbook cross-fit AIPW
correction

```
psi = (1{...}-theta) + (1/2 eps_S) 1{|tau_S|<eps_S, tau_Y<0} xi_S
                     - (1/2 eps_Y) 1{|tau_Y|<eps_Y, tau_S>=0} xi_Y,
```

with AIPW residuals `xi = (W/e-(1-W)/(1-e))(outcome-mu_W)`, is **mis-calibrated**:
the raw 1/eps boundary weight has a huge variance so the interval over-covers, and
adding the first-order correction over-shoots (the plug-in is already near-unbiased,
so the correction injects noise and bias). See `harm_share_dml(..., debias=...)`.

## What works: the two-band sieve-Riesz

The paper's variance **projects** the boundary Riesz weight onto a linear basis
`b(x) = [psi_t(x); -psi_c(x)]` — `alpha_hat(x) = b(x)' G^{-1} Bun`, with `Bun` the
band integral of the basis. That projected Riesz has the right (much smaller)
variance, and it is exactly `two_band_sieve_variance` from `estimator.py`.
`harm_share_riesz_dml(df, nuisance=...)`:

- **sieve / rf** — the first stage is itself linear-in-features (B-spline sieve or
  random-feature ridge via `opttreat`), so it *supplies* the feature maps the Riesz
  needs. RF scales to high dimension where the tensor sieve cannot.
- **gbr** — gradient boosting has no feature map, so the CATE is cross-fitted by GBR
  and the Riesz basis is a *separate* random-feature/sieve fit (the "sieve-DML" mix).

## Coverage results

`run_dml_study.py`, n = 4000, 100 reps, band δ = 0.10, random-feature count scaled with dimension. Two
DGPs: the exact-truth **KRR** oracle (d=2, θ=0.162) and the **faithful full-surrogate WGAN** oracle
(d=26, θ=0.126 — GAN2 generates CR's full 21-dim two-year outcome vector, GAN3 forecasts Y on all of it):

| DGP | nuisance | bias | MC-SD | mean-SE | 95% coverage |
|---|---|---|---|---|---|
| KRR (d=2, **exact** truth) | sieve | +0.000 | 0.018 | 0.017 | **0.92** |
| KRR (d=2, exact) | RF | −0.008 | 0.017 | 0.017 | **0.90** |
| KRR (d=2, exact) | GBR | +0.013 | 0.018 | 0.018 | **0.93** |
| WGAN (d=26, faithful) | RF | **+0.069** | 0.017 | 0.027 | **0.15** |
| WGAN (d=26, faithful) | GBR | **+0.056** | 0.020 | 0.042 | 0.92 |

**Read.** On the **exact-truth KRR DGP the method is well-calibrated** (0.90–0.93): the two-band
sieve-Riesz SE tracks the Monte-Carlo spread and the plug-in is near-unbiased. The **faithful
full-surrogate WGAN DGP is genuinely harder, and honestly so**: both ML nuisances carry a large
**positive bias (~+0.06, about half of θ)**. GBR's 0.92 coverage is **bias masked by a large SE**
(mean-SE 0.042 ≫ MC-SD 0.020), not genuine calibration; RF's tighter SE exposes the bias (0.15).

**Why the bias appears only on the full-surrogate DGP — a real property, not a bug.** Y is generated
from the *whole* 21-dim two-year surrogate vector, but the CATE nuisance may condition only on baseline
X (the surrogates are post-treatment, so the policy rule and τ_Y(X) are functions of X alone). The 20
unobserved surrogates become irreducible noise: `sd(Y|X,W) ≈ 31` against `ATE_Y ≈ 6.7`. A level-set
functional like θ = Pr(τ_Y<0) is biased **upward** by a noisy τ̂_Y: symmetric estimation noise smears
mass across the τ_Y=0 threshold, and since τ_Y is mostly positive (+6.7 ± 5.7), the smearing *adds*
harm probability. The bias is of order Var(τ̂_Y), so it **shrinks with n** (the estimator is
consistent) — RF nuisance, δ=0.10, 40 reps:

| n | θ̂ | bias | SD | 95% cov |
|---|---|---|---|---|
| 4 000 | 0.198 | +0.072 | 0.021 | 0.18 |
| 8 000 | 0.186 | +0.060 | 0.017 | 0.25 |
| 16 000 | 0.171 | +0.045 | 0.018 | 0.40 |
| 32 000 | 0.163 | +0.036 | 0.014 | 0.40 |

The bias decays at roughly **n^−0.3** — the slow, sub-√n rate of a thin-set functional — so it is
consistent but still dominates the (also shrinking) SE at feasible n. This slow bias decay under a
high-noise DGP is itself the operational face of the estimand's irregularity.

**Tuning knobs** (as for any irregular functional): the band width δ and the random-feature count
`n_features` (which sizes the Riesz projection and hence the SE). These move the SE but do **not**
remove the full-surrogate bias — only a lower-variance τ̂_Y (more regularization, or larger n) does.

**Bottom line:** with an *exact / low-noise* nuisance the two-band sieve-Riesz delivers nominal
inference for the double-threshold harm share. Under the **faithful full-surrogate DGP**, where 20 of
21 surrogates are unobserved and `Var(Y|X,W)` is large, the level-set estimator inherits a finite-n
**smearing bias** that dominates at n=4000 — consistent (vanishing with n) but not negligible at this
sample size. This is an honest limitation of level-set inference under a realistic, high-noise
long-term-outcome DGP, not a defect of the debiasing.

## Reproduce

```bash
PYTHONPATH=src python run_dml_study.py        # coverage across nuisances x DGPs
PYTHONPATH=src python -m pytest tests/test_dml.py -q
```
