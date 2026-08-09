# How the calibrated dataset is built

This note documents exactly how the simulation's **calibrated population** ("the oracle") is
constructed from real data, and how the construction mirrors — and where it deviates from — the
Generative-Adversarial-Network calibration of Chen & Ritzwoller (2023, App. D.2/D.3). The code is
`src/harm_share/calibration.py`; the entry point is `build_oracle()`.

The guiding principle is the **calibrated-simulation** design of Athey, Imbens, Metzger & Munro
(2021), also used by Chen & Ritzwoller: fit a *generative model* to a real experiment, then treat a
draw from that model as the population, so that (i) the data look realistic and (ii) the **true**
target parameter is known because the generative model is known.

---

## 1. Source data

The Banerjee et al. (2015) multi-faceted "graduation" poverty-program RCT, Pakistan/Sindh subset,
shipped with the Chen & Ritzwoller `longterm` R package as `data/graduation.rda` (read in Python via
`pyreadr`). **854 households, treatment randomized** (52.2% treated, so the propensity `e = P(W=1)`
is known and exogenous — no propensity nuisance). Each household has baseline (`*_bsl`), 2-year
(`*_end`) and 3-year (`*_fup`) waves.

We use the **same economic outcome at two horizons**:

| role | variable | meaning |
|---|---|---|
| short-run `S` | `ctotal_pcmonth_end` | 2-year per-capita monthly total consumption |
| long-run `Y` | `ctotal_pcmonth_fup` | 3-year per-capita monthly total consumption |
| covariates `X` (d=2) | `ctotal_pcmonth_bsl`, `asset_index_bsl` | baseline consumption, baseline asset index |
| treatment `W` | `treatment` | randomized program assignment |

The program's average effect **fades** from ATE(S) ≈ +12.3 (2-yr) to ATE(Y) ≈ +6.0 (3-yr), and the
2-yr and 3-yr CATEs are only weakly correlated (≈ 0.3–0.5), so a non-trivial **harm quadrant**
`{τ_S ≥ 0, τ_Y < 0}` genuinely exists (economic fade-out / mean reversion). This was confirmed on the
raw data before any modelling.

---

## 2. Correspondence to Chen & Ritzwoller's GAN cascade

CR calibrate their simulation with **three cascaded WGANs** (their `ds-wgan` package; App. D.2,
Table 1):

```
GAN1:  X            (marginal of pre-treatment covariates)
GAN2:  S | X, W     (short-term outcomes given covariates and treatment)
GAN3:  Y | S, X, W  (long-term given short-term, covariates, treatment)
```

They generate `X → (S(0), S(1)) → (Y(0), Y(1))`, assign experimental/observational 50/50, and take
**10^7 draws as the population** — a job they report at ≈ **60 CPU-years on a cluster**.

Our oracle reproduces the **same cascade structure** with lighter, exact-truth components:

| CR component | our component |
|---|---|
| GAN1: marginal of `X` | Gaussian **KDE** on quantile-transformed covariates |
| GAN2: `S \| X, W` | smooth **kernel-ridge** conditional mean `μ_{S,w}(x)` + resampled empirical residuals |
| GAN3: `Y \| S, X, W` | smooth **kernel-ridge** conditional mean `μ_{Y,w}(x)` + coupling to `S` + resampled residuals |
| 10^7 draws = population | `draw_X` + closed-form surfaces = population; **truth computed, not sampled** |

**Why this substitution is exact for our estimand.** The target `θ = Pr(τ_S(X) ≥ 0, τ_Y(X) ≤ 0)` is a
functional of the two **conditional means** (through `τ_S = μ_{S,1}−μ_{S,0}`, `τ_Y = μ_{Y,1}−μ_{Y,0}`)
and of the covariate density `f`. The GAN's added value — realistic conditional *shape*
(heteroskedasticity, skew, tails) — affects **finite-sample estimation noise**, not the population
value of `θ`. So swapping the WGANs for smooth conditional means + nonparametric residual resampling
changes the *hardness* of estimation, not the *truth*. CR themselves generate `S(1), S(0)` (and
`Y(1), Y(0)`) as **independent conditional draws** given `X` — there is no cross-world joint — so our
per-arm cascade has the same structure, not a simplification of it.

A GPU WGAN backend (installing `torch`+CUDA on the RTX 5080 and porting `ds-wgan`) is a **drop-in
replacement** for the conditional sampler only; it leaves the oracle surfaces, and therefore the
truth, unchanged.

---

## 3. Construction, step by step (`HarmShareOracle.fit`)

1. **Covariate transform.** Push each baseline covariate through a Gaussian **quantile transform**
   (`sklearn.QuantileTransformer`, fit on the 854 households). This tames the heavy right tail of
   consumption and the asset-index outliers, giving a well-behaved, near-standard-normal support —
   important for clean B-spline knots downstream. All subsequent objects live in this `z`-space.

2. **Per-arm conditional means (the oracle surfaces).** For each arm `w ∈ {0,1}` fit a smooth
   **kernel-ridge** (RBF) regression of the outcome on `z`:
   `μ_{S,w}(z)`, `μ_{Y,w}(z)`. The CATE surfaces are the contrasts
   `τ_S(z) = μ_{S,1}(z) − μ_{S,0}(z)`, `τ_Y(z) = μ_{Y,1}(z) − μ_{Y,0}(z)`.
   These fitted surfaces **are the truth** for the simulation.
   - **Why kernel ridge, not a polynomial T-learner.** A low-degree polynomial T-learner makes
     `τ_S` and `τ_Y` nearly **collinear** (corr ≈ 0.96–0.99), which collapses the two decision
     margins onto one another and destroys the codimension-2 corner. Kernel ridge retains genuinely
     two-directional heterogeneity (corr ≈ 0.5) and a transversal corner. Smoothing is tuned
     (`gamma_S=0.5, alpha_S=0.5, gamma_Y=0.4, alpha_Y=0.6`) so that (a) the harm quadrant keeps
     ≈ 0.15 mass, (b) both zero level sets have non-vanishing gradient (regular margins), and
     (c) the two gradients are transversal at the corner.

3. **Covariate density.** A Gaussian **KDE** (`scipy.gaussian_kde`, bandwidth factor 0.35) on the
   transformed covariates gives `f`.

4. **Residual pools.** Store the per-arm empirical residuals
   `S_i − μ_{S,w}(z_i)` and `Y_i − μ_{Y,w}(z_i)` for nonparametric resampling (this carries the real
   conditional shape into the sampler without a GAN).

---

## 4. Drawing a finite experiment (`sample_experiment`)

```
X  ~ f_hat  (KDE), TRUNCATED to [-3, 3]^d by rejection sampling
W  ~ Bernoulli(e = 1/2)
S  = μ_{S,W}(X) + noise_scale · eps_S,          eps_S resampled from arm-W S-residuals
Y  = μ_{Y,W}(X) + sl_coupling · eps_S + noise_scale · eps_Y,   eps_Y resampled from arm-W Y-residuals
```

Only the realized potential outcome for the drawn `W` is returned, as in a real experiment. Two
design knobs, both clearly labelled:

- **`sl_coupling` (default 0.6).** Adds a short→long dependence so `corr(S, Y)` is realistic
  (≈ 0.56). Because the added term `sl_coupling · eps_S` is **mean-zero given `(X, W)`**, it does
  **not** change `τ_Y(x)` — the truth is untouched.
- **`noise_scale` (default 0.34).** The oracle surfaces and the truth are fixed; `noise_scale`
  multiplies the resampled residuals. Consumption has a low CATE signal-to-noise ratio
  (ATE ≈ 11 on residual SD ≈ 53), so `noise_scale = 0.34` gives the clean **SNR ≈ 1** regime used
  for the methods demonstration (the CCG regime), while `noise_scale = 1.0` recovers the realistic
  low-SNR case reported as robustness. This is a design knob on the *nuisance* noise, not a change
  to the estimand.

**Truncation, not clipping.** `draw_X` uses **rejection sampling** to the box `[-3,3]^d`, i.e. the
KDE *truncated* to the box. (An earlier version *clipped* out-of-range draws to the boundary, which
piled mass on the box edge and biased `θ` downward by ≈ 0.025; truncation makes the sampled law
equal to the KDE renormalized on the box — exactly the density the grid-quadrature truth integrates,
so the Monte-Carlo truth and the grid truth agree to < 0.001.)

---

## 5. Why the truth is exact, and how it is computed

Because `θ`, the four quadrant probabilities, `ρ`, the treat-share, and the boundary geometry are
all **functionals of the known surfaces `τ_S, τ_Y` and the known density `f`**, they are computed —
not estimated:

- `mc_truth`: draw `N = 1.5 × 10^6` covariates from `draw_X`, evaluate the surfaces, average the
  indicators. Monte-Carlo error ≈ `1/√N ≈ 8 × 10^{-4}`.
- `grid_truth` (d = 2): `500 × 500` trapezoidal quadrature of `1{·} f` over `[-3,3]^2`, plus the
  geometry diagnostics (gradient norms on each margin, corner transversality `|cos∠(∇τ_S,∇τ_Y)|`,
  Hausdorff lengths of `M_S, M_Y`).

Both agree to < 0.001. As an independent **zero-error** check, `affine_dgp.py` supplies an affine
Gaussian companion where `(τ_S(X), τ_Y(X))` is bivariate normal and `θ` is an exact orthant
probability (scipy); its exact `θ` matches the grid and Monte-Carlo machinery to 5 decimals.

---

## 6. Reproducing / extending

```bash
PYTHONPATH=src python -c "from harm_share import build_oracle, mc_truth; print(mc_truth(build_oracle()))"
```

- Change outcomes/covariates: `build_oracle(covariates=[...], s_col=..., y_col=...)` or edit
  `OracleConfig`. A d=3 variant (adds `index_foodsecurity_bsl`) makes the corner a line.
- Swap the calibration source: `build_oracle(rda_path="...")`.
- Full-fidelity GAN calibration: replace the conditional-mean + residual sampler in
  `sample_experiment` with draws from a trained `ds-wgan` cascade; the truth-computation code is
  unchanged.

All randomness is seeded (`numpy.random.default_rng` / `SeedSequence`); the truth grids are
deterministic.
