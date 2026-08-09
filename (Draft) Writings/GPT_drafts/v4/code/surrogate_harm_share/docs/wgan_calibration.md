# The second calibration: a truthful WGAN population

The primary calibration (`docs/calibration.md`) fits **smooth kernel-ridge conditional means** to the
graduation RCT and takes those surfaces as the population truth, so `θ` is closed-form and exact. This
second calibration does what **Chen & Ritzwoller (2023, App. D.2/D.3)** actually do: it **trains the
three cascaded Wasserstein GANs** — on CR's **full 20-covariate baseline design** — and takes a
**large sample from the generative model as the population**. The goal here is faithful *data
replication*: the generated joint distribution should match the real graduation data as closely as
CR's does (§3.1). A 2-covariate variant keeps the harm-share **inference** demonstration in a low,
sieve-tractable dimension (§3.3). Together with the primary oracle this gives a robustness pairing —
exact-truth smooth surfaces vs. realistic, faithfully-calibrated GAN-generated data.

## 1. What is trained (faithful `ds-wgan`)

We vendor a faithful re-implementation of the `wgan` package of Athey, Imbens, Metzger & Munro
(2021, [`ds-wgan`](https://github.com/gsbDBI/ds-wgan)) — CR's tool — in `src/harm_share/wgan_backend.py`
(the package is not on PyPI and pins old torch). It matches the `ds-wgan` source element-for-element:

| element | choice |
|---|---|
| Standardization | per-variable `(x−mean)/std` of outputs and context (ds-wgan `DataWrapper`) |
| Generator | MLP `(noise, context) → output`, **ReLU** + dropout, noise dim = output dim |
| Generator output | **clamped to the data's [min,max]** in standardized units (ds-wgan `_transform`) |
| Critic | MLP `(data, context) → ℝ`, ReLU + dropout, no normalization |
| Gradient penalty | **one-sided** `relu(‖∇‖₂ − 1)` (ds-wgan), *not* the two-sided `(‖∇‖−1)²` |
| Optimizer | Adam, PyTorch-default betas (0.9, 0.999); step-based critic/generator alternation |

Tuning follows **CR's Table** where they state it — **batch 256, learning rate 1e‑4 for both
networks, gradient penalty 20, dropout 0.1** (0 for the X‑critic); everything else is the `ds-wgan`
default (three 128‑wide hidden layers, 15 critic steps). CR run 30000 epochs (X, S) / 5000 (Y) on a
cluster; on this smaller problem we train for fewer epochs (chosen so the generated joint distribution
matches the data — §3.1 — minutes on one GPU). This is a *compute* deviation, not a *procedure*
deviation.

**Two engineering points that matter.** The one-sided penalty and the output clamp are the two
`ds-wgan` choices (vs. textbook Gulrajani WGAN-GP) that make training stable on this small sample: the
clamp bounds the generated support so the mean cannot run off, and the one-sided penalty leaves the
critic free to be flat where appropriate instead of being forced near-linear (which otherwise sends
the generator/critic into a limit cycle). With the textbook two-sided penalty and no clamp, even
fitting a 1-D N(0,1) diverges here.

**Conditioning on treatment.** As in CR, GAN2/GAN3 take the binary treatment `W` as a **context
input** to a single generator (GAN2 = `S | X, W`; GAN3 = `Y | S, X, W`); the CATE is read off by
contrasting the generator at `W=1` vs `W=0`.

**The short-term outcome is CR's full 21-vector, not one surrogate.** Faithful to CR (App. §Data),
`S` is the **whole 21-dimensional vector of two-year (Endline-1) measurements**, and GAN2 generates it
jointly as `S | X, W`; GAN3 then forecasts `Y | S(full), X, W` conditioning on **all** of them. This
is the point of the surrogate approach — a single surrogate is not a sufficient statistic for the long
run, so a credible three-year forecast must condition on enough two-year outcomes. The harm-share
estimand still needs a **scalar** short-run outcome for its rule `1{τ_S(X) ≥ 0}`, so the threshold
surrogate (total consumption) is carried as **raw column 0** of the S vector — `τ_S` reads straight off
it with no back-transform — while the other 20 surrogates are encoded (continuous / binary-softmax /
hurdle, via a second `_CovariateEncoder`) purely as conditioning richness for the forecast. Set
`full_surrogate=False` for the earlier scalar-S cascade.

**The covariate set (CR's full baseline design).** To replicate CR's calibration we train GAN1 on
**all 20 pre-treatment baseline covariates** CR use, across their five categories —
consumption (×4), food security (index + 5 binary indicators), assets (×3 indices), finance
(loans ×3, savings), and income/revenue (×3) — the constant `COVARIATES_CR_FULL`. GAN2/GAN3 condition
on this full covariate vector, so the generated short/long outcomes reflect the whole pre-treatment
state, as in CR.

**Covariate encoding (ds-wgan-faithful mixed DataWrapper).** `_CovariateEncoder` routes each of the 20
covariates by type into a **model representation** that GAN1 generates (continuous heads + softmax
categorical heads), and decodes back to original units for the dataset:
- **Continuous, well-behaved** → Gaussian **quantile transform** to `z`.
- **Continuous, heavy-tailed** (|skew| > 2: consumption, asset indices) → **signed-log1p**
  (`sign(x)·log(1+|x|)`, which also handles negative ag-incomes) *then* quantile transform, so the
  generator's inverse map does not amplify tails.
- **Binary** (the 5 food-security indicators) → a **softmax categorical head** (CR's treatment of
  categoricals), so they are generated as exact {0,1} with the right proportions.
- **Heavily censored** (>40% zero: loans, savings, ag/animal income) → a **hurdle**: a softmax
  nonzero-indicator {x≠0} + a continuous signed-log positive-part (zeros filled with the nonzero mean
  so the continuous head carries no spike). This lets the GAN model the zero mass as a clean
  Bernoulli and the positive tail as a smooth continuous — reproducing the spike-and-slab shape a
  single continuous head cannot.

The encoder's encode→decode is exact on the training data; `raw_covariates()` decodes generated
model-rows to the 20 covariates in original units.

**Support box.** `draw_X` truncates to a **wide box (±6.5 in z-space)** that covers the real covariate
support (the quantile transform clips tails at ±5.2); a tight box would drop mass and distort the
covariate spread and the average treatment effect.

The **same encoding is applied to the 20 non-threshold surrogates** (a second `_CovariateEncoder`):
food-security indicators → softmax, censored monetary two-year outcomes → hurdle, the rest continuous.

The cascade for one unit (CR App. D.3):

```
X ~ GAN1                            (all 20 baseline covariates)
S(0), S(1) ~ GAN2(X, W=0), GAN2(X, W=1)          (full 21-dim short-term vector)
Y(0), Y(1) ~ GAN3(S(0), X, W=0), GAN3(S(1), X, W=1)
```

## 2. How the truth is computed

`θ = Pr(τ_S(X) ≥ 0, τ_Y(X) ≤ 0)` is a functional of the two conditional-mean CATEs, so unlike the
primary oracle it is **not** closed form — it is a property of the trained generators, computed to
controllable Monte-Carlo precision by conditional MC over the generator noise with **common random
numbers** (same noise across the two arms, a large variance reduction for the difference):

```
τ_S(x) = E_U[ G^1_S(x,U) − G^0_S(x,U) ]
τ_Y(x) = E_{U,V}[ G^1_Y(G^1_S(x,U), x, V) − G^0_Y(G^0_S(x,U), x, V) ]
```

evaluated on a large fixed population of `X` draws (`N_pop` points, `M` noise draws each). The truth is
the harm share of the WGAN's own conditional means; the finite-sample experiments drawn for the study
are genuine cascade draws carrying the GAN's realistic (heteroskedastic, skewed) conditional shapes,
which is what makes this a harder, more realistic estimation target than the smooth primary oracle.

## 3. Results

### 3.1 Calibration fidelity — replicating CR's validation exhibit (all 22 variables)

CR validate their calibration by showing that the **means, variances, and correlations** of the
generated data match the real data (on a log scale, across all variables), and that the
**long-term-outcome histograms** (by treatment) align closely. `make_wgan_figures.py` reproduces both
checks for the full 20-covariate + S + Y calibration (`results/figures/wgan_validation.png`,
`wgan_histograms.png`). As in CR, the match is close:

- **Correlations** — all 231 covariate+S+Y pairs — cluster on the 45° line (mean |Δcorr| = **0.046**);
  the **full 21-surrogate** vector's 210 pairs match at mean |Δcorr| = **0.091** (`wgan_surrogates.png`).
- **Variances** line up on the 45° line across many orders of magnitude, though on this richer joint they
  are somewhat **compressed** (generated sd below real for the high-variance monetary/consumption vars —
  the cost of learning a 40-output joint on 854 rows).
- **Means** sit on the line; only the hardest censored variables are visible outliers.
- The S and Y **histograms by treatment** overlay the real ones with matching supports and right tails.

| variable | mean real/gen | sd real/gen |
|---|---|---|
| S (2-yr consumption) | 93.6 / 83.9 | 55.5 / 37.7 |
| Y (3-yr consumption) | 87.4 / 80.8 | 53.1 / 31.3 |
| total consumption (bsl) | 134 / 116 | 148 / 76 |
| fs indicators (5, **softmax**) | 0.31/0.30 … 0.74/0.76 | exact {0,1} |
| **loan_totalamt** (**hurdle**) | 1036 / 1132 | 3688 / 4473 |
| **loan_informalamt** (**hurdle**) | 916 / 937 | 3634 / 4001 |
| **loan_formalamt** (**hurdle**, 99% zero) | 7.0 / 4.9 | 106 / 68 |
| perceived economic status | 3.07 / 3.03 | 1.88 / 1.85 |
| E[S\|W=1] − E[S\|W=0] | +12.3 / +8.6 | — |
| E[Y\|W=1] − E[Y\|W=0] | +6.0 / **+6.7** | — |

**The full surrogate fixes the long-run effect.** Conditioning `Y` on the whole 21-dim two-year vector
recovers the three-year treatment effect: **ATE_Y = +6.7 vs the data's +6.0** (the earlier scalar-S
cascade collapsed it to +1.3). This is exactly the surrogate rationale — one number under-forecasts the
long run; the full vector does not. Softmax + hurdle keep the censored marginals honest (binaries exact
{0,1}; `loan_totalamt` 1036 vs 1132, `loan_formalamt` 7.0 vs 4.9). Two variables remain hard —
`ranimals_month`, `iagri_month` (95–96% zero **with rare huge values**) — and the high-variance vars
are compressed, the price of a 40-output joint on 854 rows.

### 3.2 WGAN population truth

Read off `N_pop = 10⁵` draws with `M = 300` CRN noise draws per point (θ stable across `M`/seed) for the
**faithful full-surrogate calibration** (26-column X, 21-dim S, softmax + hurdle):

- **Harm share** θ = Pr(τ_S ≥ 0, τ_Y < 0) = **0.126**
- ATE_S = **+8.6** (τ_S = +8.6 ± 2.7), ATE_Y = **+6.7** (τ_Y = +6.7 ± 5.7); W_Y = E[max(τ_Y,0)] = 6.95

**A consequence of the full-surrogate forecast: only the long-run boundary M_Y binds.** Conditioning Y
on the whole surrogate vector makes the three-year forecast track the two-year outcomes, and in the
process the **short-run consumption CATE flattens to a near-constant +8.6** — so `τ_S ≥ 0` for *every*
population point (`Pr(τ_S<0) = 0`) and the harm share equals `Pr(τ_Y < 0) = 0.126`. The double
threshold is thus **inert on the M_S side** here: the short-run rule treats everyone, and all the harm
comes from the long-run boundary. (The earlier *scalar*-S cascade had `τ_S<0` for ~30%, so both
boundaries bound — but at the cost of a badly under-forecast ATE_Y of +1.3.) This is the honest,
CR-faithful output of a single joint GAN2 over the full S vector: a credible long-run forecast and a
non-degenerate θ, with the short-run boundary not binding for *consumption* on these 854 rows. The
**two decision boundaries meeting at a codimension-2 corner** are exhibited by the primary exact-truth
calibration (`docs/calibration.md`), where τ_S and τ_Y are prescribed smooth surfaces that both cross
zero; the two-band estimator below is still the correct estimator (its M_S band simply carries ≈0 mass
on this DGP).

### 3.3 Inference

Inference on the faithful full-surrogate DGP (d=26 model covariates) is the **sieve-DML** procedure with
ML nuisances and the two-band sieve-Riesz debiasing — the tensor sieve is infeasible at d=26, so the
nuisance must be ML. See [`docs/sieve_dml.md`](sieve_dml.md) for the estimator and coverage; the
`run_dml_study.py` driver evaluates it on both this WGAN oracle and the exact-truth KRR oracle.

> Note: the low-dimensional 2-covariate variant (`train_wgan.py --2d`, used by the older
> `run_wgan_study.py` sieve demonstration) is **degenerate under the full-surrogate design** — with only
> two covariates there is no X-heterogeneity to cross either boundary and θ collapses to 0. The
> full-surrogate inference therefore lives entirely in the d=26 sieve-DML study, not the 2-covariate
> sieve run.

## 4. Reproducing

```bash
# faithful full-covariate calibration (replicating CR's dataset)
PYTHONPATH=src python train_wgan.py         # train 20-covariate cascade + cache + fidelity table
PYTHONPATH=src python make_wgan_figures.py  # CR-style validation: all-variable moments + histograms

# 2-covariate inference variant (low-dim, sieve-tractable decision rule)
PYTHONPATH=src python train_wgan.py --2d    # train + cache the 2-covariate variant
PYTHONPATH=src python run_wgan_study.py     # truth + MC coverage + rate + bootstrap on it

PYTHONPATH=src python -m pytest tests/test_wgan.py -q
```
