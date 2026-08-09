# Quadratic (corner-aware) debiasing and the doubly debiased estimator

Companion note to `src/harm_share/quadratic.py`, `run_debias_study.py`,
`run_corner_anatomy.py`. Theory: the `two_threshold_inference_v*.tex` sitting beside this `code/` folder.

## The problem

theta = Pr(tau_S >= 0, tau_Y < 0) is a **two-threshold** value functional. Its
first pathwise derivative is a sum of two margin (codim-1) Hausdorff integrals
-- that part is in `estimator.py` (two-band sieve-Riesz SE). The **second**
derivative, new here, has three parts:

    D^2 Theta = Q_S (margin-S curvature)  +  Q_Y (margin-Y curvature)
              + 2 * C (CORNER cross term over {tau_S=0} ∩ {tau_Y=0}, codim 2)

plus diagonal corner "edge-flux" pieces inside Q_S/Q_Y. The corner cross
diagonal has expectation of order

    K^{-2s/d}  +  rho_SY * K/n     (rho_SY = within-unit corr of S,Y residuals)

— the **same order** as the margin diagonals. Consequence: applying Chen & Gao
(2026) SS/LOO debiasing **margin by margin misses a first-order-in-K/n bias
term**; the correction must perturb both surfaces jointly ("corner-aware").

## Estimators (quadratic.py)

### `ss_debiased_estimate` — split-sample, tuning-free jackknife form

Halves A/B (stratified by W), half-sample sieve fits, and the step-1/2 second
difference identity:

    theta_SS = 2*Theta(tau_bar) - (Theta(tau_A) + Theta(tau_B)) / 2

No numerical-differentiation step size, no curvature/normal/angle computation;
exact when t -> Theta(bar + t*Delta) is quadratic. Polarization isolates the
corner:

    quad_full - quad_S - quad_Y  =  2 * corner cross form,

so `theta_ss_margins` (margins-only) and `theta_ss` (corner-aware) differ by
exactly the estimated corner correction — the ablation the MC exploits.

### `dd_estimate` — doubly debiased (DML + quadratic), the Remark-7 combination

2-fold cross-fitted ML (GBR/RF) first stages + **projected** two-band
sieve-Riesz first-order correction + the SS quadratic correction from the same
two fold fits:

    theta_DD = [2*Theta(tau_bar) - (Theta(tau^A)+Theta(tau^B))/2]  +  P_n[ vhat* eps_oof ]

Key structural fact (tested in `test_quadratic.py`): for an own-basis sieve LS
first stage the projected Riesz correction is **identically zero** (LS
orthogonality). That is why the earlier *raw* AIPW band correction
(`harm_share_dml(debias=True)`) over-shoots: it approximates a projection that
is exactly zero, so it adds pure 1/eps noise. The projected correction is
active precisely for generic ML first stages — its intended use.

All variants are studentized by the same two-band sieve-Riesz SE
(`riesz_correction_and_variance` returns correction + variance from shared
ingredients; cross-fitted residuals for the ML path).

## Studies

- `run_debias_study.py`: 3 calibrated DGPs (KRR d=2 smooth; WGAN-2d CR-cascade
  with ReLU (rough, s~1) truth; affine exact) x 6 estimators x
  n in {1000,2000,4000,8000} x 500 reps. Output:
  `results/tables/debias_study.csv`, rendered by `write_debias_results.py`
  into `results/tables/debias_table.tex` + `docs/debias_results.md`;
  figures by `make_debias_figures.py`.
- `run_corner_anatomy.py`: locates the exact corner points of the KRR oracle,
  measures E[Delta_g * Delta_h] there across MC reps, checks the ~1/n scaling
  and the sign predicted by rho_SY, and compares the implied plug-in corner
  bias with the SS corner correction. Output `results/logs/corner_anatomy.json`.

## Expected qualitative results (theory predictions)

1. Smooth KRR truth: plug-in already fine; SS ~ plug-in; corrections tiny.
2. Rough WGAN truth: plug-in bias large and slowly decaying; SS removes most;
   corner-aware vs margins-only differ measurably.
3. ML (GBR) first stages: cross-fit plug-in biased; Riesz correction then
   quadratic correction reduce bias monotonically; DD best-centered at large n
   (variance inflation at small n is real and honest).
