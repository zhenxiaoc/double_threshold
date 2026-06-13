# Findings: Value-Functional Inference at d_x = 10 and d_x = 50

*Script: `rf_sieve_v_highdim_explore.py`; results stem `rf_sieve_v_explore_rep200_M8192`
in `results/`. 200 Monte Carlo reps per cell; RF-OLS first stage (cos features, intercept,
shared across arms); eps-band sieve t-statistic for V; iota = 0.01; tau_scale = 3,
sd(tau) ~ 0.7, sd(eps) = 1.*

## Headline result

**Inference on the value functional V works at d_x = 50.** Dense-index DGP (CATE driven by
two linear indices over all 50 coordinates), n = 4000, K = 200 random features:

| cell | h_rmse | V bias | V coverage |
|---|---|---|---|
| d_x = 10, n = 1000, K = 50  | 0.63 | +0.006 | 1.000 |
| d_x = 50, n = 4000, K = 200 | 0.64 | +0.003 | 1.000 |
| d_x = 50, n = 16000, K = 400 | 0.44 | +0.011 | 0.995 |

The same machinery with tensor B-splines is simply unavailable at these dimensions.
V_true ~ 0.58 in all designs, so the biases above are ~0.5-2% of the estimand.

## The five main findings

### 1. First-stage RMSE is the gatekeeper for V — and it is a *bias* story

Across all 14 cells, V coverage >= 0.94 exactly when h_rmse <~ 0.65 (~ 0.9 x sd(tau));
every failure has h_rmse >= 0.85. The failures are bias failures (underfit first stage →
boundary located wrong → V biased), and the band SE does not — and cannot — absorb bias:

- d_x = 50, n = 1000, K = 50: h_rmse 0.87, V bias +0.068, coverage 0.845;
- gamma = 6 (too-wiggly features): h_rmse 0.89, V bias +0.075, coverage 0.670;
- q = 2-sparse features with only K = 50 (random supports): h_rmse 0.62 *on average but
  bimodal*, V bias +0.18, coverage 0.230 — only ~K/C(50,2) features hit the relevant
  support, so most draws underfit catastrophically.

Practical rule suggested by the data: monitor a cross-validated first-stage RMSE and
require it comfortably below sd(tau_hat) before trusting the V interval.

### 2. W and V behave completely differently (confirming earlier exploration)

W (regular, root-n) fails in **every** non-oracle cell — bias 0.04-0.33, coverage 0.00-0.60
— while V is fine in most. The mechanism: W(h_hat) inherits the Jensen/ReLU second-order
bias E[(h_hat - tau)^2-type] near the boundary, but its sqrt(n)-scaled SE does *not* grow
with first-stage noise; V's eps-band SE self-normalizes (it inflates exactly when the
first stage is noisy). The "irregular" functional is the *robust* one in high dimension.
This deserves a remark in any write-up: irregularity here buys honesty.

### 3. The sieve-Riesz growth is a design choice, and it is empirically *flat* in K

n * Var_hat(V) across cells ranges 1.5-11.7 depending on the feature law and gamma — the
growth of the sieve variance is tuned by the sieve design, not fixed. Strikingly, at
fixed (n, d_x = 50), doubling K from 200 to 400 moves n*Var_hat(V) only from 5.70 to 5.93
(~4%): essentially flat, consistent with the spline-theory exponent K^{1/d_x} = K^{0.02}
carrying over to random features at d_x = 50. (The increase from 5.7 to 11.7 at
n = 16000 co-moves with the shrinking band and finer first stage; worth a dedicated
log-log regression of n*Var on (K, n) by dimension — the draws CSV has everything needed.)

### 4. Feature-design knobs matter, with sane defaults

- gamma (feature scale): 1.5 and 3 both fine; 6 breaks inference. Moderate scales,
  calibrated so gamma * sd(w'x + b) ~ 1-2, are safe.
- Support sparsity q at adequate K = 200 (sparse DGP): q = 1 (additive features) gives
  the best first stage (h_rmse 0.45 vs 0.54 for q = 2 vs 0.61 dense), and all three give
  valid V inference. Tuning the feature *support law* is a cheap, theory-compatible way
  to encode structure — each draw is still a fixed linear sieve conditional on features.

### 5. Support knowledge is enormously valuable — and rescues W

The oracle cell (all K = 50 features supported on the true 2 relevant coordinates,
sparse DGP, d_x = 50, n = 4000):

| | h_rmse | V bias | V cover | W bias | W cover |
|---|---|---|---|---|---|
| random q=2 supports, K=50 | 0.62 | +0.179 | 0.230 | +0.143 | 0.315 |
| **oracle supports, K=50** | **0.16** | **+0.003** | **0.945** | **+0.004** | **0.950** |

With the right low-dimensional support, *both* functionals — including the fragile
welfare functional — are accurately estimable at d_x = 50 with a tiny sieve. The gap
between these two rows is precisely the value of support recovery (screening).

## Directions going forward

1. **Screen-then-sieve (most promising practical direction).** Estimate the relevant
   coordinates/indices first (lasso on D-interactions, marginal CATE screening, or
   group lasso over q-sparse RF blocks), then build the sieve on the screened support.
   The oracle row bounds the payoff. Crucially, the draft's **Appendix D
   (sieve-influence-function + cross-fitting) is the natural vehicle for post-selection
   validity**: the first stage may be any ML learner (including screened/penalized fits)
   while v*_K is built on a low-dimensional post-screening sieve; cross-fitting handles
   the data-dependence of the selection step.

2. **Theory: "cylinder boundary" / effective dimension.** If tau(x) = g(x_S) with
   |S| = s (or g of s linear indices — note the *dense-index* DGP that worked at
   d_x = 50 has index-sparsity s = 2 without coordinate sparsity), the boundary
   {tau = 0} is a cylinder (s-1 dimensional set) x [0,1]^{d_x - s}: the Hausdorff
   boundary integral collapses to an (s-1)-dimensional integral. Conjecture: the minimax
   rate for V is governed by the *effective* dimension s, not d_x, under an exact-
   sparsity or index-sparsity condition, and a support-adapted sieve attains it. This
   would formalize "we average over most of the dimensions." Index sparsity is the more
   elegant assumption (invariant to rotations; matches single/multi-index literature).

3. **W needs debiasing, and we have the tool.** The Jensen bias that kills W is exactly
   the diagonal quadratic term that the SS/LOO estimators (Theorem 5 of the draft)
   remove. A natural next experiment: apply SS/LOO debiasing to W in this script
   (the D^2 W form has the same boundary-integral structure) and see how much of the
   0.07-0.33 bias it removes at d_x = 50.

4. **Data-driven tuning.** The h_rmse-coverage link suggests selecting (K, gamma, q) by
   cross-validated first-stage RMSE is approximately the right surrogate for valid V
   inference; verify post-selection coverage numerically, and connect to the
   bootstrap-Lepski sieve-dimension selection in Chen & Gao's appendix.

5. **Refinements.** (i) The V intervals are conservative (SE ~ 1.5-2x SD) — revisit the
   iota/band choice and consider scrambled Sobol for the band derivative in d_x = 50,
   where plain Sobol equidistribution is weak. (ii) Run the log-log growth regression of
   n*Var_hat(V) on (n, K) from the draws files to quantify finding 3. (iii) Replicate
   with relu/tanh activations.

## Caveats

- Designs here have strong signal (sd(tau) ~ 0.7 vs sd(eps) = 1) and favorable
  (Barron-type) structure; that is the point — the claim is "high-d works *under
  low-effective-dimension structure*," not unconditionally. The minimax lower bound
  (Theorem 4) shows unconditional high-d cannot work at a useful rate.
- 200 reps => coverage MC error ~ +/-1.5-3%; the qualitative contrasts (1.00 vs 0.23)
  are far outside noise, but fine distinctions (0.94 vs 0.97) are not.
- Conditional-on-features framing: all inference statements are conditional on the
  realized feature draw (a fixed linear sieve); unconditional statements would average
  over the feature law.
