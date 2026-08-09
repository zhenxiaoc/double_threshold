# Why does sieve-DML undercover?  Diagnosis

Short answer: **it is not the dataset.** On the *same* KRR-calibrated DGP, the
same samples and the same variance formula, the tensor-sieve plug-in covers
0.95 while a boosted-tree first stage covers 0.67-0.88. Four distinct causes,
in order of size, plus one genuine bug that has now been fixed.

## 0. The bug (fixed): a missing variance component

`two_band_sieve_variance` computed only the boundary/Riesz part of the
variance and omitted the **empirical-measure term** `theta(1-theta)/n`.
`theta_hat = P_n[1{tau_S>=0, tau_Y<0}]` has two error sources: the estimated
surfaces moving the region (the Riesz part, ~K/n) and the sampling error of the
average itself given the surfaces (~1/n). The asymptotic theory drops the
second because the first dominates; at n = 1000-8000 it is 6-12% of the SE.
`regular_companion_welfare` already included the analogous term (with a comment
saying it "MUST be included"); the harm share did not.

Effect on the sieve plug-in (KRR, 500 reps):

| n | SE/SD before | SE/SD after | coverage before | coverage after |
|---|---|---|---|---|
| 2000 | 0.89 | 1.01 | 0.948 | 0.948 |
| 4000 | 0.95 | 1.01 | 0.940 | 0.958 |
| 8000 | 0.89 | 1.06 | 0.932 | 0.970 |

This is the analogue of the `Var([h_0]_+)` term CCG's unknown-density welfare
theorem adds to the known-density variance. Default is now `include_empirical=True`.

## 1. Assumption 5(d) fails for trees: the correction can only see the
##    IN-SPAN part of the first-stage error, and the rest ADDS variance

The exact decomposition (CCG Appendix B) is

    theta_tilde - theta_0 = E_n[v* eps] + R1 + ... + R5,
    R5 := D_mu Theta(mu_0)[ (I - Pi_Kn)(mu_hat - mu_0) ].

The Riesz correction subtracts exactly `D[Pi_Kn Delta]` — the projection of the
first-stage error on the sieve span — and substitutes the clean score. The
orthogonal component `R5` is **never touched**. Worse: because cross-fitting
makes `mu_hat_{-k}` measurable w.r.t. the training folds while the score is
conditionally mean-zero, `Cov(score, R5) = 0` **exactly**, so the variances ADD:

    Var(DML) = sigma_n^2/n + Var(R5) + ...   >=  sigma_n^2/n

while `sigma_hat_n^2` estimates only `sigma_n^2`. Hence SE/SD < 1 and one-sided
undercoverage — never overcoverage. In the fully-orthogonal limit DML is
strictly *worse* than the plug-in: it adds `sigma_n/sqrt(n)` of independent
noise on top of an uncancelled plug-in error.

**Two regimes, one mechanism.** On a *smooth* truth (KRR) the boosted plug-in
error is modest, so the added out-of-span variance is small (the direct test
`(Var(DML)-Var(plug))/se^2` is about 0 on KRR) and the DML shortfall there is
*bias*-dominated — the correction removes only 31% of an out-of-span boundary
bias. On a *rough* truth (WGAN, s about 1) the boosted error is large and its
out-of-span part inflates the variance markedly, so the shortfall is
*variance*-dominated and the "variances add" prediction bites. Both are the
same Assumption-5(d) failure — the sieve cannot see the tree's error near the
margin — surfacing in the bias channel or the variance channel depending on how
rough the truth is.

**Measured directly** (KRR, n=4000, oracle arm means as truth): the R^2 of the
GBR error on the tensor-B-spline Riesz span is only **0.23-0.29**, and the
correction removes only **18%** of the plug-in's boundary bias. Boosting error
is piecewise constant with jumps at adaptively chosen splits — and boosting puts
its splits where the loss gradient is largest, i.e. near the very level set the
derivative integrates over. A smooth 25-dimensional sieve cannot see it.

### This is not specific to two thresholds — the parent paper shows it too

CCG's extended draft, Table 7 (`dx=2`, `K_n=81`, K=5 folds, 250 iterations),
value functional `V`, two first stages run on **identical samples and folds**:

| first stage | n | DML bias | SD | SE | SE/SD | coverage |
|---|---|---|---|---|---|---|
| gradient boosting | 1500 | +0.0018 | 0.0839 | 0.0652 | **0.78** | 0.900 |
| | 3000 | +0.0024 | 0.0420 | 0.0351 | **0.84** | 0.896 |
| | 6000 | −0.0027 | 0.0274 | 0.0242 | **0.88** | 0.900 |
| random-feature net (smooth) | 1500 | −0.0155 | 0.0974 | 0.0813 | 0.84 | 0.908 |
| | 3000 | +0.0007 | 0.0512 | 0.0482 | 0.94 | 0.924 |
| | 6000 | −0.0041 | 0.0315 | 0.0312 | **0.99** | 0.928 |

Two things stand out. First, under boosting the DML estimator is **essentially
unbiased** (|bias| <= 0.003 at every n) yet coverage is stuck at 0.90 — the
shortfall is *entirely* an SE/SD gap, which is precisely the "variances add"
signature: the SE estimates only `sigma_n^2/n` while the SD also contains the
independent out-of-span component. The draft attributes this to "the
second-order remainder that limits the rate"; the decomposition above gives the
sharper reason, and predicts the sign (undercoverage only).

Second, the **smooth** learner's SE/SD rises to 0.99 while boosting's stalls at
0.88 — the same learner contrast this project finds at two thresholds. The
draft's own high-dimensional section reaches the compatible conclusion from a
different direction: the irregular value functional "favors a lower-bias
(early-stopped, capacity-adaptive) learner" and a *smoother* representer sieve,
because its representer concentrates on the decision boundary.

### The decomposition experiment (KRR, n=4000, 300 reps)

`run_dml_diagnosis.py` varies one axis at a time. Coverage of the
Riesz-corrected estimator, studentized by the (corrected) two-band SE:

(Corrected two-band SE, i.e. including the theta(1-theta)/n term.)

| configuration | bias | plug-in bias | SD | SE/SD | coverage |
|---|---|---|---|---|---|
| tensor-sieve plug-in (reference) | +0.003 | +0.003 | 0.018 | 1.03 | **0.94** |
| gbr K=2 riesz=sieve | +0.022 | +0.032 | 0.019 | 0.96 | 0.76 |
| gbr K=5 riesz=sieve | +0.011 | +0.016 | 0.019 | 0.94 | 0.90 |
| gbr K=10 riesz=sieve | +0.008 | +0.013 | 0.020 | 0.92 | 0.91 |
| rf K=5 riesz=sieve | +0.012 | +0.045 | 0.020 | 1.01 | 0.91 |
| **krr (smooth) K=5 riesz=sieve** | +0.004 | +0.004 | 0.018 | 0.90 | 0.92 |
| gbr K=5 riesz=**rf** | +0.008 | +0.016 | 0.020 | 0.96 | **0.94** |
| krr K=5 riesz=rf | +0.005 | +0.004 | 0.018 | 0.98 | **0.94** |
| gbr **high-capacity** K=5 riesz=sieve | +0.044 | +0.046 | 0.013 | ~0.87 | **~0.07** |
| gbr K=5 **no correction** | +0.016 | +0.016 | 0.019 | ~0.89 | **~0.85** |

(The last two rows are from the pre-fix run and will shift by a percent or two
under the corrected SE; the high-cap collapse is bias-driven, |b|/SE=3.85, so it
stays near 0.07 regardless.)

Reading: **folds** K=2->5->10 lift coverage 0.76->0.90->0.91 (shrinking the
own-observation bias). **Matching the representer** to the learner (riesz=rf)
lifts gbr from 0.90 to 0.94 and the smooth learner from 0.92 to 0.94 — because a
random-feature representer spans a random-feature/tree error far better than a
25-dim smooth B-spline does. The **smooth learner** starts from a plug-in bias
of only +0.004 (vs +0.016 gbr, +0.045 rf) — almost nothing to correct.
**Higher-capacity boosting is catastrophic**: more capacity sharpens the
axis-aligned staircase and the boundary bias, it does not reduce it. Even the
best ML cell (gbr+rf / krr+rf, 0.94) sits a hair below the sieve plug-in's 0.94,
because the residual out-of-span component the SE cannot see never fully closes.
The corrected SE — the theta(1-theta)/n fix — is what moved these from the
0.88-0.92 band up to 0.90-0.94.

## 2. Fold count (dominant cause of the new-vs-old discrepancy)

`dd_estimate` uses 2 stratified halves (the SS quadratic correction needs two
independent half-sample fits), so each GBR sees n/2 rows, ~n/4 per arm.
`_gbr_riesz_out` uses K=5. Measured on identical draws at n=4000, varying only K:

    K=2 (1000 rows/arm): bias +0.0297
    K=5 (1600 rows/arm): bias +0.0121

That factor of ~2.5 in bias accounts for roughly 70% of the gap between the
older study's 0.93 and the new study's 0.666 at the same n.

## 3. Riesz basis: tensor B-spline (K=25) vs random features (K=200)

`run_dml_study.py` used `riesz="rf"` (200 exp-activation random features);
`dd_estimate` defaults to `riesz="sieve"` (25 tensor B-splines at segments=2).
The 200-dimensional random-feature space spans tree error far better, removing
roughly twice as much bias. Neither basis dimension scales with n anywhere in
the code — a separate defect worth fixing if the sieve-DML route is kept.

## 4. Trees violate conditions that are not merely violated but ill-defined

- `Assumption 4(b)`: `||mu_hat-mu_0||_inf * ||grad(mu_hat-mu_0)||_inf = o_p(.)`.
  For a piecewise-constant ensemble `grad(mu_hat)` is a sum of Diracs on the
  split hyperplanes — the condition has no finite value.
- The epsilon-band derivative degenerates: `{|tau_hat| < eps}` is a union of
  whole leaf cells, of measure O(1) or exactly 0, never O(eps).
- `{tau_hat >= 0}` is an axis-aligned staircase, so its H^{d-1} surface measure
  over-counts an oblique true boundary by up to sqrt(d) — the estimate is not
  rotation-invariant although the estimand is.
- Sup-norm consistency fails outright for fixed-depth, fixed-round boosting:
  the leaf-diameter floor does not shrink with n.

## 5. The attenuation mechanism (why boundary bias appears at all)

With `b(x) := E[tau_hat(x) - tau_0(x)]`, the boundary moves by
`-b/||grad tau_0||` and

    Bias(theta) = int_{margin} b(x) w(x) / ||grad tau_0(x)|| dH^{d-1}(x)
                = E[b(X) v_0(X) | tau_0(X)=0] * p_{tau_0(X)}(0).

Note the subtlety: *pure* multiplicative attenuation `b = -c*tau_0` vanishes on
the margin and would NOT move the boundary. The boundary moves because per-arm
boosting shrinks **each arm toward its own mean**, giving `b = -c(tau_0 - taubar)`
and hence `b|margin = c*taubar != 0`. Measured attenuation at n=4000:
`atten_S = 0.83`, `atten_Y = 0.93` (regression slope of fitted on true CATE).

## 6. So is the dataset ever the problem?

Not for the coverage failure — but n=854 *is* a real constraint on the
**calibration**: it starves the generative model and caps how large a simulated
n can be drawn without extrapolating. See `docs/datasets_for_calibration.md`.
The SNR ladder (`run_dml_diagnosis.py` section E) tests directly whether a
higher signal-to-noise DGP rescues sieve-DML coverage.

## Practical guidance

1. Use the tensor-sieve plug-in (or the SS-debiased version) with the corrected
   two-band SE. It is the estimator the variance formula describes.
2. If an ML first stage is required (d too large for a tensor sieve), use a
   **smooth** learner whose error the Riesz basis can represent (RBF
   random-feature ridge, i.e. `learner="krr"`), and match the Riesz basis to
   the learner's own feature map — then the projection is near-exact and the
   correction behaves as the theory intends.
3. Never use 2-fold cross-fitting for the point estimate; K=5 or 10.
4. For tree ensembles specifically, pair the point estimate with a
   full-refit bootstrap rather than the analytic SE.
