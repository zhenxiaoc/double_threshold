# Inference Derivation for `V_11^*`

This document derives the inference procedure the code implements and states the
expected convergence rate, drawing on the local theory (`may_2026.tex`) and the
supporting literature that was read for this project. Claims attributed to a paper were
checked against its source; where a paper does *not* establish something, that is said.

---

## 1. Why `V_11^*` is irregular while `V^*` is regular

From `docs/theory_summary.md` §4–5, the pathwise derivative of the (1,1) component is

```
V̇_11^* = (I) interior  +  (II) second-stage boundary over M2  +  (III) first-stage boundary over M1,
```

whereas for the **total** welfare `V^*` the boundary terms cancel (their combined weights
are `δ` on `M2` and `κ` on `M1`, both zero on their margins). This is the dynamic
analogue of the welfare-vs-value distinction:

* **Whitehouse, Chen, Austern & Syrgkanis (2026)** and **Chen, Austern & Syrgkanis
  (2023)** show the *optimal policy value* is **root-`n` regular** — asymptotically
  linear with a bona fide influence function — under a margin (anti-concentration)
  condition, using softmax smoothing of the max. Efficiency holds in the
  zero-non-response case. These papers target the max/total value, exactly the object
  for which the envelope cancellation restores regularity.
* **Feng, Hong & Nekipelov (2026)** reach the same conclusion for allocation/welfare
  values via a functional-delta method: the first-order effect of estimating the
  decision rule is a boundary (coarea) integral over the tie set, which vanishes by an
  envelope argument under a margin condition, so the plug-in/one-step value estimator is
  root-`n` and multiplier-bootstrap CIs are valid.
* **None of these four papers derives inference for a path *component*** such as
  `V_11^*`; that derivation is the contribution of `may_2026.tex`. For a path component,
  the boundary weight (`μ_1`, `μ_0`, or `G_ab`) does **not** vanish on the margin, so the
  thin-set boundary term survives and drives the first-order behaviour.

## 2. The scalar-state rate (Chen & Gao 2026)

**Chen & Gao (2026), "Thin Sets Are Not Equally Thin"** give the minimax rate for
estimating a submanifold-integral functional `∫_M h_0 w dH^m` of a smoothness-`s`
function over an `m`-dimensional submanifold `M ⊂ R^d` (codimension `d−m`):

```
r_n^*  =  n^{ − s / (2s + d − m) }.
```

Special cases they state explicitly:
* `m = d` (full-dimensional Lebesgue integral): `r_n^* = n^{−1/2}` — **root-`n`, regular**.
* `m = 0`: `r_n^* = n^{−s/(2s+d)}`, the **Stone (1980) point-evaluation** rate (`H^0` =
  counting measure).
* upper-contour / level-set (codimension 1): `n^{−s/(2s+1)}` for any ambient `d`.

The message of "not equally thin": every `m < d` set has zero Lebesgue measure and is
therefore irregular (no `√n` influence function), but difficulty is governed by the
**codimension** `d − m` — integrating over the `m` intrinsic directions averages out
that part of the noise; only the `d − m` transverse directions stay hard.

**Specialization to our scalar problem.** With `d_x = d_s = 1`, the margins
`M2 = {x : δ(x)=0}` and `M1 = {s : κ(s)=0}` are **finite sets of points** — `m = 0`,
codimension `d−m = 1`. Plugging `d = 1, m = 0`:

```
r_n^*  =  n^{ − s / (2s + 1) }   <   n^{−1/2}   for every finite s.
```

(The `m=0` formula `n^{−s/(2s+d)}` with `d=1` and the codimension-1 upper-contour rate
`n^{−s/(2s+1)}` coincide here.) So a plug-in estimator of the path component `V_11^*`
converges at the **nonparametric point-evaluation rate `n^{−s/(2s+1)}`, strictly slower
than root-`n`** — the thin-set irregularity, specialized to a scalar state.

*Caveat (do not over-claim).* Chen & Gao's theory is written for an integral of a single
nonparametric object over a level set of a (possibly estimated) function. `V_11^*` has
both an estimated boundary *location* (`δ`, `κ`) and an estimated boundary *weight*
(`μ_1`, `G_11`); importing the rate is justified by their nonlinear/unknown-submanifold
extension, but the derivation of `V̇_11^*` itself is the `may_2026.tex` contribution.

## 3. Sieve-Riesz construction (implemented in `riesz.py`)

Write `μ_a(x) = b_K(x)' β_a`. The population-sieve functional `V_11(β_0, β_1)` (with the
estimated densities `m, p_a` held fixed) is computed by quadrature with **exact sub-grid
region weights** on both the `x`-integral (`{δ≥0}`) and the `s`-integral (`{κ≥0}`), so it
is a smooth function of `(β_0, β_1)`. Its analytic gradient is the three-term
moving-boundary derivative of §4 of the theory summary, with the Hausdorff integrals
reduced to sums over the roots of `δ` and `κ` weighted by `1/|δ'|` and `1/|κ'|`.

* **Validation.** The analytic gradient is checked against central finite differences of
  the smooth population functional at several step sizes. In the implemented runs the
  relative discrepancy is `< 1e-4` and step-independent (`tests/test_riesz_derivative.py`),
  confirming the interior + `M2` + `M1` terms are coded correctly.
* **Riesz representer.** `α_a = G_a^{-1} ∇_{β_a} V_11`, where `G_a` is the empirical sieve
  Gram matrix `(1/n_a) Σ_{T2=a} b_K(X) b_K(X)'`. Chen & Gao's sieve-representer norm grows
  like `‖α‖² ≍ K^{(d−m)/d} = K` in the scalar case — the sieve signature of the
  codimension-1 irregularity (the representer is *not* `O(1)`).
* **Influence score (conditional on densities).**
  `ψ_i = Σ_a 1{T2_i=a}/P(T2=a) · (α_a' b_K(X_i)) · (Y_i − μ_a(X_i))`.
  The reported variance is `Var(ψ)/n`.

## 4. What variance is included, and what is omitted

The implemented sieve-Riesz variance is **conditional on the estimated densities `m` and
`p_a`** (task §11.2, option C). It captures the `μ`-driven boundary and interior
uncertainty (the boundary sensitivity to `μ` is inside `∇V_11`), but it **omits**:

1. the sampling uncertainty in the baseline density `m(s)`;
2. the sampling uncertainty in the transition law `p_a(x|s)` — which is large here,
   because `κ(s) = ∫ V2 r(x|s) dx` and hence the entire first-stage margin `M1` are
   driven by `p_a`.

**Monte Carlo evidence (this project).** On the calibrated regular DGP, the conditional
SE is only ≈ 40–45% of the true Monte Carlo SD (`se_ratio ≈ 0.43`), and the conditional
95% interval covers ≈ 62%. This is the honest, expected consequence of omitting the
transition-law uncertainty; the conditional interval is therefore **labelled explicitly
and is not presented as the complete sampling variance.**

**Complete-variance interval.** The **full-refit participant bootstrap** re-estimates all
nuisances (including `m`, `p_a`) on each resample and therefore reflects the full
uncertainty. In the Monte Carlo study it attains ≈ nominal coverage with informative
length (≈ 0.45 outcome SD). It is the **recommended primary interval**; the multiplier
bootstrap and `m`-out-of-`n` subsampling are reported as robustness. A naive percentile
bootstrap is *not* asserted to be theoretically valid for the hard-threshold target — its
behaviour is checked, not assumed (task §11.4).

*Toward option A/B.* A fully analytic complete-variance expansion would add the regular
influence contributions of `m` and `p_a` to the irregular `μ`-representer (a hybrid
expansion). Because the boundary component converges slower than root-`n`, those regular
terms are asymptotically smaller — but this must be argued and checked, not assumed. The
present implementation instead relies on the participant bootstrap for the complete
variance and reports the conditional analytic SE alongside it, with the `se_ratio`
diagnostic making the gap explicit.

## 5. Undersmoothing and sieve sensitivity

Because prediction-optimal `K` may be too small for inference (bias not negligible
relative to the inflated boundary-rate SE), results are reported for the CV-selected `K`,
`K+1`, `K+2`, and a smaller `K`. The **primary inferential `K` is fixed from the
calibrated simulation before the final point estimate is read** (`inference.primary_dim`).
The sieve-variance scaling `K/n` means the interval width shrinks at the slower boundary
rate, as it must for an irregular functional.
