# Theory Summary

**Primary source.** `may_2026.tex` (Zhenxiao Chen, "Inference on Policy Value under
Admissibility-Constrained Optimal Treatment Rules"), Section 2.4 "Dynamic Treatment
Regime". This document restates that theory exactly, in the notation used by the code.
Where this file paraphrases, the underlying equations are those of `may_2026.tex`; no
result here is invented.

The target parameter of the whole project is the **welfare contribution of the (1,1)
treatment path under the estimated optimal two-stage regime**, `V_11^*`. It is *not* the
value of a fixed policy and *not* the total optimal value `V^*`. This distinction is the
central point of the paper and is preserved throughout the code.

---

## 1. Observed-data model

One independent record per experimental unit:

```
O_i = (S_i, T1_i, X_i, T2_i, Y_i)
```

with the causal ordering

```
S  ->  T1  ->  X  ->  T2  ->  Y
```

* `S ∈ 𝒮` — first-period (baseline) state, measured before `T1`. Scalar, continuous.
* `T1 ∈ {0,1}` — first-period binary treatment.
* `X ∈ 𝒳` — second-period (intermediate) state, measured after `T1` and before `T2`.
  Scalar, continuous.
* `T2 ∈ {0,1}` — second-period binary treatment.
* `Y ∈ ℝ` — final scalar outcome, measured after `T2`.

Dimensions in the **primary specification**: `d_s := dim(𝒮) = 1`, `d_x := dim(𝒳) = 1`.

Potential outcomes. For static assignments `(τ1,τ2) ∈ {0,1}²`, `Y^(τ1,τ2)` is the
potential final outcome and `X^(τ1)` the potential intermediate state. A first-period
policy is `π1: 𝒮 → {0,1}`; a second-period policy is `π2: 𝒳 → {0,1}`. Given `π2`,

```
Y^(τ1, π2) := Σ_{τ2} 1{π2(X^(τ1)) = τ2} · Y^(τ1,τ2),
Y^(π1, π2) := Σ_{τ1} 1{π1(S)   = τ1} · Y^(τ1, π2),
V(π1,π2)  := E[ Y^(π1,π2) ].
```

The optimal regime is `(π1^*, π2^*) ∈ argmax V(π1,π2)`, with optimal value `V^* :=
V(π1^*, π2^*)`.

---

## 2. Assumptions (Assumption 2.5 of `may_2026.tex`)

1. **Consistency.** If `T1 = τ1` then `X = X^(τ1)`; if `(T1,T2) = (τ1,τ2)` then
   `Y = Y^(τ1,τ2)`.
2. **First-period sequential independence.** For every `τ1` and every admissible `π2`,
   `(X^(τ1), Y^(τ1,π2)) ⟂ T1 | S`.
3. **Second-period sequential independence.** For every `τ1,τ2`,
   `Y^(τ1,τ2) ⟂ T2 | S, T1=τ1, X`.
4. **Markov sufficiency.**
   `E[Y | S=s, T1=τ1, X=x, T2=τ2] = E[Y | X=x, T2=τ2]`.
   *Randomization does NOT imply this.* It is a modelling restriction that makes the
   scalar-state estimator theorem-aligned; the code treats it as testable and reports
   diagnostics (see `docs/identification.md`, `identification.py`).
5. **Positivity.** For all relevant `s,x` and all `τ1,τ2`,
   `P(T1=τ1 | S=s) > 0` and `P(T2=τ2 | S=s, T1=τ1, X=x) > 0`.
   With history-dependent second-stage availability, replace the second inequality by an
   *availability-relative* positivity condition (see §7).

---

## 3. Definitions

For `a ∈ {0,1}`:

| Symbol | Definition | Meaning |
|---|---|---|
| `p_a(x\|s)` | `f_{X\|S,T1=a}(x\|s)` | transition law of `X` given `S`, under `T1=a` |
| `r(x\|s)` | `p_1(x\|s) − p_0(x\|s)` | first-stage density contrast |
| `m(s)` | `f_S(s)` | baseline-state density |
| `μ_a(x)` | `E[Y \| X=x, T2=a]` | second-stage outcome regression |
| `δ(x)` | `μ_1(x) − μ_0(x)` | second-stage treatment contrast |
| `D2⁺` | `{x : δ(x) ≥ 0}` | second-stage treated region (tie → 1) |
| `D2⁻` | `{x : δ(x) < 0}` | second-stage untreated region |
| `M2` | `{x : δ(x) = 0}` | second-stage margin (level set) |
| `V2(x)` | `max{μ0(x),μ1(x)} = μ0(x) + [δ(x)]₊` | second-stage value function |
| `A_a(s)` | `E[V2(X) \| S=s, T1=a] = ∫ V2(x) p_a(x\|s) dx` | continuation value |
| `κ(s)` | `A_1(s) − A_0(s) = ∫ V2(x) r(x\|s) dx` | first-stage contrast |
| `D1⁺` | `{s : κ(s) ≥ 0}` | first-stage treated region (tie → 1) |
| `D1⁻` | `{s : κ(s) < 0}` | first-stage untreated region |
| `M1` | `{s : κ(s) = 0}` | first-stage margin (level set) |
| `G_11(s)` | `∫_{D2⁺} μ1(x) p_1(x\|s) dx` | (1,1) second-stage component |

**Tie-breaking convention:** action `1` is chosen when a contrast is exactly zero (hence
the `≥ 0` in `D1⁺`, `D2⁺`). This is encoded once, in `estimator._TIE_TO_ONE`.

**Optimal-regime characterization** (Proposition 2.6): `1{π2^*(X)=1} = 1{δ(X) ≥ 0}` and
`1{π1^*(S)=1} = 1{κ(S) ≥ 0}`.

### Target parameter

```
V_11^* := E[ Y^(1,1) · 1{π1^*(S)=1, π2^*(X^(1))=1} ]
        = ∫_{D1⁺} G_11(s) m(s) ds
        = ∫_{D1⁺} ∫_{D2⁺} μ1(x) p_1(x|s) m(s) dx ds
        = E[ Y^(1,1) · 1{δ(X^(1)) ≥ 0} · 1{κ(S) ≥ 0} ].
```

This is an **unconditional population contribution**: units whom the optimal regime sends
down any other path contribute exactly zero. It is *not* `E[Y^(1,1)]`, *not* the value of
"always (1,1)", and *not* `V^*`.

**Identification** (last display of the derivation in `may_2026.tex`, lines 358–406) uses
consistency + sequential independence + positivity to replace `f_{X^(1)|S}` by `p_1`, and
Markov sufficiency to replace `E[Y|S,T1=1,X,T2=1]` by `μ_1(x)`.

### Path decomposition and the sum check

```
G_11(s)=∫_{D2⁺} μ1 p1 dx   G_10(s)=∫_{D2⁻} μ0 p1 dx
G_01(s)=∫_{D2⁺} μ1 p0 dx   G_00(s)=∫_{D2⁻} μ0 p0 dx
A1 = G_11 + G_10           A0 = G_01 + G_00
V_11^*=∫_{D1⁺} G_11 m ds   V_10^*=∫_{D1⁺} G_10 m ds
V_01^*=∫_{D1⁻} G_01 m ds   V_00^*=∫_{D1⁻} G_00 m ds
V^*   = V_11^* + V_10^* + V_01^* + V_00^*
      = ∫_{D1⁺} A1 m ds + ∫_{D1⁻} A0 m ds.
```

The four components **must** sum to total welfare `V^*` up to numerical / cross-fitting
error. `tests/test_path_sum.py` enforces this; `estimate_all_paths()` reports the residual.

---

## 4. Moving-boundary derivative

Perturb the second-stage regressions only: `μ_{a,t} = μ_a + t·μ̇_a`, holding `p_a` and
`m` fixed. Write `δ̇ = μ̇_1 − μ̇_0`. For a smooth region integral with a moving boundary,

```
d/dt ∫_{h_t ≥ 0} q_t(z) dz |_{t=0}
   = ∫_{h ≥ 0} q̇(z) dz                                     (interior)
   + ∫_{h = 0} [ ḣ(z) q(z) / ‖∇h(z)‖ ] dH^{d-1}(z)          (boundary)
```

with the boundary sign reversed for `{h_t < 0}`. Applying this twice gives the derivative
of `V_11^*` as **three terms**:

```
V̇_11^* =
  (I)  ∫_{D1⁺} ∫_{D2⁺} μ̇1(x) p1(x|s) m(s) dx ds                                   interior
  (II) ∫_{D1⁺} ∫_{M2} [ δ̇(x) μ1(x) p1(x|s) / ‖∇δ(x)‖ ] m(s) dH^{d_x−1}(x) ds     2nd-stage boundary
  (III)∫_{M1} [ κ̇(s) G_11(s) m(s) / ‖∇κ(s)‖ ] dH^{d_s−1}(s)                        1st-stage boundary
```

where the first-stage boundary velocity uses

```
κ̇(s) = ∫_{D2⁺} μ̇1(x) r(x|s) dx + ∫_{D2⁻} μ̇0(x) r(x|s) dx.
```

* **Term (I) interior** — perturbs `μ1` for states that remain on the (1,1) path. This is
  an ordinary `L2`-type (regular) contribution.
* **Term (II) second-stage boundary** — the first-order mass of second-stage states
  crossing `M2`. The weight on `M2` is `μ1(x)` (generally nonzero).
* **Term (III) first-stage boundary** — the first-order mass of first-stage states
  crossing `M1`. The weight on `M1` is `G_11(s)` (generally nonzero).

`δ̇/‖∇δ‖` and `κ̇/‖∇κ‖` are the **normal velocities** of the two decision boundaries.

### Scalar-state reduction (the case we implement)

With `d_x = d_s = 1`, the level sets `M2` and `M1` are **finite sets of roots**, and each
Hausdorff `H^{d−1}` integral collapses to a **sum over roots** with `dH^0 =` counting
measure:

```
Term (II)  →  ∫_{D1⁺} Σ_{x_j ∈ roots(δ)} [ δ̇(x_j) μ1(x_j) p1(x_j|s) / |δ'(x_j)| ] m(s) ds
Term (III) →  Σ_{s_k ∈ roots(κ)} [ κ̇(s_k) G_11(s_k) m(s_k) / |κ'(s_k)| ]
```

`|δ'(x_j)|` and `|κ'(s_k)|` are the absolute analytic spline derivatives at the roots.
This is exactly the "sum over roots" form the code implements in `riesz.py` /
`boundaries.py`.

---

## 5. Total welfare vs. one path component

Differentiating the four components and summing, **the boundary terms cancel**:

* Second-stage: `Σ_ab B_ab^{M2}` has combined weight `μ1(x) − μ0(x) = δ(x)`, which is `0`
  on `M2` ⇒ cancels.
* First-stage: `Σ_ab B_ab^{M1}` has combined weight `A1(s) − A0(s) = κ(s)`, which is `0`
  on `M1` ⇒ cancels.

Hence

```
V̇^* = ∫_{D1⁺}∫_{D2⁺} μ̇1 p1 m + ∫_{D1⁺}∫_{D2⁻} μ̇0 p1 m
     + ∫_{D1⁻}∫_{D2⁺} μ̇1 p0 m + ∫_{D1⁻}∫_{D2⁻} μ̇0 p0 m,
```

which contains **only full-dimensional Lebesgue integrals** — an ordinary `L2`-regular
derivative. The **total welfare is regular** (envelope / self-selection cancellation),
while **individual path components are generally irregular**: their derivatives retain the
thin-set (Hausdorff) boundary functionals with nonzero weights `μ1, μ0, G_ab`. This is the
dynamic analogue of the "welfare (regular) vs. value (irregular)" distinction, and it is
*the* reason `V_11^*` needs the special inference of §7 rather than a plain root-n
influence-function standard error.

---

## 6. Regular-level-set (regular-margin) conditions

The derivative is valid only if `M2` and `M1` are **thin, regular** margins:

* `δ` and `κ` are `C¹` near `M2` and `M1`;
* `0` is a **regular value** of both: `‖∇_x δ(x)‖ > 0` on `M2`, `‖∇_s κ(s)‖ > 0` on `M1`;
* consequently `M2, M1` are smooth `(d_x−1)`- and `(d_s−1)`-dimensional level sets — in
  the scalar case, isolated regular roots with nonzero derivative.

If these fail (a level set of positive Lebesgue measure, or a vanishing gradient on the
margin — a tangential/degenerate crossing), the moving-boundary expansion with
`1/‖∇δ‖`, `1/‖∇κ‖` need not hold: the component may be only directionally
differentiable, or lack a linear pathwise derivative. The code therefore **checks
regularity at every estimated root** (nonzero derivative, interior location, adequate
local support, stability across folds and sieve dimensions) and *flags weak margins*
rather than forcing a root — see `boundaries.py`, `boundary_diagnostics()`.

---

## 7. Role of the sieve Riesz representer

Write the second-stage regressions in a sieve basis, `μ_a(x) = b_K(x)' β_a`. Then every
functional above (`δ, V2, κ, G_11, V_11`) is a known map of `(β_0, β_1)` (given the fixed
`p_a, m`), computable by quadrature, and the perturbation `μ̇_a = b_K' β̇_a` makes the
derivative `V̇_11^*` **linear in `(β̇_0, β̇_1)`**:

```
V̇_11^* = ⟨ ∇_{β0} V_11 , β̇_0 ⟩ + ⟨ ∇_{β1} V_11 , β̇_1 ⟩,
```

with the two gradient vectors assembled from terms (I)–(III) of §4 (interior integrals +
root sums). The **sieve Riesz representer** `α_a` solves, in the sieve inner product
defined by the Gram matrix `G_a = E[b_K(X) b_K(X)' | T2=a]` (with design/propensity
weights),

```
⟨ α_a , b_K ⟩_{G_a} = ∇_{β_a} V_11   ⇒   α_a = G_a^{-1} ∇_{β_a} V_11.
```

The associated influence contribution for a treated-at-stage-2 observation is the
Riesz-representer evaluated against the least-squares score, e.g. for `T2=a`:

```
ψ_a(O) = 1{T2=a}/e2 · α_a' b_K(X) · (Y − μ_a(X)),
```

plus the regular contributions from `m` and `p_a` (§ inference_derivation.md). The
**sieve variance** is the empirical variance of the total influence score; the
**studentized statistic** is `(V̂_11 − V_11^*)/ ŝe`. Analytic moving-boundary gradients
are checked against **central finite differences** applied to the quadrature-defined
population-sieve functional at several step sizes (`tests/test_riesz_derivative.py`).

Because term (II)/(III) are **thin-set** (point-evaluation) functionals, the representer
`α_a` has sieve norm that **grows with `K`**: `‖α_a‖` is not `O(1)`. This is the sieve
signature of irregularity and drives the rate discussion below.

---

## 8. Expected convergence rate in the scalar-state case

Decompose `V̂_11 − V_11^*` into the interior part (term I) and the boundary parts
(terms II, III).

* **Interior (term I)** is a region-integral / regular functional: `√n`-estimable with an
  ordinary `L2` influence function.
* **Boundary (terms II, III)** are **thin-set functionals**. In `d_x = d_s = 1` the
  margins are 0-dimensional (points), and the boundary contribution is a finite sum of
  quantities of the form `μ1(x_j) p1(x_j|s) / |δ'(x_j)|` (and the `κ` analogue). Its
  plug-in error is dominated by (a) **root-location error** `|x̂_j − x_j| = O_p(‖δ̂−δ‖_∞ /
  |δ'(x_j)|)` and (b) **point-evaluation error** of `μ`, `p_1`, `m` at the root.

Point evaluation of a spline/sieve estimate with `K` terms and Hölder smoothness `s` has
pointwise rate

```
| μ̂(x) − μ(x) |  ≍  (K/n)^{1/2}  (variance)  +  K^{−s}  (bias),
```

minimized at `K ≍ n^{1/(2s+1)}` to give the nonparametric pointwise rate
`n^{−s/(2s+1)}`, which is **slower than `n^{−1/2}`** for every finite `s`. The 0-dimensional
Hausdorff integral inherits this pointwise rate, so **`V̂_11` converges at the
nonparametric point-evaluation rate `n^{−s/(2s+1)}`, not at `√n`** — this is the "thin set"
irregularity of Chen & Gao (2026), specialized to codimension-`d` (here `d_x = d_s = 1`)
level sets. (See `docs/inference_derivation.md` for the formal statement drawn from the
supporting literature; this file states the expected rate, the derivation lives there.)

**Practical consequence for inference.** (i) Do **not** attach a plain root-`n` regression
standard error to `V̂_11` (§11 of the task). (ii) Use the **sieve variance** of the
influence score, which scales like `‖α‖²/n ≍ K/n` — the interval width shrinks at the
slower boundary rate, as it must. (iii) **Undersmooth** (choose `K` larger than the
prediction-optimal value) so the bias is asymptotically negligible relative to this
inflated standard error, making the studentized statistic asymptotically normal. (iv) The
total welfare `V^*`, by contrast, *is* `√n`-regular (the boundary terms cancel), so its
inference is standard — the code reports both, to make the regular/irregular contrast
concrete.

---

## 9. What the code must never do (guard-rails from the theory)

* Never replace `V_11^*` by a fixed-policy value or by `V^*`.
* Never treat the four path components as anything other than a decomposition that *sums*
  to `V^*`.
* Never attach an ordinary √n influence-function SE to the plug-in `V̂_11`.
* Never force a unique root, and never manipulate spline dimension / cost / transform to
  manufacture a zero crossing.
* Never differentiate the hard indicator directly; always use the analytic moving-boundary
  derivative (roots + `1/|δ'|`, `1/|κ'|`), verified against finite differences.
* Never call the Markov restriction "verified"; report diagnostics and label the scalar
  estimator's interpretation model-dependent when the rich model improves fit.
