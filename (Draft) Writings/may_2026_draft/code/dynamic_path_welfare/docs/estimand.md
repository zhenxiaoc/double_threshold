# Estimand

The target is the **welfare contributed by the (1,1) treatment path under the estimated
optimal two-stage regime**:

```
V_11^* = E[ Y^(1,1) · 1{π1^*(S)=1, π2^*(X^(1))=1} ]
       = E[ Y^(1,1) · 1{δ(X^(1)) ≥ 0} · 1{κ(S) ≥ 0} ]
       = ∫_{D1+} ∫_{D2+} μ_1(x) p_1(x|s) m(s) dx ds
       = ∫_{D1+} G_11(s) m(s) ds,    G_11(s) = ∫_{D2+} μ_1(x) p_1(x|s) dx.
```

## What it is NOT (guard-rails)

* **Not** `E[Y^(1,1)]` (the value of always assigning (1,1)).
* **Not** the value of any *fixed* policy — the regions `D1+, D2+` are the *estimated
  optimal* decision regions.
* **Not** the total optimal welfare `V^* = V_11^* + V_10^* + V_01^* + V_00^*`. `V_11^*` is
  one of four components.

Units whom the optimal regime routes down any path other than (1,1) contribute **exactly
zero** to `V_11^*`. It is an *unconditional population contribution*, so
`V_11^* ≤ E[Y^(1,1)]` and `V_11^* ≤ V^*`.

## Identification (from `may_2026.tex`)

Under consistency, first- and second-period sequential independence, positivity, and
Markov sufficiency:

1. condition on the potential intermediate state `X^(1)`;
2. sequential independence + positivity ⇒ `f_{X^(1)|S}(x|s) = p_1(x|s)`;
3. Markov sufficiency ⇒ `E[Y|S,T1=1,X,T2=1] = μ_1(x)`;
4. the optimal-rule thresholds `1{δ≥0}`, `1{κ≥0}` (tie → action 1) give the region form.

Each identifying step maps to a specific assumption, so a failure of any one is
attributable (see `docs/identification.md`).

## The four components and the sum check

```
V_11^* = ∫_{D1+} ∫_{D2+} μ1 p1 m     V_10^* = ∫_{D1+} ∫_{D2-} μ0 p1 m
V_01^* = ∫_{D1-} ∫_{D2+} μ1 p0 m     V_00^* = ∫_{D1-} ∫_{D2-} μ0 p0 m
V^*    = V_11^* + V_10^* + V_01^* + V_00^*   (component-sum identity).
```

The estimator enforces this identity numerically: `estimate_all_paths()` reports the
component-sum residual, `tests/test_path_sum.py` checks it is `< 1e-9` (cross-fitted
components) and `< 1e-6` against the direct total, and the residual is reported in
Table 11.

## Estimation (see `docs/theory_summary.md` §3 and `estimator.py`)

Cross-fitted direct-regression plug-in with honest inner cross-fitting:
`μ_a` (stage-two spline regressions) → `δ, V2` → `A_a, κ` and `G_ab` (stage-one spline
regressions on honest pseudo-outcomes) → held-out contributions
`G_ab(S_i) · 1{κ(S_i) (≥0 for a·1st-stage=1 else <0)}` → sample mean. A separate
**density-based plug-in** (`riesz.SievePopFunctional`) recomputes `V_11` by quadrature
from estimated `m, p_a` and is compared to the direct estimate (task §8.3); large
disagreement is reported, not averaged away.
