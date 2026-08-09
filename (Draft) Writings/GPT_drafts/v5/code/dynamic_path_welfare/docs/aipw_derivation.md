# Fixed-Policy IPW / AIPW Benchmarks

These benchmarks **evaluate the learned fixed policy** `(g1, g2)` on held-out data. This
is a *different object* from the population optimal-path component `V_11^*`: it is the
(1,1) contribution of a *particular, data-dependent but treated-as-fixed* policy. The
AIPW interval is a useful benchmark but does **not** by itself solve moving-boundary
inference for `V_11^*` (task §9).

## Learned policies

For each outer fold, from the training data:
```
g1(s) = 1{κ̂(s) ≥ 0},   g2(x) = 1{δ̂(x) ≥ 0}.
```

## IPW benchmark (`aipw.ipw_11`)

```
V̂_11^IPW = mean[ 1{T1=1} 1{T2=1} g1(S) g2(X) Y / (e1 e2) ],
```
using the **known design probabilities** `e1, e2` (never ML-estimated when the design is
known, task §7.3).

## Sequential AIPW score (`aipw.aipw_11`)

```
Q2(H2) = E[Y | H2, T2=1]                          (= μ_1(X) under Markov sufficiency)
M1(S)  = E[ g2(X) Q2(H2) | S, T1=1 ]              (= G_11(S) with the learned g2)

score  = g1(S) M1(S)
       + g1(S) 1{T1=1}/e1 ( g2(X) Q2(H2) − M1(S) )
       + g1(S) 1{T1=1}/e1 g2(X) 1{T2=1}/e2 ( Y − Q2(H2) ).
```

The estimator is `mean(score)`; the influence-based SE is `sd(score)/√n`.

**Neyman orthogonality.** Under a correct DGP with known probabilities, the two
augmentation terms have mean ≈ 0 (`aipw.augmentation_mean`,
`tests/test_aipw_score.py`), so the estimator is first-order insensitive to small errors
in the nuisances `Q2` and `M1`.

**Markov and a richer `Q2`.** Under the maintained Markov restriction `Q2(H2) = μ_1(X)`.
A richer-history `Q2 = E[Y | S, T1, X, T2=1]` can be substituted as a sensitivity model;
if it changes the estimate materially, the scalar-Markov interpretation is model-dependent
(cross-referenced with the Markov diagnostic in `docs/identification.md`).

## Interpretation and the key distinction

* **Fixed learned policy on held-out data** (what IPW/AIPW estimate): a *regular*,
  root-`n` object — the AIPW interval attains ≈ nominal coverage in the Monte Carlo study
  (≈ 0.92–0.95), with length ≈ 0.35 outcome SD.
* **Population optimal-path component `V_11^*`** (the paper's target): an *irregular*
  object with a slower rate `n^{−s/(2s+1)}` (see `docs/inference_derivation.md`).

Because the learned policy converges to the optimal policy, the AIPW estimate converges to
`V_11^*`, and at moderate `n` its interval is a reasonable *benchmark*. But its nominal
validity is for the fixed-policy value, and it does not account for the moving-boundary
(thin-set) contribution to the *optimal-component* target. The recommended interval for
`V_11^*` itself is the full-refit participant bootstrap (task §11.4), with the AIPW and
sieve-Riesz intervals reported alongside for comparison (Tables 12–13).
