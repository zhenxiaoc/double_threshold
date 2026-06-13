# Theory Note: Effective Dimension of the Value Functional under Sparsity

*Companion to `explore_D2_effective_dim.py`. Sketch-level — for discussion with
Xiaohong/Wayne before formalization.*

## 1. Setup

Let tau(x) = g(x_S) depend only on coordinates S, |S| = s << d_x, with
g in Lambda^sigma([0,1]^s), and let the weight w = v0 * f also depend on x_S only
(or be integrable in the remaining coordinates). Assume the s-dimensional regular
level set condition: ||grad g|| >= c > 0 on N := {x_S : g(x_S) = 0}.

## 2. The cylinder factorization

The boundary is a cylinder:

    M_0 = {x : tau(x) = 0} = N x [0,1]^{d_x - s},

with N an (s-1)-dimensional submanifold of [0,1]^s, and grad tau = (grad g, 0).
The (d_x - 1)-dimensional Hausdorff integral over a cylinder factorizes:

    DV(h0)[v] = Int_{M_0} v(x) w(x) / ||grad tau(x)|| dH^{d_x - 1}(x)
              = Int_N  v_bar(x_S) w(x_S) / ||grad g(x_S)|| dH^{s - 1}(x_S),

where  v_bar(x_S) := Int_{[0,1]^{d_x - s}} v(x_S, x_{S^c}) dx_{S^c}.

**The derivative of V only sees the S-marginal average of the first-stage error.**
This is the precise sense in which "we average over most of the dimensions":
estimation error in the nuisance coordinates is integrated out at first order.

## 3. Consequences (conjectures to formalize)

(C1) **Rate exponent is already dimension-free; sparsity relaxes the conditions.**
For V the boundary always has codimension 1, so the CG minimax rate
n^{-sigma/(2 sigma + d - m)} = n^{-sigma/(2 sigma + 1)} does not involve d_x even
without sparsity. What sparsity buys is the *side conditions*:
- restricting the sieve to S-supported functions makes the problem literally the
  s-dimensional one => plug-in rate optimality under sigma >= s (instead of
  sigma >= d_x), SS/LOO under sigma > (s+1)/2, and sieve-Riesz growth K^{1/s}
  (instead of K^{1/d_x});
- the lower bound is unchanged (the s-dimensional perturbation construction embeds
  in the sparse class), so the rate n^{-sigma/(2 sigma + 1)} remains minimax and is
  now *attainable under d_x-free smoothness conditions*.

(C2) **Index sparsity is the elegant version.** If tau(x) = g(A'x) with
A in R^{d_x x s} (unknown directions), the boundary is a generalized cylinder and
the same factorization holds along directions orthogonal to col(A): the derivative
sees only the col(A)-projection of v. This matches the numerical finding that the
*dense-index* DGP (s = 2 indices over all 50 coordinates, no coordinate sparsity)
worked at d_x = 50 with generic random features: RF capture low-index structure
without knowing A. Candidate assumption class: g sigma-smooth, s fixed; or a
dimension-free Barron-norm bound on tau (covers both versions, and gives the
K^{-1/2} RF approximation rate).

(C3) **Unknown support: screening + sample splitting.** With S unknown, a
screening step (lasso on D-interactions, etc.) recovers S under a beta-min
condition; honest inference follows by (i) split-sample screening (screen on one
half, estimate on the other — variant `screen_split` in D1), or (ii) the draft's
Appendix D cross-fitted sieve-influence-function estimator, which already
accommodates generic (including post-selection) first stages: build v*_K on the
post-screening low-dimensional sieve, cross-fit the screened first stage.

(C4) **W vs V under sparsity.** The same factorization applied to W's derivative
leaves its regular full-dimensional term intact — W's sqrt(n) rate never depended
on d_x. W's high-d failure is the second-order (Jensen) bias, i.e., a first-stage
MSE problem; sparsity rescues W by shrinking the *first stage* to s dimensions
(numerically: the oracle-support cell gives W coverage 0.95 at d_x = 50). So:
sparsity helps V through the smoothness condition and helps W through the
first-stage MSE — different channels, same structural assumption.

## 4. What the D2 simulation tests

With oracle S-supported features (the s-dimensional sieve realized exactly):
1. invariance of RMSE(V_hat), SE, and coverage in d_x in {10, 50, 100} at each n;
2. fast empirical convergence rate (tau analytic => near n^{-1/2} for the V RMSE);
3. flatness of n * Var_hat(V) in d_x.

If these hold, (C1) has direct numerical support; D1 then quantifies the cost of
*learning* S (C3), and D3 tests the W channel of (C4).

## 5. Formalization plan (suggested)

1. Lemma (cylinder factorization of the pathwise derivative) — pure calculus,
   co-area formula; should be short.
2. Theorem (upper bound): S-supported sieve (or post-screening sieve with sample
   splitting) attains n^{-sigma/(2 sigma + 1)} for V under the s-dimensional
   versions of the CG conditions; statement parallel to Theorem 3/5 of the draft
   with (d_x, K^{1/d_x}) replaced by (s, K^{1/s}).
3. Theorem (lower bound): trivial embedding of the s-dimensional construction.
4. Index version: replace coordinates by directions; the screening step becomes
   index estimation (ADE / sliced inverse regression) or is bypassed entirely by
   RF + a Barron-class assumption (the route our d_x = 50 dense-index simulation
   validates numerically).
