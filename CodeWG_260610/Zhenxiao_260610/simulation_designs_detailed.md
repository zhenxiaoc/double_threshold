# Detailed Simulation Designs in `CodeWG_260610`

This note documents the simulation and empirical-design files under
`CodeWG_260610/rf_sieve`.  It is written as a design reference, with emphasis on
the mathematical data-generating processes, estimands, estimators, variance
estimators, debiasing corrections, and Monte Carlo grids used by each script.

Source files covered:

- `rf_sieve_highd_sim.py`
- `rf_sieve_v_highdim_explore.py`
- `rf_sieve_lib.py`
- `explore_D1_screening.py`
- `explore_D2_effective_dim.py`
- `explore_D3_W_debias.py`
- `explore_D4_tuning.py`
- `sweep_S1_robustness.py`
- `sweep_S2_smoothness.py`
- `sweep_S3_pipeline.py`
- `sweep_S4_failure_modes.py`
- `sweep_S5_dml_if.py`
- `sweep_S6_fixes.py`
- `sweep_S7_extreme_share.py`
- `sweep_S7b_aug_se.py`
- `check_loo_v_delta.py`
- `jtpa_rf_application.py`

The `r_run_logs` folder contains an audit of R-script executability and missing
packages; it does not itself define additional Python simulation designs.

## 1. Common Statistical Setup

Each synthetic simulation observes independent draws

$$
Z_i = (Y_i, D_i, X_i), \qquad i = 1,\ldots,n,
$$

where

$$
X_i \sim \mathrm{Unif}([0,1]^{d_x}), \qquad
D_i \mid X_i \sim \mathrm{Bernoulli}(p_0(X_i)),
$$

and

$$
Y_i = b_0(X_i) + D_i \tau_0(X_i) + \sigma(D_i, X_i)\varepsilon_i,
\qquad
\varepsilon_i \sim N(0,1).
$$

The treatment-specific conditional means are

$$
\mu_0(x,0) = b_0(x), \qquad
\mu_0(x,1) = b_0(x) + \tau_0(x),
$$

and the conditional average treatment effect is

$$
h_0(x) = \tau_0(x) = \mu_0(x,1) - \mu_0(x,0).
$$

The default synthetic simulations use homoskedastic errors,

$$
\sigma(D_i,X_i) = 1.
$$

The heteroskedastic cells use treatment-arm heteroskedasticity,

$$
\sigma(D_i,X_i)
=
\begin{cases}
1.25, & D_i = 1,\\
0.75, & D_i = 0.
\end{cases}
$$

Unless otherwise stated, the target distribution for welfare and value is the
known uniform distribution

$$
F = \mathrm{Unif}([0,1]^{d_x}).
$$

## 2. Common Estimands

Two target functionals are used throughout the synthetic simulations.

### 2.1 Welfare Functional

The welfare functional is

$$
W_0
= W(h_0)
= \int [h_0(x)]_+\,dF(x)
= E_F\{\max(\tau_0(X),0)\}.
$$

In the simulation code the welfare weight is effectively $v_0(x)=1$.

### 2.2 Value or Treated-Share Functional

The value/share functional is

$$
V_0
= V(h_0)
= \int 1\{h_0(x) \ge 0\}\,dF(x)
= P_F\{\tau_0(X) \ge 0\}.
$$

This is the irregular functional.  Its first-order derivative involves the
boundary

$$
M_0 = \{x : h_0(x)=0\}.
$$

The simulations approximate this derivative by an epsilon-band derivative
around the estimated boundary.

## 3. Core DGP Family

Let

$$
s(z) = \frac{1}{1+\exp(-z)}
$$

be the logistic function.  For dense-index designs define

$$
a_1
= \frac{1}{\sqrt{d_x}}(1,-1,1,-1,\ldots)',
\qquad
a_2
= \frac{1}{\sqrt{d_x}}(1,1,\ldots,1)',
$$

and center the two indices at the center of the unit cube:

$$
m_1 = \frac{1}{2}\sum_{j=1}^{d_x} a_{1j},
\qquad
m_2 = \frac{1}{2}\sum_{j=1}^{d_x} a_{2j}.
$$

The center of the unit cube is

$$
x_c = (1/2,\ldots,1/2)'.
$$

For any direction $a$, the scalar index $a'x$ places $x$ on the
one-dimensional axis defined by $a$.  The "projection of the center" onto
that axis is simply the dot product

$$
a'x_c
=
\frac{1}{2}\sum_{j=1}^{d_x}a_j.
$$

Thus $m_1=a_1'x_c$ and $m_2=a_2'x_c$.  Subtracting $m_\ell$ from
$a_\ell'x$ centers the index at the cube center:

$$
a_\ell'x - m_\ell
=
a_\ell'x - a_\ell'x_c
=
a_\ell'(x-x_c),
\qquad \ell\in\{1,2\}.
$$

So when $x=x_c$, both centered indices are zero.  This keeps the sigmoid
terms centered at the middle of the covariate space and prevents their scale or
location from drifting as $d_x$ changes.

The default constants are

$$
\mathrm{TAU\_SCALE}=3,\qquad \mathrm{shift}=-0.70,\qquad
\mathrm{overlap}=1.
$$

### 3.1 Dense-Index DGP

The dense-index CATE depends on two dense linear indices:

$$
\tau_0(x)
= \mathrm{TAU\_SCALE}
\left[
s\{3(a_1'x-m_1)\}
+ \frac{1}{2}s\{3(a_2'x-m_2)\}
+ \mathrm{shift}
\right].
$$

The baseline regression is

$$
b_0(x)
= 2\tanh\{2(a_2'x-m_2)\}
+ \sin\{2\pi(a_1'x-m_1)\}.
$$

The propensity score is

$$
p_0(x)
= s\left(
\mathrm{overlap}
\left[
1.2(a_1'x-m_1) + 0.4(a_2'x-m_2)
\right]
\right).
$$

Here `overlap` is a scalar multiplier on the propensity-score logit index.
The name is slightly counterintuitive: larger values of `overlap` make
treatment assignment more deterministic and therefore make covariate overlap
worse.  For example, `overlap = 1` is the default comfortable-overlap design,
`overlap = 3` or `overlap = 6` pushes propensities closer to 0 and 1, and
`overlap = 0` would give $p_0(x)=1/2$ for every $x$, i.e. perfect
randomized assignment.

This DGP is high-dimensional in ambient coordinates but low-dimensional in
index structure.

### 3.2 Sparse-Coordinate DGP

The sparse DGP depends only on the first two coordinates:

$$
\tau_0(x)
= \mathrm{TAU\_SCALE}
\left[
s\{3(x_1-1/2)\}
+ \frac{1}{2}s\{3(x_2-1/2)\}
+ \mathrm{shift}
\right].
$$

The baseline regression is

$$
b_0(x)
= 2\tanh\{2(x_1-1/2)\}
+ \sin\{2\pi(x_2-1/2)\}.
$$

The propensity score is

$$
p_0(x)
= s\left(
\mathrm{overlap}
\left[
1.2(x_1-1/2) + 0.4(x_2-1/2)
\right]
\right).
$$

The remaining coordinates $x_3,\ldots,x_{d_x}$ are pure noise.  The true
active-coordinate set is

$$
J_0 = \{1,2\}.
$$

Here "support" is being used in the variable-selection sense: it is a set of
coordinate labels, not the support of a function or distribution on
$[0,1]^{d_x}$.  Thus $J_0=\{1,2\}$ means that only the first and second
coordinates, $x_1$ and $x_2$, enter the DGP.  In the Python scripts, this same
active-coordinate set is written as the list literal `[0, 1]`.  The square
brackets here are Python syntax for a finite list, not the mathematical
interval $[0,1]$.  Python counts coordinates from zero, so list entry `0`
means coordinate $x_1$ and list entry `1` means coordinate $x_2$.

### 3.3 Kink DGP

The kink DGP is used to test low-smoothness behavior.  For exponent $p$,

$$
\tau_0(x)
= \mathrm{TAU\_SCALE}
\left(
|x_1-0.35|^p - 0.25^p
\right).
$$

The boundary is the set of full covariate vectors $x\in[0,1]^{d_x}$ at which
$\tau_0(x)=0$:

$$
M_0
=
\{x\in[0,1]^{d_x}: |x_1-0.35|^p = 0.25^p\}.
$$

Equivalently,

$$
M_0
=
\{x\in[0,1]^{d_x}: x_1=0.10\}
\cup
\{x\in[0,1]^{d_x}: x_1=0.60\}.
$$

So the first-coordinate boundary values are $0.10$ and $0.60$, but the boundary
itself is not the two-point set $\{0.10,0.60\}$.  In $d_x=2$, it is two
vertical line segments; in general $d_x$, it is the union of two
$(d_x-1)$-dimensional hyperplane slices:

$$
\big(\{0.10\}\times[0,1]^{d_x-1}\big)
\cup
\big(\{0.60\}\times[0,1]^{d_x-1}\big).
$$

This boundary is away from the kink at $x_1=0.35$, so the regular-level-set
condition holds even though global Holder smoothness is only $p$.  The baseline
and propensity use the same two-coordinate forms as the sparse DGP:

$$
b_0(x)
= 2\tanh\{2(x_1-1/2)\}
+ \sin\{2\pi(x_2-1/2)\},
$$

$$
p_0(x)
= s\left(
\mathrm{overlap}
\left[
1.2(x_1-1/2) + 0.4(x_2-1/2)
\right]
\right).
$$

### 3.4 Cubic Failure-Mode DGP

The cubic DGP intentionally violates the nonvanishing-gradient condition:

$$
\tau_0(x) = \mathrm{TAU\_SCALE}(x_1-1/2)^3.
$$

The boundary is

$$
M_0 = \{x : x_1 = 1/2\},
$$

but

$$
\nabla \tau_0(x)
= 3\,\mathrm{TAU\_SCALE}(x_1-1/2)^2 e_1
= 0
\quad \text{on } M_0.
$$

Thus the regular-level-set assumption fails on the whole boundary.  This is a
deliberate stress test.

## 4. Random-Feature Linear Sieve

The simulations treat a shallow random-feature neural net as a linear sieve.
Conditional on the random features, the basis functions are fixed.

### 4.1 Dense Random Features

For $k=1,\ldots,K$, draw

$$
Z_k = (w_k', b_k)' \sim \mathrm{Unif}(S^{d_x}),
$$

implemented by drawing a standard normal vector in $\mathbb R^{d_x+1}$ and
normalizing it to length one.  The feature is

$$
\psi_k(x) =
\mathrm{act}\{\gamma(w_k'x+b_k)\}.
$$

The full feature vector includes an intercept:

$$
\psi(x) =
\left(
1,\psi_1(x),\ldots,\psi_K(x)
\right)'.
$$

The activation function is one of

$$
\mathrm{act}(z)=\cos(z),\qquad
\mathrm{act}(z)=\max(z,0),\qquad
\mathrm{act}(z)=\tanh(z).
$$

Most baseline designs use cosine features.

### 4.2 Support-Restricted Features

If a support set $S \subset \{1,\ldots,d_x\}$ is supplied, features are drawn
only on those coordinates.  Let $q=|S|$.  Draw

$$
(w_{k,S}',b_k)' \sim \mathrm{Unif}(S^q),
$$

set $w_{kj}=0$ for $j \notin S$, and use

$$
\psi_k(x)
= \mathrm{act}\{\gamma(w_{k,S}'x_S+b_k)\}.
$$

This is used for oracle-support features and post-screening features.

### 4.3 Random $q$-Sparse Features

If a sparsity level $q$ is supplied but no fixed support is supplied, then
for each feature $k$:

1. Draw a subset $S_k$ of $q$ coordinates uniformly without replacement.
2. Draw $(w_{k,S_k}',b_k)'\sim \mathrm{Unif}(S^q)$.
3. Set $w_{kj}=0$ outside $S_k$.

This creates a union-of-low-dimensional-subspaces feature dictionary.

## 5. Per-Arm OLS First Stage

Let

$$
\Psi_a
=
\begin{bmatrix}
\psi(X_i)' : D_i=a
\end{bmatrix}
$$

be the feature matrix in arm $a\in\{0,1\}$, and let $Y_a$ be the
corresponding outcome vector.  The code fits separate OLS regressions by
Moore-Penrose inverse:

$$
\hat\beta_a
=
(\Psi_a'\Psi_a)^+ \Psi_a'Y_a.
$$

The fitted treatment effect is

$$
\hat h(x)
= \psi(x)'(\hat\beta_1-\hat\beta_0).
$$

If either arm has too few observations relative to the number of features,
the draw is skipped.  The usual rule is

$$
\min(n_1,n_0) \ge K+1+10.
$$

Here $K+1$ counts the intercept plus the $K$ random features.

## 6. Sandwich Blocks and Sieve Standard Errors

For arm $a$, define residuals

$$
\hat e_{ia} = Y_i - \psi(X_i)'\hat\beta_a,
\qquad D_i=a.
$$

The arm-specific sandwich block is

$$
\widehat P_a
=
(\Psi_a'\Psi_a)^+
\Psi_a'\operatorname{diag}(\hat e_a^2)\Psi_a
(\Psi_a'\Psi_a)^+.
$$

Equivalently, if

$$
A_a = (\Psi_a'\Psi_a)^+(\Psi_a'\operatorname{diag}(\hat e_a)),
$$

then

$$
\widehat P_a = A_aA_a'.
$$

Let $X_m^S$, $m=1,\ldots,M$, denote Sobol quasi-Monte Carlo points used to
approximate integration under $F$.  Write

$$
\hat h_m = \hat h(X_m^S), \qquad \psi_m = \psi(X_m^S).
$$

### 6.1 Welfare Plug-In and Standard Error

The plug-in welfare estimator is

$$
\widehat W
=
\frac{1}{M}
\sum_{m=1}^M [\hat h_m]_+.
$$

The derivative vector, called `bun_W` in the code, is

$$
\widehat b_W
=
\frac{1}{M}
\sum_{m=1}^M
1\{\hat h_m \ge 0\}\psi_m.
$$

Thus $\widehat b_W$ is the vectorized pathwise derivative
$\widehat{DW}[\psi]$: its $j$th entry is
$DW(\hat h)[\psi_j]$ approximated over the Sobol target points.  This is the
$\widehat{D\Phi}[b^K]$ vector in the generic Chen-Gao sieve Riesz formula, with
$\Phi=W$ and $b^K=\psi$.

The sieve standard error is

$$
\widehat{\mathrm{se}}_W
=
\left(
\widehat b_W'\widehat P_1\widehat b_W
+
\widehat b_W'\widehat P_0\widehat b_W
\right)^{1/2}.
$$

This is the RF-OLS specialization of the generic Chen-Gao sieve Riesz
representer standard-error formula.  For a generic functional
$\Phi(h_0)$ and sieve basis $b^K(x)$, let

$$
\widehat G
:=
\frac{1}{n}
\sum_{i=1}^n b^K(X_i)b^K(X_i)'.
$$

The population sieve Riesz representer $v^*_{K,\Phi}\in\mathcal H_K$ is
defined by

$$
D\Phi(h_0)[\nu]
=
\langle v^*_{K,\Phi},\nu\rangle_{L^2(P_X)}
,
\qquad
\nu\in\mathcal H_K,
$$

and has the closed-form sieve representation

$$
v^*_{K,\Phi}(x)
=
b^K(x)'G^{-1}D\Phi(h_0)[b^K],
\qquad
G:=E[b^K(X_i)b^K(X_i)'].
$$

The plug-in estimator used for inference is

$$
\widehat v^*_{K,\Phi}(x)
:=
b^K(x)'\widehat G^{-1}
\widehat{D\Phi}[b^K],
\qquad
\widehat{D\Phi}[b^K]
:=
D\Phi(\hat h_K)[b^K].
$$

With residuals $\hat e_i=Y_i-\hat h_K(X_i)$, the estimated sieve influence
function and generic sieve variance are

$$
\widehat\psi_{i,\Phi}(K)
=
\widehat v^*_{K,\Phi}(X_i)\hat e_i,
\qquad
\widehat\sigma^2_{*,K}
:=
\frac{1}{n}
\sum_{i=1}^n
\left[\widehat\psi_{i,\Phi}(K)\right]^2.
$$

Equivalently,

$$
\widehat\sigma^2_{*,K}
=
\widehat{D\Phi}[b^K]'
\widehat G^{-1}
\left(
\frac{1}{n}\sum_{i=1}^n
\hat e_i^2 b^K(X_i)b^K(X_i)'
\right)
\widehat G^{-1}
\widehat{D\Phi}[b^K].
$$

The standard error for the estimator $\Phi(\hat h_K)$ is
$\widehat\sigma_{*,K}/\sqrt n$.  In the RF treatment-effect code this generic
formula is applied separately to the treated and control OLS regressions, using
the unnormalized Gram matrices $\Psi_a'\Psi_a$; this yields
$\widehat b_W'\widehat P_1\widehat b_W+
\widehat b_W'\widehat P_0\widehat b_W$.

### 6.2 Value Plug-In and Epsilon-Band Standard Error

The plug-in value/share estimator is

$$
\widehat V
=
\frac{1}{M}
\sum_{m=1}^M 1\{\hat h_m \ge 0\}.
$$

The epsilon band is chosen as

$$
\widehat\epsilon
=
\iota \cdot \widehat{\operatorname{sd}}_M(\hat h_m),
$$

where $\iota=0.01$ unless an iota sweep is being run.  The value derivative
vector is

$$
\widehat b_V
=
\frac{1}{2\widehat\epsilon M}
\sum_{m=1}^M
1\{|\hat h_m|<\widehat\epsilon\}\psi_m.
$$

Thus $\widehat b_V$ is the vectorized pathwise derivative
$\widehat{DV}[\psi]$: its $j$th entry estimates $DV(\hat h)[\psi_j]$.  Unlike
$W$, the derivative of $V$ is a level-set integral on $\{\hat h=0\}$, so the
code estimates it with the epsilon band
$1\{|\hat h_m|<\widehat\epsilon\}/(2\widehat\epsilon)$.  This is again the
$\widehat{D\Phi}[b^K]$ vector in the generic Chen-Gao formula, now with
$\Phi=V$.

The value standard error is

$$
\widehat{\mathrm{se}}_V
=
\left(
\widehat b_V'\widehat P_1\widehat b_V
+
\widehat b_V'\widehat P_0\widehat b_V
\right)^{1/2}.
$$

The number of Sobol points inside the band is recorded as

$$
\widehat n_{\mathrm{band}}
=
\sum_{m=1}^M 1\{|\hat h_m|<\widehat\epsilon\}.
$$

### 6.3 Monte Carlo Truth

Truth is computed by a larger Sobol design:

$$
W_0 \approx \frac{1}{M_{\mathrm{truth}}}
\sum_{m=1}^{M_{\mathrm{truth}}}
[\tau_0(\widetilde X_m)]_+,
$$

$$
V_0 \approx \frac{1}{M_{\mathrm{truth}}}
\sum_{m=1}^{M_{\mathrm{truth}}}
1\{\tau_0(\widetilde X_m)\ge 0\}.
$$

### 6.4 Coverage Indicators

Each draw records

$$
1\{|\widehat W-W_0| \le z_{0.975}\widehat{\mathrm{se}}_W\},
\qquad
1\{|\widehat V-V_0| \le z_{0.975}\widehat{\mathrm{se}}_V\},
$$

where

$$
z_{0.975}=1.959963984540054.
$$

The diagnostic

$$
n\widehat{\operatorname{Var}}(\widehat V)
= n\widehat{\mathrm{se}}_V^2
$$

is used to track empirical sieve-Riesz growth.

## 7. Debiasing Designs

Several scripts use second-order debiasing for $W$ and $V$.  Write
$F$ for either $W$ or $V$.

### 7.1 Numerical Second Derivative

Given an estimated function $h$ and direction $\Delta$, the code computes
a central-difference approximation to

$$
D^2F(h)[\Delta,\Delta].
$$

Let

$$
s_\Delta = \widehat{\operatorname{sd}}_M(\Delta_m),
\qquad
u_m = \Delta_m/s_\Delta,
\qquad
\delta = \delta_0\max\{\widehat{\operatorname{sd}}_M(h_m),10^{-12}\}.
$$

Then

$$
\widehat D^2F(h)[\Delta,\Delta]
=
\frac{
F(h+\delta u)-2F(h)+F(h-\delta u)
}{\delta^2}
s_\Delta^2.
$$

The default values are typically

$$
\delta_0=0.05 \quad \text{for } W,
\qquad
\delta_0=0.05 \text{ or } 0.2 \quad \text{for } V.
$$

### 7.2 Split-Sample Debiasing

The split-sample correction randomly splits the sample into two halves, fits
two first-stage CATE estimates $\hat h_1$ and $\hat h_2$, and defines

$$
\bar h = \frac{1}{2}(\hat h_1+\hat h_2).
$$

The split-sample debiased estimator is

$$
\widehat F_{\mathrm{SS}}
=
F(\bar h)
-
\frac{1}{8}
\widehat D^2F(\bar h)[\hat h_1-\hat h_2,\hat h_1-\hat h_2].
$$

### 7.3 Leave-One-Out Debiasing

For arm $a$, define the OLS leverage

$$
H_{ia}
=
\psi(X_i)'(\Psi_a'\Psi_a)^+\psi(X_i),
\qquad D_i=a.
$$

The leave-one-out residual is

$$
\hat e_{ia}^{(-i)}
=
\frac{\hat e_{ia}}{1-H_{ia}}.
$$

For observation $i$ in arm $a$, define the score direction

$$
s_{ia}(x)
=
n\,\psi(x)'(\Psi_a'\Psi_a)^+\psi(X_i).
$$

The sign differs between treated and control directions for $\hat h$, but the
quadratic form $D^2F[s_{ia},s_{ia}]$ removes the sign.

The LOO debiased estimator is

$$
\widehat F_{\mathrm{LOO}}
=
F(\hat h)
-
\frac{1}{2n^2}
\sum_{a\in\{0,1\}}
\sum_{i:D_i=a}
\widehat D^2F(\hat h)[s_{ia},s_{ia}]
\left(\hat e_{ia}^{(-i)}\right)^2.
$$

Some large-$n$ designs subsample at most 2000 observations per arm for this
sum and scale the result back up.

### 7.4 Augmented Standard Error for Debiased Estimators

For finite-sample LOO and DML corrections, some sweeps use

$$
\widehat{\mathrm{se}}_{\mathrm{aug}}^2
=
\widehat{\mathrm{se}}_{\mathrm{plug}}^2
+
\widehat{\operatorname{Var}}(\widehat{\mathrm{correction}}).
$$

This is intentionally conservative in the code because it ignores possible
negative covariance between the plug-in part and the correction term.

## 8. Screening Rule

The lasso screening rule standardizes the covariates and regresses

$$
\widetilde X_{ij}
:=
\frac{X_{ij}-\bar X_j}{\widehat{\operatorname{sd}}(X_{\cdot j})},
$$

$$
Y_i-\bar Y
$$

on

$$
\left[
\widetilde X_i,\,
D_i\widetilde X_i,\,
D_i-\bar D
\right].
$$

For coordinate $j$, define the screening score

$$
\mathrm{score}_j
=
|\hat\alpha_j| + |\hat\eta_j|,
$$

where $\hat\alpha_j$ is the lasso coefficient on $\widetilde X_{ij}$ and
$\hat\eta_j$ is the coefficient on $D_i\widetilde X_{ij}$.  Coordinate
$j$ is selected if

$$
\mathrm{score}_j > 10^{-10}.
$$

If no coordinate is selected, the top two scores are used.  If more than 10
coordinates are selected, only the top 10 by score are kept.  The resulting
screened support is $\widehat S$.

## 9. Design A: Base RF-Sieve High-Dimensional Smoke Simulation

Source: `rf_sieve_highd_sim.py`.

### 9.1 Purpose

This is the original smoke-size RF-sieve design.  It checks whether the
paper's sieve inference formulas work when tensor-product splines are replaced
by random features and the second stage is OLS.

### 9.2 DGP

The DGP is the dense-index `HighDimDGP` from Section 3.1 with

$$
\mathrm{TAU\_SCALE}=3,\qquad \mathrm{shift}=-0.70,\qquad
\mathrm{overlap}=1.
$$

Thus

$$
\tau_0(x)
= 3\left[
s\{3(a_1'x-m_1)\}
+ \frac{1}{2}s\{3(a_2'x-m_2)\}
-0.70
\right],
$$

$$
b_0(x)
=2\tanh\{2(a_2'x-m_2)\}
+\sin\{2\pi(a_1'x-m_1)\},
$$

$$
p_0(x)
=s\{1.2(a_1'x-m_1)+0.4(a_2'x-m_2)\}.
$$

### 9.3 Monte Carlo Grid

| Object | Value |
|---|---:|
| Ambient dimensions $d_x$ | $3,5$ |
| Sample sizes $n$ | $500,1000$ |
| Random features $K$ | $25,50$ |
| Replications | 50 |
| Activation | cosine |
| Feature scale $\gamma$ | 3.0 |
| Sobol points for functionals | 4096 |
| Sobol points for truth | $2^{15}$ |
| Epsilon multiplier $\iota$ | 0.01 |
| Heteroskedasticity | false |
| Shared features across arms | true |
| LOO debiasing | false by default |

The seed for a draw is based on

$$
2026 + 100003\,r + 7919\,d_x + n + K,
$$

where $r$ is the replication index.

### 9.4 Outputs

For each $(d_x,n,K)$, the script records bias, Monte Carlo standard
deviation, mean sieve standard error, coverage, the mean epsilon-band size,
and

$$
n\widehat{\mathrm{se}}_V^2.
$$

## 10. Design B: High-Dimensional $V$-Functional Exploration

Source: `rf_sieve_v_highdim_explore.py`.

### 10.1 Purpose

This design asks whether inference for $V=P_F\{\tau_0(X)\ge0\}$ continues to
work at $d_x=10$ and $d_x=50$, and whether low effective dimension matters
more than ambient dimension.

### 10.2 Common Settings

| Object | Value |
|---|---:|
| Replications | 200 |
| Sobol points for functionals | 8192 |
| Sobol points for truth | $2^{16}$ |
| Epsilon multiplier $\iota$ | 0.01 |
| $\mathrm{TAU\_SCALE}$ | 3.0 |
| Activation | cosine |
| Base seed | 20260609 |

The generated data are homoskedastic.

### 10.3 Experiment Cells

The experiment cells are:

| Label | DGP | $d_x$ | Feature support | $\gamma$ | $n$ | $K$ |
|---|---|---:|---|---:|---:|---:|
| `A_dense_d10_n1000_K50` | dense | 10 | dense sphere | 3.0 | 1000 | 50 |
| `A_dense_d10_n1000_K200` | dense | 10 | dense sphere | 3.0 | 1000 | 200 |
| `A_dense_d10_n4000_K200` | dense | 10 | dense sphere | 3.0 | 4000 | 200 |
| `A_dense_d50_n1000_K50` | dense | 50 | dense sphere | 3.0 | 1000 | 50 |
| `A_dense_d50_n4000_K200` | dense | 50 | dense sphere | 3.0 | 4000 | 200 |
| `A_dense_d50_n4000_K400` | dense | 50 | dense sphere | 3.0 | 4000 | 400 |
| `B_dense_d50_gamma1.5` | dense | 50 | dense sphere | 1.5 | 4000 | 200 |
| `B_dense_d50_gamma6` | dense | 50 | dense sphere | 6.0 | 4000 | 200 |
| `C_sparse_d50_featdense` | sparse | 50 | dense sphere | 3.0 | 4000 | 200 |
| `C_sparse_d50_featq2` | sparse | 50 | random $q=2$ sparse | 3.0 | 4000 | 200 |
| `C_sparse_d50_featq1` | sparse | 50 | random $q=1$ sparse | 3.0 | 4000 | 200 |
| `C_sparse_d50_featq2_K50` | sparse | 50 | random $q=2$ sparse | 3.0 | 4000 | 50 |
| `C_sparse_d50_oracle_K50` | sparse | 50 | oracle $S_0=\{1,2\}$ | 3.0 | 4000 | 50 |
| `A_dense_d50_n16000_K400` | dense | 50 | dense sphere | 3.0 | 16000 | 400 |

The cell seed has the form

$$
20260609 + 1000003\,r + \mathrm{hash}(\mathrm{label}) \bmod 99991.
$$

The script reports the first-stage CATE RMSE

$$
\left[
\frac{1}{M}\sum_{m=1}^M
\{\hat h(X_m^S)-\tau_0(X_m^S)\}^2
\right]^{1/2},
$$

as well as $W$ and $V$ bias and coverage.

## 11. Design D1: Screen-Then-Sieve

Source: `explore_D1_screening.py`.

### 11.1 Purpose

This design tests whether lasso screening can recover the sparse support
$S_0=\{1,2\}$ in a high-dimensional sparse DGP, and whether random features
restricted to $\widehat S$ restore low-dimensional inference.

### 11.2 DGP and Settings

The DGP is sparse with $d_x=50$:

$$
\tau_0(x)
=3\left[
s\{3(x_1-1/2)\}
+\frac{1}{2}s\{3(x_2-1/2)\}
-0.70
\right].
$$

Common simulation settings:

| Object | Value |
|---|---:|
| Replications | 150 |
| $d_x$ | 50 |
| Sample sizes $n$ | $1000,4000$ |
| $K$ after screening | 50 |
| Dense baseline $K$ | 200 |
| $\gamma$ | 3.0 |
| Sobol points | 8192 |
| Seed | 31 |

### 11.3 Variants

For each $n$ and replication:

1. `dense_K200`: no screening, dense sphere features with $K=200$.
2. `screen_full`: lasso screening and estimation are both done on the full
   sample.  This is the post-selection version.
3. `screen_split`: a random half is used for lasso screening, and the other
   half is used for OLS and inference.  This is the honest version.
4. `oracle`: features are restricted to the true support $S_0=\{1,2\}$.

The support-recovery indicator is

$$
1\{S_0 \subseteq \widehat S\}.
$$

The support size $|\widehat S|$, CATE RMSE, bias, standard errors, and
coverage are recorded.

## 12. Design D2: Effective-Dimension Check

Source: `explore_D2_effective_dim.py`.

### 12.1 Purpose

This design tests the cylinder-boundary conjecture.  If the CATE depends only
on $S_0=\{1,2\}$, then

$$
M_0
=
\{x\in[0,1]^{d_x}: \tau_0(x)=0\}
=
\{x_S:\tau_0(x_S)=0\}\times[0,1]^{d_x-2}.
$$

The value derivative integrates over a cylinder.  The conjecture is that
ambient $d_x$ should not affect RMSE, standard errors, or coverage once the
first-stage sieve is correctly restricted to the two relevant coordinates.

### 12.2 DGP and Grid

The DGP is sparse.  Features are oracle-supported on $S_0=\{1,2\}$.

| Object | Value |
|---|---:|
| Replications | 150 |
| Ambient dimensions $d_x$ | $10,50,100$ |
| Sample sizes $n$ | $1000,4000,16000$ |
| Random features $K$ | 50 |
| $\gamma$ | 3.0 |
| Sobol points | 8192 |
| Seed | 47 |

### 12.3 Rate Diagnostic

For each $d_x$, the script estimates the empirical convergence exponent by
regressing

$$
\log \operatorname{RMSE}(\widehat V)
$$

on

$$
\log n.
$$

The parametric benchmark is slope $-1/2$.

## 13. Design D3: Split-Sample and LOO Debiasing for $W$

Source: `explore_D3_W_debias.py`.

### 13.1 Purpose

This design targets cells where plug-in $W(\hat h)$ has visible upward
Jensen/ReLU bias:

$$
E\{[\hat h(X)]_+\}-[h_0(X)]_+
$$

is positive near the boundary because the positive-part map is convex.

The design checks whether second-order split-sample and LOO corrections remove
this bias.

### 13.2 Cells

All cells use the dense DGP, $\gamma=3.0$, $M=8192$ Sobol points, and
150 replications.

| DGP | $d_x$ | $n$ | $K$ | Motivation |
|---|---:|---:|---:|---|
| dense | 10 | 4000 | 200 | plug-in $W$ undercoverage cell |
| dense | 50 | 4000 | 200 | high-dimensional plug-in $W$ undercoverage |
| dense | 50 | 4000 | 400 | worst high-dimensional plug-in $W$ cell |

### 13.3 Estimators Compared

For $W$:

$$
\widehat W_{\mathrm{plug}} = W(\hat h),
$$

$$
\widehat W_{\mathrm{SS}}
=
W(\bar h)
-
\frac{1}{8}
\widehat D^2W(\bar h)[\hat h_1-\hat h_2,\hat h_1-\hat h_2],
$$

and

$$
\widehat W_{\mathrm{LOO}}
=
W(\hat h)
-
\frac{1}{2n^2}
\sum_{a=0}^1\sum_{i:D_i=a}
\widehat D^2W(\hat h)[s_{ia},s_{ia}]
(\hat e_{ia}^{(-i)})^2.
$$

For $V$, the script records plug-in $V$ and split-sample $V$ as controls.
All $W$ variants use the plug-in sieve standard error for coverage.

## 14. Design D4: Data-Driven Tuning and Inference Refinements

Source: `explore_D4_tuning.py`.

### 14.1 Purpose

This design studies whether ordinary first-stage cross-validation can select a
random-feature sieve that still gives valid inference for $V$.  It also
checks the sensitivity of the epsilon-band width and whether scrambled Sobol
points matter.

### 14.2 Common Settings

| Object | Value |
|---|---:|
| DGPs | dense and sparse |
| $d_x$ | 50 |
| $n$ | 4000 |
| Replications | 120 |
| Sobol points | 8192 |
| Seed | 71 |

### 14.3 Split-Half CV Selection

The sample is randomly split into two halves $A$ and $B$.  For each
$(K,\gamma)$ in

$$
K\in\{50,100,200,400\},
\qquad
\gamma\in\{1.5,3.0,6.0\},
$$

the estimator is fit on $A$, and validation MSE on $B$ is computed as

$$
\mathrm{MSE}_{\mathrm{val}}(K,\gamma)
=
\frac{1}{|B|}
\sum_{i\in B}
\left[
Y_i
-
1\{D_i=1\}\psi(X_i)'\hat\beta_1^{A}
-
1\{D_i=0\}\psi(X_i)'\hat\beta_0^{A}
\right]^2.
$$

The selected tuning pair is

$$
(\widehat K,\widehat\gamma)
=
\arg\min_{K,\gamma}\mathrm{MSE}_{\mathrm{val}}(K,\gamma).
$$

The model is then refit on the full sample at
$(\widehat K,\widehat\gamma)$, and plug-in $W,V$ inference is computed.

### 14.4 Iota and Sobol Sweep

At the reference configuration

$$
(K,\gamma)=(200,3.0),
$$

the script evaluates

$$
\iota\in\{0.005,0.01,0.02,0.05\}
$$

using plain Sobol points.  It also compares plain Sobol to scrambled Sobol at
$\iota=0.01$.

## 15. Design S1: Robustness Sweep

Source: `sweep_S1_robustness.py`.

### 15.1 Purpose

This is a one-factor-at-a-time robustness sweep for plug-in $V$ and
LOO-debiased $W$.

### 15.2 Baseline

The baseline is

| Object | Value |
|---|---:|
| DGP | dense |
| $d_x$ | 50 |
| $n$ | 4000 |
| $K$ | 200 |
| $\gamma$ | 1.5 |
| Activation | cosine |
| Shift | -0.70 |
| Overlap multiplier | 1.0 |
| Heteroskedasticity | false |
| Replications | 150 |
| Sobol points | 8192 |
| Seed | 83 |

### 15.3 Deviations

| Cell | Change from baseline |
|---|---|
| `baseline` | no change |
| `relu` | activation ReLU, $\gamma=3.0$ |
| `tanh` | activation tanh, $\gamma=3.0$ |
| `hetero` | heteroskedastic errors with $\sigma_1=1.25,\sigma_0=0.75$ |
| `share85` | shift changed to -0.40 to raise $V_0$ |
| `weak_overlap` | overlap multiplier changed to 3.0 |
| `d10` | $d_x=10$ |
| `sparse_dgp` | sparse DGP instead of dense DGP |

For each cell, the script records plug-in $V$, plug-in $W$, LOO $W$, CATE
RMSE, coverage, and the minimum sample propensity.

## 16. Design S2: Smoothness and Theorem-5 LOO Test

Source: `sweep_S2_smoothness.py`.

### 16.1 Purpose

This sweep asks whether LOO debiasing matters for $V$ in a low-smoothness
kink design.  The key DGP is

$$
\tau_0(x)
= 3\left(|x_1-0.35|^p - 0.25^p\right),
\qquad d_x=2.
$$

The boundary is $x_1=0.10$ or $x_1=0.60$.

### 16.2 Smoothness Values

Two smoothness exponents are tested:

$$
p\in\{1.6,2.5\}.
$$

The script comments interpret these as:

- $p=1.6$: plug-in $V$ smoothness condition $\sigma>d=2$ fails, while
  the SS/LOO condition $\sigma>(d+1)/2=1.5$ holds.
- $p=2.5$: both conditions hold.

### 16.3 Sieve Dimension Rule

The sieve dimension is set by

$$
K_n
=
\max
\left\{
12,\,
\operatorname{round}
\left[
3n^{d_x/(2p+1)}
\right]
\right\}.
$$

Since $d_x=2$, this is

$$
K_n
=
\max
\left\{
12,\,
\operatorname{round}
\left[
3n^{2/(2p+1)}
\right]
\right\}.
$$

### 16.4 Grid

| Object | Value |
|---|---:|
| $d_x$ | 2 |
| $n$ | $1000,4000,16000$ |
| $p$ | $1.6,2.5$ |
| Replications | 150 |
| $\gamma$ | 3.0 |
| Sobol points | 8192 |
| Max LOO observations per arm | 2000 |
| Seed | 97 |

The script compares plug-in and LOO versions of both $V$ and $W$, all
studentized by the plug-in sieve standard error.

## 17. Design S3: Full Practical Pipeline Under Signal-Strength Stress

Source: `sweep_S3_pipeline.py`.

### 17.1 Purpose

This sweep combines screening, support-restricted RF features, per-arm OLS,
plug-in $V$, and LOO $W$.  It stresses the screening step by changing the
CATE signal strength.

### 17.2 DGP

The DGP is sparse with $d_x=50$, $n=4000$, $K=50$, and
$\gamma=3.0$.  The CATE is

$$
\tau_0(x)
= \mathrm{TAU\_SCALE}
\left[
s\{3(x_1-1/2)\}
+\frac{1}{2}s\{3(x_2-1/2)\}
-0.70
\right],
$$

with

$$
\mathrm{TAU\_SCALE}\in\{1,2,3\}.
$$

### 17.3 Variants

For each signal strength:

1. `screen_full`: screen and estimate on the same sample.
2. `screen_split`: screen on half the sample and estimate on the other half.

The script records support recovery, support size, CATE RMSE, plug-in $V$
coverage, plug-in $W$ coverage, and LOO $W$ coverage.

## 18. Design S4: Known Failure Modes

Source: `sweep_S4_failure_modes.py`.

### 18.1 Purpose

This design tests whether known assumption violations fail visibly or silently.
The desired diagnostic distinction is:

- graceful failure: standard errors widen or coverage becomes conservative;
- dangerous failure: biased estimates have tight, misleading intervals.

### 18.2 Cells

All cells use $n=4000$, $M=8192$ Sobol points, 150 replications, and
$\gamma=3.0$.

| Cell | DGP | $d_x$ | $K$ | Feature support | Violation |
|---|---|---:|---:|---|---|
| `F1_cubic` | cubic | 10 | 100 | dense | $\nabla\tau_0=0$ on boundary |
| `F2_screen_miss` | sparse | 50 | 50 | $\{1\}$ only | drops true coordinate $x_2$ |
| `F2_control_S01` | sparse | 50 | 50 | $\{1,2\}$ | oracle-support control |
| `F3_overlap6` | dense, overlap 6 | 50 | 200 | dense | extreme propensity variation |

The overlap-6 propensity is

$$
p_0(x)
=s\left(
6\left[
1.2(a_1'x-m_1)+0.4(a_2'x-m_2)
\right]
\right).
$$

The summary records $V$ bias, $V$ standard deviation, $V$ standard error,
$\widehat{\mathrm{se}}_V/\operatorname{sd}(\widehat V)$, coverage, band size,
and $W$ coverage.

## 19. Design S5: Appendix-D Cross-Fitted Sieve-Influence-Function Estimator

Source: `sweep_S5_dml_if.py`.

### 19.1 Purpose

This design tests a cross-fitted influence-function correction for a generic
machine-learning first stage, here gradient boosting.  The target is the value
functional $V$.

### 19.2 Grid

| Object | Value |
|---|---:|
| DGP | dense |
| $d_x$ | $10,50$ |
| $n$ | 4000 |
| RF sieve features $K$ | 200 |
| $\gamma$ | 1.5 |
| Sobol points | 8192 |
| Replications | 100 |
| Cross-fitting folds | 2 |
| GBM model | `HistGradientBoostingRegressor(max_iter=150)` |
| Epsilon multiplier $\iota$ | 0.01 |
| Seed | 127 |

### 19.3 Reference RF Estimator

The reference estimator is the RF-OLS plug-in estimator

$$
\widehat V_{\mathrm{RF}}=V(\hat h_{\mathrm{RF}})
$$

with the epsilon-band RF sieve standard error from Section 6.2.

### 19.4 GBM Plug-In Estimator

For each fold $k$, fit gradient-boosting regressors
$\hat\mu_{1,-k}$ and $\hat\mu_{0,-k}$ on the training folds.  On Sobol
points define

$$
\hat h_k(x)
=
\hat\mu_{1,-k}(x)-\hat\mu_{0,-k}(x).
$$

The averaged GBM CATE on Sobol points is

$$
\hat h_{\mathrm{GBM}}(x)
=
\frac{1}{K_{\mathrm{fold}}}
\sum_{k=1}^{K_{\mathrm{fold}}}
\hat h_k(x).
$$

The naive GBM plug-in value is

$$
\widehat V_{\mathrm{GBM}}
=
\frac{1}{M}
\sum_{m=1}^M
1\{\hat h_{\mathrm{GBM}}(X_m^S)\ge0\}.
$$

### 19.5 Sieve-Influence Correction

For each fold $k$, compute an epsilon-band derivative at the fold-specific
GBM CATE:

$$
\widehat b_k
=
\frac{1}{2\epsilon_k M}
\sum_{m=1}^M
1\{|\hat h_k(X_m^S)|<\epsilon_k\}\psi(X_m^S),
$$

where

$$
\epsilon_k
=
\iota\,\widehat{\operatorname{sd}}_M(\hat h_k(X_m^S)).
$$

Let $T_k$ be the training set for fold $k$, and let $n_{T_k}=|T_k|$.
The arm-specific Gram matrices are

$$
\widehat G_{1,k}
=
\frac{1}{n_{T_k}}
\sum_{i\in T_k:D_i=1}
\psi(X_i)\psi(X_i)',
$$

$$
\widehat G_{0,k}
=
\frac{1}{n_{T_k}}
\sum_{i\in T_k:D_i=0}
\psi(X_i)\psi(X_i)'.
$$

The Riesz weights are

$$
\widehat w_{1,k}
=
\widehat G_{1,k}^+\widehat b_k,
\qquad
\widehat w_{0,k}
=
\widehat G_{0,k}^+(-\widehat b_k).
$$

For a held-out observation $i$ in fold $k$,

$$
\widehat v_i^*
=
1\{D_i=1\}\psi(X_i)'\widehat w_{1,k}
+
1\{D_i=0\}\psi(X_i)'\widehat w_{0,k}.
$$

The fold residual is

$$
\widehat r_i
=
Y_i-\hat\mu_{D_i,-k}(X_i).
$$

The code's correction is

$$
\widehat C
=
\frac{1}{K_{\mathrm{fold}}}
\sum_{k=1}^{K_{\mathrm{fold}}}
\frac{1}{|I_k|}
\sum_{i\in I_k}
\widehat v_i^*\widehat r_i.
$$

The DML estimator is

$$
\widehat V_{\mathrm{DML}}
=
\widehat V_{\mathrm{GBM}}+\widehat C.
$$

In S5, all three estimators are studentized by the RF sieve $V$ standard
error.

## 20. Design S6: Fixes for Overlap and DML Undercoverage

Source: `sweep_S6_fixes.py`.

### 20.1 Part A: Oracle Common-Support Trimming

This part revisits the S4 overlap failure with dense $d_x=50$, overlap
multiplier 6, $n=4000$, $K=200$, $\gamma=1.5$, $M=8192$, and 150
replications.

The target population is trimmed to

$$
\mathcal X_{\mathrm{trim}}
=
\{x: 0.05 \le p_0(x) \le 0.95\}.
$$

The sample is also trimmed using the oracle propensity:

$$
\widehat{\mathcal I}_{\mathrm{trim}}
=
\{i:0.05\le p_0(X_i)\le0.95\}.
$$

The estimands become conditional-on-trim analogs:

$$
W_{0,\mathrm{trim}}
=
E\{[\tau_0(X)]_+ \mid X\in\mathcal X_{\mathrm{trim}}\},
$$

$$
V_{0,\mathrm{trim}}
=
P\{\tau_0(X)\ge0 \mid X\in\mathcal X_{\mathrm{trim}}\}.
$$

The same RF-OLS and sieve SE formulas are applied on the trimmed sample and
trimmed Sobol target points.

### 20.2 Part B: Five-Fold DML with Augmented SE

This part repeats the dense DML design for $d_x\in\{10,50\}$, $n=4000$,
$K=200$, $\gamma=1.5$, $M=8192$, and 100 replications, but uses
five-fold cross-fitting instead of two-fold cross-fitting.

It stores observation-level correction terms

$$
\widehat q_i
=
\widehat v_i^*(Y_i-\hat\mu_{D_i,-k(i)}(X_i)),
$$

and defines

$$
\widehat C
=
\frac{1}{n_{\mathrm{ok}}}
\sum_{i:\widehat q_i\ \mathrm{finite}}
\widehat q_i.
$$

The augmented correction variance is

$$
\widehat{\operatorname{Var}}(\widehat C)
=
\frac{\widehat{\operatorname{Var}}(\widehat q_i)}
{n_{\mathrm{ok}}}.
$$

The augmented standard error is

$$
\widehat{\mathrm{se}}_{\mathrm{aug}}
=
\left(
\widehat{\mathrm{se}}_{\mathrm{plug}}^2
+
\widehat{\operatorname{Var}}(\widehat C)
\right)^{1/2}.
$$

Coverage is reported using both the plug-in and augmented standard errors.

## 21. Design S7: Extreme-Share Bias for $V$

Source: `sweep_S7_extreme_share.py`.

### 21.1 Purpose

When $V_0$ is close to one, first-stage noise flips signs asymmetrically:
there is substantial mass that can be pushed from positive to negative, but
little negative mass that can be pushed back.  This creates a one-sided bias in
$\widehat V$.

### 21.2 Grid

All cells use the dense DGP with $d_x=50$, $K=200$, $\gamma=1.5$,
$M=8192$, and 150 replications.

| Shift | Intended share regime | $n$ values |
|---:|---|---|
| -0.55 | high share, roughly 0.85 | $4000,16000$ |
| -0.40 | extreme share, roughly 0.95 | $4000,16000$ |

The estimators compared are

$$
\widehat V_{\mathrm{plug}}
=V(\hat h)
$$

and

$$
\widehat V_{\mathrm{LOO}}
=
V(\hat h)
-
\frac{1}{2n^2}
\sum_{a=0}^1\sum_{i:D_i=a}
\widehat D^2V(\hat h)[s_{ia},s_{ia}]
(\hat e_{ia}^{(-i)})^2.
$$

The LOO sum uses at most 2000 observations per arm.

## 22. Design S7b: Augmented SE for LOO-Debiased Estimators

Source: `sweep_S7b_aug_se.py`.

### 22.1 Purpose

S7 showed that LOO $V$ can remove extreme-share bias but may undercover if
the correction term's finite-sample noise is not included.  S7b adds the
augmented standard error from Section 7.4.

### 22.2 Grid

All cells use dense $d_x=50$, $n=4000$, $K=200$, $\gamma=1.5$,
$M=8192$, and 150 replications.

| Cell | Shift | Functional |
|---|---:|---|
| `V_share81` | -0.55 | $V$ |
| `V_share95` | -0.40 | $V$ |
| `W_baseline` | -0.70 | $W$ |

For each cell the script compares:

$$
\widehat F_{\mathrm{plug}},\qquad
\widehat F_{\mathrm{LOO}},
$$

with coverage using both

$$
\widehat{\mathrm{se}}_{\mathrm{plug}}
$$

and

$$
\widehat{\mathrm{se}}_{\mathrm{aug}}
=
\left(
\widehat{\mathrm{se}}_{\mathrm{plug}}^2
+
\widehat{\operatorname{Var}}(\widehat{\mathrm{correction}})
\right)^{1/2}.
$$

## 23. Design C: LOO-$V$ Delta Diagnostic

Source: `check_loo_v_delta.py`.

### 23.1 Purpose

This quick diagnostic checks whether the high variance of LOO $V$ in the
extreme-share cell is driven by the central-difference step size
$\delta_0$.

### 23.2 Grid

| Object | Value |
|---|---:|
| DGP | dense |
| $d_x$ | 50 |
| Shift | -0.40 |
| $n$ | 4000 |
| $K$ | 200 |
| $\gamma$ | 1.5 |
| Replications | 60 |
| Sobol points | 8192 |
| $\delta_0$ values | $0.05,0.2,0.5$ |

For each replication, the same fitted RF-OLS model is used and the script
computes

$$
\widehat V_{\mathrm{LOO}}(\delta_0)
$$

for each value of $\delta_0$.  It reports bias and standard deviation across
replications.

## 24. Design J: JTPA Empirical RF-Sieve Application

Source: `jtpa_rf_application.py`.

This is not a synthetic simulation.  It is an empirical application of the
validated RF-sieve pipeline to the JTPA/Kitagawa-Tetenov data.

### 24.1 Data and Outcomes

The script expects `KT_Data1.csv` with:

- treatment indicator $D$;
- 30-month earnings;
- pre-program earnings `prevearn`;
- education `edu`.

The covariate vector is

$$
X_i=(\mathrm{prevearn}_i,\mathrm{edu}_i)'.
$$

Each coordinate is min-max scaled to $[0,1]$.

Two outcomes are analyzed:

$$
Y_i^{\mathrm{no\ cost}}
=
\mathrm{earnings}_i,
$$

and

$$
Y_i^{\mathrm{cost}}
=
\mathrm{earnings}_i - 774D_i.
$$

The script analyzes both trimmed and untrimmed samples.

### 24.2 Min-Max Common-Support Trimming

For each covariate $j$, define the overlap interval

$$
L_j
=
\max\left\{
\min_{i:D_i=1}X_{ij},
\min_{i:D_i=0}X_{ij}
\right\},
$$

$$
U_j
=
\min\left\{
\max_{i:D_i=1}X_{ij},
\max_{i:D_i=0}X_{ij}
\right\}.
$$

An observation is kept if

$$
L_j\le X_{ij}\le U_j
\quad \text{for every covariate } j.
$$

### 24.3 CV Tuning

The tuning menu is

$$
K\in\{25,50,100,200\},
\qquad
\gamma\in\{1.5,3.0,6.0\}.
$$

The sample is split in half.  For each $(K,\gamma)$, per-arm RF-OLS is fit
on the first half, and prediction MSE is computed on the second half:

$$
\mathrm{MSE}_{\mathrm{val}}(K,\gamma)
=
\frac{1}{|B|}
\sum_{i\in B}
\left[
Y_i-\hat\mu_{D_i,A}^{K,\gamma}(X_i)
\right]^2.
$$

The minimizing pair is selected and then reused across 20 independent feature
draws.

### 24.4 Unknown-$F$ Empirical Estimators

Unlike the synthetic simulations, the target distribution is the empirical
distribution of the analyzed sample.  Let

$$
\hat h_i = \psi(X_i)'(\hat\beta_1-\hat\beta_0).
$$

The empirical welfare estimator is

$$
\widehat W
=
\frac{1}{n}
\sum_{i=1}^n
[\hat h_i]_+.
$$

The empirical value/share estimator is

$$
\widehat V
=
\frac{1}{n}
\sum_{i=1}^n
1\{\hat h_i\ge0\}.
$$

### 24.5 Empirical-Measure Standard Errors

For welfare,

$$
\widehat b_W
=
\frac{1}{n}
\sum_{i=1}^n
1\{\hat h_i\ge0\}\psi(X_i).
$$

The code uses

$$
\widehat{\mathrm{se}}_W^2
=
\frac{1}{n}
\widehat{\operatorname{Var}}_n([\hat h_i]_+)
+
\widehat b_W'\widehat P_1\widehat b_W
+
\widehat b_W'\widehat P_0\widehat b_W.
$$

For value, with

$$
\widehat\epsilon
=
0.01\,\widehat{\operatorname{sd}}_n(\hat h_i),
$$

the empirical derivative is

$$
\widehat b_V
=
\frac{1}{2\widehat\epsilon n}
\sum_{i=1}^n
1\{|\hat h_i|<\widehat\epsilon\}\psi(X_i).
$$

The value standard error is

$$
\widehat{\mathrm{se}}_V^2
=
\widehat b_V'\widehat P_1\widehat b_V
+
\widehat b_V'\widehat P_0\widehat b_V.
$$

### 24.6 Debiased Robustness and Feature-Draw Stability

The empirical application reports:

$$
\widehat W_{\mathrm{LOO}}
\quad \text{with } \delta_0=0.05,
$$

and

$$
\widehat V_{\mathrm{LOO}}
\quad \text{with } \delta_0=0.2,
$$

because the treated share is close to the extreme-share regime studied in S7.
The LOO sums use at most 2000 observations per arm.

The full analysis is repeated over

$$
N_{\mathrm{seeds}}=20
$$

independent feature draws.  The reported point estimates and standard errors
are medians across feature draws, and the feature-draw standard deviations are
also recorded.

## 25. Compact Map from Scripts to Design Questions

| Script | Main design question |
|---|---|
| `rf_sieve_highd_sim.py` | Can RF-OLS be treated as a sieve for $W,V$ inference in small high-dimensional smoke runs? |
| `rf_sieve_v_highdim_explore.py` | Does $V$ inference work at $d_x=10,50$, and how do dense, sparse, and oracle features differ? |
| `explore_D1_screening.py` | Does lasso screen-then-sieve recover low-dimensional performance in sparse $d_x=50$ designs? |
| `explore_D2_effective_dim.py` | Is $V$ governed by effective dimension $s=2$ rather than ambient $d_x$? |
| `explore_D3_W_debias.py` | Can SS/LOO debiasing fix plug-in $W$'s Jensen bias? |
| `explore_D4_tuning.py` | Does first-stage CV over $(K,\gamma)$ preserve $V$ inference, and how sensitive is the band width? |
| `sweep_S1_robustness.py` | Is $V$ and LOO-$W$ robust to activation, heteroskedasticity, share, overlap, dimension, and sparse DGP changes? |
| `sweep_S2_smoothness.py` | Does LOO matter for low-smoothness kink designs? |
| `sweep_S3_pipeline.py` | Does the full screened pipeline work under weaker signal strengths? |
| `sweep_S4_failure_modes.py` | Which assumption violations fail gracefully, and which are dangerous? |
| `sweep_S5_dml_if.py` | Does the Appendix-D cross-fitted sieve-IF estimator work with a GBM first stage? |
| `sweep_S6_fixes.py` | Do common-support trimming and augmented DML SEs fix the S4/S5 anomalies? |
| `sweep_S7_extreme_share.py` | When $V_0$ is high, does sign-flip bias appear, vanish with $n$, or get fixed by LOO? |
| `sweep_S7b_aug_se.py` | Does augmented SE fix finite-sample undercoverage of LOO-debiased estimators? |
| `check_loo_v_delta.py` | Which central-difference step $\delta_0$ stabilizes LOO-$V$? |
| `jtpa_rf_application.py` | How does the validated RF-sieve pipeline perform on the JTPA empirical application? |
