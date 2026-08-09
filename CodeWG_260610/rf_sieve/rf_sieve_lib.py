"""Shared infrastructure for the RF-sieve high-dimensional explorations (ClaudeWS).

Used by:
  explore_D1_screening.py      -- screen-then-sieve
  explore_D2_effective_dim.py  -- cylinder-boundary / effective-dimension check
  explore_D3_W_debias.py       -- SS/LOO debiasing of the welfare functional
  explore_D4_tuning.py         -- CV-based (K, gamma) selection, iota sweep, scrambled Sobol

Conventions follow rf_sieve_v_highdim_explore.py: per-arm OLS (pinv) on shared
random features with intercept; eps-band sieve variance for V, indicator-bun
variance for W; F = Uniform[0,1]^{d_x} known target; Sobol quasi-MC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats.qmc import Sobol

Z975 = 1.959963984540054
TAU_SCALE = 3.0


# =====================================================================
# DGPs
# =====================================================================
def _sig(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_dgp(kind: str, dim: int, shift: float = -0.70, overlap: float = 1.0,
             kink_pow: float = 1.6, tau_scale: float = TAU_SCALE) -> dict:
    """DGP factory.

    kind:
      'dense'  -- CATE from two dense linear indices (index sparsity s = 2);
      'sparse' -- signal in (x1, x2) only, the rest pure noise;
      'kink'   -- tau = tau_scale*(|x1 - 0.35|^kink_pow - 0.25^kink_pow):
                  Holder smoothness exactly kink_pow, boundary away from the
                  kink with nonvanishing gradient (Theorem-5 test bed);
      'cubic'  -- tau = tau_scale*(x1 - 0.5)^3: gradient VANISHES on the whole
                  boundary {x1 = 0.5} (regular-level-set violation, Assn 2(c)).
    shift   : constant in the dense/sparse CATE (controls the treated share);
    overlap : multiplies the propensity index (larger => weaker overlap).
    """
    if kind == "dense":
        a1 = np.array([(-1.0) ** j for j in range(dim)]) / np.sqrt(dim)
        a2 = np.ones(dim) / np.sqrt(dim)
        m1, m2 = 0.5 * a1.sum(), 0.5 * a2.sum()

        def tau(X):
            return tau_scale * (_sig(3.0 * (X @ a1 - m1))
                                + 0.5 * _sig(3.0 * (X @ a2 - m2)) + shift)

        def baseline(X):
            return 2.0 * np.tanh(2.0 * (X @ a2 - m2)) + np.sin(2.0 * np.pi * (X @ a1 - m1))

        def propensity(X):
            return _sig(overlap * (1.2 * (X @ a1 - m1) + 0.4 * (X @ a2 - m2)))

    elif kind == "sparse":
        def tau(X):
            return tau_scale * (_sig(3.0 * (X[:, 0] - 0.5))
                                + 0.5 * _sig(3.0 * (X[:, 1] - 0.5)) + shift)

        def baseline(X):
            return 2.0 * np.tanh(2.0 * (X[:, 0] - 0.5)) + np.sin(2.0 * np.pi * (X[:, 1] - 0.5))

        def propensity(X):
            return _sig(overlap * (1.2 * (X[:, 0] - 0.5) + 0.4 * (X[:, 1] - 0.5)))

    elif kind == "kink":
        c0 = 0.25 ** kink_pow

        def tau(X):
            return tau_scale * (np.abs(X[:, 0] - 0.35) ** kink_pow - c0)

        def baseline(X):
            return 2.0 * np.tanh(2.0 * (X[:, 0] - 0.5)) + np.sin(2.0 * np.pi * (X[:, 1] - 0.5))

        def propensity(X):
            return _sig(overlap * (1.2 * (X[:, 0] - 0.5) + 0.4 * (X[:, 1] - 0.5)))

    elif kind == "cubic":
        def tau(X):
            return tau_scale * (X[:, 0] - 0.5) ** 3

        def baseline(X):
            return 2.0 * np.tanh(2.0 * (X[:, 0] - 0.5)) + np.sin(2.0 * np.pi * (X[:, 1] - 0.5))

        def propensity(X):
            return _sig(overlap * (1.2 * (X[:, 0] - 0.5) + 0.4 * (X[:, 1] - 0.5)))

    else:
        raise ValueError(kind)
    return {"tau": tau, "baseline": baseline, "propensity": propensity,
            "kind": kind, "dim": dim, "true_support": (0, 1) if kind == "sparse" else None}


def generate(dgp: dict, n: int, rng: np.random.Generator, hetero: bool = False) -> dict:
    X = rng.uniform(size=(n, dgp["dim"]))
    D = (rng.uniform(size=n) < dgp["propensity"](X)).astype(float)
    sd = np.where(D == 1.0, 1.25, 0.75) if hetero else 1.0
    Y = dgp["baseline"](X) + D * dgp["tau"](X) + sd * rng.normal(size=n)
    return {"X": X, "D": D, "Y": Y}


def compute_truth(dgp: dict, m: int = 2**16) -> tuple[float, float]:
    Xs = Sobol(d=dgp["dim"], scramble=False).random(m)
    t = dgp["tau"](Xs)
    return float(np.maximum(t, 0.0).mean()), float((t >= 0.0).mean())


# =====================================================================
# Random-feature maps
# =====================================================================
def _activation(name: str):
    if name == "cos":
        return np.cos
    if name == "relu":
        return lambda z: np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh
    raise ValueError(name)


def draw_feature_map(dim: int, K: int, rng: np.random.Generator, gamma: float = 3.0,
                     activation: str = "cos", feat_q: int | None = None,
                     support: np.ndarray | list | None = None):
    """psi: X (m, dim) -> (m, K+1) with intercept column.

    support given : all features supported on those coordinates (oracle/screened);
    feat_q given  : each feature on q random coordinates;
    neither       : dense directions, (w, b) uniform on S^{dim}.
    """
    act = _activation(activation)
    if support is not None:
        S = np.asarray(support, dtype=int)
        q = len(S)
        W = np.zeros((K, dim))
        Z = rng.normal(size=(K, q + 1))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        W[:, S] = Z[:, :q]
        b = Z[:, q]
    elif feat_q is not None:
        q = int(feat_q)
        W = np.zeros((K, dim))
        Z = rng.normal(size=(K, q + 1))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        b = Z[:, q]
        for k in range(K):
            Sk = rng.choice(dim, size=q, replace=False)
            W[k, Sk] = Z[k, :q]
    else:
        Z = rng.normal(size=(K, dim + 1))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        W, b = Z[:, :dim], Z[:, dim]

    def psi(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.hstack([np.ones((X.shape[0], 1)),
                          act(gamma * (X @ W.T + b[None, :]))])

    return psi


# =====================================================================
# Per-arm OLS and functional inference
# =====================================================================
@dataclass
class ArmFit:
    beta: np.ndarray
    resid: np.ndarray
    Psi: np.ndarray
    MtM_pinv: np.ndarray
    patty: np.ndarray


def fit_arm(psi, X: np.ndarray, Y: np.ndarray) -> ArmFit:
    Psi = psi(X)
    MtM_pinv = np.linalg.pinv(Psi.T @ Psi, rcond=1e-10)
    beta = MtM_pinv @ (Psi.T @ Y)
    resid = Y - Psi @ beta
    A = MtM_pinv @ (Psi.T * resid[None, :])
    return ArmFit(beta=beta, resid=resid, Psi=Psi, MtM_pinv=MtM_pinv, patty=A @ A.T)


def fit_both_arms(psi, data: dict) -> tuple[ArmFit, ArmFit] | None:
    Dt = data["D"] == 1.0
    K1 = psi(data["X"][:1]).shape[1]
    if min(int(Dt.sum()), int((~Dt).sum())) < K1 + 10:
        return None
    return (fit_arm(psi, data["X"][Dt], data["Y"][Dt]),
            fit_arm(psi, data["X"][~Dt], data["Y"][~Dt]))


def rf_inference(fit_t: ArmFit, fit_c: ArmFit, psi, X_sobol: np.ndarray,
                 tau_sobol: np.ndarray | None = None, iota: float = 0.01) -> dict:
    """Plug-in W, V + sieve SEs; returns h_hat values for downstream debiasing."""
    Psi_s = psi(X_sobol)
    h_hat = Psi_s @ (fit_t.beta - fit_c.beta)
    M = X_sobol.shape[0]
    out: dict = {"h_hat": h_hat}

    if tau_sobol is not None:
        out["h_rmse"] = float(np.sqrt(np.mean((h_hat - tau_sobol) ** 2)))

    out["W_hat"] = float(np.maximum(h_hat, 0.0).mean())
    out["V_hat"] = float((h_hat >= 0.0).mean())

    pos = h_hat >= 0.0
    bun_W = Psi_s[pos].sum(axis=0) / M if pos.any() else np.zeros(Psi_s.shape[1])
    var_W = float(bun_W @ fit_t.patty @ bun_W + bun_W @ fit_c.patty @ bun_W)
    out["W_se"] = float(np.sqrt(max(var_W, 0.0)))

    eps = iota * float(h_hat.std())
    band = np.abs(h_hat) < eps
    out["eps_band"], out["n_band"] = eps, int(band.sum())
    if band.any() and eps > 0:
        bun_V = Psi_s[band].sum(axis=0) / (2.0 * eps * M)
        var_V = float(bun_V @ fit_t.patty @ bun_V + bun_V @ fit_c.patty @ bun_V)
        out["V_se"] = float(np.sqrt(max(var_V, 0.0)))
    else:
        out["V_se"] = np.nan
    return out


# =====================================================================
# Functionals as column-wise maps (for numerical second derivatives)
# =====================================================================
def F_W(vals: np.ndarray) -> np.ndarray | float:
    """Welfare value(s): mean over Sobol axis 0; vals may be (M,) or (M, b)."""
    return np.maximum(vals, 0.0).mean(axis=0)


def F_V(vals: np.ndarray) -> np.ndarray | float:
    return (vals >= 0.0).mean(axis=0)


def d2_quadform(F, h_bar: np.ndarray, Delta: np.ndarray, delta0: float = 0.1) -> float:
    """D^2 F(h_bar)[Delta, Delta] by a central difference along Delta."""
    sdD = float(Delta.std())
    if sdD <= 0:
        return 0.0
    u = Delta / sdD
    delta = delta0 * max(float(h_bar.std()), 1e-12)
    d2u = (float(F(h_bar + delta * u)) - 2.0 * float(F(h_bar))
           + float(F(h_bar - delta * u))) / delta**2
    return d2u * sdD**2


def ss_debias(F, psi, data: dict, X_sobol: np.ndarray,
              rng: np.random.Generator, delta0: float = 0.1) -> float | None:
    """Split-sample debiased estimator of F (Theorem 5 form):

        F(h_bar) - (1/8) D^2F(h_bar)[h1 - h2, h1 - h2].
    """
    n = data["X"].shape[0]
    perm = rng.permutation(n)
    half1, half2 = perm[: n // 2], perm[n // 2:]
    h_vals = []
    for idx in (half1, half2):
        sub = {k: v[idx] for k, v in data.items()}
        fits = fit_both_arms(psi, sub)
        if fits is None:
            return None
        h_vals.append(psi(X_sobol) @ (fits[0].beta - fits[1].beta))
    h1, h2 = h_vals
    h_bar = 0.5 * (h1 + h2)
    return float(F(h_bar)) - 0.125 * d2_quadform(F, h_bar, h1 - h2, delta0)


def loo_debias(F_colwise, fit_t: ArmFit, fit_c: ArmFit, psi, X_sobol: np.ndarray,
               plugin: float, n_total: int, delta0: float = 0.05,
               chunk: int = 256, max_per_arm: int | None = None,
               rng: np.random.Generator | None = None,
               return_var: bool = False):
    """LOO debiased estimator of F (Theorem 5 form), F_colwise in {F_W, F_V}.

    max_per_arm: if set, the per-arm sum over observations i is estimated
    unbiasedly from a random subsample of that size (scaled by n_arm / n_sub),
    which speeds up large-n cells by n_arm / max_per_arm.

    return_var: also return a (heuristic, independence-based) estimate of the
    sampling variance of the correction term, for SE augmentation:
    se_aug^2 = se_plug^2 + corr_var (cf. anomaly A1/A2 in the investigation log).
    """
    Psi_s = psi(X_sobol)
    h_hat = Psi_s @ (fit_t.beta - fit_c.beta)
    delta = delta0 * max(float(h_hat.std()), 1e-12)
    total = 0.0
    var_total = 0.0
    for fit in (fit_t, fit_c):   # sign of s_i cancels in the quadratic form
        n_a = fit.Psi.shape[0]
        if max_per_arm is not None and n_a > max_per_arm:
            idx = (rng or np.random.default_rng(0)).choice(n_a, size=max_per_arm,
                                                           replace=False)
            Psi_a, resid_a, scale_up = fit.Psi[idx], fit.resid[idx], n_a / max_per_arm
        else:
            Psi_a, resid_a, scale_up = fit.Psi, fit.resid, 1.0
        n_use = Psi_a.shape[0]
        terms = []
        for start in range(0, n_use, chunk):
            block = Psi_a[start: start + chunk]
            H_diag = np.einsum("ij,jk,ik->i", block, fit.MtM_pinv, block)
            H_diag = np.clip(H_diag, 0.0, 0.999)
            e_loo = resid_a[start: start + chunk] / (1.0 - H_diag)
            S = n_total * (Psi_s @ (fit.MtM_pinv @ block.T))
            scale = S.std(axis=0)
            scale[scale == 0.0] = 1.0
            U = S / scale[None, :]
            d2 = (F_colwise(h_hat[:, None] + delta * U)
                  - 2.0 * plugin
                  + F_colwise(h_hat[:, None] - delta * U)) / delta**2 * scale**2
            terms.append(d2 * e_loo**2)
        t_arm = np.concatenate(terms)
        total += scale_up * float(t_arm.sum())
        # Var(sum over arm) ~ scale_up^2 * n_use * var(t_i)  (independence heuristic)
        var_total += scale_up**2 * n_use * float(t_arm.var(ddof=1))
    estimate = plugin - total / (2.0 * n_total**2)
    corr_var = var_total / (2.0 * n_total**2) ** 2
    return (estimate, corr_var) if return_var else estimate


# =====================================================================
# Lasso screening of relevant coordinates
# =====================================================================
def screen_lasso(X: np.ndarray, D: np.ndarray, Y: np.ndarray,
                 cap: int = 10, seed: int = 0) -> np.ndarray:
    """Select coordinates relevant for the outcome or the treatment interaction.

    LassoCV of Y on [X_std, D*X_std, D]; coordinate j is selected if either its
    main-effect or its D-interaction coefficient is nonzero; capped at `cap`
    coordinates by coefficient magnitude; falls back to the top 2 if empty.
    """
    from sklearn.linear_model import LassoCV

    d = X.shape[1]
    Xs = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    Z = np.hstack([Xs, D[:, None] * Xs, (D - D.mean())[:, None]])
    las = LassoCV(cv=5, n_alphas=40, max_iter=20000, random_state=seed)
    las.fit(Z, Y - Y.mean())
    score = np.abs(las.coef_[:d]) + np.abs(las.coef_[d: 2 * d])
    sel = np.where(score > 1e-10)[0]
    if sel.size == 0:
        sel = np.argsort(-score)[:2]
    elif sel.size > cap:
        sel = sel[np.argsort(-score[sel])[:cap]]
    return np.sort(sel)
