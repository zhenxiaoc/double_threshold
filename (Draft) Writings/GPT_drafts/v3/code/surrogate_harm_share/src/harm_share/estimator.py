"""Finite-sample estimation of the surrogate-induced harm share via SIEVE nuisances.

We reuse the Chen-Chen-Gao `opttreat` sieve machinery (B-spline / random-feature
least squares with analytic gradients) so the estimator is aligned with the
paper's theory rather than a black-box learner.  The harm share is a
DOUBLE-threshold generalization of CCG's single-threshold `value` functional:

    theta = Pr( tau_S(X) >= 0, tau_Y(X) <= 0 ),

so we fit TWO CATE surfaces -- tau_S (short-run) and tau_Y (long-run) -- each
with a separate treated/control sieve regression (four arm regressions total),
and plug in

    theta_hat = P_n[ 1{tau_hat_S(X) >= 0} 1{tau_hat_Y(X) < 0} ].

Inference.
  * IRREGULAR (paper-aligned) analytic SE: a TWO-BAND sieve-Riesz variance that
    generalizes CCG's single boundary band `Bun` to the two decision surfaces
    M_S={tau_S=0, tau_Y<0} and M_Y={tau_Y=0, tau_S>0}.  The S-band enters with a
    `+` sign (raising tau_S expands {tau_S>=0}) and the Y-band with a `-` sign
    (raising tau_Y shrinks {tau_Y<=0}); the two per-unit influence contributions
    are SUMMED before squaring, so their covariance (S and Y are correlated
    within a unit) is captured.
  * FULL-REFIT nonparametric bootstrap as a robustness cross-check.

Because theta is a level-set (thin-set) functional it admits no sqrt(n)
influence function; the sieve variance scales like K/n and the interval width
therefore shrinks at the slower boundary rate, exactly as for CCG's value.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# make the CCG opttreat package importable
_OPTTREAT_ROOT = (
    Path(__file__).resolve().parents[3] / "longterm-main" / "Optimal-Treatment-main"
)
if str(_OPTTREAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPTTREAT_ROOT))

from opttreat.config import EstimatorConfig          # noqa: E402
from opttreat.data import split_treated_control       # noqa: E402
from opttreat.estimation import get_estimator         # noqa: E402

TIE_TO_ONE = True  # tau == 0 counts as tau >= 0 (treated), matching the paper's `>= 0`

# CCG-style sieve: separate treated/control tensor-B-spline bases, pinv solver.
DEFAULT_SIEVE = {
    "solver": "pinv", "share_features": False, "J_x_degree": 3,
    "J_x_segments_t": 1, "J_x_segments_c": 1, "knots": "uniform", "basis": "tensor",
}


def _Xcols(df):
    return [c for c in df.columns if c.startswith("X") and c[1:].isdigit()]


def _sieve_opts(d, segments):
    o = dict(DEFAULT_SIEVE)
    o["J_x_segments_t"] = segments
    o["J_x_segments_c"] = segments
    return o


def fit_cate_surface(df, ycol, sieve_opts):
    """Fit one CATE surface tau_hat(.) = mu_hat(.,1)-mu_hat(.,0) by a sieve.

    Returns the opttreat estimator-output dict (h_hat, Psi_t/c, e_t/c,
    feature_map_t/c, alpha, solver, ...).  `d` is the treatment W; `y` is ycol.
    """
    Xc = _Xcols(df)
    work = df[Xc].copy()
    work["d"] = df["W"].to_numpy()
    work["y"] = df[ycol].to_numpy()
    parsed = split_treated_control(work)
    est = get_estimator(EstimatorConfig("sieve", sieve_opts))
    return est.fit(parsed)


@dataclass
class HarmEstimate:
    theta_hat: float
    quadrants: dict
    rho_hat: float
    treat_share_S_hat: float
    se_sieve: float | None = None
    ci_sieve: tuple[float, float] | None = None
    se_boot: float | None = None
    ci_boot: tuple[float, float] | None = None
    reg_companion: float | None = None
    reg_companion_se: float | None = None
    diag: dict = field(default_factory=dict)


def _quadrants(tS, tY):
    ge = (tS >= 0) if TIE_TO_ONE else (tS > 0)
    pp = float(np.mean(ge & (tY >= 0)))
    pm = float(np.mean(ge & (tY < 0)))
    mp = float(np.mean(~ge & (tY >= 0)))
    mm = float(np.mean(~ge & (tY < 0)))
    return {"pp": pp, "pm": pm, "mp": mp, "mm": mm}


# --------------------------------------------------------------------------- #
# Two-band sieve-Riesz variance (paper-aligned analytic SE)
# --------------------------------------------------------------------------- #
def _gram_inv(Psi, alpha, solver):
    n = Psi.shape[0]
    G = (Psi.T @ Psi) / n
    if str(solver).lower() == "pinv" or float(alpha) == 0.0:
        return np.linalg.pinv(G, rcond=np.sqrt(np.finfo(float).eps))
    K = Psi.shape[1]
    return np.linalg.solve(G + alpha * np.eye(K) / n, np.eye(K))


def _riesz_influence(out, Bun_t, Bun_c):
    """Per-arm influence vectors e_a * (Psi_a @ G_a^{-1} Bun_a), scaled by 1/n_a."""
    Psi_t, Psi_c = np.asarray(out["Psi_t"]), np.asarray(out["Psi_c"])
    e_t, e_c = np.asarray(out["e_t"]).ravel(), np.asarray(out["e_c"]).ravel()
    alpha, solver = out.get("alpha", 0.0), out.get("solver", "pinv")
    w_t = _gram_inv(Psi_t, alpha, solver) @ Bun_t
    w_c = _gram_inv(Psi_c, alpha, solver) @ Bun_c
    return (e_t * (Psi_t @ w_t)) / Psi_t.shape[0], (e_c * (Psi_c @ w_c)) / Psi_c.shape[0]


def two_band_sieve_variance(out_S, out_Y, Xeval, tS, tY, delta=0.05,
                            include_empirical=True):
    """Analytic variance of theta_hat for the double threshold.

    Bun_S = +(1/(2 eps_S)) mean[ 1{|tS|<eps_S, tY<0} b_S ]   (S surface, + sign)
    Bun_Y = -(1/(2 eps_Y)) mean[ 1{|tY|<eps_Y, tS>=0} b_Y ]  (Y surface, - sign)
    with b = [Psi_t, -Psi_c] at Xeval.  Per-unit influence sums the S and Y
    contributions before squaring (captures their within-unit covariance).

    `include_empirical` adds the EMPIRICAL-MEASURE term theta(1-theta)/n.  The
    estimator theta_hat = P_n[1{tau_S>=0, tau_Y<0}] has two error sources: the
    boundary/Riesz part above, and the sampling error of the empirical average
    itself given the surfaces.  The boundary part dominates asymptotically
    (it is ~ K/n against 1/n), which is why the asymptotic theory omits it --
    but at these sample sizes it is 6-12% of the SE and its omission shows up
    as se_ratio < 1.  This is the exact analogue of the Var([h_0]_+) term that
    CCG's unknown-density welfare theorem adds to the known-density variance,
    and of `regular_companion_welfare`'s var_emp below.  The two parts are
    orthogonal at first order (one is driven by the regression residuals, the
    other by the covariate draw), so they add.
    """
    from opttreat.data import trimmed_std

    fmS_t, fmS_c = out_S["feature_map_t"], out_S["feature_map_c"]
    fmY_t, fmY_c = out_Y["feature_map_t"], out_Y["feature_map_c"]
    n_eval = Xeval.shape[0]

    def band_bun(fm_t, fm_c, tau_here, tau_other_mask, sign, min_band=5):
        eps = max(delta * trimmed_std(tau_here), 1e-8)
        good = (np.abs(tau_here) < eps) & tau_other_mask
        # widen the band if too few points fall in it (adaptive, cf. opttreat)
        for _ in range(6):
            if good.sum() >= min_band:
                break
            eps *= 1.5
            good = (np.abs(tau_here) < eps) & tau_other_mask
        b = np.hstack([np.asarray(fm_t(Xeval)), -np.asarray(fm_c(Xeval))])
        Kt = np.asarray(fm_t(Xeval[:1])).shape[1]
        if good.any():
            Bun = sign * b[good, :].sum(axis=0) / n_eval / (2.0 * eps)
        else:
            Bun = np.zeros(b.shape[1])
        return Bun[:Kt], Bun[Kt:], int(good.sum()), eps

    BunS_t, BunS_c, nbandS, epsS = band_bun(fmS_t, fmS_c, tS, tY < 0, +1.0)
    BunY_t, BunY_c, nbandY, epsY = band_bun(fmY_t, fmY_c, tY, (tS >= 0) if TIE_TO_ONE else (tS > 0), -1.0)

    # per-arm influence for each surface, then SUM per unit across surfaces.
    iS_t, iS_c = _riesz_influence(out_S, BunS_t, BunS_c)
    iY_t, iY_c = _riesz_influence(out_Y, BunY_t, BunY_c)
    # treated units align across S and Y fits (same df row order, same d); ditto control.
    infl_t = iS_t + iY_t
    infl_c = iS_c + iY_c
    var = float(np.sum(infl_t ** 2) + np.sum(infl_c ** 2))
    var_emp = 0.0
    if include_empirical:
        ge = (tS >= 0) if TIE_TO_ONE else (tS > 0)
        th = float(np.mean(ge & (tY < 0)))
        var_emp = th * (1.0 - th) / n_eval
        var += var_emp
    return var, {"n_band_S": nbandS, "n_band_Y": nbandY, "eps_S": epsS,
                 "eps_Y": epsY, "var_emp": var_emp}


# --------------------------------------------------------------------------- #
# Point estimate + inference
# --------------------------------------------------------------------------- #
def estimate_harm_share(df, segments=1, delta=0.05, with_sieve_se=True, z=1.959964):
    """Sieve plug-in of the harm share, with the two-band analytic SE."""
    Xc = _Xcols(df)
    d = len(Xc)
    opts = _sieve_opts(d, segments)
    out_S = fit_cate_surface(df, "S", opts)
    out_Y = fit_cate_surface(df, "Y", opts)
    Xeval = df[Xc].to_numpy(float)
    tS = np.asarray(out_S["h_hat"](Xeval)).ravel()
    tY = np.asarray(out_Y["h_hat"](Xeval)).ravel()
    q = _quadrants(tS, tY)
    est = HarmEstimate(
        theta_hat=q["pm"], quadrants=q,
        rho_hat=q["pm"] / max(q["pp"] + q["pm"], 1e-12),
        treat_share_S_hat=q["pp"] + q["pm"],
    )
    if with_sieve_se:
        var, diag = two_band_sieve_variance(out_S, out_Y, Xeval, tS, tY, delta)
        se = float(np.sqrt(max(var, 0.0)))
        est.se_sieve = se
        est.ci_sieve = (est.theta_hat - z * se, est.theta_hat + z * se)
        est.diag = diag
    return est


def bootstrap_ci(df, B=200, level=0.95, segments=1, seed=0, n_jobs=1):
    """Full-refit nonparametric bootstrap interval for theta (percentile)."""
    from joblib import Parallel, delayed
    rng = np.random.default_rng(seed)
    n = len(df)
    seeds = rng.integers(1 << 31, size=B)

    def one(b):
        r = np.random.default_rng(seeds[b])
        boot = df.iloc[r.integers(0, n, n)].reset_index(drop=True)
        return estimate_harm_share(boot, segments=segments, with_sieve_se=False).theta_hat

    vals = np.array(Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(b) for b in range(B)))
    a = (1 - level) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return float(lo), float(hi), float(np.std(vals, ddof=1))


# --------------------------------------------------------------------------- #
# Regular companion: value of the self-optimal long-run rule (envelope-regular)
# --------------------------------------------------------------------------- #
def regular_companion_welfare(df, segments=1):
    """Plug-in of the REGULAR companion W_Y = E[max(tau_Y(X), 0)] (long-run
    self-optimal policy value).  Its boundary weight is tau_Y, which vanishes on
    {tau_Y=0} -> envelope cancellation -> root-n regular, unlike the harm share.
    Returned with a CCG-style single-surface welfare sieve SE for contrast.
    """
    Xc = _Xcols(df)
    opts = _sieve_opts(len(Xc), segments)
    out_Y = fit_cate_surface(df, "Y", opts)
    Xeval = df[Xc].to_numpy(float)
    tY = np.asarray(out_Y["h_hat"](Xeval)).ravel()
    W = float(np.mean(np.maximum(tY, 0.0)))
    # welfare Riesz vector: Bun = mean[ 1{tY>=0} b ]  (no 1/(2eps) blow-up -> regular)
    fm_t, fm_c = out_Y["feature_map_t"], out_Y["feature_map_c"]
    b = np.hstack([np.asarray(fm_t(Xeval)), -np.asarray(fm_c(Xeval))])
    Kt = np.asarray(fm_t(Xeval[:1])).shape[1]
    good = tY >= 0
    Bun = b[good, :].sum(axis=0) / Xeval.shape[0] if good.any() else np.zeros(b.shape[1])
    it, ic = _riesz_influence(out_Y, Bun[:Kt], Bun[Kt:])
    # W_Y is REGULAR: its influence function has a nuisance/Riesz part AND the
    # empirical-average part g(X)-W with g=max(tau_Y,0).  The latter is the
    # dominant, root-n term (cf. opttreat welfare-unknown, sieve_var.py:272-277)
    # and MUST be included -- it is what makes W_Y sqrt(n)-regular.
    var_emp = float(np.var(np.maximum(tY, 0.0), ddof=0)) / len(tY)
    se = float(np.sqrt(np.sum(it ** 2) + np.sum(ic ** 2) + var_emp))
    return W, se
