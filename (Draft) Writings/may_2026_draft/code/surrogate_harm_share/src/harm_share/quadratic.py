"""Quadratic (second-order) debiasing for the DOUBLE-threshold harm share.

theta = Pr( tau_S(X) >= 0, tau_Y(X) < 0 )  is a two-threshold value functional
Theta(g,h) = E[ 1{g(X)>=0} 1{h(X)<0} ] with (g,h) = (tau_S, tau_Y).  Its
second-order pathwise derivative has THREE parts:

  D^2 Theta[(u,v),(u,v)] = Q_S[u,u]  +  Q_Y[v,v]  +  2 * C[u,v],

where Q_S / Q_Y are the single-threshold curvature terms on the two margins
M_S = {tau_S=0, tau_Y<0} and M_Y = {tau_Y=0, tau_S>=0} (as in Chen & Gao 2026,
Lemma PathD_Level), and C is a NEW codimension-2 CORNER integral over
{tau_S=0} \\cap {tau_Y=0} with weight u*v / (|grad tau_S| |grad tau_Y| sin
angle).  The corner's diagonal is the SAME order (K/n) as the margin diagonals,
so a margins-only quadratic correction leaves a first-order-in-K/n bias term:
corner-aware debiasing is necessary.  This module implements

  1. `ss_debiased_estimate`  -- the split-sample (SS) debiased estimator of
     Chen & Gao (2026) adapted to two thresholds.  With the second difference
     taken at step delta = 1/2 the correction is EXACT in closed form:

        theta_SS = 2*Theta(tau_bar) - (Theta(tau_A) + Theta(tau_B)) / 2,

     where tau_A, tau_B are half-sample fits and tau_bar their average -- a
     half-sample generalized jackknife, free of numerical-differentiation
     tuning.  A margins-only variant (perturbing one surface at a time) and
     the implied corner correction are returned alongside for the ablation.

  2. `dd_estimate` -- the "doubly debiased" (DD) estimator combining, in one
     procedure, (i) 2-fold cross-fitted generic-ML (GBR) first stages, (ii) the
     PROJECTED two-band sieve-Riesz first-order correction (the local influence
     function of the sieve-DML construction; identically zero for a sieve LS
     first stage by LS orthogonality, active for ML first stages), and (iii)
     the SS quadratic correction computed from the two fold fits.  This is the
     combination left open in Remark 7 of Chen, Chen & Gao (2026+): the
     first-order correction accommodates ML first stages, the second-order
     correction removes the quadratic own-observation diagonal.

Inference: all point estimates are studentized with the two-band sieve-Riesz
variance (Chen & Gao 2026 show the sieve variance remains consistent for the
debiased estimators, so the same studentized CI applies).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .estimator import (
    TIE_TO_ONE, _Xcols, _riesz_influence, _sieve_opts,
    fit_cate_surface, two_band_sieve_variance,
)

Z95 = 1.959964


# --------------------------------------------------------------------------- #
# Functional evaluation
# --------------------------------------------------------------------------- #
def theta_of(tS, tY, tie_to_one=TIE_TO_ONE):
    """Theta(g,h) = mean of 1{tau_S >= 0, tau_Y < 0} over the evaluation points."""
    ge = (tS >= 0) if tie_to_one else (tS > 0)
    return float(np.mean(ge & (tY < 0)))


def _predict_pair(out_S, out_Y, Xeval):
    tS = np.asarray(out_S["h_hat"](Xeval)).ravel()
    tY = np.asarray(out_Y["h_hat"](Xeval)).ravel()
    return tS, tY


# --------------------------------------------------------------------------- #
# Split-sample (SS) quadratic debiasing -- sieve first stage
# --------------------------------------------------------------------------- #
@dataclass
class SSEstimate:
    theta_plugin: float          # full-sample sieve plug-in
    theta_ss: float              # corner-aware SS debiased (full D^2)
    theta_ss_margins: float      # margins-only SS debiased (no corner)
    quad_full: float             # D^2Theta[(dS,dY)] at step 1/2 (x1/8 = correction)
    quad_S: float                # D^2Theta[(dS,0)]  margin-S diagonal
    quad_Y: float                # D^2Theta[(0,dY)]  margin-Y diagonal
    corner: float                # implied corner cross term (quad_full-quad_S-quad_Y)
    se_sieve: float | None = None
    ci_plugin: tuple | None = None
    ci_ss: tuple | None = None
    ci_ss_margins: tuple | None = None
    diag: dict = field(default_factory=dict)


def ss_debiased_estimate(df, segments=2, delta=0.08, seed=0, z=Z95,
                         with_sieve_se=True, tie_to_one=TIE_TO_ONE) -> SSEstimate:
    """Split-sample debiased harm share (Chen & Gao 2026 SS, two thresholds).

    The sample is split into halves A/B stratified by treatment W; each half
    fits its own two CATE surfaces with the SAME sieve dimension.  The step-1/2
    second difference gives the closed-form jackknife correction, decomposed
    into margin-S, margin-Y, and corner parts via polarization:

       quad_full = 4 [ Theta(A) + Theta(B) - 2 Theta(bar) ]        (joint move)
       quad_S    = 4 [ Theta(gA, hbar) + Theta(gB, hbar) - 2 Theta(bar) ]
       quad_Y    = 4 [ Theta(gbar, hA) + Theta(gbar, hB) - 2 Theta(bar) ]
       corner    = quad_full - quad_S - quad_Y                      (cross term)

    theta_ss = Theta(bar) - quad_full/8;  theta_ss_margins uses (quad_S+quad_Y)/8.
    The SE is the full-sample two-band sieve-Riesz SE (consistent for the
    debiased estimator; same studentization).
    """
    Xc = _Xcols(df)
    d = len(Xc)
    opts = _sieve_opts(d, segments)

    # full-sample fit: plug-in + SE
    out_S = fit_cate_surface(df, "S", opts)
    out_Y = fit_cate_surface(df, "Y", opts)
    Xeval = df[Xc].to_numpy(float)
    tS, tY = _predict_pair(out_S, out_Y, Xeval)
    theta_plugin = theta_of(tS, tY, tie_to_one)

    # stratified half split (by W so both arms appear in both halves)
    rng = np.random.default_rng(seed)
    W = df["W"].to_numpy().astype(int)
    idx_A = np.zeros(len(df), dtype=bool)
    for w in (0, 1):
        ids = np.flatnonzero(W == w)
        ids = ids[rng.permutation(len(ids))]
        idx_A[ids[: len(ids) // 2]] = True
    dfA = df.loc[idx_A].reset_index(drop=True)
    dfB = df.loc[~idx_A].reset_index(drop=True)

    outs = {}
    for tag, half in (("A", dfA), ("B", dfB)):
        outs[tag] = (fit_cate_surface(half, "S", opts),
                     fit_cate_surface(half, "Y", opts))
    gA, hA = _predict_pair(*outs["A"], Xeval)
    gB, hB = _predict_pair(*outs["B"], Xeval)
    gbar, hbar = 0.5 * (gA + gB), 0.5 * (hA + hB)

    th_bar = theta_of(gbar, hbar, tie_to_one)
    th_A = theta_of(gA, hA, tie_to_one)
    th_B = theta_of(gB, hB, tie_to_one)
    quad_full = 4.0 * (th_A + th_B - 2.0 * th_bar)
    quad_S = 4.0 * (theta_of(gA, hbar, tie_to_one) + theta_of(gB, hbar, tie_to_one)
                    - 2.0 * th_bar)
    quad_Y = 4.0 * (theta_of(gbar, hA, tie_to_one) + theta_of(gbar, hB, tie_to_one)
                    - 2.0 * th_bar)
    corner = quad_full - quad_S - quad_Y

    theta_ss = th_bar - quad_full / 8.0
    theta_ss_margins = th_bar - (quad_S + quad_Y) / 8.0

    est = SSEstimate(
        theta_plugin=theta_plugin, theta_ss=theta_ss,
        theta_ss_margins=theta_ss_margins,
        quad_full=quad_full, quad_S=quad_S, quad_Y=quad_Y, corner=corner,
        diag={"theta_bar": th_bar, "theta_A": th_A, "theta_B": th_B,
              "n_A": int(idx_A.sum()), "n_B": int((~idx_A).sum())},
    )
    if with_sieve_se:
        var, vdiag = two_band_sieve_variance(out_S, out_Y, Xeval, tS, tY, delta)
        se = float(np.sqrt(max(var, 0.0)))
        est.se_sieve = se
        est.ci_plugin = (theta_plugin - z * se, theta_plugin + z * se)
        est.ci_ss = (theta_ss - z * se, theta_ss + z * se)
        est.ci_ss_margins = (theta_ss_margins - z * se, theta_ss_margins + z * se)
        est.diag.update(vdiag)
    return est


# --------------------------------------------------------------------------- #
# Two-band Riesz pieces (band vectors + influence), reusable for cross-fitting
# --------------------------------------------------------------------------- #
def _band_bun(fm_t, fm_c, Xeval, tau_here, tau_other_mask, sign, delta, min_band=5):
    """Band Riesz vector for one surface: sign/(2 eps) * mean[1{|tau|<eps, mask} b],
    with b = [psi_t, -psi_c] and adaptive band widening (mirrors estimator.py)."""
    from opttreat.data import trimmed_std
    n_eval = Xeval.shape[0]
    eps = max(delta * trimmed_std(tau_here), 1e-8)
    good = (np.abs(tau_here) < eps) & tau_other_mask
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


def riesz_correction_and_variance(out_S, out_Y, Xeval, tS, tY, delta=0.10,
                                  tie_to_one=TIE_TO_ONE, include_empirical=True):
    """Projected two-band sieve-Riesz first-order CORRECTION and VARIANCE.

    correction = P_n[ vhat*_Kn(X,W) * resid ]  summed over the S and Y surfaces
    (the local-influence-function recentering of the sieve-DML construction);
    variance  = sum_i (per-unit summed influence)^2 + theta(1-theta)/n, matching
    `two_band_sieve_variance` (see its docstring for the empirical-measure term).
    For a least-squares sieve first stage whose residuals are orthogonal to its
    own basis the correction is ~0 by construction; for generic ML residuals
    (cross-fitted GBR) it is active.
    """
    ge = (tS >= 0) if tie_to_one else (tS > 0)
    BunS_t, BunS_c, nbS, epsS = _band_bun(
        out_S["feature_map_t"], out_S["feature_map_c"], Xeval, tS, tY < 0, +1.0, delta)
    BunY_t, BunY_c, nbY, epsY = _band_bun(
        out_Y["feature_map_t"], out_Y["feature_map_c"], Xeval, tY, ge, -1.0, delta)
    iS_t, iS_c = _riesz_influence(out_S, BunS_t, BunS_c)
    iY_t, iY_c = _riesz_influence(out_Y, BunY_t, BunY_c)
    infl_t, infl_c = iS_t + iY_t, iS_c + iY_c
    correction = float(np.sum(infl_t) + np.sum(infl_c))
    var = float(np.sum(infl_t ** 2) + np.sum(infl_c ** 2))
    var_emp = 0.0
    if include_empirical:
        th = float(np.mean(ge & (tY < 0)))
        var_emp = th * (1.0 - th) / Xeval.shape[0]
        var += var_emp
    diag = {"n_band_S": nbS, "n_band_Y": nbY, "eps_S": epsS, "eps_Y": epsY,
            "var_emp": var_emp}
    return correction, var, diag


# --------------------------------------------------------------------------- #
# Doubly debiased (DD): cross-fitted ML + Riesz correction + SS quadratic
# --------------------------------------------------------------------------- #
@dataclass
class DDEstimate:
    theta_cf: float              # 2-fold cross-fit ML plug-in (no corrections)
    theta_cf_riesz: float        # + projected Riesz first-order correction
    theta_dd: float              # + SS quadratic correction as well (headline)
    correction_riesz: float
    quad_full: float
    corner: float
    se_sieve: float | None = None
    ci_cf: tuple | None = None
    ci_cf_riesz: tuple | None = None
    ci_dd: tuple | None = None
    diag: dict = field(default_factory=dict)


def _fit_ml_arm_means(X, W, yv, tr, nuisance, seed):
    """Fit E[y|X,W=w] for w=0,1 on rows `tr`; return predictors over any X."""
    if nuisance == "gbr":
        from sklearn.ensemble import HistGradientBoostingRegressor
        mk = lambda s: HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=250,
            l2_regularization=1.0, random_state=s)
    elif nuisance == "rf":
        from sklearn.ensemble import RandomForestRegressor
        mk = lambda s: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, max_features="sqrt",
            n_jobs=1, random_state=s)
    else:
        raise ValueError(f"unknown ML nuisance '{nuisance}' (use gbr|rf)")
    models = {}
    for w in (0, 1):
        m = tr[W[tr] == w]
        r = mk(seed + w)
        r.fit(X[m], yv[m])
        models[w] = r
    return models


def dd_estimate(df, nuisance="gbr", segments=2, riesz="sieve", n_features=200,
                delta=0.10, seed=0, z=Z95, tie_to_one=TIE_TO_ONE) -> DDEstimate:
    """Doubly debiased harm share: 2-fold cross-fitted ML first stages, the
    projected two-band sieve-Riesz correction, and the SS quadratic correction.

    Fold structure: halves A/B (stratified by W).  tau^A is the surface pair
    fit on A (used to predict everywhere), likewise tau^B; the cross-fitted
    (out-of-fold) surfaces use tau^B on A's rows and tau^A on B's rows.

      theta_cf       = Theta( tau_oof )                       [plug-in]
      theta_cf_riesz = theta_cf + P_n[ vhat* resid_oof ]      [first-order]
      theta_dd       = 2 Theta(tau_bar) - (Theta(tau^A)+Theta(tau^B))/2
                       + P_n[ vhat* resid_oof ]               [+ second-order]

    The Riesz basis is a tensor B-spline (d<=3) or random features (`riesz`),
    with Gram/bands from the full sample and CROSS-FITTED residuals; the same
    ingredients give the two-band sieve variance for studentization.
    """
    from .sieve_dml import _linear_sieve_out

    Xc = _Xcols(df)
    X = df[Xc].to_numpy(float)
    W = df["W"].to_numpy().astype(int)
    S = df["S"].to_numpy(float)
    Y = df["Y"].to_numpy(float)
    n = len(df)

    # stratified halves
    rng = np.random.default_rng(seed)
    inA = np.zeros(n, dtype=bool)
    for w in (0, 1):
        ids = np.flatnonzero(W == w)
        ids = ids[rng.permutation(len(ids))]
        inA[ids[: len(ids) // 2]] = True
    rows_A, rows_B = np.flatnonzero(inA), np.flatnonzero(~inA)

    # half-sample ML fits, predicted everywhere
    tau_half = {}
    mu_oof = {"S": {0: np.empty(n), 1: np.empty(n)},
              "Y": {0: np.empty(n), 1: np.empty(n)}}
    for tag, tr, te in (("A", rows_A, rows_B), ("B", rows_B, rows_A)):
        for name, yv, off in (("S", S, 0), ("Y", Y, 1000)):
            models = _fit_ml_arm_means(X, W, yv, tr, nuisance, seed + off)
            mu0, mu1 = models[0].predict(X), models[1].predict(X)
            tau_half[(tag, name)] = mu1 - mu0
            # out-of-fold arm means on the held-out rows
            mu_oof[name][0][te] = mu0[te]
            mu_oof[name][1][te] = mu1[te]

    gA, gB = tau_half[("A", "S")], tau_half[("B", "S")]
    hA, hB = tau_half[("A", "Y")], tau_half[("B", "Y")]
    gbar, hbar = 0.5 * (gA + gB), 0.5 * (hA + hB)
    tS_oof = np.where(inA, gB, gA)                      # oof: other half's fit
    tY_oof = np.where(inA, hB, hA)

    theta_cf = theta_of(tS_oof, tY_oof, tie_to_one)

    # SS quadratic correction from the two half fits (step-1/2 jackknife)
    th_bar = theta_of(gbar, hbar, tie_to_one)
    th_A = theta_of(gA, hA, tie_to_one)
    th_B = theta_of(gB, hB, tie_to_one)
    quad_full = 4.0 * (th_A + th_B - 2.0 * th_bar)
    quad_S = 4.0 * (theta_of(gA, hbar, tie_to_one) + theta_of(gB, hbar, tie_to_one)
                    - 2.0 * th_bar)
    quad_Y = 4.0 * (theta_of(gbar, hA, tie_to_one) + theta_of(gbar, hB, tie_to_one)
                    - 2.0 * th_bar)
    corner = quad_full - quad_S - quad_Y

    # Riesz basis (linear-in-features) for the projection; oof ML residuals
    out_S = _linear_sieve_out(df, "S", riesz, segments, n_features, seed)
    out_Y = _linear_sieve_out(df, "Y", riesz, segments, n_features, seed + 7)
    mt, mc = W == 1, W == 0
    out_S = dict(out_S); out_Y = dict(out_Y)
    out_S["e_t"] = S[mt] - mu_oof["S"][1][mt]
    out_S["e_c"] = S[mc] - mu_oof["S"][0][mc]
    out_Y["e_t"] = Y[mt] - mu_oof["Y"][1][mt]
    out_Y["e_c"] = Y[mc] - mu_oof["Y"][0][mc]

    corr, var, vdiag = riesz_correction_and_variance(
        out_S, out_Y, X, tS_oof, tY_oof, delta, tie_to_one)
    se = float(np.sqrt(max(var, 0.0)))

    theta_cf_riesz = theta_cf + corr
    theta_dd = (2.0 * th_bar - 0.5 * (th_A + th_B)) + corr

    est = DDEstimate(
        theta_cf=theta_cf, theta_cf_riesz=theta_cf_riesz, theta_dd=theta_dd,
        correction_riesz=corr, quad_full=quad_full, corner=corner,
        se_sieve=se,
        ci_cf=(theta_cf - z * se, theta_cf + z * se),
        ci_cf_riesz=(theta_cf_riesz - z * se, theta_cf_riesz + z * se),
        ci_dd=(theta_dd - z * se, theta_dd + z * se),
        diag={"nuisance": nuisance, "riesz": riesz, "quad_S": quad_S,
              "quad_Y": quad_Y, "theta_bar": th_bar, **vdiag},
    )
    return est
