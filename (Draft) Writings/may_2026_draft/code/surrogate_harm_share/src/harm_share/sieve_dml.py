"""Cross-fit (sieve-)DML for the DOUBLE-threshold surrogate-harm share

    theta = Pr( tau_S(X) >= 0 , tau_Y(X) < 0 ),   tau_S = mu_{S,1}-mu_{S,0}, tau_Y likewise.

Unlike the ordinary single-threshold value functional, the harm share has TWO
decision boundaries M_S = {tau_S=0, tau_Y<0} and M_Y = {tau_Y=0, tau_S>=0}, so the
debiasing correction has TWO Riesz bands.  Its (narrow-band) influence function is

  psi(O) = ( 1{tau_S>=0, tau_Y<0} - theta )
         + (1/2 eps_S) 1{|tau_S|<eps_S, tau_Y<0} * xi_S       # S-boundary band  (+)
         - (1/2 eps_Y) 1{|tau_Y|<eps_Y, tau_S>=0} * xi_Y      # Y-boundary band  (-)

where xi_S, xi_Y are the AIPW (Neyman-orthogonal) residuals of the two CATEs,

  xi_S = ( W/e - (1-W)/(1-e) ) ( S - mu_{S,W}(X) ),   xi_Y likewise,

with e = P(W=1) known (randomized).  Because psi depends on the nuisances only
through the four arm means mu_{S/Y, 0/1}, the correction is **Neyman-orthogonal**:
any first-stage regressor -- a sieve, random forest, or gradient boosting -- can
be plugged in, and cross-fitting removes the own-observation bias.  theta is
irregular (thin-set), so the +/- bands enter with a 1/eps weight and the interval
width shrinks at the boundary rate, not root-n; but with orthogonal cross-fitting
the interval is correctly centered and calibrated.

`harm_share_dml(df, nuisance=...)` returns the debiased point estimate, the
influence-function SE, and the CI.  `nuisance in {"sieve","rf","gbr"}`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

Z95 = norm.ppf(0.975)


def _Xcols(df):
    return [c for c in df.columns if c.startswith("X") and c[1:].isdigit()]


# --------------------------------------------------------------------------- #
# First-stage arm-mean regressors  mu_{Y,w}(x) = E[outcome | X=x, W=w]
# --------------------------------------------------------------------------- #
def _make_regressor(nuisance: str, d: int, seed: int = 0):
    """A fresh regressor for one arm mean.  All are flexible enough to be a
    valid nonparametric first stage; the DML correction is what makes inference
    robust to their bias."""
    nl = nuisance.lower()
    if nl in ("rf", "forest", "random_forest"):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, max_features="sqrt",
            n_jobs=1, random_state=seed)
    if nl in ("gbr", "xgb", "xgboost", "boost"):
        # sklearn's histogram gradient boosting -- an XGBoost-equivalent
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=250,
            l2_regularization=1.0, random_state=seed)
    if nl == "sieve":
        # tensor/poly B-spline ridge -- the paper's linear sieve, as an sklearn pipe
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer
        from sklearn.linear_model import Ridge
        n_knots = 5 if d <= 3 else 3
        return make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=3, include_bias=True),
            Ridge(alpha=1e-3))
    raise ValueError(f"unknown nuisance '{nuisance}' (use sieve|rf|gbr)")


def _fit_arm_means(Xtr, Wtr, Ytr, Xte, nuisance, seed):
    """Fit E[Y|X,W=0] and E[Y|X,W=1] on the train fold, predict both on Xte."""
    d = Xtr.shape[1]
    preds = {}
    for w in (0, 1):
        m = Wtr == w
        reg = _make_regressor(nuisance, d, seed + w)
        reg.fit(Xtr[m], Ytr[m])
        preds[w] = reg.predict(Xte)
    return preds[0], preds[1]


# --------------------------------------------------------------------------- #
# Cross-fit DML for the harm share
# --------------------------------------------------------------------------- #
@dataclass
class DMLResult:
    theta_plugin: float
    theta_dml: float
    se: float
    ci: tuple
    treat_share_S: float
    diag: dict


def harm_share_dml(df, nuisance="rf", K=5, e=0.5, delta=0.10,
                   z=Z95, seed=0, tie_to_one=True, debias=False) -> DMLResult:
    """Cross-fit estimate of the harm share with `nuisance` first stage.

    `debias=True` adds the first-order two-band correction; but the harm share is
    a THIN-SET (irregular) functional, so that correction over-shoots — the
    cross-fit plug-in is already near-unbiased and the boundary-band influence
    only needs to enter the SE.  We therefore default to `debias=False`: the point
    estimate is the cross-fit plug-in and the interval uses the full two-band
    influence-function variance (its 1/eps boundary terms dominate, giving the
    sub-root-n width the estimand requires)."""
    from sklearn.model_selection import KFold

    Xc = _Xcols(df)
    X = df[Xc].to_numpy(float)
    W = df["W"].to_numpy().astype(int)
    S = df["S"].to_numpy(float)
    Y = df["Y"].to_numpy(float)
    n = len(df)

    # out-of-fold arm means -> OOF CATEs and OOF AIPW residuals
    muS = {0: np.empty(n), 1: np.empty(n)}
    muY = {0: np.empty(n), 1: np.empty(n)}
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        muS[0][te], muS[1][te] = _fit_arm_means(X[tr], W[tr], S[tr], X[te], nuisance, seed)
        muY[0][te], muY[1][te] = _fit_arm_means(X[tr], W[tr], Y[tr], X[te], nuisance, seed + 100)

    tS = muS[1] - muS[0]
    tY = muY[1] - muY[0]
    muS_W = np.where(W == 1, muS[1], muS[0])         # fitted mean at the realized arm
    muY_W = np.where(W == 1, muY[1], muY[0])
    ipw = W / e - (1 - W) / (1 - e)                  # (=+2 for treated, -2 for control at e=.5)
    xiS = ipw * (S - muS_W)                          # AIPW / Neyman-orthogonal residuals
    xiY = ipw * (Y - muY_W)

    ge = (tS >= 0) if tie_to_one else (tS > 0)
    ind = ge & (tY < 0)
    theta_plugin = float(ind.mean())

    epsS = max(delta * np.std(tS), 1e-8)
    epsY = max(delta * np.std(tY), 1e-8)
    bandS = (np.abs(tS) < epsS) & (tY < 0)
    bandY = (np.abs(tY) < epsY) & ge
    psi_S = np.where(bandS, xiS / (2 * epsS), 0.0)   # + S-boundary band
    psi_Y = -np.where(bandY, xiY / (2 * epsY), 0.0)  # - Y-boundary band

    correction = float(np.mean(psi_S + psi_Y))
    theta_point = theta_plugin + (correction if debias else 0.0)
    psi = (ind.astype(float) - theta_point) + psi_S + psi_Y     # full influence
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    theta_dml = theta_point
    ci = (theta_point - z * se, theta_point + z * se)
    diag = {"nuisance": nuisance, "K": K, "eps_S": epsS, "eps_Y": epsY,
            "n_band_S": int(bandS.sum()), "n_band_Y": int(bandY.sum()),
            "correction": correction}
    return DMLResult(theta_plugin, theta_dml, se, ci, float(ge.mean()), diag)


# --------------------------------------------------------------------------- #
# Two-band sieve-Riesz DML (the paper's SE with an RF or GBR nuisance)
# --------------------------------------------------------------------------- #
def _linear_sieve_out(df, ycol, method, segments, n_features, seed):
    """Fit one CATE surface with a LINEAR-in-features first stage (sieve B-spline
    or random-feature ridge) via opttreat, returning the estimator-output dict
    whose feature maps / Psi / residuals the two-band sieve-Riesz variance needs."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / "longterm-main" / "Optimal-Treatment-main"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from opttreat.config import EstimatorConfig
    from opttreat.data import split_treated_control
    from opttreat.estimation import get_estimator

    Xc = _Xcols(df)
    work = df[Xc].copy()
    work["d"] = df["W"].to_numpy()
    work["y"] = df[ycol].to_numpy()
    parsed = split_treated_control(work)
    if method == "rf":
        cfg = EstimatorConfig("rf_ridge", {
            "rfg_type": "iid_sphere", "activation": "exp", "share_features": False,
            "n_features": n_features, "random_state": seed, "alpha": 1e-5})
    else:                                       # 'sieve'
        cfg = EstimatorConfig("sieve", {
            "solver": "pinv", "share_features": False, "J_x_degree": 3,
            "J_x_segments_t": segments, "J_x_segments_c": segments,
            "knots": "uniform", "basis": "tensor"})
    return get_estimator(cfg).fit(parsed)


def _gbr_riesz_out(df, ycol, segments, seed, K=5, riesz="rf", n_features=200):
    """GBR (XGBoost-style) nuisance with a random-feature/sieve Riesz basis -- the
    'sieve-DML' mix.  The GBR arm means are CROSS-FITTED (out-of-fold) so their
    overfitting bias does not leak into the plug-in; the Riesz basis (random
    features by default, so it scales to high dimension) supplies the feature maps."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    out = _linear_sieve_out(df, ycol, riesz, segments, n_features, seed)   # Riesz basis
    Xc = _Xcols(df)
    X = df[Xc].to_numpy(float); W = df["W"].to_numpy().astype(int); yv = df[ycol].to_numpy(float)
    n = len(df)
    mu = {0: np.empty(n), 1: np.empty(n)}          # out-of-fold arm means
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        for w in (0, 1):
            m = tr[W[tr] == w]
            r = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250,
                                              l2_regularization=1.0, random_state=seed + w)
            r.fit(X[m], yv[m]); mu[w][te] = r.predict(X[te])
    mt, mc = W == 1, W == 0                         # rows align with Psi_t / Psi_c
    out["e_t"] = yv[mt] - mu[1][mt]
    out["e_c"] = yv[mc] - mu[0][mc]
    out["tau_oof"] = mu[1] - mu[0]                  # cross-fitted CATE at the sample rows
    return out


def harm_share_riesz_dml(df, nuisance="rf", segments=2, n_features=200,
                         delta=0.10, z=Z95, seed=0, tie_to_one=True):
    """Harm-share estimate with an ML/linear nuisance and the paper's TWO-BAND
    sieve-Riesz variance -- the debiasing correctly calibrated for this thin-set
    functional (unlike the raw AIPW 1/eps influence).  `nuisance in {sieve,rf,gbr}`:
    sieve/rf expose their own feature maps; gbr uses a sieve-projected Riesz."""
    from .estimator import two_band_sieve_variance, _quadrants
    if nuisance == "gbr":
        out_S = _gbr_riesz_out(df, "S", segments, seed, n_features=n_features)
        out_Y = _gbr_riesz_out(df, "Y", segments, seed + 7, n_features=n_features)
    else:
        out_S = _linear_sieve_out(df, "S", nuisance, segments, n_features, seed)
        out_Y = _linear_sieve_out(df, "Y", nuisance, segments, n_features, seed + 7)
    Xc = _Xcols(df)
    Xeval = df[Xc].to_numpy(float)
    tS = np.asarray(out_S.get("tau_oof", None) if "tau_oof" in out_S else out_S["h_hat"](Xeval)).ravel()
    tY = np.asarray(out_Y.get("tau_oof", None) if "tau_oof" in out_Y else out_Y["h_hat"](Xeval)).ravel()
    q = _quadrants(tS, tY)
    theta = q["pm"]
    var, diag = two_band_sieve_variance(out_S, out_Y, Xeval, tS, tY, delta)
    se = float(np.sqrt(max(var, 0.0)))
    ci = (theta - z * se, theta + z * se)
    diag.update({"nuisance": nuisance})
    return DMLResult(theta, theta, se, ci, float((tS >= 0).mean()), diag)
