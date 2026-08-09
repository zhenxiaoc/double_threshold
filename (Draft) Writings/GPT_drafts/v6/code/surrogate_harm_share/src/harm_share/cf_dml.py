"""Configurable cross-fitted two-band sieve-Riesz DML, for diagnosing coverage.

Generalizes `sieve_dml.harm_share_riesz_dml` along the axes that the theory
says should matter for whether the studentized interval is valid:

  * `learner`   -- first-stage arm-mean estimator: gbr | rf | krr | mlp | sieve
                   (`krr` = RBF random-feature ridge, a SMOOTH ML learner whose
                   error lives close to the span of a random-feature Riesz
                   basis; `mlp` = smooth but out-of-span; `sieve` = the linear
                   B-spline LS estimator, for which the projected correction is
                   exactly zero by least-squares orthogonality)
  * `K`         -- number of cross-fitting folds (nuisance trained on
                   (K-1)/K of the sample; K=2 doubles the regularization bias
                   relative to K=5)
  * `riesz`     -- basis for the sieve-Riesz projection: sieve | rf
  * `correct`   -- whether to add the projected first-order correction

Everything is studentized by the same two-band sieve-Riesz variance, so the
comparisons isolate the first stage rather than the variance formula.

`projection_diagnostics` measures, against the ORACLE arm means, how much of a
learner's error the Riesz basis can see -- the direct empirical test of the
sieve-approximation condition ("V_Kn tracks the first-stage error").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .estimator import TIE_TO_ONE, _Xcols, _quadrants
from .quadratic import riesz_correction_and_variance

Z95 = 1.959964


# --------------------------------------------------------------------------- #
# First-stage learners
# --------------------------------------------------------------------------- #
def make_learner(name: str, seed: int = 0, capacity: str = "default"):
    """One arm-mean regressor.  `capacity` in {default, high} for gbr/rf."""
    nl = name.lower()
    if nl == "gbr":
        from sklearn.ensemble import HistGradientBoostingRegressor
        if capacity == "high":
            return HistGradientBoostingRegressor(
                max_depth=6, learning_rate=0.03, max_iter=800,
                l2_regularization=0.0, min_samples_leaf=5, random_state=seed)
        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=250,
            l2_regularization=1.0, random_state=seed)
    if nl == "rf":
        from sklearn.ensemble import RandomForestRegressor
        leaf = 2 if capacity == "high" else 10
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=leaf, max_features="sqrt",
            n_jobs=1, random_state=seed)
    if nl == "krr":
        # RBF random-feature ridge: a SMOOTH nonparametric learner
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.kernel_approximation import RBFSampler
        from sklearn.linear_model import Ridge
        return make_pipeline(
            StandardScaler(),
            RBFSampler(gamma=0.5, n_components=300, random_state=seed),
            Ridge(alpha=1e-3))
    if nl == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh",
                         alpha=1e-4, max_iter=200, early_stopping=True,
                         n_iter_no_change=8, random_state=seed))
    if nl == "sieve":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer
        from sklearn.linear_model import Ridge
        return make_pipeline(
            SplineTransformer(n_knots=5, degree=3, include_bias=True),
            Ridge(alpha=1e-6))
    raise ValueError(f"unknown learner '{name}'")


def crossfit_arm_means(X, W, y, K, learner, seed, capacity="default"):
    """Out-of-fold mu_hat(x, 0) and mu_hat(x, 1) at the sample rows."""
    from sklearn.model_selection import KFold
    n = len(y)
    mu = {0: np.empty(n), 1: np.empty(n)}
    if K <= 1:                                   # no cross-fitting
        for w in (0, 1):
            m = W == w
            r = make_learner(learner, seed + w, capacity)
            r.fit(X[m], y[m])
            mu[w][:] = r.predict(X)
        return mu
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        for w in (0, 1):
            m = tr[W[tr] == w]
            r = make_learner(learner, seed + w, capacity)
            r.fit(X[m], y[m])
            mu[w][te] = r.predict(X[te])
    return mu


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
@dataclass
class CFResult:
    theta_plugin: float
    theta: float                 # plugin + correction (if correct=True)
    correction: float
    se: float
    ci: tuple
    diag: dict = field(default_factory=dict)


def cf_riesz_dml(df, learner="gbr", K=5, riesz="sieve", segments=2,
                 n_features=200, delta=0.10, seed=0, correct=True,
                 capacity="default", z=Z95) -> CFResult:
    """Cross-fitted ML first stage + projected two-band sieve-Riesz correction,
    studentized by the two-band sieve-Riesz variance."""
    from .sieve_dml import _linear_sieve_out

    Xc = _Xcols(df)
    X = df[Xc].to_numpy(float)
    W = df["W"].to_numpy().astype(int)
    S = df["S"].to_numpy(float)
    Y = df["Y"].to_numpy(float)

    muS = crossfit_arm_means(X, W, S, K, learner, seed, capacity)
    muY = crossfit_arm_means(X, W, Y, K, learner, seed + 1000, capacity)
    tS, tY = muS[1] - muS[0], muY[1] - muY[0]

    q = _quadrants(tS, tY)
    theta_plugin = q["pm"]

    # Riesz basis (linear-in-features, full sample) with OOF ML residuals
    out_S = dict(_linear_sieve_out(df, "S", riesz, segments, n_features, seed))
    out_Y = dict(_linear_sieve_out(df, "Y", riesz, segments, n_features, seed + 7))
    mt, mc = W == 1, W == 0
    out_S["e_t"] = S[mt] - muS[1][mt]
    out_S["e_c"] = S[mc] - muS[0][mc]
    out_Y["e_t"] = Y[mt] - muY[1][mt]
    out_Y["e_c"] = Y[mc] - muY[0][mc]

    corr, var, vdiag = riesz_correction_and_variance(
        out_S, out_Y, X, tS, tY, delta, TIE_TO_ONE)
    se = float(np.sqrt(max(var, 0.0)))
    theta = theta_plugin + (corr if correct else 0.0)
    return CFResult(
        theta_plugin=theta_plugin, theta=theta, correction=corr, se=se,
        ci=(theta - z * se, theta + z * se),
        diag={"learner": learner, "K": K, "riesz": riesz,
              "capacity": capacity, **vdiag})


# --------------------------------------------------------------------------- #
# Projection diagnostics: does the Riesz span see the learner's error?
# --------------------------------------------------------------------------- #
def projection_diagnostics(df, oracle, learner="gbr", K=5, riesz="sieve",
                           segments=2, n_features=200, delta=0.10, seed=0,
                           capacity="default"):
    """Compare the learner's arm-mean error with its projection on the Riesz
    basis, using the ORACLE arm means as truth.

    Returns, per outcome and arm:
      r2_proj  -- fraction of the error's L2 norm captured by the basis span
      bias_l2  -- ||mu_hat - mu_0||_2 (the error the correction must remove)
    and, for the functional:
      frac_bias_removed -- |correction| / |plug-in bias| computed with the
      oracle CATEs, i.e. how much of the boundary-relevant bias the projected
      correction can actually see.
    """
    from .sieve_dml import _linear_sieve_out

    Xc = _Xcols(df)
    X = df[Xc].to_numpy(float)
    W = df["W"].to_numpy().astype(int)
    S = df["S"].to_numpy(float)
    Y = df["Y"].to_numpy(float)

    muS = crossfit_arm_means(X, W, S, K, learner, seed, capacity)
    muY = crossfit_arm_means(X, W, Y, K, learner, seed + 1000, capacity)

    truth = {
        ("S", 0): oracle.mu_S(X, 0), ("S", 1): oracle.mu_S(X, 1),
        ("Y", 0): oracle.mu_Y(X, 0), ("Y", 1): oracle.mu_Y(X, 1),
    }
    out = {"S": _linear_sieve_out(df, "S", riesz, segments, n_features, seed),
           "Y": _linear_sieve_out(df, "Y", riesz, segments, n_features, seed + 7)}

    res = {}
    for name, mu in (("S", muS), ("Y", muY)):
        for w in (0, 1):
            err = np.asarray(mu[w]).ravel() - np.asarray(truth[(name, w)]).ravel()
            fm = out[name]["feature_map_t" if w == 1 else "feature_map_c"]
            B = np.asarray(fm(X))
            coef, *_ = np.linalg.lstsq(B, err, rcond=None)
            resid = err - B @ coef
            denom = float(err @ err)
            res[f"r2_{name}{w}"] = float(1 - (resid @ resid) / denom) if denom > 0 else np.nan
            res[f"bias_l2_{name}{w}"] = float(np.sqrt(denom / len(err)))
            res[f"mean_err_{name}{w}"] = float(err.mean())

    # attenuation of the CATE surfaces (shrinkage toward zero)
    tS_hat, tY_hat = muS[1] - muS[0], muY[1] - muY[0]
    tS_0 = np.asarray(oracle.tau_S(X)).ravel()
    tY_0 = np.asarray(oracle.tau_Y(X)).ravel()
    for nm, hat, tru in (("S", tS_hat, tS_0), ("Y", tY_hat, tY_0)):
        # regression slope of hat on true: <1 means attenuation
        res[f"atten_{nm}"] = float((hat @ tru) / (tru @ tru))
        res[f"sd_ratio_{nm}"] = float(hat.std() / tru.std())
        res[f"rmse_tau_{nm}"] = float(np.sqrt(np.mean((hat - tru) ** 2)))

    # how much of the plug-in's boundary bias does the correction see?
    # (reuse the cross-fit above -- refitting would double the cost)
    W_ = W
    out_S = dict(out["S"]); out_Y = dict(out["Y"])
    mt, mc = W_ == 1, W_ == 0
    out_S["e_t"] = S[mt] - muS[1][mt]
    out_S["e_c"] = S[mc] - muS[0][mc]
    out_Y["e_t"] = Y[mt] - muY[1][mt]
    out_Y["e_c"] = Y[mc] - muY[0][mc]
    corr, var, _ = riesz_correction_and_variance(
        out_S, out_Y, X, tS_hat, tY_hat, delta, TIE_TO_ONE)

    theta_plugin = float(np.mean((tS_hat >= 0) & (tY_hat < 0)))
    theta_oracle_plugin = float(np.mean((tS_0 >= 0) & (tY_0 < 0)))
    res["theta_plugin"] = theta_plugin
    res["theta_oracle_plugin"] = theta_oracle_plugin
    res["plugin_err"] = theta_plugin - theta_oracle_plugin
    res["correction"] = corr
    res["frac_removed"] = (float(-corr / (theta_plugin - theta_oracle_plugin))
                           if abs(theta_plugin - theta_oracle_plugin) > 1e-9 else np.nan)
    res["se"] = float(np.sqrt(max(var, 0.0)))
    return res
