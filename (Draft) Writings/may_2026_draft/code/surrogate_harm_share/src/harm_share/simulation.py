"""Monte Carlo study of the surrogate-induced harm share.

Compares plug-in sieve estimates against the calibrated oracle truth and reports,
for each sample size n:
  * bias, MC-SD, RMSE of theta_hat (the double-threshold harm share);
  * mean two-band sieve SE, se_ratio = mean(SE)/MC-SD, and its 95% coverage;
  * the same for the SINGLE-threshold companion Pr(tau_S>=0) (also irregular) and
    for the REGULAR companion W_Y = E[max(tau_Y,0)] (root-n regular);
  * the four-quadrant confusion matrix means and the conditional harm ratio rho.

The regular/irregular CONTRAST is the headline: theta and the treat-share are
level-set functionals (no sqrt(n) IF), while W_Y is envelope-regular; their RMSE
should decay at different rates in n.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm

from .estimator import estimate_harm_share, regular_companion_welfare, bootstrap_ci

Z95 = norm.ppf(0.975)


def _child_seed(seed, *ix):
    ss = np.random.SeedSequence([seed, *ix])
    return int(ss.generate_state(1)[0])


def run_mc(oracle, n, n_rep, truth, *, segments=2, delta=0.08, seed=20260713,
           n_jobs=-1, with_companion=True):
    """One (n) cell of the Monte Carlo.  Returns aggregate dict + raw arrays."""
    th_true = truth["theta_harm"]
    ts_true = truth["treat_share_S"]

    def one(rep):
        rng = np.random.default_rng(_child_seed(seed, n, rep))
        df = oracle.sample_experiment(n, rng)
        est = estimate_harm_share(df, segments=segments, delta=delta)
        row = {
            "theta": est.theta_hat, "se": est.se_sieve,
            "treatS": est.treat_share_S_hat, "rho": est.rho_hat,
            "pp": est.quadrants["pp"], "pm": est.quadrants["pm"],
            "mp": est.quadrants["mp"], "mm": est.quadrants["mm"],
            "cov": float(est.ci_sieve[0] <= th_true <= est.ci_sieve[1]),
        }
        if with_companion:
            W, seW = regular_companion_welfare(df, segments=segments)
            row["W"] = W; row["seW"] = seW
        return row

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(r) for r in range(n_rep))
    col = lambda k: np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])
    theta = col("theta"); se = col("se")
    agg = {
        "n": n, "n_rep": n_rep, "segments": segments,
        "theta_true": th_true,
        "bias": float(theta.mean() - th_true),
        "mc_sd": float(theta.std(ddof=1)),
        "rmse": float(np.sqrt(np.mean((theta - th_true) ** 2))),
        "mean_se": float(se.mean()),
        "se_ratio": float(se.mean() / theta.std(ddof=1)) if theta.std() > 0 else np.nan,
        "cov95_sieve": float(col("cov").mean()),
        "treatS_bias": float(col("treatS").mean() - ts_true),
        "treatS_rmse": float(np.sqrt(np.mean((col("treatS") - ts_true) ** 2))),
        "rho_mean": float(col("rho").mean()),
        "pp": float(col("pp").mean()), "pm": float(col("pm").mean()),
        "mp": float(col("mp").mean()), "mm": float(col("mm").mean()),
    }
    if with_companion and col("W").size:
        W = col("W"); seW = col("seW")
        agg["W_mean"] = float(W.mean())
        agg["W_mc_sd"] = float(W.std(ddof=1))
        agg["W_mean_se"] = float(seW.mean()) if seW.size else float("nan")
        # regular companion: analytic root-n SE should track the MC-SD (ratio ~ 1)
        agg["W_se_ratio"] = (float(seW.mean() / W.std(ddof=1))
                             if seW.size and W.std() > 0 else float("nan"))
    return {"agg": agg, "raw": rows}


def rate_experiment(oracle, n_list, truth, *, segments=2, seg_of_n=None, delta=0.08,
                    n_rep=400, seed=20260713, n_jobs=-1):
    """Run run_mc across n and fit log-RMSE / log-SD vs log-n slopes.

    For a regular (root-n) functional the slope is ~ -0.5; an irregular level-set
    functional decays more slowly (shallower slope) and/or hits a bias floor.

    `seg_of_n` (dict n->segments or callable) implements an UNDERSMOOTHING schedule
    that grows the sieve dimension K with n, so the sieve approximation bias shrinks
    with n instead of hitting a fixed-K floor -- this keeps the plug-in interval from
    losing coverage as n grows (the coverage decay seen at fixed K).
    """
    def seg_for(n):
        if seg_of_n is None:
            return segments
        return seg_of_n(n) if callable(seg_of_n) else seg_of_n[n]
    cells = [run_mc(oracle, n, n_rep, truth, segments=seg_for(n), delta=delta,
                    seed=seed, n_jobs=n_jobs)["agg"] for n in n_list]
    logn = np.log(np.array(n_list, float))

    def slope(key):
        y = np.log(np.array([c[key] for c in cells], float))
        A = np.vstack([logn, np.ones_like(logn)]).T
        return float(np.linalg.lstsq(A, y, rcond=None)[0][0])

    slopes = {
        "theta_rmse_slope": slope("rmse"),
        "theta_sd_slope": slope("mc_sd"),
        "treatS_rmse_slope": slope("treatS_rmse"),
    }
    if all("W_mc_sd" in c for c in cells):
        slopes["W_sd_slope"] = slope("W_mc_sd")
    return {"cells": cells, "slopes": slopes, "n_list": list(n_list)}


def bootstrap_coverage(oracle, n, n_rep, truth, *, segments=2, B=200, level=0.95,
                       seed=20260713, n_jobs=-1):
    """Coverage of the FULL-REFIT bootstrap interval (the primary interval)."""
    th_true = truth["theta_harm"]

    def one(rep):
        rng = np.random.default_rng(_child_seed(seed, n, rep, 99))
        df = oracle.sample_experiment(n, rng)
        lo, hi, sd = bootstrap_ci(df, B=B, level=level, segments=segments,
                                  seed=_child_seed(seed, n, rep, 7), n_jobs=1)
        pt = estimate_harm_share(df, segments=segments, with_sieve_se=False).theta_hat
        return {"cov": float(lo <= th_true <= hi), "len": hi - lo, "pt": pt, "sd": sd}

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(r) for r in range(n_rep))
    cov = np.array([r["cov"] for r in rows])
    return {
        "n": n, "n_rep": n_rep, "B": B,
        "cov95_boot": float(cov.mean()),
        "mean_len": float(np.mean([r["len"] for r in rows])),
        "mean_boot_sd": float(np.mean([r["sd"] for r in rows])),
    }
