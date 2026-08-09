"""Resampling inference: participant-level full-refit bootstrap, multiplier bootstrap
from the estimated influence score, and m-out-of-n subsampling (task section 11.4).

A naive percentile bootstrap is NOT presented as theoretically valid for the
hard-threshold optimal-path target; it is reported alongside the sieve-Riesz interval
and the subsampling interval, and its behaviour is checked in the Monte Carlo study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from .crossfit import child_seed


@dataclass
class BootstrapResult:
    method: str
    point: float
    se: float
    ci: tuple[float, float]
    n_rep: int
    reps: np.ndarray
    note: str = ""


def _fit_once(df, config, seed, path_label="11"):
    from .estimator import TwoStagePathWelfareEstimator

    cfg = config.model_copy(deep=True)
    cfg.crossfit.seed = seed
    est = TwoStagePathWelfareEstimator(cfg).fit(df)
    return est.point_[path_label]


def participant_bootstrap(
    est, *, n_rep=499, seed=909090, path_label="11", n_jobs=1, alpha=0.05
) -> BootstrapResult:
    """Full-refit bootstrap resampling independent units (participants) with replacement."""
    df = est.data_
    config = est.config
    if "group" in df.columns:
        units = df["group"].to_numpy()
        uniq = np.unique(units)
        by_unit = {u: np.where(units == u)[0] for u in uniq}
    else:
        uniq = np.arange(len(df))
        by_unit = None

    def one(b):
        rng = np.random.default_rng(child_seed(seed, b))
        drawn = rng.choice(uniq, size=uniq.size, replace=True)
        if by_unit is None:
            idx = drawn
        else:
            idx = np.concatenate([by_unit[u] for u in drawn])
        boot = df.iloc[idx].reset_index(drop=True)
        # relabel groups so resampled copies are distinct clusters
        if "group" in boot.columns:
            boot = boot.copy()
            boot["group"] = np.repeat(np.arange(uniq.size),
                                      [len(by_unit[u]) for u in drawn]) if by_unit else np.arange(len(boot))
        try:
            return _fit_once(boot, config, child_seed(seed, b, 7), path_label)
        except Exception:
            return np.nan

    reps = np.array(Parallel(n_jobs=n_jobs, prefer='threads')(delayed(one)(b) for b in range(n_rep)))
    reps = reps[~np.isnan(reps)]
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        "participant full-refit bootstrap (percentile)",
        point=est.point_[path_label], se=float(np.std(reps, ddof=1)),
        ci=(float(lo), float(hi)), n_rep=len(reps), reps=reps,
        note="percentile CI; not guaranteed valid for the hard-threshold target -- see MC study",
    )


def multiplier_bootstrap(
    est, *, K=None, n_rep=999, seed=717171, alpha=0.05, kind="gaussian",
    influence: np.ndarray | None = None, point: float | None = None,
) -> BootstrapResult:
    """Multiplier bootstrap using the sieve-Riesz influence score psi_i.

    V11* + (1/n) sum_i xi_i psi_i, xi_i ~ N(0,1) (or Rademacher), giving a distribution
    for the studentized statistic conditional on the estimated densities.
    """
    if influence is None or point is None:
        from .riesz import sieve_riesz_inference

        res = sieve_riesz_inference(est, K=K)
        # recompute psi via a light call: store on the result’s diagnostics if available
        influence = res.diagnostics.get("influence")
        point = res.estimate
        if influence is None:
            raise ValueError("influence score not available; pass influence=... explicitly")
    influence = np.asarray(influence, dtype=float)
    n = influence.size
    rng = np.random.default_rng(seed)
    reps = np.empty(n_rep)
    for b in range(n_rep):
        if kind == "rademacher":
            xi = rng.choice([-1.0, 1.0], size=n)
        else:
            xi = rng.standard_normal(n)
        reps[b] = point + np.mean(xi * influence)
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        f"multiplier bootstrap ({kind})", point=point,
        se=float(np.std(reps, ddof=1)), ci=(float(lo), float(hi)),
        n_rep=n_rep, reps=reps,
        note="conditional on estimated densities m, p_a",
    )


def subsample_mn(
    est, *, m_values=None, n_rep=200, seed=515151, path_label="11", alpha=0.05, n_jobs=1
) -> dict[int, BootstrapResult]:
    """m-out-of-n subsampling (without replacement) for several subsample sizes m."""
    df = est.data_
    config = est.config
    n = len(df)
    if m_values is None:
        m_values = [int(0.3 * n), int(0.5 * n), int(0.7 * n)]
    theta_hat = est.point_[path_label]
    out: dict[int, BootstrapResult] = {}
    for m in m_values:
        def one(b, m=m):
            rng = np.random.default_rng(child_seed(seed, m, b))
            idx = rng.choice(n, size=m, replace=False)
            sub = df.iloc[idx].reset_index(drop=True)
            try:
                return _fit_once(sub, config, child_seed(seed, m, b, 3), path_label)
            except Exception:
                return np.nan
        reps = np.array(Parallel(n_jobs=n_jobs, prefer='threads')(delayed(one)(b) for b in range(n_rep)))
        reps = reps[~np.isnan(reps)]
        # subsampling CI: center on theta_hat, scale root-m statistic to root-n
        root = np.sqrt(m) * (reps - theta_hat)
        qlo, qhi = np.percentile(root, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci = (theta_hat - qhi / np.sqrt(n), theta_hat - qlo / np.sqrt(n))
        out[m] = BootstrapResult(
            f"m-out-of-n subsampling (m={m})", point=theta_hat,
            se=float(np.std(reps, ddof=1)), ci=(float(ci[0]), float(ci[1])),
            n_rep=len(reps), reps=reps,
            note="root-m rescaling; robust to irregular limit distributions",
        )
    return out


def run_bootstrap(est, *, method="participant", **kwargs) -> BootstrapResult | dict:
    if method == "participant":
        return participant_bootstrap(est, **kwargs)
    if method == "multiplier":
        return multiplier_bootstrap(est, **kwargs)
    if method == "subsample":
        return subsample_mn(est, **kwargs)
    raise ValueError(f"unknown bootstrap method {method}")
