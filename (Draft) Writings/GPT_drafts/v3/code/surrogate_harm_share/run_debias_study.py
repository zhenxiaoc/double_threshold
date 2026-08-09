"""Monte Carlo study of quadratic (SS) debiasing and the doubly debiased (DD)
estimator for the two-threshold harm share, on CR-style calibrated DGPs.

Estimator menu (all studentized by the two-band sieve-Riesz SE):
  sieve first stage : plugin | ss_margins (margins-only quad) | ss (corner-aware)
  gbr   first stage : cf (2-fold cross-fit plug-in) | cf_riesz (+ projected
                      Riesz first-order corr) | dd (+ SS quadratic corr)

DGPs (calibrated to the Banerjee et al. graduation data, Chen & Ritzwoller
2023 style):
  krr    -- primary kernel-ridge exact-truth oracle (d=2)
  wgan2d -- CR 3-WGAN cascade oracle, 2-covariate cache (both margins bind)
  affine -- Gaussian/affine DGP with EXACT orthant truth (validation)

Usage:  python run_debias_study.py [--quick] [--wgan-scalar]

--wgan-scalar: run ONLY the scalar-cascade 2d WGAN DGP (cache
wgan_oracle_2d_scalar.npz, both margins bind, rough ReLU truth) and append its
cells to the same outputs under dgp="wgan2ds".  NOTE: the plain "wgan2d" DGP
(cache wgan_oracle_2d.npz, full-surrogate design) is DEGENERATE -- true
theta=0, treat share 1, no binding margins -- and serves as a boundary-case
arm, not the rough-truth arm.
"""
from __future__ import annotations

# pin BLAS/OpenMP threads before numpy/sklearn import (joblib runs JOBS
# worker threads; each fit would otherwise spawn its own BLAS pool)
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent / "src"))

from harm_share.calibration import build_oracle                  # noqa: E402
from harm_share.functionals import mc_truth                      # noqa: E402
from harm_share.affine_dgp import AffineDGP                      # noqa: E402
from harm_share.quadratic import ss_debiased_estimate, dd_estimate  # noqa: E402

QUICK = "--quick" in sys.argv
N_GRID = [1000, 2000, 4000, 8000] if not QUICK else [1000, 2000]
N_REP = 500 if not QUICK else 24
JOBS = 12
SEED = 20260719
DELTA_SIEVE = 0.08
DELTA_ML = 0.10
SEGMENTS = 2

RESULTS = Path(__file__).parent / "results"
LOGS = RESULTS / "logs"
TABLES = RESULTS / "tables"


def _child_seed(seed, *ix):
    return int(np.random.SeedSequence([seed, *ix]).generate_state(1)[0])


def one_rep(oracle, n, rep, th_true):
    rng = np.random.default_rng(_child_seed(SEED, n, rep))
    df = oracle.sample_experiment(n, rng)
    rows = []

    ss = ss_debiased_estimate(df, segments=SEGMENTS, delta=DELTA_SIEVE,
                              seed=_child_seed(SEED, n, rep, 1))
    se = ss.se_sieve
    for name, th, ci in (("plugin", ss.theta_plugin, ss.ci_plugin),
                         ("ss_margins", ss.theta_ss_margins, ss.ci_ss_margins),
                         ("ss", ss.theta_ss, ss.ci_ss)):
        rows.append({"est": name, "theta": th, "se": se,
                     "cov": float(ci[0] <= th_true <= ci[1]),
                     "len": ci[1] - ci[0]})
    rows[-1].update({"quad_full": ss.quad_full, "quad_S": ss.quad_S,
                     "quad_Y": ss.quad_Y, "corner": ss.corner})

    dd = dd_estimate(df, nuisance="gbr", segments=SEGMENTS, delta=DELTA_ML,
                     seed=_child_seed(SEED, n, rep, 2))
    for name, th, ci in (("cf_gbr", dd.theta_cf, dd.ci_cf),
                         ("cf_riesz_gbr", dd.theta_cf_riesz, dd.ci_cf_riesz),
                         ("dd_gbr", dd.theta_dd, dd.ci_dd)):
        rows.append({"est": name, "theta": th, "se": dd.se_sieve,
                     "cov": float(ci[0] <= th_true <= ci[1]),
                     "len": ci[1] - ci[0]})
    rows[-1].update({"quad_full": dd.quad_full, "corner": dd.corner,
                     "corr_riesz": dd.correction_riesz})
    return rows


def run_cell(oracle, n, th_true):
    t0 = time.time()
    reps = Parallel(n_jobs=JOBS, prefer="threads")(
        delayed(one_rep)(oracle, n, r, th_true) for r in range(N_REP))
    flat = [row | {"rep": i} for i, rows in enumerate(reps) for row in rows]
    dfres = pd.DataFrame(flat)
    aggs = []
    for est, g in dfres.groupby("est"):
        th = g["theta"].to_numpy()
        se_arr = g["se"].to_numpy()
        agg = {
            "est": est, "n": n, "n_rep": N_REP, "theta_true": th_true,
            "bias": float(th.mean() - th_true),
            "mc_sd": float(th.std(ddof=1)),
            "rmse": float(np.sqrt(np.mean((th - th_true) ** 2))),
            "mean_se": float(np.nanmean(se_arr)),
            "se_ratio": float(np.nanmean(se_arr) / th.std(ddof=1)) if th.std() > 0 else np.nan,
            "cov95": float(g["cov"].mean()),
            "mean_len": float(g["len"].mean()),
        }
        for k in ("quad_full", "quad_S", "quad_Y", "corner", "corr_riesz"):
            if k in g and g[k].notna().any():
                agg[f"{k}_mean"] = float(g[k].mean())
                agg[f"{k}_sd"] = float(g[k].std(ddof=1))
        aggs.append(agg)
    print(f"    n={n}: {time.time()-t0:.0f}s")
    for a in sorted(aggs, key=lambda a: a["est"]):
        print(f"      {a['est']:>14} bias={a['bias']:+.4f} sd={a['mc_sd']:.4f} "
              f"rmse={a['rmse']:.4f} se_ratio={a['se_ratio']:.2f} cov={a['cov95']:.3f}")
    return aggs


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    dgps = {}

    if "--wgan-scalar" in sys.argv:
        from harm_share.wgan_calibration import WGANOracle
        oracle_ws = WGANOracle.load(RESULTS / "wgan" / "wgan_oracle_2d_scalar.npz")
        truth_ws = oracle_ws.truth()
        dgps["wgan2ds"] = (oracle_ws, truth_ws["theta_harm"])
        print(f"wgan2ds truth: {truth_ws['theta_harm']:.4f}")
    else:
        oracle_krr = build_oracle()
        truth_krr = mc_truth(oracle_krr, n_draw=1_500_000, seed=7)
        dgps["krr"] = (oracle_krr, truth_krr["theta_harm"])
        print(f"krr truth: {truth_krr['theta_harm']:.4f}")

        wgan_path = RESULTS / "wgan" / "wgan_oracle_2d.npz"
        if wgan_path.exists():
            from harm_share.wgan_calibration import WGANOracle
            oracle_w = WGANOracle.load(wgan_path)
            truth_w = oracle_w.truth()
            dgps["wgan2d"] = (oracle_w, truth_w["theta_harm"])
            print(f"wgan2d truth (degenerate boundary case): "
                  f"{truth_w['theta_harm']:.4f}")
        else:
            print("wgan2d cache missing -- skipped")

        aff = AffineDGP()
        truth_aff = aff.exact_truth()
        dgps["affine"] = (aff, truth_aff["theta_harm"])
        print(f"affine truth: {truth_aff['theta_harm']:.4f}")

    # the scalar arm writes to ITS OWN files so a concurrently running main
    # study cannot clobber (and be clobbered by) its checkpoints
    suffix = "_wgan2ds" if "--wgan-scalar" in sys.argv else ""
    out_json = LOGS / f"debias_study{suffix}.json"
    out_csv = TABLES / f"debias_study{suffix}.csv"
    all_cells = []
    for dname, (oracle, th_true) in dgps.items():
        print(f"[{dname}]")
        for n in N_GRID:
            cells = run_cell(oracle, n, th_true)
            for c in cells:
                c["dgp"] = dname
            all_cells.extend(cells)
            # checkpoint after every cell
            out_json.write_text(json.dumps(all_cells, indent=1))
            pd.DataFrame(all_cells).to_csv(out_csv, index=False)

    print("done ->", out_csv)


if __name__ == "__main__":
    main()
