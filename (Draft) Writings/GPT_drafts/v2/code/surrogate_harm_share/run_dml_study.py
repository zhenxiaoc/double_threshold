"""Sieve-DML inference study for the DOUBLE-threshold harm share.

Experiments with three first-stage nuisances -- a B-spline **sieve**, a
**random-feature** ridge (RF), and **gradient boosting** (GBR, an XGBoost
equivalent) -- combined with the paper's TWO-BAND sieve-Riesz variance (one
Riesz band per decision boundary M_S, M_Y).  It asks: can we obtain ~95%
coverage of theta = Pr(tau_S>=0, tau_Y<0) with ML nuisances?

Two DGPs:
  * the primary KRR oracle (2 covariates, EXACT closed-form truth) -- the clean
    validation where the sieve tensor basis is tractable;
  * the faithful full WGAN DGP (26-column model covariates) -- where the tensor
    sieve is infeasible and the nuisance MUST be ML (RF / GBR).

Usage:  PYTHONPATH=src python run_dml_study.py
Writes: results/logs/dml_coverage.json, results/tables/dml_coverage.csv
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from harm_share.calibration import build_oracle, load_graduation
from harm_share.functionals import mc_truth
from harm_share.wgan_calibration import WGANOracle
from harm_share.sieve_dml import harm_share_riesz_dml

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "results" / "logs"
TABS = ROOT / "results" / "tables"
for d in (LOGS, TABS):
    d.mkdir(parents=True, exist_ok=True)

N, REPS, JOBS = 4000, 100, 8


def coverage(orc, theta, nuisance, n=N, reps=REPS, **kw):
    nf = max(200, 12 * orc.d)          # random-feature count scales with dimension
    def one(rep):
        df = orc.sample_experiment(n, np.random.default_rng(7000 + rep))
        r = harm_share_riesz_dml(df, nuisance=nuisance, n_features=nf, delta=0.10, seed=rep, **kw)
        return r.theta_dml, r.se, float(r.ci[0] <= theta <= r.ci[1])
    out = Parallel(n_jobs=JOBS, prefer="threads")(delayed(one)(r) for r in range(reps))
    est = np.array([o[0] for o in out]); se = np.array([o[1] for o in out])
    cov = np.array([o[2] for o in out])
    return {"nuisance": nuisance, "n": n, "reps": reps, "theta": float(theta),
            "bias": float(est.mean() - theta), "mc_sd": float(est.std(ddof=1)),
            "mean_se": float(se.mean()), "cover95": float(cov.mean())}


def main():
    rows = []
    t0 = time.time()

    print("[1/2] Primary KRR oracle (EXACT truth, d=2): sieve / rf / gbr ...")
    krr = build_oracle()
    theta_krr = mc_truth(krr)["theta_harm"]
    print(f"      truth theta = {theta_krr:.4f}")
    for nu in ("sieve", "rf", "gbr"):
        r = coverage(krr, theta_krr, nu); r["dgp"] = "KRR (d=2, exact)"; rows.append(r)
        print(f"      {nu:5s}: bias={r['bias']:+.4f} MC-SD={r['mc_sd']:.4f} "
              f"mean-SE={r['mean_se']:.4f} cover95={r['cover95']:.3f}")

    print("[2/2] Faithful WGAN DGP (d=26): rf / gbr (sieve tensor infeasible) ...")
    wgan = WGANOracle.load(ROOT / "results" / "wgan" / "wgan_oracle.npz", load_graduation())
    theta_w = wgan.truth()["theta_harm"]
    print(f"      truth theta = {theta_w:.4f}  (d_model={wgan.d})")
    for nu in ("rf", "gbr"):
        r = coverage(wgan, theta_w, nu); r["dgp"] = f"WGAN (d={wgan.d}, faithful)"; rows.append(r)
        print(f"      {nu:5s}: bias={r['bias']:+.4f} MC-SD={r['mc_sd']:.4f} "
              f"mean-SE={r['mean_se']:.4f} cover95={r['cover95']:.3f}")

    (LOGS / "dml_coverage.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(TABS / "dml_coverage.csv", index=False)
    print(f"\nDONE in {time.time()-t0:.0f}s. Wrote results/logs/dml_coverage.json")


if __name__ == "__main__":
    main()
