"""Run the full surrogate-induced-harm-share simulation study and save all outputs.

Usage:  PYTHONPATH=src python run_study.py
Writes JSON logs to results/logs and CSV/markdown tables to results/tables.
"""
import os
# single-thread BLAS so joblib threads don't oversubscribe
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from harm_share.calibration import build_oracle, COVARIATES_2D
from harm_share.functionals import mc_truth, grid_truth, analytic_derivative, fd_derivative
from harm_share.simulation import run_mc, rate_experiment, bootstrap_coverage

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "results" / "logs"
TABS = ROOT / "results" / "tables"
LOGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

N_GRID = [1000, 2000, 4000, 8000, 16000]
N_REP = 500
BOOT_N = [2000, 8000]
BOOT_REP = 120
BOOT_B = 200
JOBS = 8


def dump(obj, name):
    (LOGS / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"  wrote results/logs/{name}")


def main():
    t0 = time.time()
    print("[1/7] Building calibrated oracle (SNR~1, noise_scale=0.34) ...")
    orc = build_oracle(covariates=COVARIATES_2D)  # d=2, clean-noise primary DGP

    print("[2/7] Population truth (MC 1.5M + grid quadrature + geometry) ...")
    tr = mc_truth(orc)
    gt = grid_truth(orc, n_grid=500).as_dict()
    truth = {"mc": tr, "grid": gt}
    dump(truth, "truth.json")
    print(f"     theta_harm={tr['theta_harm']:.4f}  quad(++,+-,-+,--)="
          f"({tr['theta_pp']:.3f},{tr['theta_harm']:.3f},{tr['theta_mp']:.3f},{tr['theta_mm']:.3f})"
          f"  rho={tr['rho']:.3f}  treatS={tr['treat_share_S']:.3f}")
    print(f"     geometry: |gradS|_MS={gt['grad_S_on_MS']:.2f} |gradY|_MY={gt['grad_Y_on_MY']:.2f}"
          f"  corner|cos|={gt['corner_cos']:.3f}  len(M_S)={gt['len_M_S']:.2f} len(M_Y)={gt['len_M_Y']:.2f}")

    print("[3/7] Two-boundary derivative verification (analytic vs finite difference) ...")
    zero = lambda X: np.zeros(np.atleast_2d(X).shape[0])
    one = lambda X: np.ones(np.atleast_2d(X).shape[0])
    dSx = lambda X: 1.0 + 0.5 * np.atleast_2d(X)[:, 0]
    dY2 = lambda X: 1.0 - 0.3 * np.atleast_2d(X)[:, 1]
    dcheck = []
    for name, dS, dY in [("tauS_only", one, zero), ("tauY_only", zero, one),
                         ("tauS_varying", dSx, zero), ("both", dSx, dY2)]:
        ad = analytic_derivative(orc, dS, dY, eps=0.18, n_draw=1_200_000)
        fd = fd_derivative(orc, dS, dY, h=0.05, n_draw=1_200_000)
        dcheck.append({"case": name, "D_MS": ad["D_MS"], "D_MY": ad["D_MY"],
                       "analytic": ad["Dtheta"], "finite_diff": fd,
                       "abs_err": abs(ad["Dtheta"] - fd)})
        print(f"     {name:14s} analytic={ad['Dtheta']:+.4f} FD={fd:+.4f} err={abs(ad['Dtheta']-fd):.4f}")
    dump(dcheck, "derivative_check.json")

    print(f"[4/7] Rate experiment (FIXED K, seg=2): n={N_GRID}, n_rep={N_REP} ...")
    rate = rate_experiment(orc, N_GRID, tr, segments=2, n_rep=N_REP, n_jobs=JOBS)
    dump(rate, "rate_experiment.json")
    pd.DataFrame(rate["cells"]).to_csv(TABS / "rate_table.csv", index=False)
    print("     slopes:", {k: round(v, 3) for k, v in rate["slopes"].items()})

    print(f"[4b] Rate experiment (GROWING K undersmoothing schedule) ...")
    # slowly grow the sieve dimension with n (undersmoothing): more data -> more basis.
    SEG_SCHEDULE = {1000: 2, 2000: 2, 4000: 3, 8000: 3, 16000: 4}
    rate_g = rate_experiment(orc, N_GRID, tr, seg_of_n=SEG_SCHEDULE, n_rep=N_REP, n_jobs=JOBS)
    rate_g["schedule"] = SEG_SCHEDULE
    dump(rate_g, "rate_experiment_growK.json")
    print("     growing-K coverage:",
          {c["n"]: (c["segments"], round(c["cov95_sieve"], 2)) for c in rate_g["cells"]})

    print("[5/7] Sieve-dimension (K) sensitivity at n=4000, seg in {1,2,3} ...")
    ksens = [run_mc(orc, 4000, N_REP, tr, segments=s, n_jobs=JOBS)["agg"] for s in (1, 2, 3)]
    dump(ksens, "k_sensitivity.json")
    for c in ksens:
        print(f"     seg={c['segments']}: bias={c['bias']:+.4f} rmse={c['rmse']:.4f}"
              f" se_ratio={c['se_ratio']:.2f} cov={c['cov95_sieve']:.2f}")

    print(f"[6/7] Full-refit bootstrap coverage at n={BOOT_N} (n_rep={BOOT_REP}, B={BOOT_B}) ...")
    boot = [bootstrap_coverage(orc, n, BOOT_REP, tr, segments=2, B=BOOT_B, n_jobs=JOBS)
            for n in BOOT_N]
    dump(boot, "bootstrap_coverage.json")
    for b in boot:
        print(f"     n={b['n']}: boot cov95={b['cov95_boot']:.2f} mean_len={b['mean_len']:.4f}")

    print("[7/7] Robustness: realistic low-SNR noise (noise_scale=1.0) at n in {2000,8000} ...")
    orc_hi = build_oracle(covariates=COVARIATES_2D, noise_scale=1.0)
    tr_hi = mc_truth(orc_hi)  # same surfaces -> same truth (noise doesn't move it)
    realistic = [run_mc(orc_hi, n, 300, tr_hi, segments=2, n_jobs=JOBS)["agg"] for n in (2000, 8000)]
    dump({"truth": tr_hi, "cells": realistic}, "realistic_noise.json")
    for c in realistic:
        print(f"     n={c['n']}: bias={c['bias']:+.4f} rmse={c['rmse']:.4f} cov={c['cov95_sieve']:.2f}")

    print(f"\nDONE in {time.time()-t0:.0f}s. Logs in results/logs, tables in results/tables.")


if __name__ == "__main__":
    main()
