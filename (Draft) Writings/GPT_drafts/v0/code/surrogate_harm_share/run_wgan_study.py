"""Second-calibration study: valid inference on the *truthful WGAN population*.

Trains (or loads) the Chen & Ritzwoller 3-WGAN cascade, reads the exact WGAN
population truth, then runs the same sieve plug-in + two-band SE + full-refit
bootstrap inference used in `run_study.py` -- now against genuinely GAN-generated
finite samples (realistic heteroskedastic conditional shapes).

Sizing.  We generate ENOUGH for valid inference but keep runtime bounded:
population N_pop draws for a precise truth; a Monte-Carlo coverage study over a
few sample sizes with several hundred reps each; a rate check; and a bootstrap.

Usage:  PYTHONPATH=src python run_wgan_study.py [--retrain]
Writes: results/logs/wgan_*.json
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from harm_share.calibration import COVARIATES_2D, load_graduation
from harm_share.wgan_calibration import WGANOracle, build_wgan_oracle
from harm_share.simulation import rate_experiment, bootstrap_coverage

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "results" / "logs"
TABS = ROOT / "results" / "tables"
WGAN_DIR = ROOT / "results" / "wgan"
for d in (LOGS, TABS, WGAN_DIR):
    d.mkdir(parents=True, exist_ok=True)
# the inference study uses the 2-covariate variant (a low-dim, sieve-tractable
# decision rule); the faithful 20-covariate calibration lives in wgan_oracle.npz.
NPZ = WGAN_DIR / "wgan_oracle_2d.npz"

# sizing (bounded runtime, enough reps for meaningful coverage)
N_GRID = [2000, 4000, 8000]
N_REP = 300
BOOT_N = 4000
BOOT_REP = 100
BOOT_B = 200
JOBS = 8


def dump(obj, name):
    (LOGS / name).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    print(f"  wrote results/logs/{name}")


def main():
    retrain = "--retrain" in sys.argv
    t0 = time.time()

    if NPZ.exists() and not retrain:
        print(f"[1/5] Loading trained WGAN cascade from {NPZ} ...")
        orc = WGANOracle.load(NPZ, load_graduation())
    else:
        print("[1/5] Training WGAN cascade (GAN1 X, GAN2 S|X,W, GAN3 Y|S,X,W) ...")
        orc = build_wgan_oracle(covariates=COVARIATES_2D, verbose=True)
        orc.save(NPZ)
        print(f"      saved -> {NPZ}")

    print("[2/5] WGAN population truth ...")
    tr = orc.truth()
    dump(tr, "wgan_truth.json")
    print(f"      theta_harm={tr['theta_harm']:.4f}  quad(++,+-,-+,--)="
          f"({tr['theta_pp']:.3f},{tr['theta_harm']:.3f},{tr['theta_mp']:.3f},{tr['theta_mm']:.3f})"
          f"  rho={tr['rho']:.3f}  ATE_S={tr['ate_S']:.2f} ATE_Y={tr['ate_Y']:.2f}")

    print(f"[3/5] Monte-Carlo coverage + rate: n={N_GRID}, n_rep={N_REP} (sieve seg=2) ...")
    rate = rate_experiment(orc, N_GRID, tr, segments=2, n_rep=N_REP, n_jobs=JOBS)
    cells = rate["cells"]
    dump({"cells": cells, "slopes": rate["slopes"]}, "wgan_mc.json")
    pd.DataFrame(cells).to_csv(TABS / "wgan_rate_table.csv", index=False)
    for c in cells:
        print(f"      n={c['n']:5d}: bias={c['bias']:+.4f} rmse={c['rmse']:.4f}"
              f" se_ratio={c['se_ratio']:.2f} cov95={c['cov95_sieve']:.2f}")
    print("      slopes:", {k: round(v, 3) for k, v in rate["slopes"].items()})

    print(f"[4/5] Full-refit bootstrap coverage at n={BOOT_N} "
          f"(n_rep={BOOT_REP}, B={BOOT_B}) ...")
    boot = bootstrap_coverage(orc, BOOT_N, BOOT_REP, tr, segments=2, B=BOOT_B, n_jobs=JOBS)
    dump(boot, "wgan_bootstrap.json")
    print(f"      n={boot['n']}: boot cov95={boot['cov95_boot']:.2f} mean_len={boot['mean_len']:.4f}")

    print(f"[5/5] Done in {time.time()-t0:.0f}s. Logs in results/logs/wgan_*.json")


if __name__ == "__main__":
    main()
