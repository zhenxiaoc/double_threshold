"""Direction 2: cylinder-boundary / effective-dimension check.

If the CATE depends only on s = 2 coordinates, the boundary {tau = 0} is a
cylinder over a 1-dimensional curve, and the conjecture is that estimation and
inference of V are governed by the EFFECTIVE dimension s, invariant to the
ambient dimension d_x. Testable implications, with oracle-support features
(so the first stage is genuinely s-dimensional):

  (i)  RMSE(V_hat), SEs, and coverage should be ~identical across
       d_x in {10, 50, 100} at each n;
  (ii) the convergence rate exponent (log-RMSE vs log-n slope) should be
       fast (near -1/2, since tau is analytic => sigma >> s);
  (iii) n * Var_hat(V) should be flat in d_x.

Grid: sparse DGP, oracle features (support {0,1}), K = 50, gamma = 3,
d_x in {10, 50, 100} x n in {1000, 4000, 16000}.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from rf_sieve_lib import (Z975, make_dgp, generate, compute_truth,
                          draw_feature_map, fit_both_arms, rf_inference)

REPS = 150
DIMS = (10, 50, 100)
N_VALUES = (1000, 4000, 16000)
K = 50
GAMMA = 3.0
M_SOBOL = 8192
SEED = 47
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def main():
    t0 = time.time()
    rows = []
    for dim in DIMS:
        dgp = make_dgp("sparse", dim)
        W_true, V_true = compute_truth(dgp)
        X_sobol = Sobol(d=dim, scramble=False).random(M_SOBOL)
        tau_sobol = dgp["tau"](X_sobol)
        for n in N_VALUES:
            for rep in range(REPS):
                rng = np.random.default_rng(SEED + 104729 * rep + 31 * dim + n)
                data = generate(dgp, n, rng)
                psi = draw_feature_map(dim, K, rng, GAMMA, support=[0, 1])
                fits = fit_both_arms(psi, data)
                if fits is None:
                    continue
                res = rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol)
                res.pop("h_hat")
                rows.append({"dim": dim, "n": n, "rep": rep,
                             "W_true": W_true, "V_true": V_true, **res})
            print(f"  dim={dim} n={n} done ({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows)
    for f in ("W", "V"):
        draws[f"{f}_dev"] = draws[f"{f}_hat"] - draws[f"{f}_true"]
        draws[f"{f}_cover"] = (np.abs(draws[f"{f}_dev"]) <= Z975 * draws[f"{f}_se"]).astype(float)
    draws["V_sqerr"] = draws["V_dev"] ** 2
    draws["n_varV"] = draws["n"] * draws["V_se"] ** 2

    g = draws.groupby(["dim", "n"])
    summary = pd.DataFrame({
        "h_rmse": g["h_rmse"].mean(),
        "V_rmse": np.sqrt(g["V_sqerr"].mean()),
        "V_bias": g["V_dev"].mean(), "V_se": g["V_se"].mean(),
        "V_cover": g["V_cover"].mean(), "n_varV": g["n_varV"].mean(),
        "W_bias": g["W_dev"].mean(), "W_cover": g["W_cover"].mean(),
        "draws": g.size(),
    }).reset_index()

    # empirical convergence-rate exponent per dimension: slope of log RMSE on log n
    print("\nEmpirical rate exponents (log V_rmse ~ log n):")
    slopes = []
    for dim in DIMS:
        sub = summary[summary["dim"] == dim]
        slope = np.polyfit(np.log(sub["n"]), np.log(sub["V_rmse"]), 1)[0]
        slopes.append({"dim": dim, "rate_exponent": slope})
        print(f"  d_x = {dim:4d}: {slope:+.3f}   (parametric benchmark: -0.500)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / f"D2_effective_dim_summary_rep{REPS}.csv", index=False)
    pd.DataFrame(slopes).to_csv(OUTPUT_DIR / f"D2_effective_dim_slopes_rep{REPS}.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
