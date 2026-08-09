"""Direction 1: screen-then-sieve at d_x = 50 (sparse DGP).

Pipeline: lasso screening of relevant coordinates (main effects + D-interactions)
-> random features supported on the screened set -> per-arm OLS -> W/V inference.

Variants per (n,):
  dense_K200    : no screening, dense sphere features (baseline)
  screen_full   : screen and estimate on the SAME sample (post-selection caveat)
  screen_split  : screen on a random half, estimate on the other half (honest)
  oracle        : features on the true support {x1, x2} (upper benchmark)

Metrics: support recovery (S_hat contains {0,1}; |S_hat|), h_rmse, W/V bias + coverage.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from rf_sieve_lib import (Z975, make_dgp, generate, compute_truth,
                          draw_feature_map, fit_both_arms, rf_inference,
                          screen_lasso)

REPS = 150
DIM = 50
N_VALUES = (1000, 4000)
K_SCREENED = 50
K_DENSE = 200
GAMMA = 3.0
M_SOBOL = 8192
SEED = 31
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def infer_with(psi, data, X_sobol, tau_sobol):
    fits = fit_both_arms(psi, data)
    if fits is None:
        return None
    return rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol)


def main():
    t0 = time.time()
    dgp = make_dgp("sparse", DIM)
    W_true, V_true = compute_truth(dgp)
    X_sobol = Sobol(d=DIM, scramble=False).random(M_SOBOL)
    tau_sobol = dgp["tau"](X_sobol)
    print(f"W_true={W_true:.4f}  V_true={V_true:.4f}")

    rows = []
    for n in N_VALUES:
        for rep in range(REPS):
            rng = np.random.default_rng(SEED + 7919 * rep + n)
            data = generate(dgp, n, rng)

            # ---- variant: dense features, no screening
            psi = draw_feature_map(DIM, K_DENSE, rng, GAMMA)
            res = infer_with(psi, data, X_sobol, tau_sobol)
            if res is not None:
                rows.append({"variant": "dense_K200", "n": n, "rep": rep,
                             "S_size": DIM, "S_hit": 1.0, **res})

            # ---- screening on the full sample
            S_full = screen_lasso(data["X"], data["D"], data["Y"], seed=rep)
            psi = draw_feature_map(DIM, K_SCREENED, rng, GAMMA, support=S_full)
            res = infer_with(psi, data, X_sobol, tau_sobol)
            if res is not None:
                rows.append({"variant": "screen_full", "n": n, "rep": rep,
                             "S_size": len(S_full),
                             "S_hit": float({0, 1} <= set(S_full.tolist())), **res})

            # ---- honest split: screen on half, estimate on the other half
            perm = rng.permutation(n)
            i_scr, i_est = perm[: n // 2], perm[n // 2:]
            S_half = screen_lasso(data["X"][i_scr], data["D"][i_scr],
                                  data["Y"][i_scr], seed=rep)
            sub = {k: v[i_est] for k, v in data.items()}
            psi = draw_feature_map(DIM, K_SCREENED, rng, GAMMA, support=S_half)
            res = infer_with(psi, sub, X_sobol, tau_sobol)
            if res is not None:
                rows.append({"variant": "screen_split", "n": n, "rep": rep,
                             "S_size": len(S_half),
                             "S_hit": float({0, 1} <= set(S_half.tolist())), **res})

            # ---- oracle support
            psi = draw_feature_map(DIM, K_SCREENED, rng, GAMMA, support=[0, 1])
            res = infer_with(psi, data, X_sobol, tau_sobol)
            if res is not None:
                rows.append({"variant": "oracle", "n": n, "rep": rep,
                             "S_size": 2, "S_hit": 1.0, **res})

            if (rep + 1) % 25 == 0:
                print(f"  n={n}: rep {rep + 1}/{REPS} ({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows).drop(columns=["h_hat"])
    for f in ("W", "V"):
        draws[f"{f}_dev"] = draws[f"{f}_hat"] - (W_true if f == "W" else V_true)
        draws[f"{f}_cover"] = (np.abs(draws[f"{f}_dev"]) <= Z975 * draws[f"{f}_se"]).astype(float)

    g = draws.groupby(["variant", "n"], sort=False)
    summary = pd.DataFrame({
        "S_hit": g["S_hit"].mean(), "S_size": g["S_size"].mean(),
        "h_rmse": g["h_rmse"].mean(),
        "V_bias": g["V_dev"].mean(), "V_sd": g["V_hat"].std(),
        "V_se": g["V_se"].mean(), "V_cover": g["V_cover"].mean(),
        "W_bias": g["W_dev"].mean(), "W_se": g["W_se"].mean(),
        "W_cover": g["W_cover"].mean(), "draws": g.size(),
    }).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / f"D1_screening_summary_rep{REPS}.csv", index=False)
    draws.to_csv(OUTPUT_DIR / f"D1_screening_draws_rep{REPS}.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
