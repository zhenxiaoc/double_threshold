"""Sweep S7 (Round 4): the extreme-share anomaly (A2 from S1).

When V_true is near 1, first-stage noise flips near-boundary CATE signs
asymmetrically (there is almost no negative-CATE mass to flip back), creating a
one-sided bias the band SE cannot see. Relevant to JTPA (share ~ 0.89).

Questions: (i) at what share level does the bias bind? (ii) does it vanish with
n (it is second-order in the first-stage error)? (iii) does LOO debiasing of V
fix it (the flip bias is exactly a diagonal quadratic term)?

Grid: dense DGP d_x = 50, shift in {-0.55 (share ~0.85), -0.40 (share ~0.95)},
n in {4000, 16000}; V_plug and V_loo, both with the plug-in band SE.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from rf_sieve_lib import (Z975, make_dgp, generate, compute_truth,
                          draw_feature_map, fit_both_arms, rf_inference,
                          F_V, loo_debias)

REPS = 150
DIM, K, GAMMA = 50, 200, 1.5
M_SOBOL = 8192
SEED = 139
SHIFTS = (-0.55, -0.40)
N_VALUES = (4000, 16000)
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def main():
    t0 = time.time()
    rows = []
    X_sobol = Sobol(d=DIM, scramble=False).random(M_SOBOL)
    for shift in SHIFTS:
        dgp = make_dgp("dense", DIM, shift=shift)
        W_true, V_true = compute_truth(dgp)
        tau_sobol = dgp["tau"](X_sobol)
        print(f"shift={shift}: V_true={V_true:.4f}")
        for n in N_VALUES:
            for rep in range(REPS):
                rng = np.random.default_rng(SEED + 22801763 * rep + int(100 * shift) + n)
                data = generate(dgp, n, rng)
                psi = draw_feature_map(DIM, K, rng, GAMMA)
                fits = fit_both_arms(psi, data)
                if fits is None:
                    continue
                res = rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol)
                res.pop("h_hat")
                res["V_loo"] = loo_debias(F_V, fits[0], fits[1], psi, X_sobol,
                                          res["V_hat"], n_total=n,
                                          max_per_arm=2000, rng=rng)
                rows.append({"shift": shift, "n": n, "rep": rep,
                             "V_true": V_true, **res})
            print(f"  shift={shift} n={n} done ({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows)
    for est in ("V_hat", "V_loo"):
        draws[f"{est}_dev"] = draws[est] - draws["V_true"]
        draws[f"{est}_cover"] = (np.abs(draws[f"{est}_dev"]) <= Z975 * draws["V_se"]).astype(float)
    g = draws.groupby(["shift", "n"])
    summary = pd.DataFrame({
        "V_true": g["V_true"].first(), "h_rmse": g["h_rmse"].mean(),
        "Vplug_bias": g["V_hat_dev"].mean(), "Vplug_cover": g["V_hat_cover"].mean(),
        "Vloo_bias": g["V_loo_dev"].mean(), "Vloo_cover": g["V_loo_cover"].mean(),
        "V_se": g["V_se"].mean(), "n_band": g["n_band"].mean(),
        "draws": g.size(),
    }).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / f"S7_extreme_share_summary_rep{REPS}.csv", index=False)
    draws.to_csv(OUTPUT_DIR / f"S7_extreme_share_draws_rep{REPS}.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
