"""Sweep S7b (Round 4 close-out): augmented SE for LOO-debiased estimators.

S7 showed LOO-V removes the extreme-share bias but undercovers with the plug-in
SE (the correction term's own noise is first-order in finite samples) -- the
same mechanism as DML anomaly A1, fixed in S6B by SE augmentation. Here:

  se_aug^2 = se_plug^2 + var_hat(correction term),

with var_hat from the independence heuristic in rf_sieve_lib.loo_debias.

Cells: (i) extreme-share dense DGP (shift -0.55, -0.40), d50, n = 4000:
LOO-V with plug-in vs augmented SE; (ii) baseline dense d50 n4000: LOO-W with
plug-in vs augmented SE (its mild 0.85-0.92 undercoverage should also close).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from rf_sieve_lib import (Z975, make_dgp, generate, compute_truth,
                          draw_feature_map, fit_both_arms, rf_inference,
                          F_V, F_W, loo_debias)

REPS = 150
DIM, N, K, GAMMA = 50, 4000, 200, 1.5
M_SOBOL = 8192
SEED = 149
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

CELLS = [  # label, shift, functional
    ("V_share81", -0.55, "V"),
    ("V_share95", -0.40, "V"),
    ("W_baseline", -0.70, "W"),
]


def main():
    t0 = time.time()
    rows = []
    X_sobol = Sobol(d=DIM, scramble=False).random(M_SOBOL)
    for label, shift, func in CELLS:
        dgp = make_dgp("dense", DIM, shift=shift)
        W_true, V_true = compute_truth(dgp)
        tau_sobol = dgp["tau"](X_sobol)
        truth = V_true if func == "V" else W_true
        F = F_V if func == "V" else F_W
        for rep in range(REPS):
            rng = np.random.default_rng(SEED + 49979687 * rep + int(100 * shift))
            data = generate(dgp, N, rng)
            psi = draw_feature_map(DIM, K, rng, GAMMA)
            fits = fit_both_arms(psi, data)
            if fits is None:
                continue
            res = rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol)
            plug = res["V_hat"] if func == "V" else res["W_hat"]
            se_plug = res["V_se"] if func == "V" else res["W_se"]
            est, corr_var = loo_debias(F, fits[0], fits[1], psi, X_sobol,
                                       plug, n_total=N, rng=rng, return_var=True)
            se_aug = float(np.sqrt(se_plug**2 + corr_var))
            rows.append({"cell": label, "rep": rep, "truth": truth,
                         "plug": plug, "loo": est,
                         "se_plug": se_plug, "se_aug": se_aug,
                         "h_rmse": res["h_rmse"]})
        print(f"  {label} done ({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows)
    draws["plug_dev"] = draws["plug"] - draws["truth"]
    draws["loo_dev"] = draws["loo"] - draws["truth"]
    draws["plug_cover"] = (np.abs(draws["plug_dev"]) <= Z975 * draws["se_plug"]).astype(float)
    draws["loo_cover_plugse"] = (np.abs(draws["loo_dev"]) <= Z975 * draws["se_plug"]).astype(float)
    draws["loo_cover_augse"] = (np.abs(draws["loo_dev"]) <= Z975 * draws["se_aug"]).astype(float)

    g = draws.groupby("cell", sort=False)
    summary = pd.DataFrame({
        "truth": g["truth"].first(), "h_rmse": g["h_rmse"].mean(),
        "plug_bias": g["plug_dev"].mean(), "plug_cover": g["plug_cover"].mean(),
        "loo_bias": g["loo_dev"].mean(), "loo_sd": g["loo"].std(),
        "se_plug": g["se_plug"].mean(), "se_aug": g["se_aug"].mean(),
        "loo_cover_plugse": g["loo_cover_plugse"].mean(),
        "loo_cover_augse": g["loo_cover_augse"].mean(),
        "draws": g.size(),
    }).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / f"S7b_aug_se_summary_rep{REPS}.csv", index=False)
    draws.to_csv(OUTPUT_DIR / f"S7b_aug_se_draws_rep{REPS}.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
