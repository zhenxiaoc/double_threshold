"""Sweep S4: known violations -- do they fail gracefully (visibly/conservatively)?

  F1 'cubic'        : tau = 3(x1-0.5)^3 -- gradient vanishes on the entire
                      boundary (regular-level-set Assumption 2(c) fails).
  F2 'screen_miss'  : sparse DGP but the sieve support drops a truly relevant
                      coordinate (x2) -- screening false negative.
  F3 'overlap6'     : extreme overlap violation (propensity in ~(0.001, 0.999)).

Report bias, SD, SE, coverage, and the SE/SD ratio: graceful = wide intervals
and/or visible SE blow-up; dangerous = tight intervals around a biased point.
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
N = 4000
M_SOBOL = 8192
SEED = 109
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

CELLS = [
    # label, dgp kwargs, dim, K, feature support
    ("F1_cubic", dict(kind="cubic", dim=10), 10, 100, None),
    ("F2_screen_miss", dict(kind="sparse", dim=50), 50, 50, [0]),
    ("F2_control_S01", dict(kind="sparse", dim=50), 50, 50, [0, 1]),
    ("F3_overlap6", dict(kind="dense", dim=50, overlap=6.0), 50, 200, None),
]


def main():
    t0 = time.time()
    rows = []
    for label, dkw, dim, K, support in CELLS:
        dgp = make_dgp(**dkw)
        W_true, V_true = compute_truth(dgp)
        X_sobol = Sobol(d=dim, scramble=False).random(M_SOBOL)
        tau_sobol = dgp["tau"](X_sobol)
        for rep in range(REPS):
            rng = np.random.default_rng(SEED + 4256233 * rep + abs(hash(label)) % 65521)
            data = generate(dgp, N, rng)
            psi = draw_feature_map(dim, K, rng, 3.0, support=support)
            fits = fit_both_arms(psi, data)
            if fits is None:
                continue
            res = rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol)
            res.pop("h_hat")
            rows.append({"cell": label, "rep": rep,
                         "W_true": W_true, "V_true": V_true, **res})
        print(f"  {label} done ({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows)
    for est, se in (("V_hat", "V_se"), ("W_hat", "W_se")):
        truth = draws["V_true"] if est.startswith("V") else draws["W_true"]
        draws[f"{est}_dev"] = draws[est] - truth
        draws[f"{est}_cover"] = (np.abs(draws[f"{est}_dev"]) <= Z975 * draws[se]).astype(float)

    g = draws.groupby("cell", sort=False)
    summary = pd.DataFrame({
        "V_true": g["V_true"].first(), "h_rmse": g["h_rmse"].mean(),
        "V_bias": g["V_hat_dev"].mean(), "V_sd": g["V_hat"].std(),
        "V_se": g["V_se"].mean(),
        "V_se_over_sd": g["V_se"].mean() / g["V_hat"].std(),
        "V_cover": g["V_hat_cover"].mean(),
        "n_band": g["n_band"].mean(),
        "W_bias": g["W_hat_dev"].mean(), "W_cover": g["W_hat_cover"].mean(),
        "draws": g.size(),
    }).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / f"S4_failure_summary_rep{REPS}.csv", index=False)
    draws.to_csv(OUTPUT_DIR / f"S4_failure_draws_rep{REPS}.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
