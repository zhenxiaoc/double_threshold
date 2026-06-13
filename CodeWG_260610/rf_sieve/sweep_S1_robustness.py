"""Sweep S1: robustness of V inference and LOO-W (one-factor-at-a-time).

Baseline: dense DGP, d_x=50, n=4000, K=200, gamma=1.5, cos, homoskedastic,
share ~0.58, normal overlap. Deviations: relu activation; heteroskedastic
errors; high share (~0.85); weak overlap; d_x=10; sparse DGP; tanh activation.
All cells report plug-in V (band SE, iota=0.01) and plug-in/LOO W.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from rf_sieve_lib import (Z975, make_dgp, generate, compute_truth,
                          draw_feature_map, fit_both_arms, rf_inference,
                          F_W, loo_debias)

REPS = 150
N = 4000
K = 200
M_SOBOL = 8192
SEED = 83
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

BASE = dict(kind="dense", dim=50, shift=-0.70, overlap=1.0,
            hetero=False, activation="cos", gamma=1.5)
CELLS = [
    ("baseline", {}),
    ("relu", {"activation": "relu", "gamma": 3.0}),
    ("tanh", {"activation": "tanh", "gamma": 3.0}),
    ("hetero", {"hetero": True}),
    ("share85", {"shift": -0.40}),
    ("weak_overlap", {"overlap": 3.0}),
    ("d10", {"dim": 10}),
    ("sparse_dgp", {"kind": "sparse"}),
]


def main():
    t0 = time.time()
    rows = []
    for label, dev in CELLS:
        cfg = {**BASE, **dev}
        dgp = make_dgp(cfg["kind"], cfg["dim"], shift=cfg["shift"], overlap=cfg["overlap"])
        W_true, V_true = compute_truth(dgp)
        X_sobol = Sobol(d=cfg["dim"], scramble=False).random(M_SOBOL)
        tau_sobol = dgp["tau"](X_sobol)
        for rep in range(REPS):
            rng = np.random.default_rng(SEED + 6700417 * rep + abs(hash(label)) % 65521)
            data = generate(dgp, N, rng, hetero=cfg["hetero"])
            psi = draw_feature_map(cfg["dim"], K, rng, cfg["gamma"],
                                   activation=cfg["activation"])
            fits = fit_both_arms(psi, data)
            if fits is None:
                continue
            res = rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol)
            res.pop("h_hat")
            res["W_loo"] = loo_debias(F_W, fits[0], fits[1], psi, X_sobol,
                                      res["W_hat"], n_total=N)
            res["p_min"] = float(dgp["propensity"](data["X"]).min())
            rows.append({"cell": label, "rep": rep,
                         "W_true": W_true, "V_true": V_true, **res})
        print(f"  {label} done ({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows)
    for est, se in (("V_hat", "V_se"), ("W_hat", "W_se"), ("W_loo", "W_se")):
        truth = draws["V_true"] if est.startswith("V") else draws["W_true"]
        draws[f"{est}_dev"] = draws[est] - truth
        draws[f"{est}_cover"] = (np.abs(draws[f"{est}_dev"]) <= Z975 * draws[se]).astype(float)

    g = draws.groupby("cell", sort=False)
    summary = pd.DataFrame({
        "V_true": g["V_true"].first(), "h_rmse": g["h_rmse"].mean(),
        "V_bias": g["V_hat_dev"].mean(), "V_sd": g["V_hat"].std(),
        "V_se": g["V_se"].mean(), "V_cover": g["V_hat_cover"].mean(),
        "W_bias": g["W_hat_dev"].mean(), "W_cover": g["W_hat_cover"].mean(),
        "Wloo_bias": g["W_loo_dev"].mean(), "Wloo_cover": g["W_loo_cover"].mean(),
        "p_min": g["p_min"].mean(), "draws": g.size(),
    }).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / f"S1_robustness_summary_rep{REPS}.csv", index=False)
    draws.to_csv(OUTPUT_DIR / f"S1_robustness_draws_rep{REPS}.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
