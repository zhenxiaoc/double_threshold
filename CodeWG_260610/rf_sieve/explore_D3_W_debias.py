"""Direction 3: can SS/LOO debiasing (Theorem 5 of the draft) rescue the
welfare functional W in high dimensions?

The Jensen/ReLU bias of W(h_hat) is (to second order) the diagonal quadratic
term (1/2) E D^2W(h0)[h_hat - h0, h_hat - h0], which is exactly what the
split-sample (SS) and leave-one-out (LOO) corrections remove. We compare, at
the cells where plug-in W failed badly:

  W_plug : plug-in W(h_hat)
  W_ss   : split-sample debiased (D^2W by central differences on Sobol points)
  W_loo  : leave-one-out debiased (exact OLS leverage residuals)

V_plug and V_ss are recorded as controls (V should not be hurt).
Coverage for all W variants uses the same plug-in sieve SE.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from rf_sieve_lib import (Z975, make_dgp, generate, compute_truth,
                          draw_feature_map, fit_both_arms, rf_inference,
                          F_W, F_V, ss_debias, loo_debias)

REPS = 150
CELLS = [  # (dgp, dim, n, K) -- plug-in W coverage from earlier runs in comments
    ("dense", 10, 4000, 200),   # W_cover ~ 0.19
    ("dense", 50, 4000, 200),   # W_cover ~ 0.08
    ("dense", 50, 4000, 400),   # W_cover ~ 0.00 (worst)
]
GAMMA = 3.0
M_SOBOL = 8192
SEED = 59
SS_DELTA0 = 0.1
LOO_DELTA0 = 0.05
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def main():
    t0 = time.time()
    rows = []
    sobols: dict[int, np.ndarray] = {}
    for (kind, dim, n, K) in CELLS:
        dgp = make_dgp(kind, dim)
        W_true, V_true = compute_truth(dgp)
        if dim not in sobols:
            sobols[dim] = Sobol(d=dim, scramble=False).random(M_SOBOL)
        X_sobol = sobols[dim]
        tau_sobol = dgp["tau"](X_sobol)

        for rep in range(REPS):
            rng = np.random.default_rng(SEED + 15485863 * rep + dim + n + K)
            data = generate(dgp, n, rng)
            psi = draw_feature_map(dim, K, rng, GAMMA)
            fits = fit_both_arms(psi, data)
            if fits is None:
                continue
            res = rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol)
            res.pop("h_hat")

            W_ss = ss_debias(F_W, psi, data, X_sobol, rng, SS_DELTA0)
            V_ss = ss_debias(F_V, psi, data, X_sobol, rng, SS_DELTA0)
            W_loo = loo_debias(F_W, fits[0], fits[1], psi, X_sobol,
                               res["W_hat"], n_total=n, delta0=LOO_DELTA0)

            rows.append({"dgp": kind, "dim": dim, "n": n, "K": K, "rep": rep,
                         "W_true": W_true, "V_true": V_true,
                         "W_ss": W_ss, "V_ss": V_ss, "W_loo": W_loo, **res})
            if (rep + 1) % 25 == 0:
                print(f"  ({kind}, d={dim}, n={n}, K={K}): rep {rep + 1}/{REPS} "
                      f"({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows)
    for est, se in (("W_hat", "W_se"), ("W_ss", "W_se"), ("W_loo", "W_se"),
                    ("V_hat", "V_se"), ("V_ss", "V_se")):
        truth = draws["W_true"] if est.startswith("W") else draws["V_true"]
        draws[f"{est}_dev"] = draws[est] - truth
        draws[f"{est}_cover"] = (np.abs(draws[f"{est}_dev"]) <= Z975 * draws[se]).astype(float)

    g = draws.groupby(["dgp", "dim", "n", "K"])
    summary = pd.DataFrame({
        "h_rmse": g["h_rmse"].mean(),
        "W_bias_plug": g["W_hat_dev"].mean(), "W_cover_plug": g["W_hat_cover"].mean(),
        "W_bias_ss": g["W_ss_dev"].mean(), "W_cover_ss": g["W_ss_cover"].mean(),
        "W_bias_loo": g["W_loo_dev"].mean(), "W_cover_loo": g["W_loo_cover"].mean(),
        "V_bias_plug": g["V_hat_dev"].mean(), "V_cover_plug": g["V_hat_cover"].mean(),
        "V_bias_ss": g["V_ss_dev"].mean(), "V_cover_ss": g["V_ss_cover"].mean(),
        "draws": g.size(),
    }).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / f"D3_W_debias_summary_rep{REPS}.csv", index=False)
    draws.to_csv(OUTPUT_DIR / f"D3_W_debias_draws_rep{REPS}.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
