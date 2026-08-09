"""Sweep S3: the full practical pipeline under signal-strength stress.

Pipeline: lasso screening -> RF features on screened support -> per-arm OLS ->
plug-in V (band SE) + LOO-debiased W (plug-in SE). Sparse DGP, d_x = 50,
n = 4000, signal strength tau_scale in {1, 2, 3} (beta-min stress for the
screening step). Variants: full-sample screening vs honest split.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from rf_sieve_lib import (Z975, make_dgp, generate, compute_truth,
                          draw_feature_map, fit_both_arms, rf_inference,
                          F_W, loo_debias, screen_lasso)

REPS = 150
DIM, N, K, GAMMA = 50, 4000, 50, 3.0
M_SOBOL = 8192
SEED = 103
TAU_SCALES = (1.0, 2.0, 3.0)
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def run_variant(label, ts, data, S, est_idx, X_sobol, tau_sobol, rng, rows, rep):
    sub = {k: v[est_idx] for k, v in data.items()} if est_idx is not None else data
    n_eff = sub["X"].shape[0]
    psi = draw_feature_map(DIM, K, rng, GAMMA, support=S)
    fits = fit_both_arms(psi, sub)
    if fits is None:
        return
    res = rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol)
    res.pop("h_hat")
    res["W_loo"] = loo_debias(F_W, fits[0], fits[1], psi, X_sobol,
                              res["W_hat"], n_total=n_eff)
    rows.append({"variant": label, "tau_scale": ts, "rep": rep,
                 "S_size": len(S), "S_hit": float({0, 1} <= set(np.asarray(S).tolist())),
                 **res})


def main():
    t0 = time.time()
    rows, truths = [], {}
    X_sobol = Sobol(d=DIM, scramble=False).random(M_SOBOL)
    for ts in TAU_SCALES:
        dgp = make_dgp("sparse", DIM, tau_scale=ts)
        truths[ts] = compute_truth(dgp)
        tau_sobol = dgp["tau"](X_sobol)
        for rep in range(REPS):
            rng = np.random.default_rng(SEED + 999983 * rep + int(ts * 7))
            data = generate(dgp, N, rng)

            S_full = screen_lasso(data["X"], data["D"], data["Y"], seed=rep)
            run_variant("screen_full", ts, data, S_full, None,
                        X_sobol, tau_sobol, rng, rows, rep)

            perm = rng.permutation(N)
            S_half = screen_lasso(data["X"][perm[: N // 2]], data["D"][perm[: N // 2]],
                                  data["Y"][perm[: N // 2]], seed=rep)
            run_variant("screen_split", ts, data, S_half, perm[N // 2:],
                        X_sobol, tau_sobol, rng, rows, rep)
        print(f"  tau_scale={ts} done ({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows)
    draws["W_true"] = draws["tau_scale"].map({t: truths[t][0] for t in TAU_SCALES})
    draws["V_true"] = draws["tau_scale"].map({t: truths[t][1] for t in TAU_SCALES})
    for est, se in (("V_hat", "V_se"), ("W_hat", "W_se"), ("W_loo", "W_se")):
        truth = draws["V_true"] if est.startswith("V") else draws["W_true"]
        draws[f"{est}_dev"] = draws[est] - truth
        draws[f"{est}_cover"] = (np.abs(draws[f"{est}_dev"]) <= Z975 * draws[se]).astype(float)

    g = draws.groupby(["variant", "tau_scale"], sort=False)
    summary = pd.DataFrame({
        "V_true": g["V_true"].first(),
        "S_hit": g["S_hit"].mean(), "S_size": g["S_size"].mean(),
        "h_rmse": g["h_rmse"].mean(),
        "V_bias": g["V_hat_dev"].mean(), "V_cover": g["V_hat_cover"].mean(),
        "W_bias": g["W_hat_dev"].mean(), "W_cover": g["W_hat_cover"].mean(),
        "Wloo_bias": g["W_loo_dev"].mean(), "Wloo_cover": g["W_loo_cover"].mean(),
        "draws": g.size(),
    }).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / f"S3_pipeline_summary_rep{REPS}.csv", index=False)
    draws.to_csv(OUTPUT_DIR / f"S3_pipeline_draws_rep{REPS}.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
