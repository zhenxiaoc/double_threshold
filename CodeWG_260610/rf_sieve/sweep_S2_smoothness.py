"""Sweep S2: Theorem-5 test for V -- does LOO debiasing matter at low smoothness?

Kink DGP at d_x = 2: tau = 3(|x1 - 0.35|^p - 0.25^p), Holder smoothness exactly p,
boundary {x1 = 0.10} U {x1 = 0.60} away from the kink (regular level set holds).

  p = 1.6: plug-in V needs sigma > d = 2 (FAILS); SS/LOO need sigma > 1.5 (holds);
  p = 2.5: both smoothness conditions hold (control).

Sieve dimension follows the rate-optimal rule K_n ~ n^{d/(2p+1)}.
V_plug vs V_loo and W_plug vs W_loo, both studentized by the plug-in sieve SE
(Theorem 5: the same studentization applies to the debiased estimators).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from rf_sieve_lib import (Z975, make_dgp, generate, compute_truth,
                          draw_feature_map, fit_both_arms, rf_inference,
                          F_W, F_V, loo_debias)

REPS = 150
DIM = 2
GAMMA = 3.0
M_SOBOL = 8192
SEED = 97
N_VALUES = (1000, 4000, 16000)
KINK_POWS = (1.6, 2.5)
LOO_MAX_PER_ARM = 2000   # unbiased subsampled LOO at large n (4x speedup)
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def k_rule(n: int, p: float) -> int:
    return max(12, int(round(3.0 * n ** (DIM / (2.0 * p + 1.0)))))


def main():
    t0 = time.time()
    rows = []
    X_sobol = Sobol(d=DIM, scramble=False).random(M_SOBOL)
    for p in KINK_POWS:
        dgp = make_dgp("kink", DIM, kink_pow=p)
        W_true, V_true = compute_truth(dgp)
        tau_sobol = dgp["tau"](X_sobol)
        for n in N_VALUES:
            K = k_rule(n, p)
            for rep in range(REPS):
                rng = np.random.default_rng(SEED + 2147483 * rep + int(10 * p) + n)
                data = generate(dgp, n, rng)
                psi = draw_feature_map(DIM, K, rng, GAMMA)
                fits = fit_both_arms(psi, data)
                if fits is None:
                    continue
                res = rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol)
                res.pop("h_hat")
                res["V_loo"] = loo_debias(F_V, fits[0], fits[1], psi, X_sobol,
                                          res["V_hat"], n_total=n,
                                          max_per_arm=LOO_MAX_PER_ARM, rng=rng)
                res["W_loo"] = loo_debias(F_W, fits[0], fits[1], psi, X_sobol,
                                          res["W_hat"], n_total=n,
                                          max_per_arm=LOO_MAX_PER_ARM, rng=rng)
                rows.append({"p": p, "n": n, "K": K, "rep": rep,
                             "W_true": W_true, "V_true": V_true, **res})
            print(f"  p={p} n={n} (K={K}) done ({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows)
    for est, se in (("V_hat", "V_se"), ("V_loo", "V_se"),
                    ("W_hat", "W_se"), ("W_loo", "W_se")):
        truth = draws["V_true"] if est.startswith("V") else draws["W_true"]
        draws[f"{est}_dev"] = draws[est] - truth
        draws[f"{est}_cover"] = (np.abs(draws[f"{est}_dev"]) <= Z975 * draws[se]).astype(float)
    draws["V_sqerr_plug"] = draws["V_hat_dev"] ** 2
    draws["V_sqerr_loo"] = draws["V_loo_dev"] ** 2

    g = draws.groupby(["p", "n"])
    summary = pd.DataFrame({
        "K": g["K"].first(), "h_rmse": g["h_rmse"].mean(),
        "Vplug_bias": g["V_hat_dev"].mean(),
        "Vplug_rmse": np.sqrt(g["V_sqerr_plug"].mean()),
        "Vplug_cover": g["V_hat_cover"].mean(),
        "Vloo_bias": g["V_loo_dev"].mean(),
        "Vloo_rmse": np.sqrt(g["V_sqerr_loo"].mean()),
        "Vloo_cover": g["V_loo_cover"].mean(),
        "Wplug_bias": g["W_hat_dev"].mean(), "Wplug_cover": g["W_hat_cover"].mean(),
        "Wloo_bias": g["W_loo_dev"].mean(), "Wloo_cover": g["W_loo_cover"].mean(),
        "draws": g.size(),
    }).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / f"S2_smoothness_summary_rep{REPS}.csv", index=False)
    draws.to_csv(OUTPUT_DIR / f"S2_smoothness_draws_rep{REPS}.csv", index=False)
    pd.set_option("display.width", 280)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
