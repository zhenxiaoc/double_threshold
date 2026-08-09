"""Direction 4: data-driven tuning of the random-feature sieve + inference refinements.

(a) CV selection of (K, gamma): split-half cross-validation of the first-stage
    MSE over a menu K x gamma; refit on the full sample at the selected config;
    check post-selection V coverage. The hypothesis (from the earlier h_rmse <->
    coverage link): selecting on first-stage fit is a valid surrogate for
    selecting on V-inference quality.

(b) iota sweep: the V intervals were conservative (SE ~ 1.5-2x SD); how does the
    band width eps = iota * SD(h_hat) trade off SE magnitude vs coverage?

(c) Scrambled vs plain Sobol for the functional/band evaluation at d_x = 50.

Cells: dense and sparse DGP, d_x = 50, n = 4000.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from rf_sieve_lib import (Z975, make_dgp, generate, compute_truth,
                          draw_feature_map, fit_both_arms, rf_inference)

REPS = 120
DIM = 50
N = 4000
K_MENU = (50, 100, 200, 400)
GAMMA_MENU = (1.5, 3.0, 6.0)
IOTA_GRID = (0.005, 0.01, 0.02, 0.05)
M_SOBOL = 8192
SEED = 71
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def cv_mse(psi, data, i_fit, i_val) -> float:
    sub = {k: v[i_fit] for k, v in data.items()}
    fits = fit_both_arms(psi, sub)
    if fits is None:
        return np.inf
    Xv, Dv, Yv = data["X"][i_val], data["D"][i_val], data["Y"][i_val]
    Psi_v = psi(Xv)
    pred = np.where(Dv == 1.0, Psi_v @ fits[0].beta, Psi_v @ fits[1].beta)
    return float(np.mean((Yv - pred) ** 2))


def main():
    t0 = time.time()
    rows_cv, rows_iota = [], []
    for kind in ("dense", "sparse"):
        dgp = make_dgp(kind, DIM)
        W_true, V_true = compute_truth(dgp)
        X_plain = Sobol(d=DIM, scramble=False).random(M_SOBOL)
        X_scram = Sobol(d=DIM, scramble=True, seed=12345).random(M_SOBOL)
        tau_plain = dgp["tau"](X_plain)
        tau_scram = dgp["tau"](X_scram)

        for rep in range(REPS):
            rng = np.random.default_rng(SEED + 32452843 * rep + (0 if kind == "dense" else 1))
            data = generate(dgp, N, rng)
            perm = rng.permutation(N)
            iA, iB = perm[: N // 2], perm[N // 2:]

            # ---------- (a) CV selection over the menu ----------
            best, best_mse = None, np.inf
            psis = {}
            for K in K_MENU:
                for gam in GAMMA_MENU:
                    psis[(K, gam)] = draw_feature_map(DIM, K, rng, gam)
                    mse = cv_mse(psis[(K, gam)], data, iA, iB)
                    if mse < best_mse:
                        best, best_mse = (K, gam), mse
            psi_sel = psis[best]
            fits = fit_both_arms(psi_sel, data)
            if fits is None:
                continue
            res_sel = rf_inference(fits[0], fits[1], psi_sel, X_plain, tau_plain)
            res_sel.pop("h_hat")
            rows_cv.append({"dgp": kind, "rep": rep, "K_sel": best[0],
                            "gamma_sel": best[1], "cv_mse": best_mse,
                            "W_true": W_true, "V_true": V_true, **res_sel})

            # ---------- (b)+(c) iota sweep and scrambled Sobol at fixed (200, 3) ----------
            psi_ref = psis[(200, 3.0)]
            fits_ref = fit_both_arms(psi_ref, data)
            if fits_ref is None:
                continue
            for iota in IOTA_GRID:
                r = rf_inference(fits_ref[0], fits_ref[1], psi_ref, X_plain,
                                 tau_plain, iota=iota)
                rows_iota.append({"dgp": kind, "rep": rep, "iota": iota,
                                  "sobol": "plain", "V_true": V_true,
                                  "V_hat": r["V_hat"], "V_se": r["V_se"],
                                  "n_band": r["n_band"]})
            r = rf_inference(fits_ref[0], fits_ref[1], psi_ref, X_scram,
                             tau_scram, iota=0.01)
            rows_iota.append({"dgp": kind, "rep": rep, "iota": 0.01,
                              "sobol": "scrambled", "V_true": V_true,
                              "V_hat": r["V_hat"], "V_se": r["V_se"],
                              "n_band": r["n_band"]})
            if (rep + 1) % 20 == 0:
                print(f"  {kind}: rep {rep + 1}/{REPS} ({time.time() - t0:.0f}s)")

    # -------------------- summaries --------------------
    cv = pd.DataFrame(rows_cv)
    cv["V_dev"] = cv["V_hat"] - cv["V_true"]
    cv["V_cover"] = (np.abs(cv["V_dev"]) <= Z975 * cv["V_se"]).astype(float)
    cv["W_dev"] = cv["W_hat"] - cv["W_true"]
    cv["W_cover"] = (np.abs(cv["W_dev"]) <= Z975 * cv["W_se"]).astype(float)
    g = cv.groupby("dgp")
    cv_summary = pd.DataFrame({
        "K_sel_med": g["K_sel"].median(), "K_sel_max": g["K_sel"].max(),
        "gam15": g["gamma_sel"].apply(lambda s: (s == 1.5).mean()),
        "gam3": g["gamma_sel"].apply(lambda s: (s == 3.0).mean()),
        "gam6": g["gamma_sel"].apply(lambda s: (s == 6.0).mean()),
        "h_rmse": g["h_rmse"].mean(),
        "V_bias": g["V_dev"].mean(), "V_cover": g["V_cover"].mean(),
        "W_bias": g["W_dev"].mean(), "W_cover": g["W_cover"].mean(),
        "draws": g.size(),
    }).reset_index()

    it = pd.DataFrame(rows_iota)
    it["V_dev"] = it["V_hat"] - it["V_true"]
    it["V_cover"] = (np.abs(it["V_dev"]) <= Z975 * it["V_se"]).astype(float)
    gi = it.groupby(["dgp", "sobol", "iota"])
    iota_summary = pd.DataFrame({
        "V_sd": gi["V_hat"].std(), "V_se": gi["V_se"].mean(),
        "se_over_sd": gi["V_se"].mean() / gi["V_hat"].std(),
        "V_cover": gi["V_cover"].mean(), "n_band": gi["n_band"].mean(),
        "draws": gi.size(),
    }).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_summary.to_csv(OUTPUT_DIR / f"D4_cv_selection_summary_rep{REPS}.csv", index=False)
    iota_summary.to_csv(OUTPUT_DIR / f"D4_iota_sobol_summary_rep{REPS}.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n--- CV-selected (K, gamma): post-selection inference ---")
    print(cv_summary.to_string(index=False))
    print("\n--- iota sweep and Sobol scrambling at (K, gamma) = (200, 3) ---")
    print(iota_summary.to_string(index=False))


if __name__ == "__main__":
    main()
