"""Sweep S5: the Appendix-D cross-fitted sieve-influence-function estimator
with a generic ML (gradient boosting) first stage for the value functional.

Estimators compared (dense DGP, d_x in {10, 50}, n = 4000):
  V_rf   : RF-OLS plug-in + band sieve SE              (reference, known valid)
  V_gbm  : naive plug-in with a gradient-boosting CATE (no correction)
  V_dml  : cross-fitted (K_folds = 2) GBM first stage + sieve-influence-function
           correction  theta = V(h_gbm) + En[ v*_K(X,D) (Y - mu_gbm(X,D)) ],
           with v*_K built in closed form on the RF sieve (eq. (D.2)-(D.3) of
           the draft): v*(x,d) = psibar(x,d)' Ghat^{-1} Dhat, Dhat the eps-band
           derivative vector evaluated at the GBM h_hat.

All three studentized by the RF sieve band SE (Appendix D, Theorem 7: the
Section-4.1 variance estimator remains consistent for the corrected estimator).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol
from sklearn.ensemble import HistGradientBoostingRegressor

from rf_sieve_lib import (Z975, make_dgp, generate, compute_truth,
                          draw_feature_map, fit_both_arms, rf_inference)

REPS = 100
N = 4000
DIMS = (10, 50)
K = 200
GAMMA = 1.5
M_SOBOL = 8192
IOTA = 0.01
N_FOLDS = 2
SEED = 127
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def gbm_fit_arms(data, idx, rng_seed):
    """Fit per-arm gradient boosting on data[idx]; return mu_t, mu_c predictors."""
    X, D, Y = data["X"][idx], data["D"][idx], data["Y"][idx]
    out = {}
    for arm, mask in (("t", D == 1.0), ("c", D == 0.0)):
        gbm = HistGradientBoostingRegressor(random_state=rng_seed, max_iter=150)
        gbm.fit(X[mask], Y[mask])
        out[arm] = gbm
    return out


def main():
    t0 = time.time()
    rows = []
    for dim in DIMS:
        dgp = make_dgp("dense", dim)
        W_true, V_true = compute_truth(dgp)
        X_sobol = Sobol(d=dim, scramble=False).random(M_SOBOL)
        tau_sobol = dgp["tau"](X_sobol)
        M = M_SOBOL

        for rep in range(REPS):
            rng = np.random.default_rng(SEED + 7368787 * rep + dim)
            data = generate(dgp, N, rng)

            # ---------- reference: RF-OLS plug-in + sieve SE ----------
            psi = draw_feature_map(dim, K, rng, GAMMA)
            fits = fit_both_arms(psi, data)
            if fits is None:
                continue
            ref = rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol, iota=IOTA)
            Psi_s = psi(X_sobol)

            # ---------- GBM cross-fitted ----------
            perm = rng.permutation(N)
            folds = np.array_split(perm, N_FOLDS)
            h_gbm_full = np.zeros(M)          # average of fold h_hat at Sobol pts
            correction = 0.0
            for kf in range(N_FOLDS):
                test = folds[kf]
                train = np.concatenate([folds[j] for j in range(N_FOLDS) if j != kf])
                gbms = gbm_fit_arms(data, train, rng_seed=rep)
                h_fold = gbms["t"].predict(X_sobol) - gbms["c"].predict(X_sobol)
                h_gbm_full += h_fold / N_FOLDS

                # sieve Riesz representer on the RF sieve, derivative at GBM h_hat
                eps = IOTA * float(h_fold.std())
                band = np.abs(h_fold) < eps
                if not band.any():
                    continue
                b_vec = Psi_s[band].sum(axis=0) / (2.0 * eps * M)
                # Ghat blocks from the TRAINING fold (per-arm Gram / n_train)
                Xtr, Dtr = data["X"][train], data["D"][train]
                Psi_tr = psi(Xtr)
                n_tr = len(train)
                G_t = Psi_tr[Dtr == 1].T @ Psi_tr[Dtr == 1] / n_tr
                G_c = Psi_tr[Dtr == 0].T @ Psi_tr[Dtr == 0] / n_tr
                w_t = np.linalg.pinv(G_t, rcond=1e-10) @ b_vec
                w_c = np.linalg.pinv(G_c, rcond=1e-10) @ (-b_vec)
                # evaluate v*(X_i, D_i) and residuals on the held-out fold
                Xte, Dte, Yte = data["X"][test], data["D"][test], data["Y"][test]
                Psi_te = psi(Xte)
                v_star = np.where(Dte == 1.0, Psi_te @ w_t, Psi_te @ w_c)
                mu_hat = np.where(Dte == 1.0,
                                  gbms["t"].predict(Xte), gbms["c"].predict(Xte))
                correction += float(np.mean(v_star * (Yte - mu_hat))) / N_FOLDS

            V_gbm = float((h_gbm_full >= 0.0).mean())
            V_dml = V_gbm + correction
            h_rmse_gbm = float(np.sqrt(np.mean((h_gbm_full - tau_sobol) ** 2)))

            rows.append({
                "dim": dim, "rep": rep, "V_true": V_true,
                "V_rf": ref["V_hat"], "V_se": ref["V_se"],
                "h_rmse_rf": ref["h_rmse"], "h_rmse_gbm": h_rmse_gbm,
                "V_gbm": V_gbm, "V_dml": V_dml, "corr": correction,
            })
            if (rep + 1) % 20 == 0:
                print(f"  dim={dim}: rep {rep + 1}/{REPS} ({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows)
    for est in ("V_rf", "V_gbm", "V_dml"):
        draws[f"{est}_dev"] = draws[est] - draws["V_true"]
        draws[f"{est}_cover"] = (np.abs(draws[f"{est}_dev"]) <= Z975 * draws["V_se"]).astype(float)

    g = draws.groupby("dim")
    summary = pd.DataFrame({
        "h_rmse_rf": g["h_rmse_rf"].mean(), "h_rmse_gbm": g["h_rmse_gbm"].mean(),
        "Vrf_bias": g["V_rf_dev"].mean(), "Vrf_cover": g["V_rf_cover"].mean(),
        "Vgbm_bias": g["V_gbm_dev"].mean(), "Vgbm_cover": g["V_gbm_cover"].mean(),
        "Vdml_bias": g["V_dml_dev"].mean(), "Vdml_cover": g["V_dml_cover"].mean(),
        "corr_mean": g["corr"].mean(), "V_se": g["V_se"].mean(),
        "draws": g.size(),
    }).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / f"S5_dml_if_summary_rep{REPS}.csv", index=False)
    draws.to_csv(OUTPUT_DIR / f"S5_dml_if_draws_rep{REPS}.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
