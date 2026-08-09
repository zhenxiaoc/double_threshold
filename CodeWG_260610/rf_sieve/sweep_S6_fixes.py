"""Sweep S6 (Round 3): fixes for the two Round-2 anomalies.

A. Overlap failure (S4-F3): does common-support trimming (the paper's empirical
   practice) restore validity? Trim observations AND the target population to
   {x : p0(x) in [0.05, 0.95]} (oracle propensity, to isolate the mechanism;
   estimated-p_hat trimming is the obvious next step). The estimand becomes the
   trimmed-population value/welfare, as in the JTPA application.

B. DML-IF undercoverage (S5, anomaly A1): 5 folds instead of 2, and an
   augmented SE  se_aug^2 = V_se^2 + var_hat(v*(X,D)(Y - mu_hat))/n  that
   accounts for the finite-sample noise of the correction term (conservative:
   ignores the negative covariance with the plug-in part).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol
from sklearn.ensemble import HistGradientBoostingRegressor

from rf_sieve_lib import (Z975, make_dgp, generate, draw_feature_map,
                          fit_both_arms, rf_inference)

REPS_A = 150
REPS_B = 100
N = 4000
M_SOBOL = 8192
IOTA = 0.01
SEED = 131
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


# ----------------------------------------------------------------------
# Part A: trimming under extreme overlap
# ----------------------------------------------------------------------
def part_A():
    print("== Part A: oracle common-support trimming under overlap=6 ==")
    dim, K, gamma = 50, 200, 1.5
    dgp = make_dgp("dense", dim, overlap=6.0)
    X_sobol = Sobol(d=dim, scramble=False).random(M_SOBOL)
    keep_s = (dgp["propensity"](X_sobol) >= 0.05) & (dgp["propensity"](X_sobol) <= 0.95)
    X_target = X_sobol[keep_s]
    tau_t = dgp["tau"](X_target)
    W_true = float(np.maximum(tau_t, 0.0).mean())
    V_true = float((tau_t >= 0.0).mean())
    print(f"  trimmed target share kept: {keep_s.mean():.3f}; "
          f"W_true={W_true:.4f} V_true={V_true:.4f}")

    rows = []
    t0 = time.time()
    for rep in range(REPS_A):
        rng = np.random.default_rng(SEED + 86028121 * rep)
        data = generate(dgp, N, rng)
        p0 = dgp["propensity"](data["X"])
        keep = (p0 >= 0.05) & (p0 <= 0.95)
        sub = {k: v[keep] for k, v in data.items()}
        psi = draw_feature_map(dim, K, rng, gamma)
        fits = fit_both_arms(psi, sub)
        if fits is None:
            continue
        res = rf_inference(fits[0], fits[1], psi, X_target, tau_t, iota=IOTA)
        res.pop("h_hat")
        rows.append({"rep": rep, "kept": float(keep.mean()), **res})
    draws = pd.DataFrame(rows)
    for est, se, truth in (("V_hat", "V_se", V_true), ("W_hat", "W_se", W_true)):
        draws[f"{est}_dev"] = draws[est] - truth
        draws[f"{est}_cover"] = (np.abs(draws[f"{est}_dev"]) <= Z975 * draws[se]).astype(float)
    s = {
        "kept": draws["kept"].mean(), "h_rmse": draws["h_rmse"].mean(),
        "V_bias": draws["V_hat_dev"].mean(), "V_sd": draws["V_hat"].std(),
        "V_se": draws["V_se"].mean(), "V_cover": draws["V_hat_cover"].mean(),
        "W_bias": draws["W_hat_dev"].mean(), "W_cover": draws["W_hat_cover"].mean(),
        "draws": len(draws),
    }
    print(pd.Series(s).to_string())
    pd.DataFrame([s]).to_csv(OUTPUT_DIR / f"S6A_trim_summary_rep{REPS_A}.csv", index=False)
    print(f"  Part A done ({time.time() - t0:.0f}s)\n")


# ----------------------------------------------------------------------
# Part B: DML-IF with 5 folds + augmented SE
# ----------------------------------------------------------------------
def part_B():
    print("== Part B: DML-IF, 5 folds, augmented SE ==")
    K, gamma, n_folds = 200, 1.5, 5
    rows = []
    t0 = time.time()
    for dim in (10, 50):
        dgp = make_dgp("dense", dim)
        X_sobol = Sobol(d=dim, scramble=False).random(M_SOBOL)
        tau_sobol = dgp["tau"](X_sobol)
        t_big = dgp["tau"](Sobol(d=dim, scramble=False).random(2**16))
        V_true = float((t_big >= 0.0).mean())
        M = M_SOBOL

        for rep in range(REPS_B):
            rng = np.random.default_rng(SEED + 15487469 * rep + dim)
            data = generate(dgp, N, rng)
            psi = draw_feature_map(dim, K, rng, gamma)
            fits = fit_both_arms(psi, data)
            if fits is None:
                continue
            ref = rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol, iota=IOTA)
            Psi_s = psi(X_sobol)

            perm = rng.permutation(N)
            folds = np.array_split(perm, n_folds)
            h_gbm = np.zeros(M)
            corr_terms = np.full(N, np.nan)
            for kf in range(n_folds):
                test = folds[kf]
                train = np.concatenate([folds[j] for j in range(n_folds) if j != kf])
                Xtr, Dtr, Ytr = data["X"][train], data["D"][train], data["Y"][train]
                gb_t = HistGradientBoostingRegressor(random_state=rep, max_iter=150)
                gb_t.fit(Xtr[Dtr == 1], Ytr[Dtr == 1])
                gb_c = HistGradientBoostingRegressor(random_state=rep + 1, max_iter=150)
                gb_c.fit(Xtr[Dtr == 0], Ytr[Dtr == 0])
                h_fold = gb_t.predict(X_sobol) - gb_c.predict(X_sobol)
                h_gbm += h_fold / n_folds

                eps = IOTA * float(h_fold.std())
                band = np.abs(h_fold) < eps
                if not band.any():
                    continue
                b_vec = Psi_s[band].sum(axis=0) / (2.0 * eps * M)
                Psi_tr = psi(Xtr)
                n_tr = len(train)
                G_t = Psi_tr[Dtr == 1].T @ Psi_tr[Dtr == 1] / n_tr
                G_c = Psi_tr[Dtr == 0].T @ Psi_tr[Dtr == 0] / n_tr
                w_t = np.linalg.pinv(G_t, rcond=1e-10) @ b_vec
                w_c = np.linalg.pinv(G_c, rcond=1e-10) @ (-b_vec)
                Xte, Dte, Yte = data["X"][test], data["D"][test], data["Y"][test]
                Psi_te = psi(Xte)
                v_star = np.where(Dte == 1.0, Psi_te @ w_t, Psi_te @ w_c)
                mu_hat = np.where(Dte == 1.0, gb_t.predict(Xte), gb_c.predict(Xte))
                corr_terms[test] = v_star * (Yte - mu_hat)

            ok = np.isfinite(corr_terms)
            correction = float(corr_terms[ok].mean()) if ok.any() else 0.0
            corr_var = float(corr_terms[ok].var(ddof=1)) / ok.sum() if ok.sum() > 1 else 0.0
            V_gbm = float((h_gbm >= 0.0).mean())
            V_dml = V_gbm + correction
            se_plug = ref["V_se"]
            se_aug = float(np.sqrt(se_plug**2 + corr_var))

            rows.append({"dim": dim, "rep": rep, "V_true": V_true,
                         "V_rf": ref["V_hat"], "se_plug": se_plug, "se_aug": se_aug,
                         "V_gbm": V_gbm, "V_dml": V_dml, "corr": correction})
            if (rep + 1) % 20 == 0:
                print(f"  dim={dim}: rep {rep + 1}/{REPS_B} ({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows)
    draws["dml_dev"] = draws["V_dml"] - draws["V_true"]
    draws["gbm_dev"] = draws["V_gbm"] - draws["V_true"]
    draws["dml_cover_plug"] = (np.abs(draws["dml_dev"]) <= Z975 * draws["se_plug"]).astype(float)
    draws["dml_cover_aug"] = (np.abs(draws["dml_dev"]) <= Z975 * draws["se_aug"]).astype(float)
    draws["gbm_cover_aug"] = (np.abs(draws["gbm_dev"]) <= Z975 * draws["se_aug"]).astype(float)
    g = draws.groupby("dim")
    summary = pd.DataFrame({
        "gbm_bias": g["gbm_dev"].mean(), "dml_bias": g["dml_dev"].mean(),
        "dml_sd": g["V_dml"].std(),
        "se_plug": g["se_plug"].mean(), "se_aug": g["se_aug"].mean(),
        "dml_cover_plug": g["dml_cover_plug"].mean(),
        "dml_cover_aug": g["dml_cover_aug"].mean(),
        "gbm_cover_aug": g["gbm_cover_aug"].mean(),
        "draws": g.size(),
    }).reset_index()
    summary.to_csv(OUTPUT_DIR / f"S6B_dml_aug_summary_rep{REPS_B}.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    part_A()
    part_B()
