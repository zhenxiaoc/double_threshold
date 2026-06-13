"""JTPA empirical application with the validated RF-sieve pipeline.

Data: OptTreat/R Codes/Code/KT_Data1.csv (n = 9,223; D, 30-month earnings,
pre-program earnings, education) -- the Kitagawa-Tetenov (2018) sample used in
the paper's Section 6.

Pipeline (per FINDINGS_comprehensive.md):
  1. covariates min-max scaled to [0,1]; common-support (min-max) trimming;
  2. (K, gamma) chosen by split-half CV of first-stage MSE;
  3. per-arm OLS on shared random cos features;
  4. unknown-F (sample-average) estimators over the trimmed sample:
       W_hat = mean [h_hat]_+,  V_hat = mean 1{h_hat >= 0};
     SEs from the stacked-OLS sandwich with EMPIRICAL-measure derivative
     vectors (no kernel density estimation needed -- the sample average
     integrates against f0 automatically; cf. Theorem 6 of the draft):
       bun_W = (1/n) sum 1{h_hat(X_i) >= 0} psi_diff(X_i)
       bun_V = (1/(2 eps n)) sum 1{|h_hat(X_i)| < eps} psi_diff(X_i)
       se_W^2 = Var_hat([h_hat]_+)/n + bun_W' Patty bun_W,
       se_V^2 = bun_V' Patty bun_V;
  5. debiased robustness: LOO-W (delta0 = 0.05) and -- because the share ~0.9
     is in the extreme-share regime -- LOO-V (delta0 = 0.2);
  6. feature-draw stability: everything repeated over N_SEEDS feature draws.

Outcomes: (a) 30-month earnings; (b) earnings - $774 x D (program cost).
Comparison targets (paper Table 6, B-spline; KT 2018 plug-in):
  no cost : share 0.89 (0.73, 1.05);  gain $1,519 ($764, $2,274);  KT: 0.91 / $1,693
  cost    : share 0.80 (0.53, 1.07);  gain $858  ($152, $1,564);   KT: 0.78 / $996
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rf_sieve_lib import (Z975, draw_feature_map, fit_both_arms, loo_debias,
                          F_V, F_W)

DATA = Path(r"C:\Users\wayne\Dropbox\D1_Academic\A0_Research\R57_OpTreat\OptTreat\R Codes\Code\KT_Data1.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
K_MENU = (25, 50, 100, 200)
GAMMA_MENU = (1.5, 3.0, 6.0)
IOTA = 0.01
N_SEEDS = 20
SEED = 163
COST = 774.0


def trim_minmax(df: pd.DataFrame, cols) -> pd.DataFrame:
    keep = np.ones(len(df), dtype=bool)
    for c in cols:
        lo = max(df.loc[df.D == 1, c].min(), df.loc[df.D == 0, c].min())
        hi = min(df.loc[df.D == 1, c].max(), df.loc[df.D == 0, c].max())
        keep &= (df[c] >= lo) & (df[c] <= hi)
    return df.loc[keep].reset_index(drop=True)


def cv_pick(data, rng):
    n = data["X"].shape[0]
    perm = rng.permutation(n)
    iA, iB = perm[: n // 2], perm[n // 2:]
    best, best_mse = None, np.inf
    for K in K_MENU:
        for gam in GAMMA_MENU:
            psi = draw_feature_map(data["X"].shape[1], K, rng, gam)
            sub = {k: v[iA] for k, v in data.items()}
            fits = fit_both_arms(psi, sub)
            if fits is None:
                continue
            Psi_v = psi(data["X"][iB])
            pred = np.where(data["D"][iB] == 1.0,
                            Psi_v @ fits[0].beta, Psi_v @ fits[1].beta)
            mse = float(np.mean((data["Y"][iB] - pred) ** 2))
            if mse < best_mse:
                best, best_mse = (K, gam), mse
    return best, best_mse


def analyze_once(data, K, gam, rng):
    """One feature draw: estimates + SEs over the (already trimmed) sample."""
    n = data["X"].shape[0]
    psi = draw_feature_map(data["X"].shape[1], K, rng, gam)
    fits = fit_both_arms(psi, data)
    if fits is None:
        return None
    fit_t, fit_c = fits
    Psi_all = psi(data["X"])
    h_hat = Psi_all @ (fit_t.beta - fit_c.beta)

    W_hat = float(np.maximum(h_hat, 0.0).mean())
    V_hat = float((h_hat >= 0.0).mean())

    # ---- empirical-measure derivative vectors (unknown-F case) ----
    pos = h_hat >= 0.0
    bun_W = Psi_all[pos].sum(axis=0) / n
    var_W_func = float(bun_W @ fit_t.patty @ bun_W + bun_W @ fit_c.patty @ bun_W)
    se_W = float(np.sqrt(np.maximum(h_hat, 0.0).var(ddof=1) / n + var_W_func))

    eps = IOTA * float(h_hat.std())
    band = np.abs(h_hat) < eps
    bun_V = Psi_all[band].sum(axis=0) / (2.0 * eps * n) if band.any() else None
    se_V = (float(np.sqrt(bun_V @ fit_t.patty @ bun_V + bun_V @ fit_c.patty @ bun_V))
            if bun_V is not None else np.nan)

    # ---- debiased robustness (recipe: LOO-W always; LOO-V for extreme share) ----
    W_loo = loo_debias(F_W, fit_t, fit_c, psi, data["X"], W_hat, n_total=n,
                       delta0=0.05, max_per_arm=2000, rng=rng)
    V_loo = loo_debias(F_V, fit_t, fit_c, psi, data["X"], V_hat, n_total=n,
                       delta0=0.2, max_per_arm=2000, rng=rng)

    return {"W_hat": W_hat, "se_W": se_W, "V_hat": V_hat, "se_V": se_V,
            "W_loo": W_loo, "V_loo": V_loo, "n_band": int(band.sum()),
            "sd_h": float(h_hat.std())}


def run_outcome(df: pd.DataFrame, label: str, cost: float, rng) -> dict:
    d = df.copy()
    d["Y"] = d["earnings"] - cost * d["D"]
    X = d[["prevearn", "edu"]].to_numpy(float)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    data = {"X": X, "D": d["D"].to_numpy(float), "Y": d["Y"].to_numpy(float)}

    (K, gam), cv_mse = cv_pick(data, rng)
    runs = []
    for s in range(N_SEEDS):
        r = analyze_once(data, K, gam, np.random.default_rng(SEED + 7717 * s))
        if r is not None:
            runs.append(r)
    R = pd.DataFrame(runs)
    out = {"outcome": label, "n": len(d), "K": K, "gamma": gam,
           "cv_rmse": np.sqrt(cv_mse)}
    for col in ("W_hat", "V_hat", "W_loo", "V_loo", "se_W", "se_V", "n_band", "sd_h"):
        out[col] = float(R[col].median())
        if col in ("W_hat", "V_hat", "W_loo", "V_loo"):
            out[f"{col}_seedSD"] = float(R[col].std())
    return out


def main():
    df0 = pd.read_csv(DATA)
    print(f"raw n = {len(df0)}, treated share = {df0.D.mean():.3f}")
    df = trim_minmax(df0, ["prevearn", "edu"])
    print(f"trimmed n = {len(df)} (removed {len(df0) - len(df)})")

    rng = np.random.default_rng(SEED)
    rows = []
    for sample, frame in (("trimmed", df), ("untrimmed", df0)):
        for label, cost in (("no_cost", 0.0), ("cost774", COST)):
            r = run_outcome(frame, f"{sample}_{label}", cost, rng)
            rows.append(r)
            print(f"  {r['outcome']}: K={r['K']} gamma={r['gamma']} done")

    res = pd.DataFrame(rows)
    for f, se in (("W_hat", "se_W"), ("V_hat", "se_V")):
        res[f"{f}_lo"] = res[f] - Z975 * res[se]
        res[f"{f}_hi"] = res[f] + Z975 * res[se]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUTPUT_DIR / "JTPA_rf_results.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["outcome", "n", "K", "gamma", "cv_rmse", "sd_h",
            "V_hat", "se_V", "V_hat_lo", "V_hat_hi", "V_loo", "V_hat_seedSD",
            "W_hat", "se_W", "W_hat_lo", "W_hat_hi", "W_loo", "W_hat_seedSD",
            "n_band"]
    print()
    print(res[cols].to_string(index=False))
    print("\nPaper (B-spline, trimmed): no-cost share 0.89 (0.73,1.05), gain 1519 (764,2274)")
    print("                           cost    share 0.80 (0.53,1.07), gain  858 (152,1564)")
    print("KT (2018):                 no-cost 0.91 / 1693;  cost 0.78 / 996")


if __name__ == "__main__":
    main()
