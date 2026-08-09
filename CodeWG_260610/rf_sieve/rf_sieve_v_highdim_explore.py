"""Can inference on the value functional V survive d_x = 10 and d_x = 50?

Exploration script (ClaudeWS) for the Chen-Chen-Gao project, following up on
rf_sieve_highd_sim.py.  Focus: the value functional V = P_F(tau(X) >= 0) with a
random-feature (shallow NN) OLS first stage, treated as a linear sieve.

Hypothesis under investigation
------------------------------
Even though the first stage is high-dimensional, the estimand is a scalar
functional that *averages over most dimensions*.  If the CATE (and hence its
zero boundary) effectively depends on a low-dimensional structure -- a few
linear indices ("dense index" DGP) or a few coordinates ("sparse" DGP) -- then:
  (i)  the boundary {tau = 0} is a cylinder over a low-dimensional set, so the
       irregular part of the functional has low *effective* dimension;
  (ii) the random-feature sieve can be *tuned* to that structure (feature
       support sparsity q, scale gamma, dimension K), trading first-stage bias
       against the sieve-Riesz norm growth -- which, unlike a fixed spline
       basis, is a design choice here;
  (iii) the eps-band sieve t-statistic for V is self-normalizing, so validity
       hinges on the first-stage bias near the boundary being small relative
       to the (growing) band SE -- not on a particular growth-rate exponent.

Experiments
-----------
  A. dense-index DGP, dense sphere features:    d_x in {10, 50}, K, n varied
  B. gamma (feature scale) sensitivity at d_x = 50
  C. sparse DGP (CATE/baseline/propensity depend on x_1, x_2 only; 48 noise
     dims), comparing dense features vs support-sparse features (q = 1, 2)

Diagnostics per draw: h_rmse (first-stage RMSE of the CATE at Sobol points),
n * Var_hat(V) (empirical sieve-Riesz growth), band size, V/W bias + coverage.

Run: python rf_sieve_v_highdim_explore.py   (~3-6 min at default settings)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

# =====================================================================
# CONFIG
# =====================================================================
REPS = 200
M_SOBOL = 8192          # Sobol points for functionals / band derivative
M_TRUTH = 2**16         # Sobol points for the truth
IOTA = 0.01             # eps = IOTA * SD(h_hat at Sobol points)
TAU_SCALE = 3.0
ACTIVATION = "cos"
SEED = 20260609
MIN_ARM_SLACK = 10
PROGRESS_EVERY = 200
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
Z975 = 1.959963984540054

# Each experiment cell: (label, dgp_kind, dim, feat_q, gamma, n, K)
#   feat_q = None -> dense sphere features; q -> q-sparse feature supports
EXPERIMENTS = [
    # --- A: dense-index DGP, dense features ---------------------------------
    ("A_dense_d10_n1000_K50",   "dense", 10, None, 3.0, 1000, 50),
    ("A_dense_d10_n1000_K200",  "dense", 10, None, 3.0, 1000, 200),
    ("A_dense_d10_n4000_K200",  "dense", 10, None, 3.0, 4000, 200),
    ("A_dense_d50_n1000_K50",   "dense", 50, None, 3.0, 1000, 50),
    ("A_dense_d50_n4000_K200",  "dense", 50, None, 3.0, 4000, 200),
    ("A_dense_d50_n4000_K400",  "dense", 50, None, 3.0, 4000, 400),
    # --- B: feature-scale sensitivity at d_x = 50 ---------------------------
    ("B_dense_d50_gamma1.5",    "dense", 50, None, 1.5, 4000, 200),
    ("B_dense_d50_gamma6",      "dense", 50, None, 6.0, 4000, 200),
    # --- C: sparse DGP (signal in x1, x2 only), d_x = 50 --------------------
    ("C_sparse_d50_featdense",  "sparse", 50, None, 3.0, 4000, 200),
    ("C_sparse_d50_featq2",     "sparse", 50, 2,    3.0, 4000, 200),
    ("C_sparse_d50_featq1",     "sparse", 50, 1,    3.0, 4000, 200),
    ("C_sparse_d50_featq2_K50", "sparse", 50, 2,    3.0, 4000, 50),
    # oracle benchmark: features supported on the TRUE relevant coords (x1, x2)
    # -- the upper bound for what support screening/selection could deliver
    ("C_sparse_d50_oracle_K50", "sparse", 50, "oracle2", 3.0, 4000, 50),
    # consistency check at d_x = 50: does the dense-DGP bias keep shrinking?
    ("A_dense_d50_n16000_K400", "dense",  50, None, 3.0, 16000, 400),
]


# =====================================================================
# DGPs
# =====================================================================
def _sig(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_dgp(kind: str, dim: int) -> dict:
    """Return dict with callables tau, baseline, propensity on (m, dim) arrays."""
    if kind == "dense":
        a1 = np.array([(-1.0) ** j for j in range(dim)]) / np.sqrt(dim)
        a2 = np.ones(dim) / np.sqrt(dim)
        m1, m2 = 0.5 * a1.sum(), 0.5 * a2.sum()

        def tau(X):
            return TAU_SCALE * (_sig(3.0 * (X @ a1 - m1))
                                + 0.5 * _sig(3.0 * (X @ a2 - m2)) - 0.70)

        def baseline(X):
            return 2.0 * np.tanh(2.0 * (X @ a2 - m2)) + np.sin(2.0 * np.pi * (X @ a1 - m1))

        def propensity(X):
            return _sig(1.2 * (X @ a1 - m1) + 0.4 * (X @ a2 - m2))

    elif kind == "sparse":
        # everything depends on (x1, x2) only; remaining dims are pure noise
        def tau(X):
            return TAU_SCALE * (_sig(3.0 * (X[:, 0] - 0.5))
                                + 0.5 * _sig(3.0 * (X[:, 1] - 0.5)) - 0.70)

        def baseline(X):
            return 2.0 * np.tanh(2.0 * (X[:, 0] - 0.5)) + np.sin(2.0 * np.pi * (X[:, 1] - 0.5))

        def propensity(X):
            return _sig(1.2 * (X[:, 0] - 0.5) + 0.4 * (X[:, 1] - 0.5))

    else:
        raise ValueError(kind)
    return {"tau": tau, "baseline": baseline, "propensity": propensity}


def generate(dgp: dict, dim: int, n: int, rng: np.random.Generator) -> dict:
    X = rng.uniform(size=(n, dim))
    D = (rng.uniform(size=n) < dgp["propensity"](X)).astype(float)
    Y = dgp["baseline"](X) + D * dgp["tau"](X) + rng.normal(size=n)
    return {"X": X, "D": D, "Y": Y}


def compute_truth(dgp: dict, dim: int) -> tuple[float, float]:
    Xs = Sobol(d=dim, scramble=False).random(M_TRUTH)
    t = dgp["tau"](Xs)
    return float(np.maximum(t, 0.0).mean()), float((t >= 0.0).mean())


# =====================================================================
# Random features: dense sphere or q-sparse supports
# =====================================================================
def _activation(name: str):
    if name == "cos":
        return np.cos
    if name == "relu":
        return lambda z: np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh
    raise ValueError(name)


def draw_feature_map(dim: int, K: int, rng: np.random.Generator,
                     gamma: float, feat_q: int | None):
    """psi: X (m, dim) -> (m, K+1), first column the intercept.

    feat_q = None: (w_k, b_k) iid uniform on S^{dim} (dense directions).
    feat_q = q:    each feature is supported on q random coordinates, with
                   (w_S, b) uniform on S^{q} -- a 'union of q-dim subspaces'
                   sieve encoding a sparsity prior.
    """
    act = _activation(ACTIVATION)
    if feat_q is None:
        Z = rng.normal(size=(K, dim + 1))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        W, b = Z[:, :dim], Z[:, dim]
    elif feat_q == "oracle2":
        # all features supported on the true relevant coordinates (x1, x2)
        W = np.zeros((K, dim))
        Z = rng.normal(size=(K, 3))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        W[:, 0], W[:, 1], b = Z[:, 0], Z[:, 1], Z[:, 2]
    else:
        q = int(feat_q)
        W = np.zeros((K, dim))
        Z = rng.normal(size=(K, q + 1))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        b = Z[:, q]
        for k in range(K):
            S = rng.choice(dim, size=q, replace=False)
            W[k, S] = Z[k, :q]

    def psi(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.hstack([np.ones((X.shape[0], 1)),
                          act(gamma * (X @ W.T + b[None, :]))])

    return psi


# =====================================================================
# OLS per arm + functional inference (as in rf_sieve_highd_sim.py)
# =====================================================================
def fit_arm(psi, X: np.ndarray, Y: np.ndarray):
    Psi = psi(X)
    MtM_pinv = np.linalg.pinv(Psi.T @ Psi, rcond=1e-10)
    beta = MtM_pinv @ (Psi.T @ Y)
    resid = Y - Psi @ beta
    A = MtM_pinv @ (Psi.T * resid[None, :])
    patty = A @ A.T
    return beta, patty


def one_draw(label, dgp, dim, feat_q, gamma, n, K,
             X_sobol, tau_sobol, rng) -> dict | None:
    data = generate(dgp, dim, n, rng)
    Dt = data["D"] == 1.0
    n_t, n_c = int(Dt.sum()), int((~Dt).sum())
    if min(n_t, n_c) < K + 1 + MIN_ARM_SLACK:
        return None

    psi = draw_feature_map(dim, K, rng, gamma, feat_q)
    beta_t, P_t = fit_arm(psi, data["X"][Dt], data["Y"][Dt])
    beta_c, P_c = fit_arm(psi, data["X"][~Dt], data["Y"][~Dt])

    Psi_s = psi(X_sobol)
    h_hat = Psi_s @ (beta_t - beta_c)
    M = X_sobol.shape[0]

    h_rmse = float(np.sqrt(np.mean((h_hat - tau_sobol) ** 2)))
    W_hat = float(np.maximum(h_hat, 0.0).mean())
    V_hat = float((h_hat >= 0.0).mean())

    pos = h_hat >= 0.0
    bun_W = Psi_s[pos].sum(axis=0) / M if pos.any() else np.zeros(Psi_s.shape[1])
    var_W = float(bun_W @ P_t @ bun_W + bun_W @ P_c @ bun_W)

    eps = IOTA * float(h_hat.std())
    band = np.abs(h_hat) < eps
    n_band = int(band.sum())
    if n_band > 0 and eps > 0:
        bun_V = Psi_s[band].sum(axis=0) / (2.0 * eps * M)
        var_V = float(bun_V @ P_t @ bun_V + bun_V @ P_c @ bun_V)
    else:
        var_V = np.nan

    return {
        "label": label, "n": n, "K": K, "dim": dim,
        "W_hat": W_hat, "W_se": float(np.sqrt(max(var_W, 0.0))),
        "V_hat": V_hat,
        "V_se": float(np.sqrt(max(var_V, 0.0))) if np.isfinite(var_V) else np.nan,
        "h_rmse": h_rmse, "eps_band": eps, "n_band": n_band,
    }


# =====================================================================
# Driver
# =====================================================================
def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    truths: dict[tuple[str, int], tuple[float, float]] = {}
    sobols: dict[int, np.ndarray] = {}
    rows = []

    for (label, kind, dim, feat_q, gamma, n, K) in EXPERIMENTS:
        dgp = make_dgp(kind, dim)
        if (kind, dim) not in truths:
            truths[(kind, dim)] = compute_truth(dgp, dim)
        if dim not in sobols:
            sobols[dim] = Sobol(d=dim, scramble=False).random(M_SOBOL)
        W_true, V_true = truths[(kind, dim)]
        X_sobol = sobols[dim]
        tau_sobol = dgp["tau"](X_sobol)

        for rep in range(REPS):
            rng = np.random.default_rng(
                SEED + 1000003 * rep + hash(label) % 99991)
            res = one_draw(label, dgp, dim, feat_q, gamma, n, K,
                           X_sobol, tau_sobol, rng)
            if res is None:
                continue
            res.update({"rep": rep, "W_true": W_true, "V_true": V_true,
                        "dgp": kind, "feat_q": -1 if feat_q is None else feat_q,
                        "gamma": gamma})
            rows.append(res)
            if len(rows) % PROGRESS_EVERY == 0:
                print(f"  ... {len(rows)} draws ({time.time() - t0:.0f}s)  [{label}]")

    draws = pd.DataFrame(rows)
    draws["W_dev"] = draws["W_hat"] - draws["W_true"]
    draws["V_dev"] = draws["V_hat"] - draws["V_true"]
    draws["n_varV"] = draws["n"] * draws["V_se"] ** 2
    draws["W_cover"] = (np.abs(draws["W_dev"]) <= Z975 * draws["W_se"]).astype(float)
    draws["V_cover"] = (np.abs(draws["V_dev"]) <= Z975 * draws["V_se"]).astype(float)

    g = draws.groupby("label", sort=False)
    summary = pd.DataFrame({
        "dgp": g["dgp"].first(), "dim": g["dim"].first(),
        "feat_q": g["feat_q"].first(), "gamma": g["gamma"].first(),
        "n": g["n"].first(), "K": g["K"].first(),
        "h_rmse": g["h_rmse"].mean(),
        "V_true": g["V_true"].first(),
        "V_bias": g["V_dev"].mean(), "V_sd": g["V_hat"].std(),
        "V_se_mean": g["V_se"].mean(), "V_cover": g["V_cover"].mean(),
        "n_varV": g["n_varV"].mean(),
        "n_band": g["n_band"].mean(),
        "W_bias": g["W_dev"].mean(), "W_cover": g["W_cover"].mean(),
        "draws": g.size(),
    }).reset_index()
    return summary, draws


def write_outputs(summary: pd.DataFrame, draws: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"rf_sieve_v_explore_rep{REPS}_M{M_SOBOL}"
    summary.to_csv(OUTPUT_DIR / f"{stem}_summary.csv", index=False)
    draws.to_csv(OUTPUT_DIR / f"{stem}_draws.csv", index=False)
    with open(OUTPUT_DIR / f"{stem}_results.md", "w", encoding="utf-8") as f:
        f.write("# V-functional inference in d_x = 10 and 50 (RF linear sieve)\n\n")
        f.write(f"- reps = {REPS}, M_sobol = {M_SOBOL}, iota = {IOTA}, "
                f"tau_scale = {TAU_SCALE}, activation = {ACTIVATION}\n")
        f.write("- feat_q = -1 denotes dense sphere features; q >= 1 denotes "
                "q-sparse feature supports\n\n")
        f.write("```\n" + summary.to_string(index=False) + "\n```\n")
    print(f"\nWrote outputs to {OUTPUT_DIR} (stem: {stem})")


if __name__ == "__main__":
    summary_df, draws_df = run()
    pd.set_option("display.width", 250)
    print()
    print(summary_df.to_string(index=False))
    write_outputs(summary_df, draws_df)
